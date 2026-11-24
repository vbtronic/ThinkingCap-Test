#!/usr/bin/env python3
# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""ColdCap: teach Qwen3-0.6B to think shorter by GRPO on cold-porridge prompts only.

    python utils/train/prepare_data.py                       # once: data/train.jsonl, data/val.jsonl
    python train_coldcap.py --out outputs/run1               # trains, validates, checkpoints
    python train_coldcap.py --out outputs/run1 --resume      # after a Kaggle disconnect

Each step:
  1. draw prompts, skip the ones known to be hot or icy (early ejection before generation);
  2. generate G answers each, capped at eject_factor x the prompt's shortest correct
     answer (early ejection during generation);
  3. grade them, classify each prompt hot / cold / icy, keep only the cold groups;
  4. reward = 1 + alpha * L_ref / L for a right answer, -beta for a wrong one;
     group-relative advantages; one LoRA update.
Every --eval-every steps the validation split is answered by the base model (once) and the
current policy on the same rollout GPU, with the same seeds; the controller moves
alpha / beta to stay within 0.2 accuracy points per 1% of tokens saved, and a checkpoint
lands in <out>/ckpt-<step>/ with its numbers.

GPUs: --train-gpu and --rollout-gpu may name different cards, even different models; each
side chooses its own precision. By default every GPU generates (the training card shares
its memory with a vLLM engine); with one GPU training and generation share it.
"""

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from coldcap.porridge import (COLD, HOT, ICY, Controller, ControllerConfig, Memory,  # noqa: E402
                              PorridgeConfig, advantages, floor_length, rewards, thinks)

BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"   # the one evaluate.py pins
# Longest prompt we accept; the Skywork maths prompts are the long ones. Longer rows are
# dropped, never truncated, so a prompt is always asked whole.
PROMPT_BUDGET = 1536


def main():
    arguments = parse_arguments()
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    from coldcap import hw
    from coldcap.policy import Trainer, load_policy, restore_adapter
    from utils.train.rewards import is_task_correct

    print("GPUs:\n" + hw.report(), flush=True)
    devices = hw.inventory()
    if not devices:
        raise SystemExit("No CUDA device")
    train_gpu = arguments.train_gpu
    if arguments.rollout_gpu:
        rollout_gpus = [int(g) for g in arguments.rollout_gpu.split(",")]
    else:
        # Training takes seconds per step, generation minutes: with two cards both generate,
        # and the training card lends vLLM part of its memory.
        others = [d["index"] for d in devices if d["index"] != train_gpu]
        rollout_gpus = others + [train_gpu] if others else [train_gpu]
    backend = arguments.rollout_backend
    if backend == "auto":
        backend = "vllm" if all(devices[g]["vllm"] for g in rollout_gpus) else "hf"
    if backend == "hf":
        rollout_gpus = [train_gpu]
    print(f"train on cuda:{train_gpu} ({devices[train_gpu]['name']}), roll out on "
          + ", ".join(f"cuda:{g} ({devices[g]['name']})" for g in rollout_gpus)
          + f" with {backend}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))

    train_rows = read_jsonl(ROOT / "data/train.jsonl")
    val_rows = pick_validation(read_jsonl(ROOT / "data/val.jsonl"), arguments.val_per_source)
    if arguments.sources:
        wanted = set(arguments.sources.split(","))
        train_rows = [r for r in train_rows if r["source"] in wanted]
        val_rows = [r for r in val_rows if r["source"] in wanted]
    def tokenize(row):
        return tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                             add_generation_prompt=True, enable_thinking=True)

    def fits(rows, name):
        kept = [r for r in rows if len(tokenize(r)) <= PROMPT_BUDGET]
        if len(kept) < len(rows):
            print(f"{name}: dropped {len(rows) - len(kept)} prompts over {PROMPT_BUDGET} tokens")
        return kept

    train_rows, val_rows = fits(train_rows, "train"), fits(val_rows, "val")
    cache = {}

    def encode(row):
        key = id(row)
        if key not in cache:
            cache[key] = tokenize(row)
        return cache[key]

    val_ids = [encode(r) for r in val_rows]

    porridge = PorridgeConfig(**pick(arguments, PorridgeConfig))
    memory = Memory(porridge, seed=arguments.seed)
    controller = Controller(ControllerConfig(**pick(arguments, ControllerConfig)))
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if arguments.resume and state_path.exists() else None
    adapter_dir = out / "latest" if state else None

    # The rollout engine goes first: on a shared card vLLM must claim its share of memory
    # before the trainer fragments it.
    engine = None
    if backend == "vllm":
        from coldcap.rollout import EnginePool, VLLMEngine
        engines = []
        for gpu in rollout_gpus:
            shared = gpu == train_gpu
            memory_share = (arguments.vllm_shared_memory if shared else arguments.vllm_memory)
            engines.append(VLLMEngine(gpu, BASE_MODEL, BASE_REVISION, eos_ids,
                                      max_model_len=PROMPT_BUDGET + porridge.max_cap,
                                      lora_rank=arguments.rank, seed=arguments.seed,
                                      memory=memory_share, dtype=arguments.rollout_dtype))
        engine = engines[0] if len(engines) == 1 else EnginePool(engines)
    model, train_dtype = load_policy(BASE_MODEL, BASE_REVISION, train_gpu, arguments.train_dtype,
                                     arguments.rank, adapter=str(adapter_dir) if adapter_dir else None)
    if backend == "hf":
        from coldcap.rollout import HFEngine
        engine = HFEngine(model, tokenizer, eos_ids)
    trainer = Trainer(model, train_dtype, lr=arguments.lr, kl=arguments.kl)
    print(f"trainer dtype {train_dtype}; rollouts {engine.info}", flush=True)

    step, cursor, baseline = 0, 0, None
    order = list(range(len(train_rows)))
    random.Random(arguments.seed).shuffle(order)
    rng = random.Random(arguments.seed + 1)
    if state:
        step, cursor, baseline = state["step"], state["cursor"], state["baseline"]
        memory.load(state["memory"])
        controller.load(state["controller"])
        trainer.load(torch.load(out / "trainer.pt", map_location=f"cuda:{train_gpu}"))
        rng.setstate(tuple_state(state["rng"]))
        print(f"resumed at step {step}, alpha {controller.alpha:.3f}, beta {controller.beta:.3f}")
    if state and baseline and baseline.get("rollout") != engine.info:
        raise SystemExit(f"Baseline was measured on {baseline.get('rollout')}, rollouts now run on "
                         f"{engine.info}; validation would compare different hardware. "
                         "Resume on the same rollout card or start a new --out.")

    def grade(rows, results):
        """-> (task-correct flags, reasoned-in-<think> flags), one per result."""
        correct, reasoned = [], []
        for row, r in zip(rows, results):
            text = tokenizer.decode(r["tokens"], skip_special_tokens=True)
            correct.append(int(r["finished"] and is_task_correct(text, row["answer"], row["task"])))
            reasoned.append(int(thinks(text, porridge.min_think_words)))
        return correct, reasoned

    def validate(adapter):
        """Same questions, same seeds, same rollout card: only the weights differ."""
        requests = [{"ids": ids, "max_tokens": porridge.max_cap, "seed": 10_000 + i * 16 + s}
                    for i, ids in enumerate(val_ids) for s in range(arguments.val_samples)]
        rows = [row for row in val_rows for _ in range(arguments.val_samples)]
        results = engine.generate(requests, adapter, fixed=True)
        correct, reasoned = grade(rows, results)   # accuracy as evaluate.py would score it
        tokens = [len(r["tokens"]) for r in results]
        per_source = {}
        for row, ok, n in zip(rows, correct, tokens):
            bucket = per_source.setdefault(row["source"], {"correct": 0, "tokens": 0, "n": 0})
            bucket["correct"] += ok
            bucket["tokens"] += n
            bucket["n"] += 1
        return {"accuracy": sum(correct) / len(correct), "tokens": sum(tokens) / len(tokens),
                "thinking": sum(reasoned) / len(reasoned),
                "per_source": {k: {"accuracy": v["correct"] / v["n"], "tokens": v["tokens"] / v["n"]}
                               for k, v in per_source.items()},
                "rollout": engine.info}

    def sync():
        """Hand the current LoRA to the rollout engine through a fresh directory."""
        if backend == "hf":
            return None
        target = out / "sync" / f"step-{step}"
        model.save_pretrained(target)
        for old in (out / "sync").iterdir():
            if old != target:
                shutil.rmtree(old, ignore_errors=True)
        return str(target)

    deadline = time.time() + arguments.hours * 3600 if arguments.hours else None
    validation_seconds = 0.0
    if baseline is None:
        print("validating the base model ...", flush=True)
        deadline_started = time.time()
        if backend == "hf":
            with model.disable_adapter():
                baseline = validate(None)
        else:
            baseline = validate(None)
        validation_seconds = time.time() - deadline_started
        print(f"base: accuracy {baseline['accuracy']:.3f}, tokens {baseline['tokens']:.0f}")
        (out / "baseline.json").write_text(json.dumps(baseline, indent=2))

    log = (out / "log.jsonl").open("a")
    adapter = sync() if step else None
    while step < arguments.steps:
        started = time.time()
        groups, stats = [], {"generated": 0, "generated_tokens": 0, "skipped": 0,
                             HOT: 0, COLD: 0, ICY: 0, "generate_s": 0.0,
                             "replayed": 0}
        used = set()             # a prompt is generated at most once per step
        for _ in range(arguments.max_rounds):
            if len(groups) >= arguments.prompts_per_step:
                break
            # Part of the pool replays prompts already known to be cold (capped, cheap, nearly
            # always useful); the rest explores prompts not seen yet.
            batch = memory.replay(round(porridge.replay * arguments.pool_chunk), exclude=used)
            stats["replayed"] += len(batch)
            while len(batch) < arguments.pool_chunk:
                index = order[cursor % len(order)]
                cursor += 1
                if index in batch or index in used:
                    continue
                if memory.worth_generating(index):
                    batch.append(index)
                else:
                    stats["skipped"] += 1
            requests = [{"ids": encode(train_rows[i]), "max_tokens": memory.cap(i),
                         "seed": rng.randrange(2**31)}
                        for i in batch for _ in range(arguments.group)]
            used.update(batch)
            generating = time.time()
            results = engine.generate(requests, adapter)
            stats["generate_s"] += time.time() - generating
            stats["generated"] += len(results)
            stats["generated_tokens"] += sum(len(r["tokens"]) for r in results)
            for k, index in enumerate(batch):
                group = results[k * arguments.group:(k + 1) * arguments.group]
                prompt = requests[k * arguments.group]["ids"]
                # Training is stricter than scoring: a right answer without reasoning in
                # <think> counts as wrong, so skipping the thinking can never pay.
                correct, reasoned = grade([train_rows[index]] * len(group), group)
                correct = [c and t for c, t in zip(correct, reasoned)]
                lengths = [len(r["tokens"]) for r in group]
                kind = memory.update(index, correct, lengths, step)
                stats[kind] += 1
                if kind != COLD:     # every cold group is used, even past --prompts-per-step
                    continue
                earned = rewards(correct, lengths, controller.alpha, controller.beta,
                                 memory.get(index).l_min, floor_length(memory.get(index), porridge))
                scores = advantages(earned, scale=arguments.std)
                groups.append([{"prompt": prompt, "completion": r["tokens"], "advantage": a}
                               for r, a in zip(group, scores)])
        training = time.time()
        update = trainer.step([s for g in groups for s in g])
        stats["train_s"] = time.time() - training
        stats["generate_s"] = round(stats["generate_s"], 1)
        stats["train_s"] = round(stats["train_s"], 1)
        step += 1
        adapter = sync()
        record = {"step": step, "seconds": round(time.time() - started, 1),
                  "cold_groups": len(groups), "alpha": controller.alpha, "beta": controller.beta,
                  **stats, **update, "memory": memory.counts()}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(f"step {step}: cold {len(groups)}  hot {stats[HOT]}  icy {stats[ICY]}  "
              f"skipped {stats['skipped']}  replayed {stats['replayed']}  mean len "
              f"{stats['generated_tokens'] / max(stats['generated'], 1):.0f}  "
              f"loss {update['loss']:.4f}  {record['seconds']}s "
              f"(generate {stats['generate_s']}s, train {stats['train_s']}s)", flush=True)

        # Kaggle kills a session at 12 h and keeps no output then: stop in time, with room for
        # one more step and a validation, and leave a checkpoint to --resume from.
        out_of_time = bool(deadline) and (
            time.time() + 2 * record["seconds"] + 2 * validation_seconds > deadline)
        if out_of_time:
            print(f"stopping at step {step}: --hours {arguments.hours} nearly used", flush=True)
        if step % arguments.eval_every == 0 or step == arguments.steps or out_of_time:
            validating = time.time()
            if backend == "hf":
                current = validate(None)
            else:
                current = validate(adapter)
            validation_seconds = max(validation_seconds, time.time() - validating)
            thinking = current["thinking"] / max(baseline.get("thinking", 1.0), 1e-9)
            decision = controller.observe(step, baseline["accuracy"], baseline["tokens"],
                                          current["accuracy"], current["tokens"], thinking)
            print(f"  val: accuracy {current['accuracy']:.3f} (base {baseline['accuracy']:.3f}), "
                  f"saved {decision['saved_pct']:+.1f}%, drop {decision['drop_pp']:+.2f} pp "
                  f"(allowed {decision['allowed_pp']:.2f}), thinking {current['thinking']:.2f} "
                  f"-> alpha {controller.alpha:.3f}, "
                  f"beta {controller.beta:.3f}", flush=True)
            checkpoint = out / f"ckpt-{step:05d}"
            model.save_pretrained(checkpoint)
            (checkpoint / "metrics.json").write_text(json.dumps(
                {"validation": current, "baseline": baseline, "controller": decision}, indent=2))
            if decision["rollback_to"] is not None and not decision["stop"]:
                good = out / f"ckpt-{decision['rollback_to']:05d}"
                restore_adapter(model, good)
                trainer.reset()
                adapter = sync()
                print(f"  out of budget: back to {good.name} (rollback {controller.rollbacks} of "
                      f"{controller.config.max_rollbacks})", flush=True)
            save_state(out, model, trainer, torch, step, cursor, baseline, memory, controller, rng)
            if decision["stop"]:
                print(f"stopping at step {step}: out of budget after "
                      f"{controller.config.max_rollbacks} rollbacks; best is "
                      f"ckpt-{controller.last_good:05d}", flush=True)
                break
        if out_of_time:
            break

    log.close()
    engine.close()
    write_curve(out)


def save_state(out, model, trainer, torch, step, cursor, baseline, memory, controller, rng):
    """Everything --resume needs; written atomically so a disconnect mid-write is harmless."""
    model.save_pretrained(out / "latest.tmp")
    shutil.rmtree(out / "latest", ignore_errors=True)
    os.replace(out / "latest.tmp", out / "latest")
    torch.save(trainer.state(), out / "trainer.tmp")
    os.replace(out / "trainer.tmp", out / "trainer.pt")
    state = {"step": step, "cursor": cursor, "baseline": baseline, "memory": memory.state(),
             "controller": controller.state(), "rng": rng.getstate()}
    (out / "state.tmp").write_text(json.dumps(state))
    os.replace(out / "state.tmp", out / "state.json")


def write_curve(out):
    """One row per checkpoint: the length/accuracy trade-off the submission asks for."""
    rows = []
    for metrics in sorted(out.glob("ckpt-*/metrics.json")):
        data = json.loads(metrics.read_text())["controller"]
        rows.append({"checkpoint": metrics.parent.name, **data})
    (out / "curve.json").write_text(json.dumps(rows, indent=2))
    print("\ncheckpoint      saved   drop pp  allowed  within")
    for r in rows:
        print(f"{r['checkpoint']}  {r['saved_pct']:+6.1f}%  {r['drop_pp']:+6.2f}  "
              f"{r['allowed_pp']:6.2f}   {'yes' if r['within_budget'] else 'no'}")


def pick_validation(rows, per_source):
    taken, counts = [], {}
    for row in rows:
        if counts.get(row["source"], 0) < per_source:
            taken.append(row)
            counts[row["source"]] = counts.get(row["source"], 0) + 1
    return taken


def tuple_state(state):
    version, internal, gauss = state
    return version, tuple(internal), gauss


def pick(arguments, config_class):
    """The command-line values that belong to one config dataclass."""
    values = {}
    for f in fields(config_class):
        value = getattr(arguments, f.name, None)
        if value is not None:
            values[f.name] = value
    return values


def read_jsonl(path):
    if not path.exists():
        raise SystemExit(f"{path} is missing; run python utils/train/prepare_data.py")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    # hardware: the two sides are chosen and configured independently
    p.add_argument("--train-gpu", type=int, default=0)
    p.add_argument("--rollout-gpu", help="Comma-separated GPUs for vLLM (default: every GPU, "
                   "the training one shared)")
    p.add_argument("--rollout-backend", default="auto", choices=("auto", "vllm", "hf"))
    p.add_argument("--train-dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--rollout-dtype", default="auto", choices=("auto", "bfloat16", "float16"))
    p.add_argument("--vllm-memory", type=float, default=0.85, help="vLLM's share of a card of its own")
    p.add_argument("--vllm-shared-memory", type=float, default=0.5,
                   help="vLLM's share of the training card when it generates there too")
    # optimisation
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--hours", type=float, help="Stop cleanly (validated, checkpointed) within this many hours")
    p.add_argument("--prompts-per-step", type=int, default=16, help="Generate until at least this many cold groups")
    p.add_argument("--group", type=int, default=6, help="Rollouts per prompt (G)")
    p.add_argument("--pool-chunk", type=int, default=32, help="Prompts generated per round")
    p.add_argument("--max-rounds", type=int, default=4, help="Rounds per step to find cold prompts")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--rank", type=int, default=16)
    # An anchor to the base model: without one (runs 1 and 2) nothing held the policy back
    # once the length reward started to win. Costs one extra no-grad forward per sequence.
    p.add_argument("--kl", type=float, default=0.04, help="KL to the base model (0 = off)")
    p.add_argument("--std", action="store_true",
                   help="Divide advantages by the group std (classic GRPO; cancels alpha)")
    # Skywork is competition maths: at 0.6B it is hot porridge almost always (0 of 10 solved
    # in the smoke run), and each first visit costs a full group of maximum-length answers.
    p.add_argument("--sources", default="gsm8k,commonsense_qa,arc_easy",
                   help="Comma-separated subset of gsm8k,skywork,commonsense_qa,arc_easy")
    # validation
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--val-per-source", type=int, default=50)
    p.add_argument("--val-samples", type=int, default=2)
    # porridge (None = the dataclass default)
    p.add_argument("--cold-pass", type=float)
    p.add_argument("--icy-spread", type=float)
    p.add_argument("--recheck", type=float)
    p.add_argument("--eject-factor", type=float)
    p.add_argument("--min-cap", type=int)
    p.add_argument("--max-cap", type=int)
    p.add_argument("--replay", type=float, help="Share of each pool replaying known cold prompts")
    p.add_argument("--floor", type=float, help="No length reward below this x a prompt's first L_min")
    p.add_argument("--min-think-words", type=int, help="Words of reasoning a correct rollout needs")
    # controller
    p.add_argument("--alpha", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--budget", type=float)
    p.add_argument("--slack", type=float)
    p.add_argument("--min-thinking", type=float, help="Share of replies that must still reason (vs base)")
    p.add_argument("--max-rollbacks", type=int, help="Returns to the last good checkpoint before stopping")
    return p.parse_args()


if __name__ == "__main__":
    main()
