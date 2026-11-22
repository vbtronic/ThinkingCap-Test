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
side chooses its own precision. With one GPU both share it.
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
                              PorridgeConfig, advantages, rewards)

BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"   # the one evaluate.py pins
PROMPT_BUDGET = 1024


def main():
    arguments = parse_arguments()
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    from coldcap import hw
    from coldcap.policy import Trainer, load_policy
    from utils.train.rewards import is_task_correct

    print("GPUs:\n" + hw.report(), flush=True)
    devices = hw.inventory()
    if not devices:
        raise SystemExit("No CUDA device")
    train_gpu = arguments.train_gpu
    rollout_gpu = arguments.rollout_gpu
    if rollout_gpu is None:
        rollout_gpu = 1 if len(devices) > 1 and train_gpu == 0 else train_gpu
    backend = arguments.rollout_backend
    if backend == "auto":
        backend = "vllm" if devices[rollout_gpu]["vllm"] else "hf"
    shared = rollout_gpu == train_gpu
    print(f"train on cuda:{train_gpu} ({devices[train_gpu]['name']}), roll out on "
          f"cuda:{rollout_gpu} ({devices[rollout_gpu]['name']}) with {backend}", flush=True)

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
    encode = lambda row: tokenizer.apply_chat_template(  # noqa: E731
        row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=True)
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
        from coldcap.rollout import VLLMEngine
        engine = VLLMEngine(rollout_gpu, BASE_MODEL, BASE_REVISION, eos_ids,
                            max_model_len=PROMPT_BUDGET + porridge.max_cap,
                            lora_rank=arguments.rank, seed=arguments.seed,
                            memory=arguments.vllm_memory or (0.35 if shared else 0.85),
                            dtype=arguments.rollout_dtype)
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
        return [int(r["finished"] and is_task_correct(
                    tokenizer.decode(r["tokens"], skip_special_tokens=True),
                    row["answer"], row["task"]))
                for row, r in zip(rows, results)]

    def validate(adapter):
        """Same questions, same seeds, same rollout card: only the weights differ."""
        requests = [{"ids": ids, "max_tokens": porridge.max_cap, "seed": 10_000 + i * 16 + s}
                    for i, ids in enumerate(val_ids) for s in range(arguments.val_samples)]
        rows = [row for row in val_rows for _ in range(arguments.val_samples)]
        results = engine.generate(requests, adapter)
        correct = grade(rows, results)
        tokens = [len(r["tokens"]) for r in results]
        per_source = {}
        for row, ok, n in zip(rows, correct, tokens):
            bucket = per_source.setdefault(row["source"], {"correct": 0, "tokens": 0, "n": 0})
            bucket["correct"] += ok
            bucket["tokens"] += n
            bucket["n"] += 1
        return {"accuracy": sum(correct) / len(correct), "tokens": sum(tokens) / len(tokens),
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

    if baseline is None:
        print("validating the base model ...", flush=True)
        if backend == "hf":
            with model.disable_adapter():
                baseline = validate(None)
        else:
            baseline = validate(None)
        print(f"base: accuracy {baseline['accuracy']:.3f}, tokens {baseline['tokens']:.0f}")
        (out / "baseline.json").write_text(json.dumps(baseline, indent=2))

    log = (out / "log.jsonl").open("a")
    adapter = sync() if step else None
    while step < arguments.steps:
        started = time.time()
        groups, stats = [], {"generated": 0, "generated_tokens": 0, "skipped": 0,
                             HOT: 0, COLD: 0, ICY: 0}
        for _ in range(arguments.max_rounds):
            if len(groups) >= arguments.prompts_per_step:
                break
            batch = []
            while len(batch) < arguments.pool_chunk:
                index = order[cursor % len(order)]
                cursor += 1
                if memory.worth_generating(index):
                    batch.append(index)
                else:
                    stats["skipped"] += 1
            requests = [{"ids": encode(train_rows[i]), "max_tokens": memory.cap(i),
                         "seed": rng.randrange(2**31)}
                        for i in batch for _ in range(arguments.group)]
            results = engine.generate(requests, adapter)
            stats["generated"] += len(results)
            stats["generated_tokens"] += sum(len(r["tokens"]) for r in results)
            for k, index in enumerate(batch):
                group = results[k * arguments.group:(k + 1) * arguments.group]
                prompt = requests[k * arguments.group]["ids"]
                correct = grade([train_rows[index]] * len(group), group)
                lengths = [len(r["tokens"]) for r in group]
                kind = memory.update(index, correct, lengths, step)
                stats[kind] += 1
                if kind != COLD or len(groups) >= arguments.prompts_per_step:
                    continue
                earned = rewards(correct, lengths, controller.alpha, controller.beta,
                                 memory.get(index).l_min)
                scores = advantages(earned, scale=not arguments.no_std)
                groups.append([{"prompt": prompt, "completion": r["tokens"], "advantage": a}
                               for r, a in zip(group, scores)])
        update = trainer.step([s for g in groups for s in g])
        step += 1
        adapter = sync()
        record = {"step": step, "seconds": round(time.time() - started, 1),
                  "cold_groups": len(groups), "alpha": controller.alpha, "beta": controller.beta,
                  **stats, **update, "memory": memory.counts()}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(f"step {step}: cold {len(groups)}  hot {stats[HOT]}  icy {stats[ICY]}  "
              f"skipped {stats['skipped']}  mean len "
              f"{stats['generated_tokens'] / max(stats['generated'], 1):.0f}  "
              f"loss {update['loss']:.4f}  {record['seconds']}s", flush=True)

        if step % arguments.eval_every == 0 or step == arguments.steps:
            if backend == "hf":
                current = validate(None)
            else:
                current = validate(adapter)
            decision = controller.observe(step, baseline["accuracy"], baseline["tokens"],
                                          current["accuracy"], current["tokens"])
            print(f"  val: accuracy {current['accuracy']:.3f} (base {baseline['accuracy']:.3f}), "
                  f"saved {decision['saved_pct']:+.1f}%, drop {decision['drop_pp']:+.2f} pp "
                  f"(allowed {decision['allowed_pp']:.2f}) -> alpha {controller.alpha:.3f}, "
                  f"beta {controller.beta:.3f}", flush=True)
            checkpoint = out / f"ckpt-{step:05d}"
            model.save_pretrained(checkpoint)
            (checkpoint / "metrics.json").write_text(json.dumps(
                {"validation": current, "baseline": baseline, "controller": decision}, indent=2))
            save_state(out, model, trainer, torch, step, cursor, baseline, memory, controller, rng)

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
    p.add_argument("--rollout-gpu", type=int, help="Default: 1 if there are two GPUs, else 0")
    p.add_argument("--rollout-backend", default="auto", choices=("auto", "vllm", "hf"))
    p.add_argument("--train-dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--rollout-dtype", default="auto", choices=("auto", "bfloat16", "float16"))
    p.add_argument("--vllm-memory", type=float, help="vLLM's share of its GPU (default 0.85, 0.35 shared)")
    # optimisation
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--prompts-per-step", type=int, default=16, help="Cold groups trained per step")
    p.add_argument("--group", type=int, default=6, help="Rollouts per prompt (G)")
    p.add_argument("--pool-chunk", type=int, default=32, help="Prompts generated per round")
    p.add_argument("--max-rounds", type=int, default=4, help="Rounds per step to find cold prompts")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--kl", type=float, default=0.0, help="KL to the base model (0 = off, cheaper)")
    p.add_argument("--no-std", action="store_true", help="Centre advantages without dividing by std")
    p.add_argument("--sources", help="Comma-separated subset of gsm8k,skywork,commonsense_qa,arc_easy")
    # validation
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--val-per-source", type=int, default=50)
    p.add_argument("--val-samples", type=int, default=2)
    # porridge (None = the dataclass default)
    p.add_argument("--cold-pass", type=float)
    p.add_argument("--icy-spread", type=float)
    p.add_argument("--recheck", type=float)
    p.add_argument("--eject-factor", type=float)
    p.add_argument("--min-cap", type=int)
    p.add_argument("--max-cap", type=int)
    # controller
    p.add_argument("--alpha", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--budget", type=float)
    p.add_argument("--slack", type=float)
    return p.parse_args()


if __name__ == "__main__":
    main()
