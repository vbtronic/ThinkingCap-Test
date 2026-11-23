# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""Rollout and trainer throughput on this machine, before committing GPU hours to a run.

    python kaggle/bench.py --out bench.json [--backends TRITON_ATTN,FLASHINFER] [--max-tokens 1024]

For each vLLM attention backend: start the rollout engine on --rollout-gpu, generate the same
training prompts with the base model and with a LoRA, and report tokens per second. Then time
one trainer step on --train-gpu with real generated answers. A backend that fails to start is
recorded with its error and skipped, so one run answers every question.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_coldcap import BASE_MODEL, BASE_REVISION, PROMPT_BUDGET, read_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="bench.json")
    p.add_argument("--backends", default="TRITON_ATTN,FLASHINFER,FLEX_ATTENTION")
    p.add_argument("--train-gpu", type=int, default=0)
    p.add_argument("--rollout-gpu", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--sizes", default="64,192", help="Sequences per generate() call")
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--train-sequences", type=int, default=24)
    arguments = p.parse_args()

    import torch
    from transformers import AutoTokenizer
    from coldcap.policy import Trainer, load_policy
    from coldcap.rollout import VLLMEngine

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    rows = read_jsonl(ROOT / "data/train.jsonl")
    random.Random(0).shuffle(rows)
    prompts = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=True)
        if len(ids) <= PROMPT_BUDGET:
            prompts.append(ids)
        if len(prompts) >= 64:
            break

    report = {"max_tokens": arguments.max_tokens, "group": arguments.group, "rollout": {}}
    out = Path(arguments.out)

    def save():
        out.write_text(json.dumps(report, indent=2))

    # A fresh LoRA (B = 0 at init still runs every LoRA kernel) and the trainer on its card.
    model, train_dtype = load_policy(BASE_MODEL, BASE_REVISION, arguments.train_gpu, "auto", 16)
    adapter = out.parent / "bench-adapter"
    model.save_pretrained(adapter)

    answers = []
    for backend in arguments.backends.split(","):
        os.environ["VLLM_ATTENTION_BACKEND"] = backend
        entry = report["rollout"][backend] = {}
        started = time.time()
        try:
            engine = VLLMEngine(arguments.rollout_gpu, BASE_MODEL, BASE_REVISION, eos_ids,
                                max_model_len=PROMPT_BUDGET + arguments.max_tokens, lora_rank=16)
        except Exception:
            entry["error"] = traceback.format_exc()[-2000:]
            print(f"{backend}: failed to start", flush=True)
            save()
            continue
        entry["startup_s"] = round(time.time() - started, 1)
        entry["info"] = engine.info
        for size in map(int, arguments.sizes.split(",")):
            for name, lora in (("base", None), ("lora", str(adapter))):
                requests = [{"ids": prompts[(i // arguments.group) % len(prompts)],
                             "max_tokens": arguments.max_tokens, "seed": i} for i in range(size)]
                started = time.time()
                try:
                    results = engine.generate(requests, lora)
                except Exception:
                    entry[f"{name}-{size}"] = {"error": traceback.format_exc()[-2000:]}
                    continue
                seconds = time.time() - started
                tokens = sum(len(r["tokens"]) for r in results)
                entry[f"{name}-{size}"] = {"seconds": round(seconds, 1), "tokens": tokens,
                                           "tok_per_s": round(tokens / seconds),
                                           "truncated": sum(not r["finished"] for r in results)}
                print(f"{backend} {name} x{size}: {tokens / seconds:.0f} tok/s "
                      f"({seconds:.0f}s)", flush=True)
                if not answers:
                    answers = [(q["ids"], r["tokens"]) for q, r in zip(requests, results)]
        engine.close()
        save()

    if answers:
        trainer = Trainer(model, train_dtype)
        batch = [{"prompt": q, "completion": a, "advantage": random.choice((-1.0, 1.0))}
                 for q, a in answers[:arguments.train_sequences]]
        for attempt in range(2):     # the first one includes warm-up
            torch.cuda.synchronize(arguments.train_gpu)
            started = time.time()
            update = trainer.step(batch)
            torch.cuda.synchronize(arguments.train_gpu)
            seconds = time.time() - started
            report[f"train_step_{attempt}"] = {
                "seconds": round(seconds, 1), "tokens": update["trained_tokens"],
                "tok_per_s": round(update["trained_tokens"] / seconds),
                "peak_gb": round(torch.cuda.max_memory_allocated(arguments.train_gpu) / 2**30, 2)}
            print(f"train step: {report[f'train_step_{attempt}']}", flush=True)
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
