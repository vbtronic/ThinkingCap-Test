#!/usr/bin/env python3
# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""Score a checkpoint and compare it with the base model scored on the very same setup.

    python eval_pair.py --adapter outputs/run1/best --out runs/run1           # full evaluation
    python eval_pair.py --adapter outputs/run1/best --out runs/run1 --smoke   # plumbing only
    python eval_pair.py --adapter ... --out ... --benchmarks svamp --samples 2  # quick dev loop

The problem it solves: the same seed draws different text on a T4 in fp16 than
on an L4 in bf16, so a baseline scored on one card and a checkpoint scored on
another differ partly because of the hardware. The fix is to never mix them:

  1. find which card evaluate.py will run on (the first visible GPU) and which
     dtype it resolves to there;
  2. look up the baseline cached for exactly that (card, dtype, selection,
     benchmarks, samples) under runs/base/; score it first if it is missing;
  3. score the candidate with the same flags on the same card;
  4. compare the two, paired question by question and seed by seed.

Both runs then share questions, seeds, prompts, sampling, token caps, card and
dtype -- the only thing left that differs is the weights. To measure what the
hardware alone does, compare two cached baselines (an A/A calibration):

    python compare.py runs/base/tesla-t4-float16-full runs/base/nvidia-l4-bfloat16-full

Pick the card with CUDA_VISIBLE_DEVICES; on a box whose GPU 0 and GPU 1 differ,
each card gets its own baseline.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    arguments, passthrough = parse_arguments()
    gpu, dtype = current_setup(arguments.dtype)
    print(f"Setup: {gpu}, {dtype}  (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})", flush=True)

    base = Path(arguments.base_root).resolve() / baseline_name(gpu, dtype, arguments)
    if finished(base):
        print(f"Baseline for this setup: {base} (cached)")
        check_cached(base, gpu, dtype)
    else:
        print(f"No baseline for this setup yet; scoring the base model into {base}", flush=True)
        evaluate(["--out", str(base)], dtype, passthrough)

    candidate = Path(arguments.out).resolve()
    if finished(candidate) and arguments.reuse:
        print(f"Candidate already scored: {candidate} (reused)")
    else:
        flags = ["--out", str(candidate)]
        if arguments.model:
            flags += ["--model", arguments.model]
        if arguments.adapter:
            flags += ["--adapter", str(Path(arguments.adapter).resolve())]
        evaluate(flags, dtype, passthrough)

    comparison = [sys.executable, str(ROOT / "compare.py"), str(base), str(candidate),
                  "--out", str(candidate / "comparison.json")]
    subprocess.run(comparison, check=True, cwd=ROOT)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Directory for the candidate's run")
    parser.add_argument("--adapter", help="LoRA to score on top of the base model")
    parser.add_argument("--model", help="Full weights to score instead of the base model")
    parser.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16"))
    parser.add_argument("--base-root", default="runs/base",
                        help="Where baselines are cached, one directory per setup")
    parser.add_argument("--reuse", action="store_true",
                        help="Do not re-score a candidate whose --out already has results")
    # Everything that changes the selection must reach both runs identically,
    # and it also names the cached baseline.
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmarks")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    arguments, rest = parser.parse_known_args()
    if not arguments.adapter and not arguments.model:
        parser.error("give --adapter and/or --model: the base model is the baseline")
    passthrough = rest + ["--dtype", "PLACEHOLDER"]
    for flag, value in (("--smoke", arguments.smoke), ("--benchmarks", arguments.benchmarks),
                        ("--samples", arguments.samples), ("--max-tokens", arguments.max_tokens)):
        if value is True:
            passthrough.append(flag)
        elif value not in (None, False):
            passthrough += [flag, str(value)]
    return arguments, passthrough


def current_setup(requested):
    """(card name, dtype) exactly as evaluate.py will resolve them in a child process."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device: evaluate.py needs a GPU for vLLM")
    sys.path.insert(0, str(ROOT))
    from evaluate import resolve_dtype
    return torch.cuda.get_device_name(0), resolve_dtype(requested)


def baseline_name(gpu, dtype, arguments):
    """One directory per setup: everything that changes the generations is in the name."""
    parts = [slug(gpu), dtype, "smoke" if arguments.smoke else "full"]
    if arguments.benchmarks:
        parts.append(slug(arguments.benchmarks))
    if arguments.samples:
        parts.append(f"s{arguments.samples}")
    if arguments.max_tokens:
        parts.append(f"cap{arguments.max_tokens}")
    return "-".join(parts)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def finished(run):
    return (run / "results.json").exists() and (run / "records.jsonl").exists()


def check_cached(base, gpu, dtype):
    """A cached baseline must really come from this setup; the name alone is not proof."""
    hardware = json.loads((base / "results.json").read_text()).get("hardware") or {}
    backend = os.environ.get("VLLM_ATTENTION_BACKEND", "auto")
    cached = (hardware.get("gpu"), hardware.get("dtype"), hardware.get("attention_backend", "auto"))
    if cached != (gpu, dtype, backend):
        raise SystemExit(f"{base} was scored on {' '.join(map(str, cached))}, not "
                         f"{gpu} {dtype} {backend}; move it away and rerun")


def evaluate(flags, dtype, passthrough):
    command = [sys.executable, str(ROOT / "evaluate.py")] + flags + [
        dtype if item == "PLACEHOLDER" else item for item in passthrough]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
