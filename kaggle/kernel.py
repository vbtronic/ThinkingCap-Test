# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""Kaggle script kernel: clone the repo, install, train ColdCap, evaluate.

Pushed by kaggle/push.py, which fills in MODE and COMMIT below. Everything worth keeping
lands in /kaggle/working (the kernel's Output); the clone, data and caches stay in /tmp.

    smoke   a few real-sized steps: checks every moving part and times a step
    train   the real training run (resumes from RESUME_FROM if given)
    eval    scores CHECKPOINT and the base model on the same card (eval_pair.py)
    bench   rollout tokens/s per attention backend, base vs LoRA, and one trainer step
"""

import os
from pathlib import Path
import subprocess
import sys
import time

MODE = "smoke"
COMMIT = "main"
REPO = "https://github.com/vbtronic/ThinkingCap-Test.git"
EXTRA = ""            # extra flags for train_coldcap.py / eval_pair.py
RESUME_FROM = ""      # e.g. /kaggle/input/<kernel-output>/outputs/run1
CHECKPOINT = ""       # for MODE=eval, relative to /kaggle/working or absolute

# vLLM on a T4 picks FlashInfer attention, which JIT-compiles CUDA code at start-up and fails
# to link on Kaggle ("cannot find -lcuda"). Triton attention needs no compiler. vLLM 0.23 no
# longer reads the variable, so coldcap.hw.vllm_attention() passes it to every LLM(...); the
# stub directory is a last resort. It applies to training *and* evaluation, so the baseline
# and the candidate run the same kernels.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# FlashInfer's JIT links with -lcuda, but the image only has the driver's libcuda.so.1 (no
# unversioned name). A symlink in a directory on LIBRARY_PATH makes FlashInfer buildable, so
# MODE=bench can compare it with Triton.
LINKS = Path("/tmp/cudalink")
for found in ("/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/local/nvidia/lib64/libcuda.so.1",
              "/usr/lib64/libcuda.so.1", "/usr/local/cuda/lib64/stubs/libcuda.so"):
    if os.path.exists(found):
        LINKS.mkdir(parents=True, exist_ok=True)
        if not (LINKS / "libcuda.so").exists():
            (LINKS / "libcuda.so").symlink_to(found)
        os.environ["LIBRARY_PATH"] = ":".join(filter(None, [os.environ.get("LIBRARY_PATH"),
                                                            str(LINKS)]))
        print("libcuda for linking:", found, flush=True)
        break

WORK = Path("/kaggle/working")
SRC = Path("/tmp/coldcap")
LOGS = WORK / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
failures = []


def sh(name, command, env=None, required=True):
    """Run one stage, tee its output to logs/<name>.txt, keep going unless required."""
    print(f"\n===== {name}: {command}", flush=True)
    started = time.time()
    with open(LOGS / f"{name}.txt", "w") as log:
        process = subprocess.Popen(command, shell=True, cwd=SRC if SRC.exists() else None,
                                   env={**os.environ, **(env or {})}, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    print(f"===== {name}: exit {code} after {time.time() - started:.0f}s", flush=True)
    if code:
        failures.append(name)
        if required:
            finish()


def finish():
    (WORK / "STATUS.txt").write_text("FAILED: " + ", ".join(failures) if failures else "OK")
    print("\nSTATUS:", (WORK / "STATUS.txt").read_text(), flush=True)
    sys.exit(1 if failures else 0)


sh("gpus", "nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv")
sh("clone", f"git clone -q {REPO} {SRC} && cd {SRC} && git checkout -q {COMMIT} && git log --oneline -1")
sh("install", f"{sys.executable} -m pip install -q uv && "
              f"uv pip install --system -q -r requirements.txt && "
              f"{sys.executable} -c 'import torch, vllm, transformers, peft; "
              f"print(torch.__version__, vllm.__version__, transformers.__version__, peft.__version__)'")
sh("data", f"{sys.executable} utils/train/prepare_data.py && {sys.executable} utils/eval/benchmarks.py")
sh("hw", f"{sys.executable} -m coldcap.hw")

out = WORK / "outputs" / "run1"
if RESUME_FROM and not out.exists():
    sh("resume-copy", f"mkdir -p {out.parent} && cp -r {RESUME_FROM} {out}")

if MODE == "smoke":
    # Real group size and pool, few steps: every part working, and how long a step takes.
    # The evaluation pipeline has its own mode (eval).
    sh("train", f"{sys.executable} train_coldcap.py --out {out} --resume --steps 3 --eval-every 3 "
                f"--prompts-per-step 8 --group 6 --pool-chunk 16 --max-rounds 2 "
                f"--val-per-source 10 --val-samples 1 {EXTRA}")
elif MODE == "train":
    sh("train", f"{sys.executable} train_coldcap.py --out {out} --resume {EXTRA}")
elif MODE == "bench":
    sh("bench", f"{sys.executable} kaggle/bench.py --out {WORK}/bench.json {EXTRA}")
elif MODE == "eval":
    checkpoint = CHECKPOINT if CHECKPOINT.startswith("/") else str(WORK / CHECKPOINT)
    sh("eval", f"{sys.executable} eval_pair.py --adapter {checkpoint} --out {WORK}/runs/eval "
               f"--base-root {WORK}/runs/base {EXTRA}", env={"CUDA_VISIBLE_DEVICES": "0"})
else:
    failures.append(f"unknown MODE {MODE}")

# Rollout sync copies are scratch; keep the output small.
sh("cleanup", f"rm -rf {out}/sync", required=False)
finish()
