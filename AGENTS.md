# AGENTS.md

Notes for coding agents (Claude Code, Codex, …) working on this repository.

## What this is

ColdCap, a submission to the BottleCapAI ThinkingCap Test: a LoRA for Qwen3-0.6B that thinks
shorter at the same accuracy. Task and rules: `TASK.md`. Method: `IDEA.md`. Overview: `README.md`.

## Hard rules (from the task — breaking them invalidates the submission)

- Never train on the evaluation benchmarks (SVAMP, BFCL, MBPP+, IFEval) or anything derived
  from them. Training data comes from `data/train.jsonl` (`utils/train/prepare_data.py`).
- Do not change evaluation settings: prompts, sampling (`utils/eval/recipe.py`), token caps,
  `PROMPT_BUDGET`, seeds. Precision (`--dtype`) may differ only if the baseline uses the same.
- The model must stay ~0.6B, think before answering, and load in stock Transformers and vLLM
  (no custom model code).
- Shortening must live in the weights, not in a prompt.

## Layout

- `coldcap/porridge.py` — pure Python, no torch: temperature filter, rewards, memory, controller.
  Keep it that way so `tests/test_porridge.py` runs anywhere.
- `coldcap/rollout.py` — one vLLM worker process per GPU, pinned via `CUDA_VISIBLE_DEVICES`
  (`EnginePool` splits requests between them); HF fallback.
- `coldcap/policy.py` — LoRA policy and loss; logits are never materialized whole (T4 memory).
- `coldcap/hw.py` — per-device capabilities. GPU 0 and GPU 1 may be different cards: never assume
  they match; each side picks its own dtype.
- `train_coldcap.py` — training loop; state in `<out>/state.json`, `trainer.pt`, `latest/`.
- `eval_pair.py` — baseline cached per (GPU, dtype); always compare on the same setup.
- `evaluate.py`, `compare.py`, `utils/` — provided by BottleCapAI; changes kept minimal and noted
  in their headers and in `NOTICE`.
- `kaggle/` — `kernel.py` (script kernel), `push.py` (pushes it), `coldcap.ipynb` (interactive).

## Commands

```bash
python -m unittest discover tests            # no GPU needed; run before every commit
python utils/train/prepare_data.py && python utils/eval/benchmarks.py
python train_coldcap.py --out outputs/run1 [--resume]
CUDA_VISIBLE_DEVICES=0 python eval_pair.py --adapter <ckpt> --out runs/<name> [--smoke]
python kaggle/push.py smoke|train|eval [...]  # Kaggle CLI: `uvx --with six kaggle` works locally
```

The Kaggle kernel clones the repo from GitHub at the pushed commit — push before `kaggle/push.py`.

## Conventions

- New files start with `# Copyright 2026 Viktor Brunclík` and `# SPDX-License-Identifier: Apache-2.0`.
  Modified BottleCapAI files keep their copyright and get a "Modified … by Viktor Brunclík" line.
  Do not edit `LICENSE`; list attributions in `NOTICE`.
- Code, comments, docs for the submission (`README.md`, `IDEA.md`) in English. Talk to the
  author in Czech.
- Never commit secrets (Kaggle token lives in `~/.kaggle/access_token`), data (`data/`,
  `questions/`), runs (`runs/`, `outputs/`).
- Commit messages: imperative summary line, a short body with the why.
