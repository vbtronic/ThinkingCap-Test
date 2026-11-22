# ColdCap — shorter thinking for Qwen3-0.6B

A submission to the [BottleCapAI ThinkingCap Test](https://github.com/BottleCapAI/ThinkingCap-Test):
make **Qwen3-0.6B** generate fewer tokens (thinking included) while keeping its accuracy.
The task, rules and evaluation are described in [TASK.md](TASK.md) (the original README).

**ColdCap** trains a LoRA with GRPO, but only on *cold porridge*: questions the model already
solves reliably yet sometimes answers with far more thinking than it needs. Hard and medium
questions (*hot*) and questions with nothing left to cut (*icy*) are skipped before any compute
is spent on them. Each answer is rewarded relative to the shortest correct answer seen for its
question, and a controller tunes the length pressure to stay within the test's budget of
0.2 accuracy points per 1 % of tokens saved. Full description: [IDEA.md](IDEA.md).

## Repository layout

| path | what |
|---|---|
| [IDEA.md](IDEA.md) | motivation, method, results |
| [TASK.md](TASK.md) | the original task description and rules |
| `coldcap/porridge.py` | the algorithm: hot / cold / icy filter, rewards, early ejection, controller |
| `coldcap/rollout.py` | answer generation: vLLM in its own process on its own GPU, HF fallback |
| `coldcap/policy.py` | Qwen3-0.6B + LoRA, memory-light GRPO loss |
| `coldcap/hw.py` | per-GPU capabilities (bf16, vLLM); GPUs may differ |
| `train_coldcap.py` | the training loop, checkpoints, `--resume` |
| `eval_pair.py` | scores the base model and a checkpoint on the same GPU and dtype, then compares |
| `evaluate.py`, `compare.py`, `utils/` | the provided evaluation (records GPU/dtype; compares across hardware) |
| `kaggle/` | Kaggle notebook, script kernel and push helper |
| `tests/` | unit tests (run without a GPU) |
| [AGENTS.md](AGENTS.md) | notes for coding agents working on this repo |

## Quick start

Needs one or two CUDA GPUs (a free Kaggle T4 x2 is enough).

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

python utils/train/prepare_data.py     # training data: GSM8K, Skywork easy, CommonsenseQA, ARC Easy
python utils/eval/benchmarks.py        # evaluation questions

python train_coldcap.py --out outputs/run1                  # GPU 0 trains, GPU 1 generates
python train_coldcap.py --out outputs/run1 --resume         # continue after a disconnect

CUDA_VISIBLE_DEVICES=0 python eval_pair.py --adapter outputs/run1/ckpt-00200 --out runs/run1 --smoke
CUDA_VISIBLE_DEVICES=0 python eval_pair.py --adapter outputs/run1/ckpt-00200 --out runs/run1
```

Each validation writes `outputs/run1/ckpt-XXXXX/` (LoRA in `.safetensors` + `metrics.json`);
`outputs/run1/curve.json` is the length/accuracy curve over the run.

### On Kaggle

```bash
python kaggle/push.py smoke                          # checks every moving part in a few minutes
python kaggle/push.py train --extra "--steps 200"
python kaggle/push.py eval --checkpoint outputs/run1/ckpt-00200 --source viktorbrunclk/coldcap-train
```

or open `kaggle/coldcap.ipynb` with *Accelerator: GPU T4 x2* and *Internet: On*.

## Fair comparison across GPUs

The same seed produces different text on a T4 in fp16 and an L4 in bf16. `eval_pair.py`
therefore always scores the base model on the candidate's own GPU and dtype (cached per setup in
`runs/base/<gpu>-<dtype>-…`). `compare.py` still compares runs from different setups and says so;
the same checkpoint on two setups is reported as an A/A calibration of the hardware effect.

## Tests

```bash
python -m unittest discover tests
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The original evaluation and
helpers are © 2026 BottleCapAI; ColdCap is © 2026 Viktor Brunclík. Benchmarks belong to their
authors (see [TASK.md](TASK.md#sources-and-licenses)).
