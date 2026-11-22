# ColdCap — train only on cold porridge

## Motivation

A reasoning model wastes tokens in two very different situations:

- on questions it **cannot** reliably solve, where the long thinking is the search for the
  answer — cutting it cuts the finding;
- on questions it **reliably** solves, where some answers go straight to the point and others
  wander for hundreds of tokens before saying the same thing.

Only the second kind of length is pure waste. So instead of penalising length everywhere and
hoping accuracy survives, ColdCap shortens the model **only where shortening is safe**, and
measures as it goes whether it still is.

The idea is carried over from my earlier pretraining test ([NoCap-Test, Smart
Filtering](https://github.com/vbtronic/NoCap-Test/blob/main/IDEA.md)): there a *Goldilocks*
filter trained only on sequences that were neither too easy nor too hard, and the main lesson
was that savings only come from compute that is never spent (masking a token still pays for its
forward and backward). Here the porridge is turned around: we do **not** want the middle.

**Why "ColdCap".** In *Goldilocks and the Three Bears* the girl tastes three bowls of porridge —
too hot, too cold, and *just right* — and picks the middle one. A Goldilocks filter does the same
with training data. ColdCap deliberately picks the **cold** bowl instead: not the *hot* questions
the model is still struggling with (where shortening hurts), and not the *icy* ones that are
already as short as they get (nothing to learn) — only the cold ones, solved reliably but with
thinking to spare. The *Cap* is a nod to the test's family of names (ThinkingCap, BottleCap, NoCap).

## Method

GRPO with LoRA on Qwen3-0.6B, thinking on, sampling exactly as in the evaluation
(T = 0.6, top-p 0.95, top-k 20). Training data: the four suggested sets (GSM8K, Skywork-OR1
easy, CommonsenseQA, ARC Easy); none of the evaluation benchmarks.

### 1. Porridge temperature of a prompt

For every prompt, G = 6 answers are generated and graded:

| temperature | rule | what we do |
|---|---|---|
| **hot** | pass rate < 0.75 (hard *and* middle) | skip — shortening here costs accuracy |
| **icy** | all correct and lengths within 10 % of each other | skip — group advantages ≈ 0, no signal |
| **cold** | reliably correct, but some answers ramble | **train** |

### 2. Reward relative to the shortest correct answer

- correct: `1 + α · L_ref / L`, where `L_ref` is the shortest correct answer ever seen for this
  prompt (in this group or before) — the best answer earns `1 + α`;
- wrong or cut off by the cap: `−β`.

Advantages are group-relative (GRPO), loss is token-normalised, one on-policy update per step.

### 3. Early ejection — compute never spent

- **before generation**: prompts already known to be hot or icy are skipped (re-tested with
  10 % probability, since the model changes);
- **during generation**: a known prompt may generate at most `2 × L_min` tokens — a longer answer
  could not earn a positive advantage anyway;
- **before backward**: sequences with zero advantage are dropped.

### 4. Adaptive controller

Every 20 steps the same 200 validation questions (held out from the training sets, fixed seeds)
are answered by the current policy on the same rollout GPU on which the base model was measured.
The controller compares with the target ratio of the test — at most **0.2 accuracy points per
1 % of tokens saved** (+ a noise margin):

- within budget → increase `α` (push for shorter), relax `β`;
- over budget → decrease `α`, increase `β` (wrong answers cost more).

Each validation saves a checkpoint, so one run yields the whole length/accuracy curve.

### 5. Hardware-aware measurement

Training runs on one GPU and vLLM rollouts on another; the two may be different cards and pick
their precision independently (bf16 where supported, fp16 on a T4). Because the same seed gives
different text on different cards/dtypes, every evaluation records its card and dtype, and
`eval_pair.py` always scores the base model on the **same setup** as the candidate (cached per
setup). Comparing across setups is still allowed and labelled; comparing the base model with
itself on two setups is an A/A calibration of the hardware effect.

## Results

*To be filled in after the runs.*

| checkpoint | tokens saved | accuracy change | notes |
|---|---|---|---|
| | | | |

## What worked, what didn't

*To be filled in.*

## Hardware

Kaggle free tier, 2 × NVIDIA T4 (16 GB): GPU 0 trains (fp16 + fp32 LoRA), GPU 1 generates
with vLLM (fp16). Evaluation on one T4 in fp16, baseline scored the same way.

## How to run

```bash
python utils/train/prepare_data.py
python utils/eval/benchmarks.py
python train_coldcap.py --out outputs/run1                    # --resume after a disconnect
CUDA_VISIBLE_DEVICES=0 python eval_pair.py --adapter outputs/run1/ckpt-00200 --out runs/run1
```

Kaggle: `kaggle/coldcap.ipynb`.
