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
(T = 0.6, top-p 0.95, top-k 20). Training data: three of the four suggested sets (GSM8K,
CommonsenseQA, ARC Easy); none of the evaluation benchmarks. Skywork-OR1 is left out: for a
0.6B model it is hot porridge almost everywhere (0 of 10 validation problems solved in the
first test run), so it would only burn full-length generations — early ejection at the level
of a whole dataset.

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

Advantages are group-relative but **not divided by the group's std** (as in Dr. GRPO), loss is
token-normalised, one on-policy update per step, with a KL anchor to the base model
(`--kl 0.04`, the LoRA switched off for the reference pass, no second copy of the model).
Dividing by the std was the root cause of the first two collapses: in a group where every
answer is right, a 10 % spread in length gets the same ±1 push as a 3× spread, and `α`
cancels out — the controller could not slow the length pressure at all.

Two guards, added after the first full run collapsed (see Results):

- **a correct answer must reason**: during training, a reply counts as correct only if its
  `<think>` block holds real reasoning (≥ 8 words besides any "Final answer" line). The
  evaluation does not check this, but the rules require the model to think before answering —
  and without the guard, thinking nothing is the cheapest way to collect the length reward;
- **a floor**: `L_ref` never goes below 25 % of the shortest correct answer first seen for the
  prompt, so once an answer is that tight, being even shorter earns nothing.

### 3. Early ejection — compute never spent

- **before generation**: prompts already known to be hot or icy are skipped (re-tested with
  10 % probability, since the model changes); an unknown prompt gets the evaluation's own
  4096-token cap, so a long but correct answer is never mistaken for a wrong one;
- **during generation**: a known prompt may generate at most `2 × L_min` tokens — a longer answer
  could not earn a positive advantage anyway — but never less than `2 ×` the floor, or a lucky
  short answer would ratchet the cap down until sane answers are cut off and count as wrong;
- **before backward**: sequences with zero advantage are dropped.

Cold prompts are also **replayed**: half of every generation pool returns to prompts already
known to be cold (least recently seen first). They are capped at `2 × L_min`, so they are
cheap, and almost every one of them yields a useful group. A prompt keeps coming back until
the model answers it tightly every time — it turns icy and drops out on its own. The other
half explores prompts not seen yet, to find new cold ones.

### 4. Adaptive controller

Every 10 steps the same 150 validation questions (held out from the training sets, fixed seeds)
are answered by the current policy on the same rollout GPU on which the base model was measured.
The controller compares with the target ratio of the test — at most **0.2 accuracy points per
1 % of tokens saved** (+ a noise margin):

- within budget → increase `α` (push for shorter), relax `β`;
- over budget → decrease `α`, increase `β` (wrong answers cost more);
- replies that stop reasoning (under 95 % of the base model's share of real `<think>` blocks)
  count as over budget whatever the accuracy: the linear budget alone allowed a model that
  saved 99 % of tokens by not thinking and lost 18 points.

An over-budget validation also **rolls training back** to the last checkpoint inside the budget
(LoRA weights restored, optimizer moments reset): lowering `α` does not undo a collapse that has
already happened. After three rollbacks training stops; the last good checkpoint is the result.

Each validation saves a checkpoint, so one run yields the whole length/accuracy curve.

### 5. Hardware-aware measurement

Training runs on one GPU and vLLM rollouts on another; the two may be different cards and pick
their precision independently (bf16 where supported, fp16 on a T4). Because the same seed gives
different text on different cards/dtypes, every evaluation records its card and dtype, and
`eval_pair.py` always scores the base model on the **same setup** as the candidate (cached per
setup). Comparing across setups is still allowed and labelled; comparing the base model with
itself on two setups is an A/A calibration of the hardware effect.

## Results

Validation (150 held-out training-style questions × 2 samples, same rollout GPU and seeds as
the base model). Final benchmark numbers follow once the full evaluation is done.

**Run 1** (no guards, std-scaled advantages; base 0.630 accuracy, 1072 tokens):

| checkpoint | tokens saved | accuracy change | notes |
|---|---|---|---|
| ckpt-10 | 51 % | +11 pp | |
| ckpt-20 | 73 % | +7 pp | full benchmark run done, baseline still missing |
| ckpt-30 | 85 % | +3 pp | 30-question benchmark smoke: 66 % saved, −9 pp |
| ckpt-40 → 400 | ~98 % | −18 pp | collapse: `<think>Final answer: X</think>` |

**Run 2** (thinking guard, floor, controller thinking rule; still std-scaled; base 0.613, 1075):

| checkpoint | tokens saved | accuracy change | notes |
|---|---|---|---|
| ckpt-10 | 43 % | +11 pp | |
| ckpt-20 | 68 % | +11 pp | |
| ckpt-30 | 83 % | +7 pp | |
| ckpt-35 | 88 % | +1 pp | |
| ckpt-40 → 51 | 95–97 % | −12 to −25 pp | collapse again, controller reacted too late |

## What worked, what didn't

- **What the LoRA is.** A behavioural task vector (task arithmetic, Ilharco et al. 2022): a
  low-rank weight delta that changes *how long* the model thinks, not *what* it knows. The
  model can already answer tersely; RL only teaches it to pick that path, which carries little
  information — hence a rank-16 LoRA moves thinking length by half within 10–20 steps.

- **Worked:** cold-porridge training finds the waste fast. Within 10–20 steps the model thinks
  half as long and is *more* accurate on held-out questions of the training kind, likely
  because rambling answers that run into the cap stop being lost.
- **Did not work: guards on the reward alone.** Requiring real reasoning in `<think>` and a
  length floor only delayed the collapse by a few steps. Two things the guards could not reach
  drove it: std-scaled advantages (full length pressure on every group, `α` irrelevant) and no
  anchor to the base model. As the model got shorter, more prompts turned hot and were skipped
  (1138 hot vs 20 cold at the end of run 2), so the last steps trained on ~20 easy prompts
  with nothing holding the rest of the behaviour in place. Fixes now in the code: unscaled
  advantages, KL anchor, floor-aware cap, rollback to the last good checkpoint.
- **Validation ≠ benchmarks.** The validation questions are the training kind (maths, multiple
  choice); the benchmarks add code, function calls and instruction following. On a small
  benchmark smoke, run 1's ckpt-30 lost 9 points while validation said +3: the shortening
  transfers, the accuracy does not always. Earlier checkpoints are the safer picks.
- **Evaluation is the real cost on a T4.** With Triton attention (FlashInfer fails on the T4)
  a full 8-sample run of one LoRA took 10.9 h, and the base model, which thinks longer, would
  need more than a 12 h Kaggle session. `evaluate.py --resume` now continues a run cut off by
  a session limit from its last whole batch, with identical results.

## Hardware

Kaggle free tier, 2 × NVIDIA T4 (16 GB). Generation is the bottleneck: a benchmark run
(`kaggle/bench.py`) measured ~460 generated tokens/s per T4 with the LoRA (vLLM, Triton
attention; FlashInfer fails on the T4, Flex attention is 2× slower), while one training
step on ~24 k tokens takes ~16 s. So both GPUs generate with vLLM (fp16), and GPU 0 also
trains (fp16 + fp32 LoRA) in the memory vLLM leaves free. Evaluation on one T4 in fp16 with
Triton attention, baseline scored the same way.

## How to run

```bash
python utils/train/prepare_data.py
python utils/eval/benchmarks.py
python train_coldcap.py --out outputs/run1                    # --resume after a disconnect
CUDA_VISIBLE_DEVICES=0 python eval_pair.py --adapter outputs/run1/ckpt-00200 --out runs/run1
```

Kaggle: `kaggle/coldcap.ipynb`.
