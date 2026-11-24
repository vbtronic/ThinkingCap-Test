# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""Cold porridge: which prompts to train on, what each rollout earns, and how hard to push.

Pure Python on purpose -- no torch -- so every rule here is unit-tested without a GPU.

Three parts:

* Temperature of a prompt, from its group of G rollouts.
    hot   the model often fails (pass rate under `cold_pass`). This includes the middle,
          which a classic Goldilocks filter would keep. Shortening here risks accuracy:
          the model is still finding the answer, and cutting its thinking cuts the finding.
    icy   the model always succeeds and every rollout is about as long as the shortest
          one. Group-relative advantages are ~0, so it costs a backward pass and teaches
          nothing.
    cold  the model reliably succeeds but some rollouts ramble. This is where length is
          pure waste and can be cut without touching accuracy -- the only prompts we train on.

* Reward of a rollout, against the shortest correct answer seen for its prompt.
    correct   1 + alpha * L_ref / L     (the shortest correct one earns 1 + alpha)
    wrong     -beta                     (truncated counts as wrong)
  L_ref is the shortest correct length in the group or in the prompt's history, whichever
  is shorter, so a prompt keeps pulling toward its best-ever answer -- but never below
  `floor` x the shortest correct answer first seen for it: past that point being shorter
  earns nothing more. "Correct" also requires real reasoning inside <think> (`thinks`).

* Controller: after each validation it compares accuracy and tokens with the base model
  and moves alpha / beta so that training stays inside the competition's budget of
  0.2 accuracy points per 1% of tokens saved. Outside the budget it also sends training
  back to the last checkpoint inside it, and stops after `max_rollbacks` such returns.
  Advantages are not divided by the group std, so alpha really sets the length pressure.

Plus the memory that makes early ejection possible: prompts already known to be hot or icy
are skipped before any token is generated, and a known prompt's generation is capped at
`eject_factor` x its shortest correct length. Known cold prompts are also *replayed*: part of
every pool comes back to them, so most generation goes to prompts that are cheap (capped)
and almost sure to teach something, until they freeze into icy and drop out by themselves.
"""

from dataclasses import dataclass, field, asdict
import math
import random

HOT, COLD, ICY, UNKNOWN = "hot", "cold", "icy", "unknown"


@dataclass
class PorridgeConfig:
    cold_pass: float = 0.75      # pass rate at or above this is cold or icy, below is hot
    icy_spread: float = 0.10     # (max - min) / max length of correct rollouts under this -> icy
    recheck: float = 0.10        # chance to re-test a prompt known to be hot or icy
    eject_factor: float = 2.0    # cap a known prompt's generation at this x its L_min
    min_cap: int = 256           # never cap a generation below this
    max_cap: int = 4096          # and never above this: the evaluation's own cap, so a
                                 # long but correct answer is not mistaken for a wrong one
    replay: float = 0.5          # share of each pool drawn from prompts known to be cold
    floor: float = 0.25          # no length reward below this x the prompt's first L_min
    min_think_words: int = 8     # a rollout must reason in <think> to count as correct


@dataclass
class PromptMemory:
    seen: int = 0                # how many times its group was generated
    pass_rate: float | None = None   # pass rate of the last group
    l_min: int | None = None     # shortest correct length ever seen
    l_first: int | None = None   # shortest correct length the first time one was seen
    temperature: str = UNKNOWN
    step: int = -1               # training step it was last generated at


class Memory:
    """What we know about each training prompt, keyed by its row index."""

    def __init__(self, config, seed=0):
        self.config = config
        self.prompts: dict[int, PromptMemory] = {}
        self.rng = random.Random(seed)

    def get(self, index):
        return self.prompts.setdefault(index, PromptMemory())

    def worth_generating(self, index):
        """Early ejection, part 1: skip prompts known to be hot or icy, mostly."""
        known = self.prompts.get(index)
        if known is None or known.temperature in (UNKNOWN, COLD):
            return True
        return self.rng.random() < self.config.recheck

    def replay(self, count, exclude=()):
        """Up to `count` prompts known to be cold, least recently generated first."""
        cold = [(m.step, self.rng.random(), i) for i, m in self.prompts.items()
                if m.temperature == COLD and i not in exclude]
        cold.sort()
        return [i for _, _, i in cold[:count]]

    def cap(self, index):
        """Early ejection, part 2: how many tokens this prompt may generate."""
        known = self.prompts.get(index)
        if known is None or known.l_min is None:
            return self.config.max_cap
        # Never below eject_factor x the floor: once l_min shrinks under it, a tighter cap
        # would only cut off answers of a sane length and count them as wrong.
        shortest = max(known.l_min, floor_length(known, self.config) or 0)
        cap = math.ceil(self.config.eject_factor * shortest)
        return max(self.config.min_cap, min(self.config.max_cap, cap))

    def update(self, index, correct, lengths, step):
        """Record a group's outcome; returns the prompt's temperature now."""
        memory = self.get(index)
        memory.seen += 1
        memory.step = step
        memory.pass_rate = sum(correct) / len(correct)
        right = [n for ok, n in zip(correct, lengths) if ok]
        if right:
            shortest = min(right)
            memory.l_min = shortest if memory.l_min is None else min(memory.l_min, shortest)
            if memory.l_first is None:
                memory.l_first = shortest
        memory.temperature = temperature(correct, lengths, self.config)
        return memory.temperature

    def counts(self):
        tally = {HOT: 0, COLD: 0, ICY: 0, UNKNOWN: 0}
        for memory in self.prompts.values():
            tally[memory.temperature] += 1
        return tally

    def state(self):
        return {"prompts": {str(k): asdict(v) for k, v in self.prompts.items()},
                "rng": self.rng.getstate()}

    def load(self, state):
        self.prompts = {int(k): PromptMemory(**v) for k, v in state["prompts"].items()}
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))


def temperature(correct, lengths, config):
    """hot / cold / icy for one group of rollouts (see the module docstring)."""
    if not correct:
        raise ValueError("empty group")
    pass_rate = sum(correct) / len(correct)
    if pass_rate < config.cold_pass:
        return HOT
    right = [n for ok, n in zip(correct, lengths) if ok]
    if pass_rate == 1.0 and len(right) > 1:
        spread = (max(right) - min(right)) / max(right)
        if spread < config.icy_spread:
            return ICY
    return COLD


def floor_length(memory, config):
    """Below this length a prompt earns no more for being shorter (None: no floor yet)."""
    if memory.l_first is None or not config.floor:
        return None
    return math.ceil(config.floor * memory.l_first)


def thinks(text, min_words=8):
    """Did the reply reason before answering? Its <think> block, minus any line that only
    states the final answer, must hold at least `min_words` words. Without this rule the
    length reward is cheapest to collect by thinking nothing at all, which the first full
    run did after ~35 steps (<think>Final answer: 72</think>)."""
    if "</think>" not in text:
        return False
    inside = text.split("</think>", 1)[0].split("<think>")[-1]
    lines = [line for line in inside.splitlines()
             if line.strip() and not line.strip().lower().startswith("final answer")]
    return sum(len(line.split()) for line in lines) >= min_words


def rewards(correct, lengths, alpha, beta, l_history=None, l_floor=None):
    """Reward of each rollout in one group; see the module docstring."""
    right = [n for ok, n in zip(correct, lengths) if ok]
    candidates = right + ([l_history] if l_history else [])
    l_ref = min(candidates) if candidates else None
    if l_ref is not None and l_floor:
        l_ref = max(l_ref, l_floor)
    out = []
    for ok, n in zip(correct, lengths):
        if ok:
            out.append(1.0 + alpha * min(1.0, l_ref / max(n, 1)))
        else:
            out.append(-beta)
    return out


def advantages(group_rewards, scale=False, eps=1e-4):
    """GRPO: reward minus the group mean, optionally divided by the group's std.

    Off by default (as in Dr. GRPO). Dividing by the std hands every group the same push,
    however small its spread: in a group where all are right and lengths differ by 10%,
    the shortest gets +1 std all the same, and alpha cancels out -- the controller could
    not slow the length pressure, which drove both early runs into collapse."""
    mean = sum(group_rewards) / len(group_rewards)
    centred = [r - mean for r in group_rewards]
    if not scale:
        return centred
    std = math.sqrt(sum(c * c for c in centred) / len(centred))
    return [c / (std + eps) for c in centred]


@dataclass
class ControllerConfig:
    alpha: float = 0.5           # starting length weight
    beta: float = 0.0            # starting wrong-answer penalty
    alpha_bounds: tuple = (0.05, 4.0)
    beta_bounds: tuple = (0.0, 2.0)
    up: float = 1.25             # multiply alpha by this while inside the budget
    down: float = 0.6            # and by this when over it
    beta_step: float = 0.25      # added to beta when over budget, decays by half otherwise
    budget: float = 0.2          # allowed accuracy points lost per 1% of tokens saved
    slack: float = 1.0           # extra points allowed, for validation noise
    min_thinking: float = 0.95   # validation replies that still reason, relative to the base
    max_rollbacks: int = 3       # out of budget: go back to the last good checkpoint, this often


@dataclass
class Controller:
    config: ControllerConfig = field(default_factory=ControllerConfig)
    alpha: float = 0.0
    beta: float = 0.0
    history: list = field(default_factory=list)
    last_good: int | None = None     # step of the last checkpoint inside the budget
    rollbacks: int = 0

    def __post_init__(self):
        self.alpha = self.alpha or self.config.alpha
        self.beta = self.beta or self.config.beta

    def allowed_drop(self, saved_pct):
        """Accuracy points we may lose for this saving (never negative)."""
        return max(0.0, self.config.budget * saved_pct) + self.config.slack

    def observe(self, step, base_acc, base_tokens, acc, tokens, thinking=1.0):
        """Validation result in, new alpha/beta out. Accuracies are fractions (0..1);
        `thinking` is the share of replies that reason, relative to the base model's."""
        saved_pct = 100.0 * (1.0 - tokens / base_tokens)
        drop_pp = 100.0 * (base_acc - acc)
        within = drop_pp <= self.allowed_drop(saved_pct) and thinking >= self.config.min_thinking
        c = self.config
        if within:
            self.alpha = min(c.alpha_bounds[1], self.alpha * c.up)
            self.beta = max(c.beta_bounds[0], self.beta * 0.5)
        else:
            self.alpha = max(c.alpha_bounds[0], self.alpha * c.down)
            self.beta = min(c.beta_bounds[1], self.beta + c.beta_step)
        # Outside the budget the weights themselves are suspect (a collapse does not undo
        # itself when alpha drops), so training goes back to the last good checkpoint.
        rollback = None
        if within:
            self.last_good = step
        elif self.last_good is not None:
            rollback = self.last_good
            self.rollbacks += 1
        record = {"step": step, "accuracy": acc, "tokens": tokens, "saved_pct": saved_pct,
                  "drop_pp": drop_pp, "allowed_pp": self.allowed_drop(saved_pct),
                  "thinking": thinking, "within_budget": within,
                  "alpha": self.alpha, "beta": self.beta, "rollback_to": rollback,
                  "stop": self.rollbacks > c.max_rollbacks}
        self.history.append(record)
        return record

    def state(self):
        return {"alpha": self.alpha, "beta": self.beta, "history": self.history,
                "last_good": self.last_good, "rollbacks": self.rollbacks}

    def load(self, state):
        self.alpha, self.beta, self.history = state["alpha"], state["beta"], state["history"]
        self.last_good, self.rollbacks = state.get("last_good"), state.get("rollbacks", 0)
