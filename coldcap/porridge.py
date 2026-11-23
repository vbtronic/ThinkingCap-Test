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
  is shorter, so a prompt keeps pulling toward its best-ever answer.

* Controller: after each validation it compares accuracy and tokens with the base model
  and moves alpha / beta so that training stays inside the competition's budget of
  0.2 accuracy points per 1% of tokens saved.

Plus the memory that makes early ejection possible: prompts already known to be hot or icy
are skipped before any token is generated, and a known prompt's generation is capped at
`eject_factor` x its shortest correct length.
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


@dataclass
class PromptMemory:
    seen: int = 0                # how many times its group was generated
    pass_rate: float | None = None   # pass rate of the last group
    l_min: int | None = None     # shortest correct length ever seen
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

    def cap(self, index):
        """Early ejection, part 2: how many tokens this prompt may generate."""
        known = self.prompts.get(index)
        if known is None or known.l_min is None:
            return self.config.max_cap
        cap = math.ceil(self.config.eject_factor * known.l_min)
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


def rewards(correct, lengths, alpha, beta, l_history=None):
    """Reward of each rollout in one group; see the module docstring."""
    right = [n for ok, n in zip(correct, lengths) if ok]
    candidates = right + ([l_history] if l_history else [])
    l_ref = min(candidates) if candidates else None
    out = []
    for ok, n in zip(correct, lengths):
        if ok:
            out.append(1.0 + alpha * min(1.0, l_ref / max(n, 1)))
        else:
            out.append(-beta)
    return out


def advantages(group_rewards, scale=True, eps=1e-4):
    """GRPO: reward minus the group mean, optionally divided by the group's std."""
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


@dataclass
class Controller:
    config: ControllerConfig = field(default_factory=ControllerConfig)
    alpha: float = 0.0
    beta: float = 0.0
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.alpha = self.alpha or self.config.alpha
        self.beta = self.beta or self.config.beta

    def allowed_drop(self, saved_pct):
        """Accuracy points we may lose for this saving (never negative)."""
        return max(0.0, self.config.budget * saved_pct) + self.config.slack

    def observe(self, step, base_acc, base_tokens, acc, tokens):
        """Validation result in, new alpha/beta out. Accuracies are fractions (0..1)."""
        saved_pct = 100.0 * (1.0 - tokens / base_tokens)
        drop_pp = 100.0 * (base_acc - acc)
        within = drop_pp <= self.allowed_drop(saved_pct)
        c = self.config
        if within:
            self.alpha = min(c.alpha_bounds[1], self.alpha * c.up)
            self.beta = max(c.beta_bounds[0], self.beta * 0.5)
        else:
            self.alpha = max(c.alpha_bounds[0], self.alpha * c.down)
            self.beta = min(c.beta_bounds[1], self.beta + c.beta_step)
        record = {"step": step, "accuracy": acc, "tokens": tokens, "saved_pct": saved_pct,
                  "drop_pp": drop_pp, "allowed_pp": self.allowed_drop(saved_pct),
                  "within_budget": within, "alpha": self.alpha, "beta": self.beta}
        self.history.append(record)
        return record

    def state(self):
        return {"alpha": self.alpha, "beta": self.beta, "history": self.history}

    def load(self, state):
        self.alpha, self.beta, self.history = state["alpha"], state["beta"], state["history"]
