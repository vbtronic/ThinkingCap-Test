# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""The policy: Qwen3-0.6B + LoRA on the training GPU, and the GRPO loss on it.

Memory on a 16 GB T4 is dominated by logits (151,936 x sequence length), so they are never
materialized whole: the transformer runs once, then the LM head and log-softmax run on
chunks of positions under activation checkpointing. Only the completion is scored, never
the prompt.

Precision follows the training card, not the rollout card:
  bf16-capable (Ampere+)  base weights bf16, autocast bf16, no scaler
  T4 / V100               base weights fp16, autocast fp16, GradScaler
  either                  LoRA weights in fp32, so small updates are not rounded away
"""

from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint

from coldcap.hw import best_dtype, torch_dtype

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CHUNK = 512


def load_policy(model_id, revision, device_index, dtype="auto", rank=16, adapter=None):
    """-> (peft model on cuda:device_index, dtype name it uses)."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = best_dtype(device_index) if dtype == "auto" else dtype
    device = torch.device(f"cuda:{device_index}" if device_index is not None else "cpu")
    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                dtype=torch_dtype(dtype)).to(device)
    base.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    else:
        config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                            target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(base, config)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, dtype


def restore_adapter(model, checkpoint):
    """Put a saved checkpoint's LoRA weights back into the live model (rollback)."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    device = next(p for p in model.parameters() if p.requires_grad).device
    weights = load_file(str(checkpoint / "adapter_model.safetensors"), device=str(device))
    result = set_peft_model_state_dict(model, weights)
    if result.unexpected_keys:
        raise RuntimeError(f"{checkpoint}: unexpected LoRA keys {result.unexpected_keys[:3]}")


def autocast(dtype):
    if dtype in ("float16", "bfloat16"):
        return torch.autocast("cuda", dtype=torch_dtype(dtype))
    return nullcontext()


def token_logps(model, prompt_ids, completion_ids, dtype):
    """Log-probability of each completion token, shape (len(completion_ids),)."""
    inner = model.get_base_model()          # Qwen3ForCausalLM, LoRA already injected in place
    device = inner.lm_head.weight.device
    ids = torch.tensor([prompt_ids + completion_ids], device=device)
    with autocast(dtype):
        hidden = inner.model(input_ids=ids).last_hidden_state[0]
    # position p predicts token p+1: the completion is predicted from len(prompt)-1 on
    start = len(prompt_ids) - 1
    hidden = hidden[start:start + len(completion_ids)]
    targets = torch.tensor(completion_ids, device=device)
    pieces = []
    for at in range(0, len(completion_ids), CHUNK):
        pieces.append(checkpoint(chunk_logps, inner.lm_head, hidden[at:at + CHUNK],
                                 targets[at:at + CHUNK], dtype, use_reentrant=False))
    return torch.cat(pieces)


def chunk_logps(head, hidden, targets, dtype):
    with autocast(dtype):
        logits = head(hidden)
    logits = logits.float()
    return logits.gather(-1, targets[:, None])[:, 0] - torch.logsumexp(logits, dim=-1)


class Trainer:
    """One optimizer step per batch of groups: on-policy GRPO, token-level normalisation."""

    def __init__(self, model, dtype, lr=2e-5, kl=0.0, max_grad_norm=1.0):
        self.model, self.dtype, self.kl = model, dtype, kl
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=0.0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
        self.max_grad_norm = max_grad_norm

    def step(self, sequences):
        """sequences: [{"prompt": ids, "completion": ids, "advantage": float}, ...].

        Sequences with zero advantage carry no gradient and are skipped before any
        forward pass -- the last place a wasted backward could hide.
        """
        sequences = [s for s in sequences if abs(s["advantage"]) > 1e-8 and s["completion"]]
        if not sequences:
            return {"loss": 0.0, "trained_sequences": 0, "trained_tokens": 0}
        self.model.train()
        total = sum(len(s["completion"]) for s in sequences)
        loss_sum, kl_sum = 0.0, 0.0
        for s in sequences:
            logps = token_logps(self.model, s["prompt"], s["completion"], self.dtype)
            # on-policy single update: ratio is 1, the gradient is that of -A * log p
            loss = -(s["advantage"] * logps).sum() / total
            if self.kl:
                with torch.no_grad(), self.model.disable_adapter():
                    reference = token_logps(self.model, s["prompt"], s["completion"], self.dtype)
                delta = reference - logps
                kl = (torch.exp(delta) - delta - 1).sum() / total     # k3 estimator
                loss = loss + self.kl * kl
                kl_sum += kl.item()
            self.scaler.scale(loss).backward()
            loss_sum += loss.item()
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return {"loss": loss_sum, "kl": kl_sum, "grad_norm": float(norm),
                "trained_sequences": len(sequences), "trained_tokens": total}

    def reset(self):
        """Fresh optimizer moments: after a rollback the old ones point toward the collapse."""
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.optimizer.defaults["lr"],
                                           weight_decay=0.0)

    def state(self):
        return {"optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict()}

    def load(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
