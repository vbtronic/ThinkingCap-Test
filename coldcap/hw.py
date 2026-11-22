# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""What each GPU can do, asked per device: on Kaggle or a home box GPU 0 and GPU 1 need not match.

    from coldcap.hw import inventory, best_dtype
    inventory()        # [{'index': 0, 'name': 'Tesla T4', 'capability': '7.5', 'bf16': False, ...}, ...]
    best_dtype(0)      # 'float16' on a T4, 'bfloat16' on an L4 / A100 / 3090

Nothing here assumes the devices are alike. The trainer and the rollout engine each pick
their own precision for the card they run on, and every run records which card and which
precision produced it, so a comparison can refuse to pair runs that differ.
"""

import torch

# vLLM needs compute capability 7.0 (Volta) or newer; a P100 is 6.0.
VLLM_MIN_CAPABILITY = (7, 0)
# Native bf16 arrived with Ampere (8.0). Older cards emulate it slowly or not at all.
BF16_MIN_CAPABILITY = (8, 0)


def describe(index):
    """One device as a plain dict, safe to write into JSON."""
    props = torch.cuda.get_device_properties(index)
    capability = (props.major, props.minor)
    return {"index": index, "name": props.name, "capability": f"{props.major}.{props.minor}",
            "memory_gb": round(props.total_memory / 2**30, 1),
            "bf16": capability >= BF16_MIN_CAPABILITY,
            "vllm": capability >= VLLM_MIN_CAPABILITY}


def inventory():
    """Every visible GPU, each described on its own; [] without CUDA."""
    if not torch.cuda.is_available():
        return []
    return [describe(i) for i in range(torch.cuda.device_count())]


def best_dtype(index):
    """'bfloat16' where the card has it natively, else 'float16'; 'float32' on CPU."""
    if index is None or not torch.cuda.is_available():
        return "float32"
    return "bfloat16" if describe(index)["bf16"] else "float16"


def torch_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def report():
    """A human-readable line per GPU, flagging when they differ."""
    devices = inventory()
    if not devices:
        return "no CUDA device"
    lines = [f"  cuda:{d['index']}  {d['name']}  cc {d['capability']}  {d['memory_gb']} GB  "
             f"bf16={'yes' if d['bf16'] else 'no'}  vllm={'yes' if d['vllm'] else 'no'}"
             for d in devices]
    if len({(d["name"], d["capability"]) for d in devices}) > 1:
        lines.append("  NOTE: the GPUs differ -- each side picks its own precision, and runs "
                     "you compare must come from the same card and dtype")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
