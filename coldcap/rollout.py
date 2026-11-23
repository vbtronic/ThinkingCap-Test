# Copyright 2026 Viktor Brunclík
# SPDX-License-Identifier: Apache-2.0
"""Rollouts: generate G answers per prompt, on whichever GPU is meant for it.

Two backends with one interface, `engine.generate(requests, adapter) -> list of results`:

* VLLMEngine runs vLLM in its own process, pinned to one GPU through CUDA_VISIBLE_DEVICES.
  It picks its own dtype for that card (bf16 on Ampere+, fp16 on a T4), independently of
  the trainer's card. The trainer hands it the current LoRA as a directory on disk.
* HFEngine generates with the trainer's own model, in-process. Slow, but it runs where
  vLLM cannot (a P100) and needs no second copy of anything.

A request is {"ids": [prompt token ids], "max_tokens": int, "seed": int}.
A result is {"tokens": [generated ids, eos included], "finished": bool}; finished is
False when the cap cut the answer off, which the grader counts as wrong -- as evaluate.py does.

Sampling is the evaluation's (temperature 0.6, top-p 0.95, top-k 20): we train on the
same distribution of answers that will be scored.
"""

import atexit
import multiprocessing as mp
import os
import time
import traceback

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


class VLLMEngine:
    def __init__(self, device, model, revision, eos_ids, max_model_len, lora_rank,
                 memory=0.85, seed=0, dtype="auto"):
        context = mp.get_context("spawn")
        self.conn, child = context.Pipe()
        config = {"model": model, "revision": revision, "eos_ids": list(eos_ids),
                  "max_model_len": max_model_len, "lora_rank": lora_rank,
                  "memory": memory, "seed": seed, "dtype": dtype}
        # The child inherits the environment at start: show it only its own card.
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible = saved.split(",") if saved else None
        os.environ["CUDA_VISIBLE_DEVICES"] = visible[device] if visible else str(device)
        try:
            # Not a daemon: vLLM starts its own engine-core process, and daemonic processes
            # may not have children. close() (also registered with atexit) shuts it down.
            self.process = context.Process(target=serve, args=(child, config))
            self.process.start()
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
        atexit.register(self.close)
        kind, payload = self.conn.recv()
        if kind != "ready":
            raise RuntimeError(f"vLLM worker failed to start:\n{payload}")
        self.info = payload   # {"gpu": ..., "dtype": ...} of the card it actually got
        self.adapter_id = 0
        self.speed = None     # generated tokens per second in the last call

    def generate(self, requests, adapter=None, fixed=False):
        """adapter: a directory saved by peft, or None for the base model. `fixed` only
        matters to a pool."""
        self.submit(requests, adapter)
        return self.collect()

    def submit(self, requests, adapter=None):
        """Start generating without waiting, so several engines can work at once."""
        if adapter is not None:
            self.adapter_id += 1
        self.conn.send(("generate", requests, adapter, self.adapter_id))

    def collect(self):
        kind, payload = self.conn.recv()
        if kind != "done":
            raise RuntimeError(f"vLLM worker failed:\n{payload}")
        tokens = sum(len(r["tokens"]) for r in payload["results"])
        if tokens and payload["seconds"] > 0:
            self.speed = tokens / payload["seconds"]
        return payload["results"]

    def close(self):
        """Stop the worker; idempotent. Python joins non-daemon children at exit, so a
        worker left waiting on the pipe would hang the trainer forever."""
        if not self.process.is_alive():
            return
        try:
            self.conn.send(("stop", None, None, None))
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=60)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)


def serve(conn, config):
    """The vLLM worker process: build the engine, then answer requests until told to stop."""
    try:
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import torch
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from coldcap.hw import best_dtype, vllm_attention

        dtype = best_dtype(0) if config["dtype"] == "auto" else config["dtype"]
        llm = LLM(model=config["model"], revision=config["revision"],
                  tokenizer_revision=config["revision"], dtype=dtype,
                  max_model_len=config["max_model_len"],
                  gpu_memory_utilization=config["memory"], seed=config["seed"],
                  enable_lora=True, max_lora_rank=config["lora_rank"], max_loras=1,
                  max_cpu_loras=2, enable_prefix_caching=True, generation_config="vllm",
                  **vllm_attention())
        conn.send(("ready", {"gpu": torch.cuda.get_device_name(0), "dtype": dtype}))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return

    while True:
        try:
            kind, requests, adapter, adapter_id = conn.recv()
        except EOFError:          # the trainer is gone
            return
        if kind == "stop":
            return
        try:
            lora = LoRARequest(f"policy-{adapter_id}", adapter_id, adapter) if adapter else None
            params = [SamplingParams(max_tokens=r["max_tokens"], seed=r["seed"],
                                     stop_token_ids=config["eos_ids"], **SAMPLING)
                      for r in requests]
            started = time.time()
            outputs = llm.generate([{"prompt_token_ids": r["ids"]} for r in requests], params,
                                   lora_request=lora, use_tqdm=False)
            seconds = time.time() - started
            results = []
            for output in outputs:
                tokens = list(output.outputs[0].token_ids)
                results.append({"tokens": tokens,
                                "finished": bool(tokens) and tokens[-1] in config["eos_ids"]})
            conn.send(("done", {"results": results, "seconds": seconds}))
        except Exception:
            conn.send(("error", traceback.format_exc()))


class EnginePool:
    """Several VLLMEngines (one per GPU) behind the one-engine interface.

    Requests are split into contiguous slices, one per engine. For training the slices follow
    each engine's measured speed, so both cards finish together (the one shared with the
    trainer has less KV-cache memory and runs fewer sequences at once). With fixed=True the
    split is even and depends only on the request count, so validation always sends request
    i to the same card: base and policy are compared like for like.
    """

    def __init__(self, engines):
        self.engines = engines
        gpus = [e.info["gpu"] for e in engines]
        dtypes = sorted({e.info["dtype"] for e in engines})
        self.info = {"gpu": " + ".join(gpus), "dtype": " + ".join(dtypes)}

    def shares(self, fixed):
        speeds = [e.speed for e in self.engines]
        if fixed or None in speeds:
            return [1.0 / len(self.engines)] * len(self.engines)
        return [v / sum(speeds) for v in speeds]

    def generate(self, requests, adapter=None, fixed=False):
        bounds, total = [0], 0.0
        for share in self.shares(fixed):
            total += share
            bounds.append(round(total * len(requests)))
        bounds[-1] = len(requests)
        slices = [requests[a:b] for a, b in zip(bounds, bounds[1:])]
        busy = [engine for engine, chunk in zip(self.engines, slices) if chunk]
        for engine, chunk in zip(busy, slices):
            engine.submit(chunk, adapter)
        results = []
        for engine in busy:
            results.extend(engine.collect())
        return results

    def close(self):
        for engine in self.engines:
            engine.close()


class HFEngine:
    """Fallback: generate with the policy itself. Ignores `adapter` -- it *is* the adapter."""

    info = None

    def __init__(self, model, tokenizer, eos_ids, batch=32):
        import torch
        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.eos_ids, self.batch = list(eos_ids), batch
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_ids[0]
        device = next(model.parameters()).device
        self.info = {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                     "dtype": str(next(model.parameters()).dtype).replace("torch.", "")}

    def generate(self, requests, adapter=None, fixed=False):
        torch = self.torch
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        results = []
        try:
            for start in range(0, len(requests), self.batch):
                chunk = requests[start:start + self.batch]
                width = max(len(r["ids"]) for r in chunk)
                ids = torch.full((len(chunk), width), self.pad, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for row, r in enumerate(chunk):          # left padding
                    ids[row, width - len(r["ids"]):] = torch.tensor(r["ids"])
                    mask[row, width - len(r["ids"]):] = 1
                torch.manual_seed(chunk[0]["seed"])
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=ids.to(device), attention_mask=mask.to(device),
                        max_new_tokens=max(r["max_tokens"] for r in chunk), do_sample=True,
                        eos_token_id=self.eos_ids, pad_token_id=self.pad, use_cache=True,
                        **SAMPLING)
                for row, r in enumerate(chunk):
                    tokens = out[row, width:].tolist()
                    results.append(cut(tokens, r["max_tokens"], self.eos_ids))
        finally:
            self.model.train(was_training)
        return results

    def close(self):
        pass


def cut(tokens, cap, eos_ids):
    """Trim padding after eos and apply this request's own cap."""
    for position, token in enumerate(tokens[:cap]):
        if token in eos_ids:
            return {"tokens": tokens[:position + 1], "finished": True}
    return {"tokens": tokens[:cap], "finished": False}
