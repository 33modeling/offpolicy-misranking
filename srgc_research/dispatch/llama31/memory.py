"""Bounded Llama generation, checkpointed backward and CPU checkpoint snapshots."""

import copy
import gc
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from srgc_rebuttal.progress import record as progress
from srgc_rebuttal.torch_backend import TorchBackend

GENERATION_BATCH = 2
LOGPROB_BATCH = 1
LOGIT_CHUNK = 64


class DecodeProgress:
    def __init__(self):
        self.last = 0.0

    def __call__(self, input_ids, scores, **kwargs):
        now = time.monotonic()
        if now - self.last >= 30:
            progress("generation", sequence_tokens=input_ids.shape[-1])
            self.last = now
        return False


def bounded_generate(original, device, *, batch_size=GENERATION_BATCH):
    """Keep all requested responses. On OOM restart the whole draw from its RNG state."""

    def generate(*args, **kwargs):
        count = kwargs.get("num_return_sequences", 1)
        if type(count) is not int or count < 1:
            raise ValueError("positive response count required")
        if kwargs.get("return_dict_in_generate", False):
            raise ValueError("SRGC generation expects sequence tensors")
        from transformers import StoppingCriteriaList

        kwargs = dict(kwargs)
        kwargs.setdefault("stopping_criteria", StoppingCriteriaList([DecodeProgress()]))
        # Llama's large vocabulary need only be projected for the last decode token.
        kwargs["logits_to_keep"] = 1
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        size = min(batch_size, count)
        while True:
            chunks = []
            failed = False
            try:
                for offset in range(0, count, size):
                    chunks.append(
                        original(
                            *args,
                            **{
                                **kwargs,
                                "num_return_sequences": min(size, count - offset),
                            },
                        )
                    )
                width = max(chunk.shape[-1] for chunk in chunks)
                pad = kwargs.get("pad_token_id")
                if pad is None:
                    raise ValueError("explicit padding token required")
                result = torch.full(
                    (count, width), pad, dtype=chunks[0].dtype, device=chunks[0].device
                )
                offset = 0
                for chunk in chunks:
                    result[offset : offset + len(chunk), : chunk.shape[-1]] = chunk
                    offset += len(chunk)
                if offset != count:
                    raise ValueError("generation returned the wrong response count")
                return result
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                failed = True
            # Leave the exception scope before freeing tensors: its traceback
            # can otherwise retain the failed generation's KV cache.
            if failed:
                chunks.clear()
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.set_rng_state(cuda_rng, device)
                torch.set_rng_state(cpu_rng)
                size = max(1, size // 2)
                progress(
                    "generation_oom_retry", micro_batch=size, requested_responses=count
                )
                print(
                    f"LLAMA generation OOM: retry all {count} responses at micro_batch={size}",
                    flush=True,
                )

    return generate


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


class LlamaBackend(TorchBackend):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("logprob_micro_batch", LOGPROB_BATCH)
        kwargs.setdefault("logit_chunk_tokens", LOGIT_CHUNK)
        super().__init__(*args, **kwargs)
        base = self.model.get_base_model()
        if base.config.attention_dropout != 0 or any(
            isinstance(m, torch.nn.Dropout) and m.p != 0 for m in self.model.modules()
        ):
            raise ValueError("checkpointed Llama experiment requires zero dropout")
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def _logps_batch(self, sequences, start):
        # HF layer checkpoints activate only in training mode. Rollout generation
        # stays in eval mode with KV caching; dropout is checked to be zero.
        previous = self.model.training
        self.model.train(True)
        try:
            return super()._logps_batch(sequences, start)
        finally:
            self.model.train(previous)

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return {
            "trainable": {n: cpu_snapshot(p) for n, p in self.train_parameters},
            "optimizer": cpu_snapshot(self.optimizer.state_dict()),
            "projection_dim": self.projection_dim,
            "projection_seed": self.projection_seed,
            "score_names": [n for n, _ in self.score_parameters],
        }


@contextmanager
def durable_checkpoints():
    """Also fsync the frozen runner's five/25-step boundary saves before rename."""
    from unittest.mock import patch

    original = torch.save

    def save(value, target, *args, **kwargs):
        if isinstance(target, (str, os.PathLike)):
            with Path(target).open("wb") as stream:
                original(value, stream, *args, **kwargs)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            original(value, target, *args, **kwargs)

    with patch.object(torch, "save", save):
        yield
