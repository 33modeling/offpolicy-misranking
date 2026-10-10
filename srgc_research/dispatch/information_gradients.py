"""Measure per-problem derivatives with the actual GRPO forward schedule."""

import hashlib
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ADAPTER_SOURCE = Path(__file__).read_text()
ADAPTER_SHA256 = hashlib.sha256(ADAPTER_SOURCE.encode()).hexdigest()


def problem_gradient(backend, records, *, batch_prompts=None):
    """Match TorchBackend.train, including ragged response micro-batches."""
    import torch

    from srgc_rebuttal.objectives import grpo_advantages
    from srgc_rebuttal.progress import record as progress
    batch_prompts = len(records) if batch_prompts is None else batch_prompts
    if not records or type(batch_prompts) is not int or batch_prompts < len(records):
        raise ValueError("problem gradients require a valid global batch size")
    saved = [None if p.grad is None else p.grad.detach().clone() for _, p in backend.train_parameters]
    try:
        backend.optimizer.zero_grad(set_to_none=True)
        for rid, record in records.items():
            sequences, rewards, start = record[:3]
            advantages = grpo_advantages(rewards)
            active = [i for i, advantage in enumerate(advantages) if advantage != 0]
            for offset in range(0, len(active), backend.logprob_micro_batch):
                indices = active[offset:offset + backend.logprob_micro_batch]
                values = backend._logps_batch([sequences[i].to(backend.device) for i in indices], start)
                losses = []
                for index, logps in zip(indices, values):
                    # The real one-step update uses its own current logps as
                    # old policy, not a separate single-response forward pass.
                    advantage = float(advantages[index])
                    ratio = torch.exp(logps - logps.detach())
                    weighted = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                    losses.append(-weighted.mean())
                # Match normalization *inside* backward as well. Scaling a
                # unit-loss gradient afterwards is different in BF16 kernels.
                (sum(losses) / (len(rewards) * batch_prompts)).backward()
                progress("information_problem_gradient", prompt=rid)
        parts = [torch.zeros_like(p).float().cpu().flatten() if p.grad is None else
                 p.grad.detach().float().cpu().flatten() for _, p in backend.train_parameters]
        gradient = torch.cat(parts) * (batch_prompts / len(records))
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite problem gradient")
        return gradient
    finally:
        for (_, parameter), gradient in zip(backend.train_parameters, saved):
            parameter.grad = gradient


@contextmanager
def aligned_problem_gradients():
    from srgc_research import information
    original_probe = information.probe_gradient
    original_inspect = information.inspect_update

    def inspect(backend, records, *args, **kwargs):
        def aligned_probe(backend, probe, *, distributed=True):
            if distributed:
                return original_probe(backend, probe, distributed=True)
            return problem_gradient(backend, probe, batch_prompts=len(records))
        with patch.object(information, "probe_gradient", aligned_probe):
            result, tensors = original_inspect(backend, records, *args, **kwargs)
        result["metrics"]["problem_gradient_definition"] = "actual-GRPO-micro-batches-and-global-normalization"
        result["metrics"]["problem_gradient_adapter_sha256"] = ADAPTER_SHA256
        tensors["problem_gradient_adapter"] = {"sha256": ADAPTER_SHA256, "source": ADAPTER_SOURCE}
        return result, tensors

    with patch.object(information, "inspect_update", inspect):
        yield
