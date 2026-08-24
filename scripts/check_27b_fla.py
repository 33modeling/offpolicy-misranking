#!/usr/bin/env python3
"""Fail fast unless Qwen3.8's FLA recurrent and chunk kernels execute."""

from __future__ import annotations

from importlib.metadata import version

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    installed = version("fla-core")
    if installed != "0.5.2":
        raise RuntimeError(f"expected FLA 0.5.2, got {installed}")
    recurrent = modeling.fused_recurrent_gated_delta_rule
    chunk = modeling.chunk_gated_delta_rule
    if recurrent is None or chunk is None:
        raise RuntimeError("FLA fused recurrent/chunk kernels are unavailable")

    torch.manual_seed(104729)
    shape = (1, 8, 2, 16)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(shape[:3], device="cuda", dtype=torch.float32)
    beta = torch.rand(shape[:3], device="cuda", dtype=torch.float32)
    kwargs = {
        "g": g,
        "beta": beta,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
    }
    for name, kernel in (("recurrent", recurrent), ("chunk", chunk)):
        output, state = kernel(q, k, v, **kwargs)
        torch.cuda.synchronize()
        if not torch.isfinite(output).all() or not torch.isfinite(state).all():
            raise RuntimeError(f"FLA {name} kernel produced non-finite values")
    print(
        f"[27b-runtime] GPU={torch.cuda.get_device_name()} FLA={installed} "
        "recurrent+chunk OK",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
