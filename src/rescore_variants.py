#!/usr/bin/env python3
"""Re-score stored behavior rollouts under alternative budgets (extension E6).

No generation is repeated. For each requested variant the four stale
selectors are recomputed from the immutable ``rollouts_behavior_train.jsonl``
with

* ``bk<K>``: only the first ``K`` behavior responses per prompt (by
  ``rollout_idx``), so the leave-one-out advantages and the gradient average
  use a smaller stale budget;
* ``clip<C>``: token-ratio products clipped to ``[1/C, C]`` instead of the
  registered ``[1/10, 10]``;
* combinations such as ``bk4-clip3``.

Outputs ``scores_offpolicy.variant-<name>.json`` next to the registered file
in the same layout, plus ``variant_protocol.json`` recording the settings.
The registered ``scores_offpolicy.json`` is never rewritten.

    PYTHONPATH=src python3 src/rescore_variants.py --run <run> --model <base> [--adapter <dir>] \
        --variants bk2 bk4 clip3 clip30 --proj-dim 4096 --grad-layers 4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from experiment import read_rollouts, split_validation_directions
from grads import (
    ESTIMATORS,
    ProjectionSpec,
    cosine,
    grad_params,
    loo_advantages,
    prompt_gradient,
    sequence_logprobs_batch,
    token_weights,
)
from rollout import load_policy

VARIANT_RE = re.compile(r"^(?:bk(?P<k>\d+))?-?(?:clip(?P<cap>\d+(?:\.\d+)?))?$")


def parse_variant(name: str, default_cap: float) -> tuple[int | None, float]:
    match = VARIANT_RE.match(name)
    if not match or not name or (match.group("k") is None and match.group("cap") is None):
        raise ValueError(f"unknown variant spec: {name!r} (expected bk<K>, clip<C>, or bk<K>-clip<C>)")
    k = int(match.group("k")) if match.group("k") else None
    cap = float(match.group("cap")) if match.group("cap") else default_cap
    if k is not None and k < 2:
        raise ValueError("a behavior budget needs at least two responses for leave-one-out advantages")
    if cap < 1.0:
        raise ValueError("clip cap must be >= 1")
    return k, cap


def rescore(run: Path, pi, beta, variants: list[str], *, proj_dim: int, grad_layers: int,
            micro_batch: int, default_cap: float) -> dict[str, Path]:
    rollouts = read_rollouts(run / "rollouts_behavior_train.jsonl")
    val_groups = torch.load(run / "val_groups.pt", weights_only=True)
    selection_val, _, _ = split_validation_directions(val_groups)
    params = grad_params(pi, grad_layers)
    spec = ProjectionSpec(dim=proj_dim)
    specs = {name: parse_variant(name, default_cap) for name in variants}
    outputs = {name: {est: {} for est in ESTIMATORS} for name in variants}
    for prompt_idx, rows in sorted(rollouts.items()):
        rows = sorted(rows, key=lambda r: int(r["rollout_idx"]))
        logps_pi = sequence_logprobs_batch(pi, rows, micro_batch=micro_batch)
        logps_beta = sequence_logprobs_batch(beta, rows, micro_batch=micro_batch)
        for name, (k, cap) in specs.items():
            subset = rows if k is None else rows[:k]
            if k is not None and len(subset) < k:
                raise ValueError(f"prompt {prompt_idx} has only {len(subset)} behavior responses; variant {name} needs {k}")
            lp, lb = logps_pi[: len(subset)], logps_beta[: len(subset)]
            advs = loo_advantages(torch.tensor([float(r["reward"]) for r in subset]))
            for est in ESTIMATORS:
                weights = [token_weights(a, b, float(adv), est, clip_cap=cap)
                           for a, b, adv in zip(lp, lb, advs, strict=True)]
                g = prompt_gradient(pi, params, subset, weights, spec, micro_batch=micro_batch)
                outputs[name][est][str(prompt_idx)] = {"score": cosine(g, selection_val), "norm": float(g.norm())}
    written = {}
    for name, (k, cap) in specs.items():
        path = run / f"scores_offpolicy.variant-{name}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(outputs[name], indent=1))
        tmp.replace(path)
        written[name] = path
    protocol = run / "variant_protocol.json"
    existing = json.loads(protocol.read_text()) if protocol.exists() else {}
    for name, (k, cap) in specs.items():
        existing[name] = {"behavior_k": k, "clip_cap": cap, "proj_dim": proj_dim, "grad_layers": grad_layers,
                          "prompts": len(rollouts)}
    protocol.write_text(json.dumps(existing, indent=1))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", default=["bk2", "bk4", "clip3", "clip30"])
    parser.add_argument("--proj-dim", type=int, default=4096)
    parser.add_argument("--grad-layers", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--clip-cap", type=float, default=10.0, help="registered cap used when a variant omits clip<C>")
    args = parser.parse_args(argv)
    config = json.loads((args.run / "run_config.json").read_text())
    adapter = args.adapter
    if adapter is None and int(config.get("drift", 0)) > 0:
        candidate = args.run / f"policy_step_{int(config['drift'])}"
        if candidate.is_dir():
            adapter = candidate
    for name in args.variants:
        parse_variant(name, args.clip_cap)
    pi, _ = load_policy(args.model, adapter)
    beta, _ = load_policy(args.model, None)
    written = rescore(args.run, pi, beta, args.variants, proj_dim=args.proj_dim, grad_layers=args.grad_layers,
                      micro_batch=args.micro_batch, default_cap=args.clip_cap)
    for name, path in written.items():
        print(f"[variant] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
