"""Split-half scores of the reuse estimators on a completed point (GPU).

The matrix scores each candidate's stored behavior responses once
(g00, g10, g01, g11 against the ranking validation direction, see
experiment.stage_score). This recomputes the same scores on the two halves of
every response group (first half and second half by response index, with
leave-one-out advantages inside each half), so that the reuse selectors carry
the same two-measurement reliability diagnostic as the fresh scores (a/b of
scores_splithalf.json) and the difficulty score. Optionally the full-group
score is recomputed for a few prompts and compared with scores_offpolicy.json
as a consistency check of the machinery.

    python src/stale_splithalf.py --run POINT --shard I --shards N [--check-full 8]   # one GPU
    python src/stale_splithalf.py --run POINT --merge --shards N                        # CPU

Writes scores_stale_splithalf.shard<I>.json per shard (also with a single
shard) and, after --merge, scores_stale_splithalf.json {est: {idx: {"a", "b"}}} with a protocol file
(parameters, generation validation, git revision, full-score check).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from grads import (ESTIMATORS, ProjectionSpec, cosine, grad_params, loo_advantages, prompt_gradient,
                   sequence_logprobs_batch, token_weights)

SCHEMA = "offpolicy-stale-splithalf/v1"


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def halves_of(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda r: int(r["rollout_idx"]))
    cut = len(ordered) // 2
    if cut < 2:
        raise ValueError("split-half scoring needs at least four responses per prompt")
    return ordered[:cut], ordered[cut:]


def group_score(pi, params, rows: list[dict], logps_pi: list, logps_beta: list, spec: ProjectionSpec,
                direction: torch.Tensor, clip_cap: float, micro_batch: int) -> dict[str, float]:
    """Estimator scores of one response group (a half or the whole group)."""
    rewards = torch.tensor([float(r["reward"]) for r in rows])
    advantages = loo_advantages(rewards)
    out = {}
    for est in ESTIMATORS:
        weights = [token_weights(lp, lb, float(a), est, clip_cap=clip_cap)
                   for lp, lb, a in zip(logps_pi, logps_beta, advantages, strict=True)]
        g = prompt_gradient(pi, params, rows, weights, spec, micro_batch=micro_batch)
        out[est] = cosine(g, direction)
    return out


def score_prompts(pi, beta_logps: dict[int, list], rollouts: dict[int, list[dict]], params, spec: ProjectionSpec,
                  direction: torch.Tensor, clip_cap: float, micro_batch: int, check_full: int = 0,
                  log=print) -> tuple[dict, dict]:
    """Half scores for every prompt of ``rollouts``; full scores for the first ``check_full``."""
    halves = {est: {} for est in ESTIMATORS}
    full = {est: {} for est in ESTIMATORS}
    started = time.time()
    for n_done, (idx, rows) in enumerate(sorted(rollouts.items()), 1):
        ordered = sorted(rows, key=lambda r: int(r["rollout_idx"]))
        logps_pi = sequence_logprobs_batch(pi, ordered, micro_batch=micro_batch)
        logps_beta = [beta_logps[idx][int(r["rollout_idx"])] for r in ordered]
        cut = len(ordered) // 2
        part_a = group_score(pi, params, ordered[:cut], logps_pi[:cut], logps_beta[:cut], spec, direction, clip_cap, micro_batch)
        part_b = group_score(pi, params, ordered[cut:], logps_pi[cut:], logps_beta[cut:], spec, direction, clip_cap, micro_batch)
        for est in ESTIMATORS:
            halves[est][idx] = {"a": part_a[est], "b": part_b[est]}
        if n_done <= check_full:
            whole = group_score(pi, params, ordered, logps_pi, logps_beta, spec, direction, clip_cap, micro_batch)
            for est in ESTIMATORS:
                full[est][idx] = whole[est]
        if n_done % 5 == 0 or n_done == len(rollouts):
            elapsed = time.time() - started
            eta = (len(rollouts) - n_done) * elapsed / n_done
            log(f"[stale-splithalf] {n_done}/{len(rollouts)} prompts ({elapsed / n_done:.0f}s per prompt, ETA {eta / 60:.0f}m)")
    return halves, full


def beta_logprobs(beta, rollouts: dict[int, list[dict]], micro_batch: int, log=print) -> dict[int, list]:
    out = {}
    for n, (idx, rows) in enumerate(sorted(rollouts.items()), 1):
        ordered = sorted(rows, key=lambda r: int(r["rollout_idx"]))
        values = sequence_logprobs_batch(beta, ordered, micro_batch=micro_batch)
        out[idx] = {int(r["rollout_idx"]): v for r, v in zip(ordered, values, strict=True)}
        if n % 25 == 0:
            log(f"[stale-splithalf] behavior log-probs {n}/{len(rollouts)}")
    return out


def compute_shard(run: Path, shard: int, shards: int, check_full: int = 0, loader=None, log=print) -> Path:
    """One shard of a point: prompts sorted(ids)[shard::shards]. ``loader`` is
    rollout.load_policy unless a test injects models."""
    from experiment import read_rollouts, split_validation_directions
    config = json.loads((run / "run_config.json").read_text())
    target = run / f"scores_stale_splithalf.shard{shard}.json"
    if target.is_file() or (run / "scores_stale_splithalf.json").is_file():
        log(f"[stale-splithalf] exists, skipped: {target}")
        return target
    rollouts = read_rollouts(run / "rollouts_behavior_train.jsonl")
    keys = sorted(rollouts)[shard::shards]
    rollouts = {k: rollouts[k] for k in keys}
    if not rollouts:
        raise ValueError("shard has no prompts")
    drift = int(config.get("drift", 0))
    adapter = run / f"policy_step_{drift}" if drift > 0 else None
    proj_dim, grad_layers = int(config.get("proj_dim", 4096)), int(config.get("grad_layers", 4))
    clip_cap, micro_batch = float(config.get("clip_cap", 10.0)), int(config.get("micro_batch", 2))
    if loader is None:
        from rollout import load_policy
        loader = load_policy
    # two passes, as in the matrix scoring: behavior log-probs first, then the policy
    beta, _ = loader(config["model"], None)
    logps_beta = beta_logprobs(beta, rollouts, micro_batch, log)
    del beta
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pi, _ = loader(config["model"], adapter)
    params = grad_params(pi, grad_layers)
    val_groups = torch.load(run / "val_groups.pt", weights_only=True)
    direction, _, _ = split_validation_directions(val_groups)
    spec = ProjectionSpec(dim=proj_dim)
    halves, full = score_prompts(pi, logps_beta, rollouts, params, spec, direction, clip_cap, micro_batch, check_full, log)
    payload = {"schema": SCHEMA, "shard": shard, "shards": shards, "prompts": len(rollouts),
               "parameters": {"proj_dim": proj_dim, "grad_layers": grad_layers, "clip_cap": clip_cap,
                              "micro_batch": micro_batch, "adapter": str(adapter) if adapter else None},
               "halves": {est: {str(i): v for i, v in halves[est].items()} for est in ESTIMATORS},
               "full_check": {est: {str(i): v for i, v in full[est].items()} for est in ESTIMATORS}}
    _atomic_text(target, json.dumps(payload, indent=1))
    log(f"[stale-splithalf] written: {target}")
    return target


def merge(run: Path, shards: int) -> Path:
    parts = []
    for shard in range(shards):
        path = run / f"scores_stale_splithalf.shard{shard}.json"
        if not path.is_file():
            raise FileNotFoundError(f"shard output missing: {path}")
        parts.append(json.loads(path.read_text()))
    merged = {est: {} for est in ESTIMATORS}
    full = {est: {} for est in ESTIMATORS}
    for part in parts:
        if part.get("schema") != SCHEMA:
            raise ValueError("unsupported shard schema")
        for est in ESTIMATORS:
            for idx, value in part["halves"][est].items():
                if idx in merged[est]:
                    raise ValueError(f"prompt {idx} scored in two shards")
                merged[est][idx] = value
            full[est].update(part["full_check"][est])
    expected = set(json.loads((run / "scores_oracle.json").read_text()))
    for est in ESTIMATORS:
        if set(merged[est]) != expected:
            raise ValueError(f"{est}: half-score coverage differs from the point's candidates "
                             f"({len(merged[est])} of {len(expected)})")
    check = {}
    if any(full[est] for est in ESTIMATORS):
        stored = json.loads((run / "scores_offpolicy.json").read_text())
        for est in ESTIMATORS:
            diffs = [abs(full[est][i] - float(stored[est][i]["score"])) for i in full[est] if i in stored[est]]
            check[est] = {"prompts": len(diffs), "max_abs_difference": max(diffs) if diffs else None}
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]).strip()
    except (OSError, subprocess.CalledProcessError):
        git = None
    protocol = {"schema": SCHEMA, "shards": shards, "prompts": len(expected), "parameters": parts[0]["parameters"],
                "halves": "first half and second half of the stored responses by response index; leave-one-out "
                          "advantages inside each half; scores are cosines with the ranking validation direction",
                "full_score_check": check, "git": git}
    _atomic_text(run / "scores_stale_splithalf.json", json.dumps(merged, indent=1))
    _atomic_text(run / "scores_stale_splithalf.protocol.json", json.dumps(protocol, indent=1))
    return run / "scores_stale_splithalf.json"


def reliability(run: Path) -> dict[str, float]:
    """Pearson correlation of the two halves per estimator, for a quick readout."""
    from gate_decision import pearson
    data = json.loads((run / "scores_stale_splithalf.json").read_text())
    out = {}
    for est in ESTIMATORS:
        a = [v["a"] for v in data[est].values()]
        b = [v["b"] for v in data[est].values()]
        out[est] = pearson(a, b)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--check-full", type=int, default=0, help="recompute the full-group score for this many prompts per shard")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.merge:
            target = merge(args.run.resolve(), args.shards)
            rel = reliability(args.run.resolve())
            print("[stale-splithalf] half-score correlations: " + " ".join(f"{k}={v:+.3f}" for k, v in rel.items() if math.isfinite(v)))
            print(f"[stale-splithalf] merged: {target}")
        else:
            compute_shard(args.run.resolve(), args.shard, args.shards, args.check_full)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
