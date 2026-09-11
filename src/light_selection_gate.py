"""One cache scan and one fixed decision; no model training or bootstrap loop.

Reliability is measured for the actual difficulty score, not raw pass rate.
The screening rule is exploratory: it does not certify downstream benefit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from pathlib import Path

import selection_gate as core

SCHEMA = "offpolicy-light-gate/v2-1"


def default_rule():
    return {
        "schema": SCHEMA, "status": "prespecified_exploratory",
        "score": "negative_absolute_passrate_minus_half", "schedule": "once_before_training",
        "min_half_score_correlation": 0., "min_mixed_fraction": 0.,
        "min_mixed_uplift": 0., "min_score_spread": 0.,
        "max_measurement_fraction": .01, "max_cache_age_steps": 400,
        "development_trajectory_ids": [],
        "claim": "screening hypothesis, not a benchmark improvement certificate",
    }


def validate_rule(rule):
    if set(rule) != set(default_rule()) or rule["schema"] != SCHEMA:
        raise ValueError("unsupported light-gate rule")
    for key in ("score", "schedule", "claim"):
        if rule[key] != default_rule()[key]:
            raise ValueError(f"unsupported {key}")
    if rule["status"] not in ("prespecified_exploratory", "development_frozen"):
        raise ValueError("unsupported rule status")
    core.number(rule["min_half_score_correlation"], "score correlation threshold", 0, 1)
    for key in ("min_mixed_fraction", "min_mixed_uplift", "min_score_spread"):
        core.number(rule[key], key, 0, 1)
    core.number(rule["max_measurement_fraction"], "cost fraction", 1e-12, .1)
    core.integer(rule["max_cache_age_steps"], "cache age")
    ids = rule["development_trajectory_ids"]
    if not isinstance(ids, list) or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid development trajectory IDs")
    if rule["status"] == "development_frozen" and not ids:
        raise ValueError("development-frozen rules need development trajectory IDs")
    return rule


def correlation(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    a, b = statistics.fmean(left), statistics.fmean(right)
    x, y = [v-a for v in left], [v-b for v in right]
    denom = math.sqrt(sum(v*v for v in x)*sum(v*v for v in y))
    return max(-1., min(1., sum(u*v for u, v in zip(x, y))/denom)) if denom else None


def summarize(groups, *, k, seed):
    n = len(groups)
    core.integer(k, "subset size", 1)
    core.integer(seed, "split seed")
    if k >= n or not n:
        raise ValueError("selection needs 0 < k < pool size")
    size = len(groups[0])
    if size < 4 or size % 2 or any(len(row) != size for row in groups):
        raise ValueError("equal even response groups of at least four required")
    if any(type(v) not in (int, float) or v not in (0, 1) for row in groups for v in row):
        raise ValueError("binary verifier rewards required")
    score = lambda values: -abs(statistics.fmean(values)-.5)
    a, b, full, mixed, pa, pb = [], [], [], [], [], []
    for idx, row in enumerate(groups):
        order = list(row)
        random.Random(seed+idx*1_000_003).shuffle(order)
        first, second = order[:size//2], order[size//2:]
        a.append(score(first)); b.append(score(second)); full.append(score(row))
        pa.append(statistics.fmean(first)); pb.append(statistics.fmean(second))
        mixed.append(int(0 < sum(row) < size))
    rng = random.Random(seed+701_000_003)
    ties = [rng.random() for _ in groups]
    selected = sorted(sorted(range(n), key=lambda i: (-full[i], ties[i]))[:k])
    frac = statistics.fmean(mixed)
    r = correlation(a, b)
    return {
        "prompts": n, "responses_per_prompt": size, "selected_indices": selected,
        "half_score_correlation": r, "half_passrate_correlation": correlation(pa, pb),
        "score_spread": statistics.pstdev(full), "mixed_fraction": frac,
        "selected_mixed_fraction": statistics.fmean(mixed[i] for i in selected),
        "mixed_uplift": statistics.fmean(mixed[i] for i in selected)-frac,
        "mean_passrate": statistics.fmean(statistics.fmean(row) for row in groups),
        "full_pool_coverage": True,
        "sampling_note": "one deterministic randomized split; correlation is descriptive, not a confidence bound",
        "utility_note": "selected mixed-group enrichment uses the same cache; it is not independent utility evidence",
        "spearman_brown": None,
        "spearman_brown_note": "nonlinear full difficulty score is not the mean of the two half scores",
    }


def measure(path, *, prompts, responses, k, seed, cache_step, target_step,
            wall_cap=30., max_bytes=256*1024*1024):
    for val, name, low in ((prompts, "prompts", 2), (responses, "responses", 4),
                           (cache_step, "cache step", 0), (target_step, "target step", 0),
                           (max_bytes, "byte cap", 1)):
        core.integer(val, name, low)
    core.number(wall_cap, "wall cap", 1e-12)
    if cache_step > target_step or responses % 2:
        raise ValueError("invalid cache time or odd response group")
    start, cpu = time.monotonic(), time.process_time()
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("cache exceeds byte cap")
    groups = [[None]*responses for _ in range(prompts)]
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for line in handle:
            count += len(line)
            if count > max_bytes or time.monotonic()-start >= wall_cap:
                raise TimeoutError("cache scan exceeded its resource cap")
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            i = core.integer(row["prompt_idx"], "prompt index")
            j = core.integer(row["rollout_idx"], "response index")
            reward = core.number(row["reward"], "reward", 0, 1)
            if i >= prompts or j >= responses or reward not in (0., 1.):
                raise ValueError("unexpected cached prompt, response or reward")
            if groups[i][j] is not None:
                raise ValueError("duplicate response; resume rows cannot be independent samples")
            groups[i][j] = reward
    if any(v is None for row in groups for v in row):
        raise ValueError("incomplete whole-pool cache")
    result = summarize(groups, k=k, seed=seed)
    elapsed = time.monotonic()-start
    if elapsed >= wall_cap:
        raise TimeoutError("cache summary exceeded its resource cap")
    return {**result, "schema": SCHEMA, "source_sha256": digest.hexdigest(),
            "cache_step": cache_step, "target_step": target_step,
            "wall_seconds": elapsed, "cpu_seconds": time.process_time()-cpu,
            "bytes": count, "split_seed": seed}


def choose(report, rule, *, measured_gpu_seconds, budget_gpu_seconds):
    validate_rule(rule)
    core.number(measured_gpu_seconds, "measurement GPU seconds", 0)
    core.number(budget_gpu_seconds, "budget GPU seconds", 1e-12)
    reasons = []
    if not report.get("full_pool_coverage"):
        reasons.append("incomplete_pool")
    if report["target_step"]-report["cache_step"] > rule["max_cache_age_steps"]:
        reasons.append("cache_age_out_of_scope")
    for field, key in (("half_score_correlation", "min_half_score_correlation"),
                       ("mixed_fraction", "min_mixed_fraction"),
                       ("mixed_uplift", "min_mixed_uplift"), ("score_spread", "min_score_spread")):
        value = report[field]
        if value is None or not math.isfinite(value) or value <= rule[key]:
            reasons.append(field)
    if measured_gpu_seconds/budget_gpu_seconds > rule["max_measurement_fraction"]:
        reasons.append("measurement_cost")
    return {"action": "random" if reasons else "select", "reasons": reasons,
            "rule_sha256": core.fingerprint(rule), "schedule": rule["schedule"],
            "claim": rule["claim"], "measurement_gpu_seconds": measured_gpu_seconds}


def counterexamples():
    noisy_gain = oracle_gain = decoded_gain = 0.
    worlds = list(itertools.product((0., 1.), (0., 1.), (-10., 10.), (-10., 10.)))
    for t0, t1, e0, e1 in worlds:
        theta, y = [t0, t1], [t0+e0, t1+e1]
        win = max(range(2), key=lambda i: y[i])
        decode = max(range(2), key=lambda i: y[i] % 10)
        noisy_gain += theta[win]-.5
        oracle_gain += max(theta)-.5
        decoded_gain += theta[decode]-.5
    n = len(worlds)
    activity = summarize([[0]*8, [0]*8, [1]*8, [1]*8], k=1, seed=0)
    return {"data_kind": "exact_synthetic_counterexamples", "worlds": n,
            "non_gaussian": {"rho": .25/100.25, "sqrt_rho": math.sqrt(.25/100.25),
                             "top_noisy_gain": noisy_gain/n, "oracle_gain": oracle_gain/n,
                             "decoded_gain": decoded_gain/n},
            "perfect_passrate_zero_activity": activity,
            "benchmark_evidence": False}


def freeze(results):
    """Freeze a development-tested rule without fitting a new predictor."""
    if (results.get("schema") != SCHEMA or results.get("role") != "development"
            or results.get("excluded") or not results.get("points")):
        raise ValueError("need a complete validated development summary")
    rule = dict(validate_rule(results["rule"]))
    ids = sorted({row["trajectory_id"] for row in results["points"]})
    if len(ids) < 2:
        raise ValueError("at least two development trajectories are required")
    rule["development_trajectory_ids"] = ids
    rule["status"] = "development_frozen"
    return validate_rule(rule)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rule", "audit"):
        p = sub.add_parser(name); p.add_argument("--out", type=Path)
    p = sub.add_parser("freeze")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("measure")
    p.add_argument("--rollouts", type=Path, required=True)
    for name, default in (("prompts", 400), ("responses", 8), ("k", 40), ("seed", 0),
                          ("cache-step", 0), ("target-step", 100)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "rule":
        result = default_rule()
    elif args.command == "audit":
        result = counterexamples()
    elif args.command == "freeze":
        result = freeze(core.read(args.results))
    else:
        result = measure(args.rollouts, prompts=args.prompts, responses=args.responses, k=args.k,
                         seed=args.seed, cache_step=args.cache_step, target_step=args.target_step)
    if args.out:
        core.atomic_json(args.out, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
