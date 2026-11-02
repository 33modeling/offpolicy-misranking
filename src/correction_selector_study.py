"""Synthetic falsification study for CFCS; no real-model/GPU claims.

Uses the existing verifiable Markov environment. Selectors see sampled data
only; exact population quantities are reserved for evaluation. Fresh validation
and independent test prompt identities are disjoint. All arms train from the
same checkpoint under a fixed update budget, which is NOT an equal-cost claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from correction_selector import (
    ENDPOINTS,
    correction_components,
    crossfit_scores,
    directional_difference,
    endpoint_groups,
    fit_corrections,
)
from select_rules import jittered_topk, topk_count
from synthetic_pool_study import (
    Environment,
    StudyConfig,
    _cosine_rows,
    _sigmoid,
    current_group_gradients,
    exact_gradients,
)

METHODS = ENDPOINTS + (
    "additive",
    "cfcs",
    "anchor_cfcs",
    "anchor_best",
    "cfcs_in_sample",
    "pass_rate",
    "uniform",
    "fresh_direction",
    "fresh_cosine",
)


def token_logps(env, theta, ids, actions):
    logits = env.logits(theta, ids)
    prev = env._previous(actions)
    rows = np.arange(len(ids))[:, None, None]
    steps = np.arange(env.config.horizon)[None, None, :]
    selected = logits[rows, steps, prev]
    return -np.logaddexp(0, np.where(actions == 1, -selected, selected))


def expected_rewards(env, theta, ids):
    probabilities = _sigmoid(env.logits(theta, ids))
    state = np.zeros((len(ids), 2))
    state[:, 0] = 1
    for t in range(env.config.horizon):
        one = (state * probabilities[:, t]).sum(axis=-1)
        state = np.stack((1 - one, one), axis=-1)
    return state[:, 1]


def train_subset(
    env, selected, test_ids, *, seed, updates, batch_size=4, rollout_k=8, lr=0.5
):
    start = env.theta_pi.copy()
    rng = np.random.default_rng(seed)
    try:
        initial = float(expected_rewards(env, start, test_ids).mean())
        for _ in range(updates):
            ids = rng.choice(selected, batch_size, replace=True)
            actions = env.sample(env.theta_pi, ids, rollout_k, rng)
            rewards = actions[:, :, -1].astype(float)
            centered = rewards - rewards.mean(axis=1, keepdims=True)
            std = rewards.std(axis=1, keepdims=True)
            advantage = np.divide(
                centered, std, out=np.zeros_like(centered), where=std > 0
            )
            gradients = env.score_function(ids, actions).sum(axis=2)
            env.theta_pi += lr * (advantage[:, :, None] * gradients).mean(axis=(0, 1))
        final = float(expected_rewards(env, env.theta_pi, test_ids).mean())
    finally:
        env.theta_pi = start
    return initial, final


def run_cell(config, drift, behavior_k, seed, updates, *, export=None, anchor_k=32):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    env = Environment(config, drift, streams[0])
    candidate = env.candidates
    validation, test = np.array_split(env.validation, 2)
    # A common independent validation direction is frozen before candidate scoring.
    val_groups = current_group_gradients(env, validation, 32, 8, streams[1])
    direction = val_groups.mean(axis=(0, 1))
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm else direction
    actions = env.sample(env.theta_beta, candidate, behavior_k, streams[2])
    beta_logps = token_logps(env, env.theta_beta, candidate, actions)
    center = token_logps(env, env.theta_pi, candidate, actions)
    epsilon = 1e-3
    plus = token_logps(env, env.theta_pi + epsilon * direction, candidate, actions)
    minus = token_logps(env, env.theta_pi - epsilon * direction, candidate, actions)
    derivative = directional_difference(plus, minus, epsilon)
    groups = endpoint_groups(
        center - beta_logps,
        derivative,
        actions[:, :, -1],
        group_size=2,
        cap=config.clip_cap,
    )
    if export is not None:
        np.savez(export, endpoint_groups=groups, prompt_ids=candidate)
    exact_derivative = np.einsum(
        "nkth,h->nkt", env.score_function(candidate, actions), direction
    )
    derivative_error = float(np.abs(exact_derivative - derivative).max())
    means = groups.mean(axis=1)
    start = time.perf_counter()
    cfcs, folds = crossfit_scores(groups, seed=seed)
    calibration_seconds = time.perf_counter() - start
    anchor_rng = np.random.default_rng(seed + 7103)
    anchor_ids = np.sort(
        anchor_rng.choice(candidate, max(8, len(candidate) // 10), replace=False)
    )
    anchor_gradient = current_group_gradients(
        env, anchor_ids, anchor_k, 2, streams[4]
    ).mean(axis=1)
    anchor_scores = anchor_gradient @ direction
    anchor_start = time.perf_counter()
    anchored, anchor_folds = crossfit_scores(
        groups, seed=seed, anchor_ids=anchor_ids, anchor_scores=anchor_scores
    )
    anchor_seconds = time.perf_counter() - anchor_start
    anchor_start = time.perf_counter()
    anchor_best, _ = crossfit_scores(
        groups,
        seed=seed,
        anchor_ids=anchor_ids,
        anchor_scores=anchor_scores,
        anchor_discrete=True,
    )
    anchor_best_seconds = time.perf_counter() - anchor_start
    if export is not None:
        np.savez(
            export.with_name("example_anchors.npz"),
            anchor_ids=anchor_ids,
            anchor_scores=anchor_scores,
        )
    weights, _ = fit_corrections(groups, groups.shape[1])
    components = correction_components(groups).mean(axis=1)
    rewards = actions[:, :, -1].mean(axis=1)
    scores = dict(zip(ENDPOINTS, means.T))
    scores.update(
        {
            "additive": means[:, 1] + means[:, 2] - means[:, 0],
            "cfcs": cfcs,
            "anchor_cfcs": anchored,
            "anchor_best": anchor_best,
            "cfcs_in_sample": components[:, 0] + components[:, 1:] @ weights,
            "pass_rate": rewards * (1 - rewards),
            "uniform": np.zeros(len(candidate)),
        }
    )
    fresh = current_group_gradients(env, candidate, behavior_k, 2, streams[3]).mean(
        axis=1
    )
    scores["fresh_direction"] = fresh @ direction
    scores["fresh_cosine"] = _cosine_rows(fresh, direction)
    # These exact quantities never enter the selector or its coefficient fit.
    population = exact_gradients(env, candidate)
    known_score = population @ direction
    test_direction = exact_gradients(env, test).mean(axis=0)
    k = topk_count(len(candidate), config.topk_frac)
    optimum = float(np.sort(known_score)[-k:].mean())
    rows = []
    for method in METHODS:
        chosen = sorted(jittered_topk(dict(enumerate(scores[method])), k, seed + 71))
        before, after = train_subset(
            env, chosen, test, seed=seed + 311, updates=updates
        )
        stale_method = method not in ("fresh_direction", "fresh_cosine", "uniform")
        fresh_method = method in ("fresh_direction", "fresh_cosine")
        needs_direction = method not in ("uniform", "pass_rate")
        behavior_tokens = (
            len(candidate) * behavior_k * config.horizon if stale_method else 0
        )
        candidate_fresh_tokens = (
            len(candidate) * behavior_k * config.horizon if fresh_method else 0
        )
        if method in ("anchor_cfcs", "anchor_best"):
            candidate_fresh_tokens = len(anchor_ids) * anchor_k * config.horizon
        validation_tokens = (
            len(validation) * 32 * config.horizon if needs_direction else 0
        )
        scalar_forwards = (
            3 * behavior_tokens if stale_method and method != "pass_rate" else 0
        )
        rows.append(
            {
                "seed": seed,
                "drift": drift,
                "behavior_k": behavior_k,
                "method": method,
                "test_reward_before": before,
                "test_reward_after": after,
                "test_reward_change": after - before,
                "population_directional_regret": optimum
                - float(known_score[chosen].mean()),
                "population_test_alignment": float(
                    (population[chosen] @ test_direction).mean()
                ),
                "behavior_generation_tokens": behavior_tokens,
                "fresh_candidate_generation_tokens": candidate_fresh_tokens,
                "validation_generation_tokens": validation_tokens,
                "candidate_forward_token_evaluations": scalar_forwards,
                "fresh_candidate_gradient_tokens": candidate_fresh_tokens,
                "validation_gradient_tokens": validation_tokens,
                "training_generation_tokens": updates * 4 * 8 * config.horizon,
                "training_gradient_tokens": updates * 4 * 8 * config.horizon,
                "calibration_cpu_seconds": {
                    "cfcs": calibration_seconds,
                    "anchor_cfcs": anchor_seconds,
                    "anchor_best": anchor_best_seconds,
                }.get(method, 0.0),
                "finite_difference_max_error": derivative_error,
            }
        )
    return rows, {
        "seed": seed,
        "drift": drift,
        "behavior_k": behavior_k,
        "folds": folds,
        "anchor_folds": anchor_folds,
    }


def summarize(rows):
    summaries = []
    cells = sorted({(r["drift"], r["behavior_k"]) for r in rows})
    for drift, kb in cells:
        subset = [r for r in rows if r["drift"] == drift and r["behavior_k"] == kb]
        reference = {r["seed"]: r for r in subset if r["method"] == "cfcs"}
        anchor_reference = {
            r["seed"]: r for r in subset if r["method"] == "anchor_cfcs"
        }
        for method in METHODS:
            records = [r for r in subset if r["method"] == method]
            gain = np.array([r["test_reward_change"] for r in records])
            diff = np.array(
                [
                    reference[r["seed"]]["test_reward_after"] - r["test_reward_after"]
                    for r in records
                ]
            )
            se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else None
            anchor_diff = np.array(
                [
                    anchor_reference[r["seed"]]["test_reward_after"]
                    - r["test_reward_after"]
                    for r in records
                ]
            )
            summaries.append(
                {
                    "drift": drift,
                    "behavior_k": kb,
                    "method": method,
                    "replicates": len(records),
                    "mean_reward_change": float(gain.mean()),
                    "mean_directional_regret": float(
                        np.mean([r["population_directional_regret"] for r in records])
                    ),
                    "cfcs_minus_method_reward": float(diff.mean()),
                    "paired_standard_error": se,
                    "cfcs_win_count": int((diff > 0).sum()),
                    "anchor_minus_method_reward": float(anchor_diff.mean()),
                    "anchor_paired_standard_error": float(
                        anchor_diff.std(ddof=1) / np.sqrt(len(anchor_diff))
                    )
                    if len(anchor_diff) > 1
                    else None,
                }
            )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument(
        "--drifts", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0]
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=160)
    parser.add_argument("--anchor-k", type=int, default=32)
    args = parser.parse_args()
    if args.replicates < 1 or args.updates < 0 or args.candidates < 8:
        parser.error(
            "replicates/candidates must be positive; updates must be nonnegative"
        )
    if any(k < 8 or k % 2 for k in args.budgets):
        parser.error("budgets must be even and >=8")
    if args.anchor_k < 2 or args.anchor_k % 2:
        parser.error("anchor-k must be even and >=2")
    config = StudyConfig(
        n_candidates=args.candidates, n_validation=64, dim=16, horizon=6
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_files = (
        "correction_selector.py",
        "correction_selector_study.py",
        "synthetic_pool_study.py",
        "select_rules.py",
    )
    source_dir = args.output_dir / "source"
    source_dir.mkdir()
    source_hashes = {}
    for filename in source_files:
        original = Path(__file__).resolve().parent / filename
        shutil.copy2(original, source_dir / filename)
        source_hashes[filename] = hashlib.sha256(original.read_bytes()).hexdigest()
    metadata = {
        "schema": "cfcs-synthetic-study/v1",
        "config": asdict(config),
        "replicates": args.replicates,
        "seed_start": args.seed_start,
        "drifts": args.drifts,
        "budgets": args.budgets,
        "updates": args.updates,
        "anchor_count": max(8, args.candidates // 10),
        "anchor_k": args.anchor_k,
        "methods": METHODS,
        "source_sha256": source_hashes,
        "selection_frozen_before_training": True,
        "heldout_test": "last 32 validation-distribution prompts; exact expected terminal reward",
        "cost_scope": "operation/token counts, NOT measured LLM GPU time or equal-total-cost evidence",
        "scope": "Synthetic exploratory experiment. Neither GPS nor full CROPI/GradAlign is reproduced.",
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    all_rows, folds = [], []
    for drift in args.drifts:
        for kb in args.budgets:
            for offset in range(args.replicates):
                export = args.output_dir / "example_input.npz" if not all_rows else None
                rows, fit = run_cell(
                    config,
                    drift,
                    kb,
                    args.seed_start + offset,
                    args.updates,
                    export=export,
                    anchor_k=args.anchor_k,
                )
                all_rows.extend(rows)
                folds.append(fit)
            print(
                json.dumps(
                    {"drift": drift, "behavior_k": kb, "replicates": args.replicates}
                ),
                flush=True,
            )
    with (args.output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summarize(all_rows), indent=2, allow_nan=False)
    )
    (args.output_dir / "folds.json").write_text(
        json.dumps(folds, indent=2, allow_nan=False)
    )
    print(str(args.output_dir / "summary.json"))


if __name__ == "__main__":
    main()
