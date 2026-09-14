"""Frozen state-dependent switch rule; no GPU imports or trial training at inference."""

from __future__ import annotations

import math
import statistics
import time

import net_gain_gate as historical
import selection_gate as core

SCHEMA = "offpolicy-selected-prefix-switch/v1"
SCHEDULE = "once_before_fixed_budget_continuation"
STEPS = (25, 50, 100)
DEV_SEEDS = (0, 1, 2)
TEST_SEEDS = (3, 4)
DEV_ARMS = ("selection_reduced", "random_reduced")
TEST_ARMS = ("selection_full", "random_full", "gated", *DEV_ARMS)
FEATURES = ("recent_reward", "recent_active_fraction", "success_rate_std", "log_prefix_updates")
MEASUREMENT = {"recent_window": 20, "max_measurement_fraction": .01, "measurement_wall_seconds": 30.}


def measurement_config(value):
    if value != MEASUREMENT:
        raise ValueError("the registered diagnostic configuration is fixed")
    return value


def measure(cache, *, stats, step, prompts, responses=8, seed=0, window=20, wall_cap=30.):
    if step not in STEPS or window != 20:
        raise ValueError("unregistered decision state or feature window")
    report = historical.measure(cache, stats=stats, step=step, prompts=prompts,
                                responses=responses, seed=seed, window=window, wall_cap=wall_cap)
    report["features"] = {k: report["features"][k] for k in FEATURES[:-1]}
    report["features"]["log_prefix_updates"] = math.log1p(step)
    report.pop("difficulty_indices")
    report.update(schema=SCHEMA, schedule=SCHEDULE)
    return report


def feature_vector(features):
    if set(features) != set(FEATURES):
        raise ValueError("missing feature or unregistered input")
    return [core.number(features[k], k, 0, 1 if k != FEATURES[-1] else None) for k in FEATURES]


def model_id(model):
    return core.fingerprint({k: v for k, v in model.items() if k != "model_id"})


def validate_model(model):
    if (model.get("schema") != SCHEMA or model.get("model_id") != model_id(model)
            or model.get("features") != list(FEATURES) or model.get("alpha") != 1.
            or model.get("margin") != 0. or model.get("data_kind") != "observed"
            or model.get("schedule") != SCHEDULE):
        raise ValueError("invalid frozen switch model")
    measurement_config(model["measurement_config"])
    if model["fit_seeds"] != list(DEV_SEEDS) or not model["fit_parent_states"]:
        raise ValueError("development provenance missing")
    for rule, n in ((model["ridge"], 4), (model["checkpoint_only"], 1)):
        for key in ("mean", "scale", "coef"):
            if len(rule[key]) != n:
                raise ValueError("invalid ridge dimensions")
            for value in rule[key]:
                core.number(value, key)
        if any(x <= 0 for x in rule["scale"]):
            raise ValueError("nonpositive scale")
        core.number(rule["intercept"], "intercept")
    return model


def fit(rows):
    """Three trajectories, unit total weight each; alpha and threshold never tuned."""
    import numpy as np
    expected = {(s, t) for s in DEV_SEEDS for t in STEPS}
    if len(rows) != len(expected) or {(r["seed"], r["step"]) for r in rows} != expected:
        raise ValueError("all nine registered development states are required")
    first = rows[0]
    for r in rows:
        if r["role"] != "development" or not r["complete"]:
            raise ValueError("test outcomes or incomplete labels cannot fit the gate")
        if not core.compatible_scope(r["scope"], first["scope"]) or r["budget_gpu_seconds"] != first["budget_gpu_seconds"]:
            raise ValueError("incompatible development conditions")
        if set(r["means"]) != set(DEV_ARMS):
            raise ValueError("labels require both diagnostic-paid controls")
        for reward in r["means"].values():
            core.number(reward, "reward", 0, 1)
    x = np.array([feature_vector(r["features"]) for r in rows])
    y = np.array([r["means"]["selection_reduced"]-r["means"]["random_reduced"] for r in rows])
    weights = np.array([1/sum(q["seed"] == r["seed"] for q in rows) for r in rows])
    def regression(values):
        mean = np.average(values, axis=0, weights=weights)
        scale = np.sqrt(np.average((values-mean)**2, axis=0, weights=weights))
        scale[scale < 1e-12] = 1.
        z = (values-mean)/scale
        intercept = float(np.average(y, weights=weights))
        coef = np.linalg.solve(z.T @ (weights[:, None]*z)+np.eye(z.shape[1]), z.T @ (weights*(y-intercept)))
        return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(), "intercept": intercept}
    model = {"schema": SCHEMA, "schedule": SCHEDULE, "data_kind": "observed", "features": list(FEATURES),
             "alpha": 1., "margin": 0., "ridge": regression(x), "checkpoint_only": regression(x[:, -1:]),
             "scope": first["scope"], "budget_gpu_seconds": first["budget_gpu_seconds"],
             "measurement_config": MEASUREMENT, "fit_seeds": list(DEV_SEEDS),
             "fit_trajectories": sorted({r["trajectory_id"] for r in rows}),
             "fit_parent_states": [r["parent"] for r in rows],
             "development_sha256": core.fingerprint(rows), "frozen_at": time.time()}
    model["model_id"] = model_id(model)
    return validate_model(model)


def check_scope(model, scope, budget, *, trajectory, parent, role, observed=False):
    validate_model(model)
    if not core.compatible_scope(model["scope"], scope) or model["budget_gpu_seconds"] != budget:
        raise ValueError("frozen model scope or budget differs")
    if role != "test" or trajectory in model["fit_trajectories"] or list(parent) in model["fit_parent_states"]:
        raise ValueError("development/test leakage")


def choose(model, features, *, checkpoint_only=False):
    validate_model(model)
    values = feature_vector(features)
    rule = model["checkpoint_only" if checkpoint_only else "ridge"]
    if checkpoint_only:
        values = values[-1:]
    prediction = rule["intercept"] + sum(w*(x-m)/s for x, m, s, w in
                                           zip(values, rule["mean"], rule["scale"], rule["coef"], strict=True))
    return {"action": "select" if prediction > 0 else "random", "prediction": prediction,
            "reason": "frozen_ridge_positive" if prediction > 0 else "frozen_ridge_nonpositive"}


def decision_audit(means, action):
    delta = means["selection_reduced"]-means["random_reduced"]
    wrong_switch = max(delta, 0.) if action == "random" else 0.
    wrong_retention = max(-delta, 0.) if action == "select" else 0.
    return {"delta": delta, "wrong_switch_loss": wrong_switch, "wrong_retention_loss": wrong_retention,
            "decision_regret": wrong_switch+wrong_retention, "near_tie": delta == 0.,
            "gate_minus_random": means["gated"]-means["random_full"],
            "gate_minus_selection": means["gated"]-means["selection_full"]}


def clustered_summary(rows):
    keys = ("delta", "decision_regret", "gate_minus_random", "gate_minus_selection")
    seeds = sorted({r["seed"] for r in rows})
    return {"independent_seeds": len(seeds), "seed_means": {
        str(seed): {k: statistics.fmean(r["audit"][k] for r in rows if r["seed"] == seed) for k in keys}
        for seed in seeds}, "uncertainty": "paired seed-level contrasts; checkpoints are not independent replicates",
        "optimal_switch_time_claim": False}
