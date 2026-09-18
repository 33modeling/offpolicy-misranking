"""CPU analysis for a preregistered on-policy/cached-selector comparison.

This is deliberately not the historical selector-versus-random gate. Labels are
GPU seconds saved to a fixed target, not rewards at a fixed update count. The
primary crossing is the first *evaluated checkpoint*, without interpolation.
"""
from __future__ import annotations

import statistics

import selection_gate as core
import selection_switch as legacy

SCHEMA = "offpolicy-selector-pair/v1"
SELECTORS = {"on_policy": "fresh_r", "cached": "difficulty"}
FEATURES = legacy.FEATURES
STEPS, DEV_SEEDS, TEST_SEEDS = legacy.STEPS, legacy.DEV_SEEDS, legacy.TEST_SEEDS


def matched_state(contracts):
    """Require identical model, optimizer, history, prompts and evaluation recipe."""
    identities = []
    for c in contracts:
        identity = {key: c[key] for key in (
            "config", "evaluation", "eval_k", "eval_seed", "n", "budget_gpu_seconds", "max_steps")}
        identity["source_hashes"] = c["source_hashes"]
        identity["prefix_certificate"] = c["selected_prefix"]["certificate_sha256"]
        identity["scope"] = {k: v for k, v in c["scope"].items() if k != "selector"}
        identities.append(identity)
    if not identities or any(item != identities[0] for item in identities[1:]):
        raise ValueError("branches do not share the same model/optimizer state and evaluation contract")
    return core.fingerprint(identities[0])


def finished_events(events):
    """Serial phase intervals; failures are included and unknown cost is fatal."""
    if not core.cost_summary(events)["complete"]:
        raise ValueError("unclosed cost events; recover receipts before analysis")
    starts, finishes, order = {}, {}, []
    active = None
    for event in events:
        key = event["event_id"]
        if event["state"] == "started":
            if key in starts:
                continue
            if active is not None:
                raise ValueError("overlapping phases in a serial branch ledger")
            starts[key], active = event, key
            order.append(key)
        else:
            if key in finishes:
                continue
            if active != key:
                raise ValueError("cost events are not ordered start/finish pairs")
            finishes[key], active = event, None
    return [(starts[key], finishes[key]) for key in order]


def cost_at_checkpoint(events, receipt, *, final=False):
    """Cost through a durable checkpoint, including startup and previous retries.

    Clock comparison is only within the checkpoint's own allocation/host. Never
    compare timestamps from different nodes or approximate by average step cost.
    Final checkpoints include the rest of their publishing train allocation.
    """
    total = training = scoring = 0.
    found = False
    for start, finish in finished_events(events):
        phase = finish["phase"]
        is_score = phase in (
            "fresh-r-validation", "fresh-r-merge-validation", "fresh-r-candidate",
            "fresh-r-merge-candidate", "difficulty-select")
        # Matched-training roots put scoring on the reporting ledger. Evaluation
        # is genuinely reporting-only; scoring must still be charged to H.
        include = finish["ledger"] != "reporting" or is_score
        charged = finish["allocated_gpu_seconds"]
        if finish["event_id"] == receipt["event_id"]:
            if phase != "train" or not include:
                raise ValueError("checkpoint receipt does not identify a train allocation")
            t0 = core.number(start["time"], "allocation start")
            t1 = core.number(finish["time"], "allocation finish")
            stamp = core.number(receipt["time"], "checkpoint timestamp")
            if abs((t1-t0)-finish["seconds"]) > max(.25, finish["seconds"]*1e-4):
                raise ValueError("wall-clock jump; checkpoint cost cannot be reconstructed safely")
            if not t0 <= stamp <= t1:
                raise ValueError("checkpoint timestamp is outside its allocation")
            if not final:
                charged = min(charged, (stamp-t0)*finish["gpus"])
            found = True
        if include:
            total += charged
            training += charged if phase == "train" else 0.
            scoring += charged if is_score else 0.
        if found:
            break
    if not found:
        raise ValueError("checkpoint has no closed allocation receipt")
    return {"gpu_seconds": total, "training_gpu_seconds": training,
            "scoring_gpu_seconds": scoring, "other_gpu_seconds": total-training-scoring}


def crossing(points, target, *, diagnosis=0., observed_gpu_seconds=None):
    """No endpoint-selected targets, extrapolation, or zero cost for non-reachers."""
    core.number(target, "target reward", 1e-12, 1.)
    core.number(diagnosis, "diagnostic cost", 0.)
    if not points or points[0]["updates"] != 0:
        raise ValueError("curve must start at the common parent")
    previous_updates, previous_cost = -1, -1.
    for point in points:
        updates = core.integer(point["updates"], "updates")
        cost = core.number(point["gpu_seconds"], "cumulative GPU seconds", 0.)
        core.number(point["reward"], "reward", 0., 1.)
        if updates <= previous_updates or cost < previous_cost:
            raise ValueError("curve updates/cost must be strictly increasing/nondecreasing")
        previous_updates, previous_cost = updates, cost
    observed = points[-1]["gpu_seconds"] if observed_gpu_seconds is None else core.number(
        observed_gpu_seconds, "observed allocation cost", points[-1]["gpu_seconds"])
    if points[0]["reward"] >= target:
        return {"status": "target_not_above_parent", "gpu_seconds": None, "updates": None}
    for point in points[1:]:
        if point["reward"] >= target:
            return {"status": "reached", "gpu_seconds": point["gpu_seconds"]+diagnosis,
                    "updates": point["updates"], "reward": point["reward"],
                    "training_gpu_seconds": point["training_gpu_seconds"],
                    "scoring_gpu_seconds": point["scoring_gpu_seconds"],
                    "diagnostic_gpu_seconds": diagnosis}
    return {"status": "right_censored", "gpu_seconds": None, "updates": None,
            "observed_through_gpu_seconds": observed+diagnosis,
            "observed_through_updates": points[-1]["updates"]}


def contrast(g, d):
    if g["status"] != "reached" or d["status"] != "reached":
        return {"h_gpu_seconds": None, "preferred": None,
                "status": "ineligible" if "target_not_above_parent" in (g["status"], d["status"]) else "censored"}
    h = d["gpu_seconds"]-g["gpu_seconds"]
    return {"h_gpu_seconds": h, "preferred": "on_policy" if h > 0 else "cached", "status": "observed"}


def fit(rows, protocol_id):
    """Fixed ridge, alpha=1, zero margin; all nine development labels required.

    Refuse incomplete/censored designs instead of silently fitting a convenient
    subset of successful states. Checkpoints within a seed have total weight 1.
    """
    import numpy as np
    expected = {(seed, step) for seed in DEV_SEEDS for step in STEPS}
    if len(rows) != len(expected) or {(r["seed"], r["step"]) for r in rows} != expected:
        raise ValueError("all nine development states required; held-out data cannot fit the rule")
    for row in rows:
        if row["role"] != "development" or row["protocol_id"] != protocol_id:
            raise ValueError("development role/protocol mismatch")
        if row["contrast"]["status"] != "observed":
            raise ValueError("unreached or ineligible target: no point label; do not fit on successful states only")
        core.number(row["contrast"]["h_gpu_seconds"], "observed H")
    x = np.array([legacy.feature_vector(r["features"]) for r in rows])
    y = np.array([r["contrast"]["h_gpu_seconds"] for r in rows])
    weights = np.array([1/sum(q["seed"] == r["seed"] for q in rows) for r in rows])
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1.
    z = (x-mean)/scale
    intercept = float(np.average(y, weights=weights))
    coef = np.linalg.solve(z.T @ (weights[:, None]*z)+np.eye(len(FEATURES)),
                           z.T @ (weights*(y-intercept)))
    model = {"schema": SCHEMA, "protocol_id": protocol_id, "data_kind": "observed",
             "features": list(FEATURES), "alpha": 1., "margin": 0., "label_unit": "GPU seconds",
             "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(), "intercept": intercept,
             "fit_seeds": list(DEV_SEEDS), "fit_states": [r["state_id"] for r in rows],
             "development_label_signs": {"positive": int(sum(y > 0)), "nonpositive": int(sum(y <= 0))},
             "development_sha256": core.fingerprint(rows)}
    model["model_id"] = core.fingerprint(model)
    return validate_model(model)


def validate_model(model):
    if (model.get("schema") != SCHEMA or model.get("data_kind") != "observed"
            or model.get("features") != list(FEATURES) or model.get("alpha") != 1.
            or model.get("margin") != 0. or model.get("fit_seeds") != list(DEV_SEEDS)
            or model.get("label_unit") != "GPU seconds"
            or model.get("model_id") != core.fingerprint({k: v for k, v in model.items() if k != "model_id"})):
        raise ValueError("invalid frozen pair model")
    for key in ("mean", "scale", "coef"):
        if len(model[key]) != len(FEATURES):
            raise ValueError("invalid model dimensions")
        for number in model[key]:
            core.number(number, key)
    if any(v <= 0 for v in model["scale"]):
        raise ValueError("invalid feature scale")
    core.number(model["intercept"], "intercept")
    return model


def choose(model, features, *, seed, state_id, protocol_id):
    validate_model(model)
    if seed not in TEST_SEEDS or state_id in model["fit_states"] or model["protocol_id"] != protocol_id:
        raise ValueError("held-out seed/state/protocol mismatch")
    values = legacy.feature_vector(features)
    prediction = model["intercept"] + sum(w*(v-m)/s for v, m, s, w in
        zip(values, model["mean"], model["scale"], model["coef"], strict=True))
    core.number(prediction, "predicted H")
    return {"selector": "on_policy" if prediction > 0 else "cached", "h_hat_gpu_seconds": prediction}


def audit(row):
    controls = row["crossings"]
    pair = contrast(controls["on_policy"], controls["cached"])
    chosen = row["decision"]["selector"]
    h = pair["h_gpu_seconds"]
    result = {**pair, "chosen": chosen, "regret_gpu_seconds": None,
              "absolute_prediction_error_gpu_seconds": None,
              "adaptive_saving_vs_on_policy": None, "adaptive_saving_vs_cached": None,
              "adaptive_saving_vs_random": None}
    if h is not None:
        result["regret_gpu_seconds"] = max(-h, 0.) if chosen == "on_policy" else max(h, 0.)
        if "h_hat_gpu_seconds" in row["decision"]:
            result["absolute_prediction_error_gpu_seconds"] = abs(row["decision"]["h_hat_gpu_seconds"]-h)
    # Actual adaptive continuation; never substitute the selected control curve.
    adaptive = controls["adaptive"]
    if adaptive["status"] == "reached":
        for name in ("on_policy", "cached", "random"):
            if controls[name]["status"] == "reached":
                result[f"adaptive_saving_vs_{name}"] = controls[name]["gpu_seconds"]-adaptive["gpu_seconds"]
    return result


def summarize(rows):
    keys = ("regret_gpu_seconds", "absolute_prediction_error_gpu_seconds",
            "adaptive_saving_vs_on_policy", "adaptive_saving_vs_cached",
            "adaptive_saving_vs_random")
    by_seed = {}
    for seed in TEST_SEEDS:
        audits = [r["audit"] for r in rows if r["seed"] == seed]
        by_seed[str(seed)] = {}
        for key in keys:
            values = [r[key] for r in audits if r.get(key) is not None]
            by_seed[str(seed)][key] = {"mean": statistics.fmean(values) if values else None,
                                      "observed_states": len(values), "expected_states": len(STEPS)}
    return {"seed_means": by_seed, "complete_test_states": len(rows), "expected_test_states": 6,
            "decision_counts": {name: sum(r["audit"].get("chosen") == name for r in rows) for name in SELECTORS},
            "uncertainty": "two independent held-out seeds; checkpoints are dependent; no question-only CI",
            "scope": "one decision at each of three separate common-prefix states; not a repeated online policy",
            "optimal_switch_time_claim": False}
