"""V3 one-shot prediction of cost-inclusive continuation reward, CPU only.

The target is a fixed-budget contrast, not an optimal switching step. Legacy
equal-update E5 reports are never converted into equal-budget training labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import struct
import time
from pathlib import Path

import selection_gate as core
import selection_gate_study as legacy

SCHEMA = "offpolicy-net-gain-gate/v3-1"
FEATURES = (*core.FEATURES, "decision_step", "cache_age_steps", "recent_available",
            "recent_reward", "recent_active_fraction", "recent_gpu_seconds_per_update")
TARGET = core.TARGET
SCHEDULE = "once_before_fixed_budget_continuation"


def measurement_config(value):
    if not isinstance(value, dict) or set(value) != {"recent_window", "max_measurement_fraction", "measurement_wall_seconds"}:
        raise ValueError("missing frozen measurement configuration")
    core.integer(value["recent_window"], "recent window", 1)
    core.number(value["max_measurement_fraction"], "measurement fraction", 1e-12, .1)
    core.number(value["measurement_wall_seconds"], "measurement wall cap", 1e-12)
    return value


def state_features(stats, step, cache_step, *, window=20, wall_cap=30.):
    core.integer(step, "decision step")
    core.integer(cache_step, "cache step")
    core.integer(window, "recent window", 1)
    core.number(wall_cap, "state wall cap", 1e-12)
    if cache_step > step:
        raise ValueError("cache is newer than the decision")
    values = {"decision_step": step, "cache_age_steps": step-cache_step,
              "recent_available": 0., "recent_reward": 0.,
              "recent_active_fraction": 0., "recent_gpu_seconds_per_update": 0.}
    if step == 0:
        if stats is not None:
            raise ValueError("base checkpoint cannot use later training statistics")
        return values, None
    if stats is None:
        raise ValueError("current checkpoint statistics required; cache alone is not current state")
    started, rows, digest, size = time.monotonic(), [], hashlib.sha256(), 0
    path = Path(stats)
    if path.stat().st_size > 32*1024*1024:
        raise ValueError("training statistics exceed byte cap")
    with path.open("rb") as handle:
        previous = 0
        for line in handle:
            size += len(line)
            if size > 32*1024*1024:
                raise ValueError("training statistics grew beyond byte cap")
            if time.monotonic()-started >= wall_cap:
                raise TimeoutError("state measurement exceeded wall cap")
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            index = core.integer(row["step"], "recorded step", 1)
            if index <= previous or index > step:
                raise ValueError("duplicate, unordered, or post-decision training statistic")
            previous = index
            if index > step-window:
                groups = core.integer(row["groups"], "groups", 1)
                active = core.integer(row["nonzero_advantage_groups"], "active groups")
                if groups != 4 or active > groups:
                    raise ValueError("expected four-rank GRPO group statistics")
                rows.append((index, core.number(row["reward_mean"], "recent reward", 0, 1),
                             active/groups, 4*core.number(row["step_seconds"], "update seconds", 1e-12)))
    if [r[0] for r in rows] != list(range(max(1, step-window+1), step+1)):
        raise ValueError("incomplete recent checkpoint window")
    values.update(recent_available=1., recent_reward=statistics.fmean(r[1] for r in rows),
                  recent_active_fraction=statistics.fmean(r[2] for r in rows),
                  recent_gpu_seconds_per_update=statistics.fmean(r[3] for r in rows))
    return values, digest.hexdigest()


def measure(cache, *, stats, step, prompts, responses=8, seed=0, cache_step=0,
            wall_cap=30., window=20):
    """Read each cached response once. No model, rollout, gradient or bootstrap."""
    core.integer(prompts, "prompts", 2)
    core.integer(responses, "responses", 4)
    core.integer(seed, "seed")
    core.number(wall_cap, "measurement wall cap", 1e-12)
    if responses % 2:
        raise ValueError("even response groups required")
    started, cpu = time.monotonic(), time.process_time()
    recent, stats_hash = state_features(stats, step, cache_step, window=window, wall_cap=wall_cap)
    path = Path(cache)
    if path.stat().st_size > 256*1024*1024:
        raise ValueError("reward cache exceeds byte cap")
    groups = [[None]*responses for _ in range(prompts)]
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for line in handle:
            size += len(line)
            if size > 256*1024*1024 or time.monotonic()-started >= wall_cap:
                raise TimeoutError("whole-pool measurement exceeded its resource cap")
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            i, j = core.integer(row["prompt_idx"], "prompt index"), core.integer(row["rollout_idx"], "response index")
            reward = core.number(row["reward"], "binary reward", 0, 1)
            if i >= prompts or j >= responses or reward not in (0., 1.):
                raise ValueError("invalid cached response")
            if groups[i][j] is not None:
                raise ValueError("duplicate cached response")
            groups[i][j] = reward
    if any(v is None for group in groups for v in group):
        raise ValueError("incomplete whole-pool cache")
    means = [statistics.fmean(group) for group in groups]
    ordered = sorted(means)
    def quantile(q):
        index = (prompts-1)*q
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)
    features = {"success_rate": statistics.fmean(means), "success_rate_std": statistics.pstdev(means),
                "mixed_group_fraction": sum(0 < x < 1 for x in means)/prompts,
                "zero_success_fraction": means.count(0.)/prompts, "all_success_fraction": means.count(1.)/prompts,
                "success_rate_p25": quantile(.25), "success_rate_p50": quantile(.5), "success_rate_p75": quantile(.75),
                **recent}
    rng = random.Random(seed+701_000_003)
    ties = [rng.random() for _ in means]
    selected = sorted(sorted(range(prompts), key=lambda i: (abs(means[i]-.5), ties[i]))[:max(1, int(.1*prompts))])
    elapsed = time.monotonic()-started
    if elapsed >= wall_cap:
        raise TimeoutError("whole-pool summary exceeded wall cap")
    return {"schema": SCHEMA, "features": features, "full_pool_coverage": True,
            "feature_step": step, "cache_step": cache_step, "source_sha256": digest.hexdigest(),
            "stats_sha256": stats_hash, "recent_window": window, "prompts": prompts,
            "responses_per_prompt": responses, "difficulty_indices": selected,
            "wall_seconds": elapsed, "cpu_seconds": time.process_time()-cpu,
            "schedule": SCHEDULE, "feature_source": "pre-continuation cached rewards and existing GRPO logs"}


def validate_features(features):
    if not isinstance(features, dict) or set(features) != set(FEATURES):
        raise ValueError("missing or unapproved feature (post-training rewards are forbidden)")
    for key, value in features.items():
        high = None if key in {"decision_step", "cache_age_steps", "recent_gpu_seconds_per_update"} else 1
        core.number(value, key, 0, high)
    for key in ("decision_step", "cache_age_steps"):
        core.integer(features[key], key)
    if features["cache_age_steps"] > features["decision_step"]:
        raise ValueError("invalid cache age")
    if features["recent_available"] not in (0., 1.) or bool(features["decision_step"]) != bool(features["recent_available"]):
        raise ValueError("current-state availability does not match checkpoint")
    if not features["recent_available"] and any(features[k] for k in
            ("recent_reward", "recent_active_fraction", "recent_gpu_seconds_per_update")):
        raise ValueError("unavailable current-state features must be explicitly zero")
    if features["recent_available"] and features["recent_gpu_seconds_per_update"] <= 0:
        raise ValueError("available current state requires positive update cost")


def validate_study(data):
    if data.get("schema") != SCHEMA or data.get("label_protocol") not in {"v3_measured", "legacy_replay_development_only"}:
        raise ValueError("not a cost-inclusive v3 study")
    measurement_config(data.get("measurement_config"))
    projected = copy.deepcopy(data)
    projected["schema"] = legacy.STUDY_SCHEMA
    for point in projected.get("points", []):
        validate_features(point["features"])
        if point["features"]["decision_step"] != point["step"]:
            raise ValueError("feature checkpoint differs from decision checkpoint")
        if data["label_protocol"] == "legacy_replay_development_only" and point["role"] != "development":
            raise ValueError("replayed legacy costs cannot certify the new diagnostic on held-out runs")
        point["features"] = {k: point["features"][k] for k in core.FEATURES}
    return legacy.validate_study(projected)


def model_id(model):
    return core.fingerprint({k: v for k, v in model.items() if k != "model_id"})


def validate_model(model):
    if model.get("schema") != SCHEMA or model.get("target") != TARGET or model.get("schedule") != SCHEDULE:
        raise ValueError("unsupported net-gain model")
    if model.get("model_id") != model_id(model):
        raise ValueError("net-gain model hash changed")
    core.validate_scope(model["scope"])
    measurement_config(model.get("measurement_config"))
    core.number(model["budget_gpu_seconds"], "model budget", 1e-12)
    core.number(model["margin"], "reward margin", 0, 1)
    if model.get("data_kind") not in {"observed", "synthetic"}:
        raise ValueError("missing observed/synthetic provenance")
    if model.get("features") != list(FEATURES) or not model.get("fit_trajectories") or not model.get("fit_parent_states"):
        raise ValueError("incomplete fitted model provenance")
    ids = model["fit_trajectories"]
    if not isinstance(ids, list) or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid fitted trajectory identities")
    for parent in model["fit_parent_states"]:
        if not isinstance(parent, (list, tuple)) or len(parent) != 2 or any(not isinstance(v, str) or not v for v in parent):
            raise ValueError("invalid fitted parent identity")
    if set(model.get("feature_ranges", {})) != set(FEATURES):
        raise ValueError("missing development feature support")
    for key, bounds in model["feature_ranges"].items():
        lo, hi = bounds
        core.number(lo, key, 0)
        core.number(hi, key, lo)
    nodes, seen = model["nodes"], set()
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 7:
        raise ValueError("tree must have depth at most two")
    def visit(i, depth):
        core.integer(i, "node")
        if i >= len(nodes) or i in seen or depth > 2:
            raise ValueError("invalid tree structure")
        seen.add(i)
        node = nodes[i]
        if set(node) == {"value"}:
            core.number(node["value"], "net reward prediction", -1, 1)
        elif set(node) == {"feature", "threshold", "left", "right"} and node["feature"] in FEATURES:
            core.number(node["threshold"], "threshold", 0)
            visit(node["left"], depth+1)
            visit(node["right"], depth+1)
        else:
            raise ValueError("invalid tree node")
    visit(0, 0)
    if len(seen) != len(nodes):
        raise ValueError("unreachable model node")
    return model


def fit(data, *, margin=0., min_leaf=2):
    summaries = validate_study(data)
    if data.get("excluded"):
        raise ValueError("resolve excluded development points before fitting; do not drop failures silently")
    core.integer(min_leaf, "min leaf", 2)
    core.number(margin, "reward margin", 0, 1)
    pairs = [(p, r) for p, r in zip(data["points"], summaries) if p["role"] == "development"]
    trajectories = {p["trajectory_id"] for p, _ in pairs}
    if len(trajectories) < 3:
        raise ValueError("at least three independent development trajectories required")
    from sklearn.tree import DecisionTreeRegressor
    counts = {name: sum(p["trajectory_id"] == name for p, _ in pairs) for name in trajectories}
    estimator = DecisionTreeRegressor(max_depth=2, min_samples_leaf=min_leaf, random_state=20260914)
    x = [[p["features"][f] for f in FEATURES] for p, _ in pairs]
    estimator.fit(x, [r["net_selection_gain"] for _, r in pairs],
                  sample_weight=[1/counts[p["trajectory_id"]] for p, _ in pairs])
    leaves = estimator.apply(x)
    for leaf in set(leaves):
        if len({p["trajectory_id"] for i, (p, _) in enumerate(pairs) if leaves[i] == leaf}) < min_leaf:
            raise ValueError("tree leaf has too few independent trajectories; do not count checkpoints as seeds")
    nodes = []
    for i in range(estimator.tree_.node_count):
        tree = estimator.tree_
        nodes.append({"value": float(tree.value[i, 0, 0])} if tree.children_left[i] == tree.children_right[i] else
                     {"feature": FEATURES[tree.feature[i]], "threshold": float(tree.threshold[i]),
                      "left": int(tree.children_left[i]), "right": int(tree.children_right[i])})
    model = {"schema": SCHEMA, "target": TARGET, "schedule": SCHEDULE, "features": list(FEATURES),
             "scope": pairs[0][0]["scope"], "budget_gpu_seconds": pairs[0][0]["budget_gpu_seconds"],
             "margin": margin, "nodes": nodes, "data_kind": data["data_kind"],
             "label_protocol": data["label_protocol"], "fit_trajectories": sorted(trajectories),
             "measurement_config": data["measurement_config"],
             "fit_parent_states": sorted({(p["branches"]["random_full"]["parent_sha256"],
                                           p["branches"]["random_full"]["optimizer_sha256"]) for p, _ in pairs}),
             "feature_ranges": {f: [min(p["features"][f] for p, _ in pairs),
                                     max(p["features"][f] for p, _ in pairs)] for f in FEATURES},
             "development_sha256": core.fingerprint([p for p, _ in pairs]),
             "claim": "experimental fixed-budget reward predictor; no optimal switch-time or individual reward guarantee"}
    model["model_id"] = model_id(model)
    return validate_model(copy.deepcopy(model))


def check_scope(model, scope, budget, *, trajectory, parent, role, observed=False):
    validate_model(model)
    if not core.compatible_scope(model["scope"], scope) or model["budget_gpu_seconds"] != budget:
        raise ValueError("model/dataset/selector/hardware scope or continuation budget differs")
    if observed and model["data_kind"] != "observed":
        raise ValueError("synthetic models cannot control a GPU experiment")
    if role not in {"development", "calibration", "test"}:
        raise ValueError("invalid split role")
    if role != "development" and (trajectory in model["fit_trajectories"] or list(parent) in
            [list(v) for v in model["fit_parent_states"]]):
        raise ValueError("held-out run shares a fitted trajectory or parent state")


def choose(model, features):
    validate_model(model)
    validate_features(features)
    for name, value in features.items():
        lo, hi = model["feature_ranges"][name]
        if not lo <= value <= hi:
            return {"action": "random", "reason": "outside_development_support", "prediction": None}
    i = 0
    while "value" not in model["nodes"][i]:
        node = model["nodes"][i]
        value = struct.unpack("f", struct.pack("f", features[node["feature"]]))[0]
        i = node["left"] if value <= node["threshold"] else node["right"]
    prediction = model["nodes"][i]["value"]
    return {"action": "select" if prediction > model["margin"] else "random",
            "reason": "predicted_positive_net_gain" if prediction > model["margin"] else "predicted_nonpositive_net_gain",
            "prediction": prediction}


def analyze(data, model):
    summaries = validate_study(data)
    if data["measurement_config"] != model["measurement_config"]:
        raise ValueError("measurement configuration differs from fitted model")
    rows = []
    for p, summary in zip(data["points"], summaries):
        branch = p["branches"]["random_full"]
        check_scope(model, p["scope"], p["budget_gpu_seconds"], trajectory=p["trajectory_id"],
                    parent=(branch["parent_sha256"], branch["optimizer_sha256"]), role=p["role"])
        if model["data_kind"] != data["data_kind"]:
            raise ValueError("model and evaluation data kinds differ")
        decision = choose(model, p["features"])
        r = summary["rewards"]
        chosen = r["selection_reduced" if decision["action"] == "select" else "random_reduced"]
        rows.append({**summary, **decision, "gate_reward": chosen,
                     "gate_minus_random": chosen-r["random_full"],
                     "gate_minus_selection_after_measurement": chosen-r["selection_reduced"],
                     "forgone_net_gain": max(0., summary["net_selection_gain"]) if decision["action"] == "random" else 0.,
                     "accounting_residual": chosen-r["random_full"]-
                     ((summary["action_advantage"] if decision["action"] == "select" else 0)-summary["measurement_budget_effect"])})
    groups = {}
    for role in ("development", "calibration", "test"):
        subset = [r for r in rows if r["role"] == role]
        ids = sorted({r["trajectory_id"] for r in subset})
        if ids:
            groups[role] = {"trajectories": len(ids), "points": len(subset),
                            "mean_gate_minus_random": statistics.fmean(statistics.fmean(r["gate_minus_random"] for r in subset
                              if r["trajectory_id"] == name) for name in ids)}
    return {"schema": SCHEMA, "data_kind": data["data_kind"], "model_id": model["model_id"],
            "rows": rows, "by_role": groups, "certified": False,
            "evaluation_kind": "offline counterfactual replay, not newly executed gated training",
            "cost_note": "random fallback uses the shortened random branch and still pays diagnosis",
            "horizon": {"budget_gpu_seconds": model["budget_gpu_seconds"], "optimal_switch_step": None}}


def import_legacy(root):
    """Read only. Preserve actual old costs; charge feature replay as research."""
    root = Path(root)
    original = core.read(root / "study.json")
    legacy.validate_study(original)
    if original.get("excluded"):
        raise ValueError("legacy study contains excluded points")
    started = time.process_time()
    result = copy.deepcopy(original)
    result.update(schema=SCHEMA, label_protocol="legacy_replay_development_only")
    result["measurement_config"] = {"recent_window": 20, "max_measurement_fraction": .01, "measurement_wall_seconds": 30.}
    for p in result["points"]:
        if p["role"] != "development":
            raise ValueError("legacy import is development-only; preserve declared held-out data")
        if Path(p["id"]).name != p["id"] or p["id"] in {".", ".."}:
            raise ValueError("invalid point path")
        contract = core.read(root / "points" / p["id"] / "contract.json")
        cfg = contract["config"]
        if cfg["drift"] != p["step"] or contract["scope"] != p["scope"]:
            raise ValueError("legacy point contract mismatch")
        parent = f"policy_step_{p['step']}"
        branch = p["branches"]["random_full"]
        for field, filename in (("parent_sha256", "adapter_model.safetensors"), ("optimizer_sha256", "optimizer.pt")):
            expected = contract["source_hashes"][f"{parent}/{filename}"]
            with (Path(contract["source_run"]) / parent / filename).open("rb") as handle:
                if branch[field] != expected or hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                    raise ValueError("legacy parent binding changed")
        p["trajectory_id"] = f"{contract['scope']['model']}:seed-{cfg['seed']}"
        recent, sha = state_features(Path(contract["source_run"]) / f"policy_step_{p['step']}" / "grpo_stats.jsonl",
                                     p["step"], 0)
        p["features"].update(recent)
        p["feature_replay_stats_sha256"] = sha
    result.update(legacy_study_sha256=core.fingerprint(original), feature_replay_cpu_seconds=time.process_time()-started,
                  cost_note="historical budget labels unchanged; replay CPU is offline research, not measured deployed diagnostic cost")
    validate_study(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("import-legacy")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    for name in ("fit", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--study", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "fit":
            p.add_argument("--margin", type=float, default=0.)
            p.add_argument("--min-leaf", type=int, default=2)
        else:
            p.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    inputs = [getattr(args, name, None) for name in ("study", "model")]
    if args.command == "import-legacy":
        inputs.append(args.source_root / "study.json")
    if any(path and args.out.resolve() == path.resolve() for path in inputs) or args.out.exists():
        parser.error("output must be a new file; never overwrite source evidence or a frozen model")
    if args.command == "import-legacy":
        result = import_legacy(args.source_root)
    elif args.command == "fit":
        result = fit(core.read(args.study), margin=args.margin, min_leaf=args.min_leaf)
    else:
        result = analyze(core.read(args.study), core.read(args.model))
    core.atomic_json(args.out, result)
    print(f"[v3] {args.command}: {args.out}")


if __name__ == "__main__":
    main()
