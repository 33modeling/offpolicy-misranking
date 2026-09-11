"""CPU preparation, fitting, and analysis for the pre-selection gate study."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import selection_gate as gate

STUDY_SCHEMA = "offpolicy-selection-gate-study/one-shot-v1"
BRANCHES = ("random_full", "random_reduced", "selection_reduced")
LINEAGE = ("parent_sha256", "optimizer_sha256", "pool_sha256", "evaluation_sha256",
           "learner_config_sha256", "gpu_type")


def cached_features(path, *, step, expected_prompts, expected_responses,
                    max_bytes=256*1024*1024, max_wall_seconds=30., allocated_gpus=0):
    """One CPU pass over an existing reward cache; no model or rollout generation."""
    gate.integer(step, "feature step")
    gate.integer(expected_prompts, "prompt count", 1)
    gate.integer(expected_responses, "response count", 1)
    gate.integer(max_bytes, "input byte cap", 1)
    gate.number(max_wall_seconds, "feature wall-time cap", 1e-12)
    gate.integer(allocated_gpus, "allocated GPUs")
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("feature input exceeds byte cap")
    start, cpu_start = time.perf_counter(), time.process_time()
    digest, groups, count = hashlib.sha256(), {}, 0
    with path.open("rb") as handle:
        for line in handle:
            if time.perf_counter()-start > max_wall_seconds:
                raise TimeoutError("whole-pool summary exceeded its wall-time cap")
            count += len(line)
            if count > max_bytes:
                raise ValueError("feature input grew beyond byte cap")
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            idx = gate.integer(row["prompt_idx"], "prompt index")
            ridx = gate.integer(row["rollout_idx"], "response index")
            reward = gate.number(row["reward"], "reward", 0, 1)
            if idx >= expected_prompts or ridx >= expected_responses or reward not in (0., 1.):
                raise ValueError("unexpected prompt/response index or nonbinary verifier reward")
            group = groups.setdefault(idx, {})
            if ridx in group:
                raise ValueError("duplicate cached response")
            group[ridx] = reward
    if len(groups) != expected_prompts or any(len(v) != expected_responses for v in groups.values()):
        raise ValueError("cached reward coverage incomplete")
    successes = [int(sum(v.values())) for v in groups.values()]
    means = [v/expected_responses for v in successes]
    ordered = sorted(means)
    def quantile(q):
        idx = (len(ordered)-1)*q
        lower, upper = math.floor(idx), math.ceil(idx)
        return ordered[lower]+(ordered[upper]-ordered[lower])*(idx-lower)
    elapsed = time.perf_counter()-start
    return {"schema": gate.SCHEMA, "feature_step": step,
            "source_sha256": digest.hexdigest(), "bytes": count,
            "features": {"success_rate": statistics.fmean(means), "success_rate_std": statistics.pstdev(means),
                         "mixed_group_fraction": sum(0 < v < 1 for v in means)/len(means),
                         "zero_success_fraction": means.count(0.)/len(means),
                         "all_success_fraction": means.count(1.)/len(means),
                         "success_rate_p25": quantile(.25), "success_rate_p50": quantile(.5),
                         "success_rate_p75": quantile(.75)},
            "success_count_histogram": [successes.count(i) for i in range(expected_responses+1)],
            "full_pool_coverage": True,
            "wall_seconds": elapsed, "cpu_seconds": time.process_time()-cpu_start,
            "allocated_gpu_seconds": elapsed*allocated_gpus, "allocated_gpus": allocated_gpus,
            "prompts": expected_prompts,
            "responses_per_prompt": expected_responses, "source": str(path.resolve())}


def reward_mean(branch):
    rewards = branch.get("rewards")
    if not isinstance(rewards, dict) or not rewards or any(not isinstance(k, str) or not k for k in rewards):
        raise ValueError("branch requires prompt-keyed independent evaluation rewards")
    return sum(gate.number(v, "evaluation reward", 0, 1) for v in rewards.values())/len(rewards)


def validate_point(point):
    for key in ("id", "trajectory_id"):
        if not isinstance(point.get(key), str) or not point[key]:
            raise ValueError(f"point needs {key}")
    if point.get("role") not in {"development", "calibration", "test"}:
        raise ValueError("invalid trajectory split role")
    gate.validate_scope(point.get("scope"))
    step = gate.integer(point["step"], "point step")
    if point.get("feature_step") != step:
        raise ValueError("features not available at the decision step")
    if point.get("full_pool_coverage") is not True or point.get("decision_schedule") != "once_before_training":
        raise ValueError("study requires a one-time full-pool distribution before training")
    if set(point.get("features", {})) != set(gate.FEATURES):
        raise ValueError("missing or post-decision feature")
    for key, value in point["features"].items():
        gate.number(value, key, 0, 1)
    budget = gate.number(point["budget_gpu_seconds"], "point budget", 1e-12)
    measured = gate.number(point["measurement_gpu_seconds"], "point measurement", 0, budget)
    if measured == budget:
        raise ValueError("no post-measurement training budget")
    branches = point.get("branches", {})
    if set(branches) != set(BRANCHES):
        raise ValueError("need all three matched-parent branches")
    reference = branches["random_full"]
    for arm, branch in branches.items():
        if branch.get("complete") is not True or branch.get("stop_reason") not in {"budget_exhausted", "no_block_fits"}:
            raise ValueError("incomplete branch or equal-update result used as matched compute")
        for key in LINEAGE:
            if not isinstance(branch.get(key), str) or not branch[key] or branch[key] != reference.get(key):
                raise ValueError(f"branch parent/lineage mismatch: {key}")
        if branch["pool_sha256"] != point["scope"]["pool_sha256"] or branch["gpu_type"] != point["scope"]["gpu_type"]:
            raise ValueError("branch scope mismatch")
        expected = budget if arm == "random_full" else budget-measured
        cap = gate.number(branch["budget_gpu_seconds"], "branch budget", 1e-12)
        if not math.isclose(cap, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("post-measurement budget mismatch")
        gate.number(branch["used_gpu_seconds"], "actual branch cost", 0, cap)
        reward_mean(branch)
        if set(branch["rewards"]) != set(reference["rewards"]):
            raise ValueError("evaluation prompt identities mismatch")
    means = {arm: reward_mean(branch) for arm, branch in branches.items()}
    delta = means["selection_reduced"]-means["random_reduced"]
    ell = means["random_full"]-means["random_reduced"]
    return {"id": point["id"], "trajectory_id": point["trajectory_id"], "role": point["role"],
            "step": step, "action_advantage": delta, "measurement_budget_effect": ell,
            "net_selection_gain": delta-ell, "rewards": means,
            "unspent_gpu_seconds": {arm: b["budget_gpu_seconds"]-b["used_gpu_seconds"]
                                    for arm, b in branches.items()}}


def validate_study(study):
    if study.get("schema") != STUDY_SCHEMA or study.get("data_kind") not in {"observed", "synthetic"}:
        raise ValueError("invalid study schema/data kind")
    points = study.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("no matched-parent training observations")
    identities, roles, parent_roles, summaries = set(), {}, {}, []
    for point in points:
        summary = validate_point(point)
        if point["id"] in identities:
            raise ValueError("duplicate study point")
        identities.add(point["id"])
        parent = point["trajectory_id"]
        if parent in roles and roles[parent] != point["role"]:
            raise ValueError("trajectory leakage across development/calibration/test")
        roles[parent] = point["role"]
        branch = point["branches"]["random_full"]
        parent_key = (branch["parent_sha256"], branch["optimizer_sha256"])
        if parent_key in parent_roles and parent_roles[parent_key] != point["role"]:
            raise ValueError("same checkpoint leakage under different trajectory names")
        parent_roles[parent_key] = point["role"]
        summaries.append(summary)
    if any(not gate.compatible_scope(p["scope"], points[0]["scope"]) for p in points):
        raise ValueError("fit separate studies for different model/dataset/selector scopes")
    if any(p["budget_gpu_seconds"] != points[0]["budget_gpu_seconds"] for p in points):
        raise ValueError("one fitted gate requires one declared training budget")
    return summaries


def fit(study, *, min_leaf=2, features=gate.FEATURES):
    summaries = validate_study(study)
    if not features or len(set(features)) != len(features) or set(features)-set(gate.FEATURES):
        raise ValueError("unapproved model features")
    gate.integer(min_leaf, "minimum leaf samples", 1)
    pairs = [(p, r) for p, r in zip(study["points"], summaries, strict=True) if p["role"] == "development"]
    if len({p["trajectory_id"] for p, _ in pairs}) < 2 or len(pairs) < 2*min_leaf:
        raise ValueError("need at least two development trajectories and two minimum-size leaves")
    import sklearn
    from sklearn.tree import DecisionTreeRegressor

    x = [[p["features"][f] for f in features] for p, _ in pairs]
    y = [r["net_selection_gain"] for _, r in pairs]
    counts = {p["trajectory_id"]: sum(q["trajectory_id"] == p["trajectory_id"] for q, _ in pairs) for p, _ in pairs}
    estimator = DecisionTreeRegressor(max_depth=2, min_samples_leaf=min_leaf, random_state=20260912)
    estimator.fit(x, y, sample_weight=[1/counts[p["trajectory_id"]] for p, _ in pairs])
    tree, nodes = estimator.tree_, []
    for i in range(tree.node_count):
        if tree.children_left[i] == tree.children_right[i]:
            nodes.append({"value": float(tree.value[i, 0, 0])})
        else:
            nodes.append({"feature": features[tree.feature[i]], "threshold": float(tree.threshold[i]),
                          "left": int(tree.children_left[i]), "right": int(tree.children_right[i])})
    model = {"schema": gate.SCHEMA, "target": gate.TARGET, "scope": pairs[0][0]["scope"],
             "budget_gpu_seconds": pairs[0][0]["budget_gpu_seconds"],
             "features": list(features), "nodes": nodes,
             "feature_ranges": {f: [min(p["features"][f] for p, _ in pairs),
                                     max(p["features"][f] for p, _ in pairs)] for f in features},
             "fit_trajectories": sorted(counts), "fit_points": [p["id"] for p, _ in pairs],
             "fit_parent_states": sorted({(p["branches"]["random_full"]["parent_sha256"],
                                           p["branches"]["random_full"]["optimizer_sha256"])
                                          for p, _ in pairs}),
             "data_kind": study["data_kind"], "sklearn_version": sklearn.__version__,
             "development_sha256": gate.fingerprint([p for p, _ in pairs]),
             "validation_status": "development fit; no whole-controller certification"}
    model["model_id"] = gate.fingerprint(model)
    return gate.validate_model(model)


def analyze(study, model=None, *, threshold=0.):
    summaries = validate_study(study)
    gate.number(threshold, "decision threshold", 0, 1)
    if model is not None:
        gate.validate_model(model)
        if (not gate.compatible_scope(model["scope"], study["points"][0]["scope"])
                or model["data_kind"] != study["data_kind"]
                or model["budget_gpu_seconds"] != study["points"][0]["budget_gpu_seconds"]):
            raise ValueError("model and evaluation scope/data kind differ")
        for p in study["points"]:
            if p["role"] != "development":
                branch = p["branches"]["random_full"]
                parent = (branch["parent_sha256"], branch["optimizer_sha256"])
                if (p["trajectory_id"] in model["fit_trajectories"]
                        or parent in {tuple(v) for v in model.get("fit_parent_states", [])}):
                    raise ValueError("evaluation trajectory or checkpoint was used to fit the gate")
    rows = []
    for p, row in zip(study["points"], summaries, strict=True):
        pred, action, reason = None, "random", "no_fitted_model"
        if model is not None:
            try:
                pred = gate.predict(model, p["features"])
                action = "select" if pred > threshold else "random"
                reason = "fitted_prediction"
            except ValueError:
                reason = "outside_feature_support"
        measured = model is not None
        chosen_arm = "selection_reduced" if action == "select" else "random_reduced"
        chosen = row["rewards"][chosen_arm if measured else "random_full"]
        delta = row["action_advantage"]
        rows.append({**row, "prediction": pred, "action": action, "reason": reason,
                     "measurement_performed": measured,
                     "gate_minus_random_full": chosen-row["rewards"]["random_full"],
                     "weighted_decision_error": abs(delta)*int((action == "select") != (delta > 0)) if measured else None,
                     "accounting_residual": chosen-row["rewards"]["random_full"]
                     -(int(action == "select")*delta-int(measured)*row["measurement_budget_effect"])})
    by_role = {}
    for role in ("development", "calibration", "test"):
        selected = [r for r in rows if r["role"] == role]
        if not selected:
            continue
        grouped = {}
        for row in selected:
            grouped.setdefault(row["trajectory_id"], []).append(row)
        def mean(key, grouped=grouped):
            groups = [[r[key] for r in group if r[key] is not None] for group in grouped.values()]
            values = [sum(group)/len(group) for group in groups if group]
            return sum(values)/len(values) if values else None
        by_role[role] = {"points": len(selected), "trajectories": len(grouped),
                         "selection_fraction": sum(r["action"] == "select" for r in selected)/len(selected),
                         "mean_gate_contrast": mean("gate_minus_random_full"),
                         "mean_weighted_error": mean("weighted_decision_error"),
                         "unsupported_points": sum(r["reason"] == "outside_feature_support" for r in selected)}
    return {"schema": STUDY_SCHEMA, "data_kind": study["data_kind"],
            "study_sha256": gate.fingerprint(study), "model_id": model["model_id"] if model else None,
            "threshold": threshold, "rows": rows, "by_role": by_role,
            "certified": False,
            "scope": "one initial decision between fixed full-horizon training continuations",
            "uncertainty": "descriptive trajectory-grouped means; no bootstrap or inferred seed multiplication"}


def discover(paths):
    reports = []
    for root in paths:
        root = Path(root)
        if not root.exists():
            continue
        found = [root] if root.is_file() else sorted(root.rglob("downstream_results.json"))
        for path in found:
            try:
                data = gate.read(path)
                rows = data.get("rows", [])
                random_rows = [r for r in rows if r.get("selector") == "random"]
                comparisons = []
                for random in random_rows:
                    for row in rows:
                        if row.get("selector") == "random" or any(row.get(k) != random.get(k) for k in ("dataset", "seed", "drift")):
                            continue
                        comparisons.append({"selector": row["selector"], "seed": row.get("seed"),
                                            "difference_vs_random": gate.number(row["reward_after"], "reward", 0, 1)
                                            -gate.number(random["reward_after"], "random reward", 0, 1)})
                reports.append({"path": str(path.resolve()), "complete": data.get("complete", False),
                                "comparisons": comparisons, "gate_labels_usable": False,
                                "reason": "legacy fixed-arm report lacks same-state switch and matched-cost contracts"})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                reports.append({"path": str(path), "error": str(exc), "gate_labels_usable": False})
    return {"legacy_reports": reports, "matched_switch_labels": 0}


def initialize(state_path, config, *, model=None, rollouts=None, prompts=None, responses=8, allocated_gpus=0):
    """Profile the full pool at most once; resume reads the frozen decision."""
    config.validate()
    gate.integer(allocated_gpus, "allocated GPUs")
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".initialize.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if state_path.exists():
            saved = gate.read(state_path)
            from dataclasses import asdict
            if saved.get("schema") != gate.SCHEMA or saved.get("contract") != gate.fingerprint(asdict(config)):
                raise ValueError("frozen initial gate contract changed")
            return saved["events"][saved["last_event"]]["decision"]
        obs = {"event_id": "initial", "step": config.start_step, "feature_step": config.start_step,
               "scope": config.scope, "total_used": 0., "measurement_used_now": 0., "measurement_wall_now": 0.}
        # Reject unsupported deployment before scanning the potentially large cache.
        supported = (model is not None and model.get("model_id") == config.model_id
                     and gate.compatible_scope(model.get("scope"), config.scope)
                     and model.get("data_kind") == config.data_kind
                     and model.get("budget_gpu_seconds") == config.total_gpu_seconds)
        if not supported:
            return gate.durable_decide(state_path, config, obs, model)
        gate.validate_model(model)
        cap = min(config.measurement_wall_seconds,
                  config.measurement_gpu_seconds/allocated_gpus if allocated_gpus else config.measurement_wall_seconds)
        if cap <= 0:
            obs["measurement_status"] = "measurement_budget_exhausted"
            return gate.durable_decide(state_path, config, obs, model)
        if rollouts is None:
            raise ValueError("a supported gate model requires the full-pool reward cache")
        profile_path = state_path.with_suffix(".profile.json")
        binding = gate.fingerprint({"config": config.__dict__, "rollouts": str(Path(rollouts).resolve()),
                                    "prompts": prompts, "responses": responses, "allocated_gpus": allocated_gpus})
        if profile_path.exists():
            profile = gate.read(profile_path)
            if profile.get("binding") != binding:
                raise ValueError("frozen distribution profile changed scope")
        else:
            started = time.perf_counter()
            try:
                profile = cached_features(rollouts, step=config.start_step, expected_prompts=prompts,
                                          expected_responses=responses, max_wall_seconds=cap,
                                          allocated_gpus=allocated_gpus)
                profile["measurement_status"] = "ok"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                elapsed = time.perf_counter()-started
                profile = {"wall_seconds": elapsed, "allocated_gpu_seconds": elapsed*allocated_gpus,
                           "measurement_status": "invalid_pool_distribution", "error": str(exc)}
            gate.atomic_json(profile_path, {**profile, "binding": binding})
        obs.update(total_used=profile["allocated_gpu_seconds"], measurement_used_now=profile["allocated_gpu_seconds"],
                   measurement_wall_now=profile["wall_seconds"], measurement_status=profile["measurement_status"],
                   features=profile.get("features", {}), full_pool_coverage=profile.get("full_pool_coverage", False))
        return gate.durable_decide(state_path, config, obs, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("features")
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--prompts", type=int, required=True)
    p.add_argument("--responses", type=int, default=8)
    p.add_argument("--allocated-gpus", type=int, default=0)
    p.add_argument("--max-wall-seconds", type=float, default=30.)
    for command in ("fit", "analyze"):
        p = sub.add_parser(command)
        p.add_argument("--study", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if command == "fit":
            p.add_argument("--min-leaf", type=int, default=2)
        else:
            p.add_argument("--model", type=Path)
            p.add_argument("--threshold", type=float, default=0.)
    p = sub.add_parser("decide")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--observation", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p = sub.add_parser("initialize")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--rollouts", type=Path)
    p.add_argument("--prompts", type=int)
    p.add_argument("--responses", type=int, default=8)
    p.add_argument("--allocated-gpus", type=int, default=0)
    p = sub.add_parser("cost")
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("status")
    p.add_argument("--state", type=Path, required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--paths", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = {"stage": "CPU gate implementation", "gpu_work": False,
                      "commands": ["cpu", "features", "fit", "analyze", "initialize", "decide", "cost", "inspect", "status"],
                      "gpu_entrypoint": "bash scripts/run_selection_gate_gpu.sh run",
                      "decision_schedule": "once_before_training"}
        elif args.command == "features":
            result = cached_features(args.rollouts, step=args.step, expected_prompts=args.prompts,
                                     expected_responses=args.responses, allocated_gpus=args.allocated_gpus,
                                     max_wall_seconds=args.max_wall_seconds)
        elif args.command == "fit":
            result = fit(gate.read(args.study), min_leaf=args.min_leaf)
        elif args.command == "analyze":
            result = analyze(gate.read(args.study), gate.read(args.model) if args.model else None,
                             threshold=args.threshold)
        elif args.command == "decide":
            result = gate.durable_decide(args.state, gate.GateConfig(**gate.read(args.config)),
                                         gate.read(args.observation), gate.read(args.model) if args.model else None)
        elif args.command == "initialize":
            result = initialize(args.state, gate.GateConfig(**gate.read(args.config)),
                                model=gate.read(args.model) if args.model else None, rollouts=args.rollouts,
                                prompts=args.prompts, responses=args.responses, allocated_gpus=args.allocated_gpus)
        elif args.command == "cost":
            result = gate.cost_summary([json.loads(line) for line in args.events.read_text().splitlines() if line.strip()])
        elif args.command == "status":
            result = gate.read(args.state) if args.state.exists() else {"state": "not started"}
        else:
            result = discover(args.paths)
        if getattr(args, "out", None) is not None:
            inputs = [getattr(args, k, None) for k in ("study", "model", "rollouts", "events")]
            if any(p is not None and p.resolve() == args.out.resolve() for p in inputs):
                raise ValueError("output must not overwrite input")
            if args.out.exists() and gate.read(args.out) != result:
                raise ValueError("output already contains different data; choose another output path")
            gate.atomic_json(args.out, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(f"[gate] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
