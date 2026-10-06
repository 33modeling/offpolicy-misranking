"""Paired question-level analysis and self-contained JSON result collection."""

import itertools
import json
import math

from srgc_rebuttal.runtime import atomic_json

from .design import Condition, tasks
from .storage import read_manifest, validate_artifacts, validate_endpoint


def paired_questions(left, right):
    if not left or set(left) != set(right):
        raise ValueError("paired evaluation requires exactly the same nonempty question IDs")
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
           for row in (left, right) for v in row.values()):
        raise ValueError("question rewards must be finite probabilities")
    differences = {i: right[i] - left[i] for i in left}
    return {"questions": len(left), "mean_difference": sum(differences.values()) / len(left),
            "observed_higher": sum(v > 1e-12 for v in differences.values()),
            "observed_lower": sum(v < -1e-12 for v in differences.values()),
            "observed_equal": sum(abs(v) <= 1e-12 for v in differences.values()),
            "per_question_difference": differences}


def pass_at_k(samples, k):
    if not samples or any(v not in (0, 1) for v in samples) or type(k) is not int or not 1 <= k <= len(samples):
        raise ValueError("pass@k requires binary samples and 1 <= k <= observed response count")
    n, successes = len(samples), int(sum(samples))
    return 1 - math.comb(n - successes, k) / math.comb(n, k) if n - successes >= k else 1.


def coverage(row):
    samples, settings = row.get("binary_samples"), row.get("sampling")
    if not samples or not settings:
        return {"pass_at_k": None, "reason": "raw binary responses and sampling settings not both recorded"}
    if set(samples) != set(row["per_question_reward"]):
        raise ValueError("raw sample IDs differ from evaluation IDs")
    n = settings["responses"]
    if type(n) is not int or n < 1:
        raise ValueError("evaluation response budget must be a positive integer")
    if any(len(r) != n or abs(sum(r) / n - row["per_question_reward"][i]) > 1e-10 for i, r in samples.items()):
        raise ValueError("raw samples disagree with response budget or mean reward")
    return {"pass_at_k": {str(k): sum(pass_at_k(r, k) for r in samples.values()) / len(samples)
                           for k in (1, 2, 4, 8, 16, 32) if k <= n}, "sampling": settings}


def analyze_archive(bundle):
    """Pair within dataset, seed and study; never infer unrecorded responses."""
    groups, errors, ambiguous = {}, [], set()
    for path, row in bundle.get("source_results", {}).items():
        if not isinstance(row, dict):
            errors.append(f"{path}: invalid raw endpoint")
            continue
        if "condition" in row:
            condition = row["condition"]
            if condition.get("kind") != "trajectory" or row.get("status") != "complete":
                continue
            arm, family = condition["key"], bundle.get("scope", condition["key"].split("-", 1)[0])
            if not row.get("result", {}).get("curve"):
                errors.append(f"{path}: missing trajectory evaluation")
                continue
            row = {**row, **row["result"]["curve"][-1], "total_updates": condition["updates"],
                   "plan_sha256": row.get("source_plan_sha256"), "sampling_protocol": row.get("protocol")}
        else:
            arm, family = row.get("arm"), "base"
            if arm not in {"on_policy", "sr", "switch", "random"} or "replicate-" in path or row.get("replicate"):
                continue
        seed = row.get("seed")
        if type(seed) is not int:
            errors.append(f"{path}: missing seed")
            continue
        dataset = row.get("dataset", bundle.get("dataset"))
        key = (dataset, seed, family)
        if arm in groups.setdefault(key, {}) or (key, arm) in ambiguous:
            errors.append(f"{path}: duplicate seed/arm; refusing ambiguous pairing")
            groups[key].pop(arm, None)
            ambiguous.add((key, arm))
            continue
        groups[key][arm] = row
    comparisons, covered = [], []
    for (dataset, seed, family), arms in sorted(groups.items(), key=lambda item: str(item[0])):
        for arm, row in arms.items():
            try:
                covered.append({"dataset": dataset, "seed": seed, "study": family, "arm": arm, **coverage(row)})
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"seed {seed} {arm}: {exc}")
                ambiguous.add(((dataset, seed, family), arm))
        for a, b in itertools.combinations(sorted(arms), 2):
            if any(((dataset, seed, family), arm) in ambiguous for arm in (a, b)):
                continue
            left, right = arms[a], arms[b]
            keys = ("plan_sha256", "input_sha256", "implementation_sha256", "total_updates", "sampling_protocol")
            if any(left.get(k) is None or left.get(k) != right.get(k) for k in keys):
                errors.append(f"seed {seed} {a}/{b}: unmatched experiment provenance")
                continue
            try:
                if left.get("sampling") != right.get("sampling"):
                    raise ValueError("evaluation sampling settings differ")
                comparisons.append({"dataset": dataset, "seed": seed, "study": family, "left": a, "right": b,
                    **paired_questions(left["per_question_reward"], right["per_question_reward"])})
            except (ValueError, KeyError) as exc:
                errors.append(f"seed {seed} {a}/{b}: {exc}")
    if not groups:
        errors.append("no complete raw base-arm endpoints; export the source_results bundle")
    summaries = {}
    for row in comparisons:
        key = (row["dataset"], row["study"], row["left"], row["right"])
        summaries.setdefault(key, []).append(row)
    summary = []
    for (dataset, family, a, b), rows in summaries.items():
        values = [r["mean_difference"] for r in rows]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean)**2 for v in values) / (len(values) - 1)) if len(values) > 1 else None
        summary.append({"dataset": dataset, "study": family, "left": a, "right": b,
                        "paired_seeds": [r["seed"] for r in rows], "mean_difference": mean,
                        "training_seed_sample_sd": sd})
    return {"scope": "n08", "dataset": bundle.get("dataset"), "comparisons": comparisons, "coverage": covered,
            "summary": summary,
            "errors": errors, "interpretation": "observed response frequencies, not definitive capability gain/loss",
            "uncertainty_units": {"training": "paired by seed", "responses": "not independent training seeds"}}


def budget_readout(curve, target=None, budgets=(), cache_cost=0.):
    usable = []
    for row in curve:
        ledger = row.get("costs_to_checkpoint", {})
        costs = ledger.get("known_gpu_seconds", {})
        cost = sum(costs.get(k, 0.) for k in ("selection_gpu_seconds", "training_gpu_seconds"))
        if ledger.get("complete") is True and cache_cost is not None:
            cost += cache_cost
            usable.append((cost, row))
    reached = next(((cost, r) for cost, r in usable if target is not None and r["reward"] >= target), None)
    return {"target_reward": target,
            "first_observed_target": None if reached is None else {"gpu_seconds": reached[0], "update": reached[1]["update"]},
            "at_gpu_budgets": [{"budget_gpu_seconds": budget, "last_observed": next((
                {"reward": r["reward"], "update": r["update"], "gpu_seconds": cost}
                for cost, r in reversed(usable) if cost <= budget), None)} for budget in budgets],
            "basis": "cold training plus selection including one SR cache where used; no interpolation",
            "one_time_sr_cache_gpu_seconds": cache_cost}


def collect(root, datasets, scope):
    report = {"scope": scope, "output_root": str(root), "rows": [], "source_results": {}, "errors": [], "pending": []}
    for dataset in datasets:
        for seed in range(5, 10):
            folder = root / dataset / f"seed-{seed}"
            if not (folder / "manifest.json").exists():
                report["pending"].extend(str(folder / c.key / "endpoint.json") for c in tasks(scope))
                continue
            try:
                manifest = read_manifest(folder / "manifest.json", dataset, seed)
            except (OSError, ValueError) as exc:
                report["errors"].append(f"{folder}: {exc}")
                continue
            for condition in tasks(scope):
                path = folder / condition.key / "endpoint.json"
                if not path.exists():
                    report["pending"].append(str(path))
                    continue
                try:
                    value = validate_endpoint(json.loads(path.read_text()), manifest, condition)
                    validate_artifacts(folder, manifest, condition, value)
                    relative = str(path.relative_to(root))
                    result = value["result"]
                    cost = value["cost_receipts"]
                    row = {"dataset": dataset, "seed": manifest["seed"], "condition": condition.key,
                           "endpoint": str(path), "costs": cost, "result": result}
                    if condition.kind == "trajectory":
                        row["reward"] = result["curve"][-1]["reward"]
                        row["coverage"] = coverage(result["curve"][-1])
                        cache_path = folder / "cache/endpoint.json"
                        row["one_time_sr_cache_gpu_seconds"] = 0.
                        cache_phase_total = 0.
                        row["one_time_sr_cache_role"] = "not used by this selector"
                        if condition.needs_cache:
                            cache_endpoint = validate_endpoint(json.loads(cache_path.read_text()), manifest,
                                                               Condition("cache", "cache", updates=0))
                            validate_artifacts(folder, manifest, Condition("cache", "cache", updates=0), cache_endpoint)
                            cache = cache_endpoint["cost_receipts"]
                            cache_phase_total = cache["total_gpu_seconds"]
                            row["one_time_sr_cache_gpu_seconds"] = sum(cache["known_gpu_seconds"].get(k, 0.)
                                for k in ("cache_generation_gpu_seconds", "cache_export_gpu_seconds")) if cache["complete"] else None
                            row["one_time_sr_cache_role"] = "cold cost counted once; shared cache artifact"
                        cache_cost = row["one_time_sr_cache_gpu_seconds"]
                        direct = sum(cost["known_gpu_seconds"].get(f"{p}_gpu_seconds", 0.) for p in ("selection", "training"))
                        row["cold_selection_training_gpu_seconds"] = direct + cache_cost \
                            if cache_cost is not None and cost["complete"] else None
                        row["warm_measured_phase_gpu_seconds"] = cost["total_gpu_seconds"]
                        row["cold_measured_phase_gpu_seconds"] = cost["total_gpu_seconds"] + cache_phase_total \
                            if cost["total_gpu_seconds"] is not None and cache_phase_total is not None else None
                        analysis = manifest.get("analysis_plan", {})
                        row["budget_comparison"] = budget_readout(result["curve"], analysis.get("target_reward"),
                            analysis.get("budgets_gpu_seconds", ()), cache_cost)
                    from srgc_rebuttal.cost_ledger import PhaseLedger
                    row["invocation_costs_nonadditive"] = PhaseLedger(path.parent / "invocations").totals() \
                        if (path.parent / "invocations").exists() else None
                    report["rows"].append(row)
                    report["source_results"][relative] = value
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report["errors"].append(f"{path}: {exc}")
    if not report["rows"] and not report["pending"]:
        report["pending"].append("no research manifests yet")
    groups = {}
    for row in report["rows"]:
        if "reward" in row:
            groups.setdefault((row["dataset"], row["condition"]), []).append(row)
    report["performance_summary"] = [{"dataset": dataset, "condition": condition,
        "completed_seeds": [r["seed"] for r in rows], "expected_seeds": list(range(5, 10)),
        "mean_reward": sum(r["reward"] for r in rows) / len(rows), "all_seeds_complete": len(rows) == 5}
        for (dataset, condition), rows in sorted(groups.items())]
    target = root / "exports" / f"{'-'.join(datasets)}-{scope}-results.json"
    report["result_path"] = str(target)
    atomic_json(target, report)
    return report
