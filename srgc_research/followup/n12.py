"""N12: fixed initial-cache mixture standardization and a once-sampled panel."""

import math
import random
from collections import Counter, defaultdict

from srgc_research.rollout_distribution import group_report
from srgc_research.stage_rollout_analysis import Analysis

from .common import file_hash, fingerprint, pool, read, seed_value

STRATA = ("0/8", "1-3/8", "4/8", "5-7/8", "8/8")


def stratum(successes):
    return STRATA[0 if successes == 0 else 1 if successes < 4 else 2 if successes == 4 else 3 if successes < 8 else 4]


def panel(data, input_hash, seed):
    candidates, _, _ = pool(data)
    if len(candidates) < 20:
        raise ValueError("the fixed panel needs at least 20 candidate prompts")
    seed_value(seed)
    draw_seed = int(fingerprint({"protocol": "srgc-n12-panel-v1", "input_sha256": input_hash, "seed": seed})[:16], 16)
    return {"schema": "srgc-n12-panel-v1", "input_sha256": input_hash, "seed": seed,
            "prompt_ids": random.Random(draw_seed).sample(candidates, 20), "responses_per_prompt": 8,
            "sampling": "uniform_without_replacement_once", "role": "training_pool_diagnostic_only",
            "used_for_training_or_selection": False}


def means(rows):
    if not rows:
        return {"mixed_fraction": None, "success_rate": None, "mean_completion_tokens": None,
                "success_histogram_fraction": None}
    lengths = [g["response_token_lengths"] for g in rows]
    histogram = Counter(g["successes"] for g in rows)
    return {"mixed_fraction": sum(g["outcome"] == "mixed" for g in rows) / len(rows),
            "success_rate": sum(g["observed_success_fraction"] for g in rows) / len(rows),
            "mean_completion_tokens": sum(sum(v) for v in lengths) / (8 * len(rows)) if all(v is not None for v in lengths) else None,
            "success_histogram_fraction": [histogram[i] / len(rows) for i in range(9)]}


def standardized_window(rows, counts):
    """No positive-weight stratum may disappear from the denominator."""
    totals = Counter(stratum(n) for n in counts.values())
    weights = {s: totals[s] / len(counts) for s in STRATA}
    by_stratum = {s: [g for g in rows if stratum(counts[g["prompt_id"]]) == s] for s in STRATA}
    missing = [s for s in STRATA if weights[s] > 0 and not by_stratum[s]]
    metrics = {s: means(by_stratum[s]) for s in STRATA}
    adjusted = {}
    for key in ("mixed_fraction", "success_rate", "mean_completion_tokens"):
        adjusted[key] = (sum(weights[s] * metrics[s][key] for s in STRATA if weights[s] > 0)
                         if not missing and all(metrics[s][key] is not None for s in STRATA if weights[s] > 0) else None)
    adjusted["success_histogram_fraction"] = ([sum(weights[s] * metrics[s]["success_histogram_fraction"][i]
        for s in STRATA if weights[s] > 0) for i in range(9)] if not missing else None)
    exposure = Counter(g["prompt_id"] for g in rows)
    return {"groups": len(rows), "responses": 8 * len(rows), "unique_prompts": len(exposure),
            "repeat_exposures": len(rows) - len(exposure), "prompt_exposures": dict(exposure),
            "pool_size": len(counts), "pool_weights": weights, "unsupported_positive_strata": missing,
            "original": means(rows), "standardized": adjusted,
            "strata": [{"stratum": s, "pool_count": totals[s], "pool_weight": weights[s],
                        "groups": len(by_stratum[s]), "unique_prompts": len({g["prompt_id"] for g in by_stratum[s]}),
                        **metrics[s]} for s in STRATA]}


def report(bundle, records, *, model=None, dataset=None):
    data = read(bundle)
    _, _, counts = pool(data)
    input_hash = file_hash(bundle)
    analysis = Analysis(model=model, dataset=dataset)
    seen = set()
    for path in records:
        source = {"path": str(path), "sha256": file_hash(path)}
        if source["sha256"] in seen:
            raise ValueError("duplicate source payload")
        seen.add(source["sha256"])
        analysis.sources.append(source)
        analysis.document(read(path), source)
    raw = analysis.result()
    cohorts = [c for c in raw["cohorts"] if c["input_sha256"] == input_hash]
    matching = {c["cohort_id"] for c in cohorts}
    buckets = defaultdict(list)
    for g in raw["groups"]:
        if g["cohort_id"] not in matching or g["phase"] != "training" or g["anchor_step"] is not None:
            continue
        prompt = g["prompt_id"]
        if prompt not in counts or g["responses"] != 8:
            raise ValueError("training group is outside the bound pool or does not have eight responses")
        if g["cached_success_rate"] is not None and g["cached_success_rate"] != counts[prompt] / 8:
            raise ValueError("recorded initial cache differs from the bound input bundle")
        selected = analysis.selections.get(analysis.key(g, g["update"]))
        if selected and prompt not in selected["selected_ids"]:
            raise ValueError("training group was not in the actually applied batch")
        if g["selected_for_training"] is False:
            raise ValueError("training group explicitly marked unused")
        end = ((g["update"] - 1) // 25 + 1) * 25
        buckets[(g["cohort_id"], g["arm"], end)].append(g)
    # Explicitly retain entirely missing windows when the source records its
    # horizon. Continuation arms start at t26; the common prefix is one arm.
    for (cohort, arm, anchor), horizon in analysis.horizons.items():
        if cohort in matching and anchor is None:
            ends = (25,) if arm == "prefix" else range(50, horizon + 1, 25)
            for end in ends:
                buckets.setdefault((cohort, arm, end), [])
    windows = []
    for (cohort, arm, end), groups in sorted(buckets.items()):
        updates = Counter(g["update"] for g in groups)
        if any(n > 4 for n in updates.values()):
            raise ValueError("more than four applied training groups per update")
        windows.append({"cohort_id": cohort, "arm": arm, "start_step": end - 24, "end_step": end,
                        "recorded_updates": len(updates), "complete_training_window": len(updates) == 25 and all(n == 4 for n in updates.values()),
                        **standardized_window(groups, counts), "metadata_breakdown": metadata_breakdown(data, groups, counts)})
    return {"schema": "srgc-n12-report-v1", "input_sha256": input_hash, "sources": analysis.sources,
            "cohorts": cohorts, "excluded_cohorts": [c for c in raw["cohorts"] if c["cohort_id"] not in matching],
            "windows_25": windows, "coverage": [c for c in raw["coverage"] if c["cohort_id"] in matching],
            "training_rewards": [p for p in raw["reward_points"] if p["cohort_id"] in matching and p["kind"] == "training"],
            "held_out_rewards": [p for p in raw["reward_points"] if p["cohort_id"] in matching and p["kind"] == "held_out"],
            "issues": raw["issues"] + ([] if any(buckets.values()) else ["No source-bound applied training rollout groups were supplied; standardization is unmeasured."]),
            "interpretation": {"strata_are_true_difficulty": False, "causal_effect_estimated": False,
                               "missing_strata_renormalized": False, "training_rewards_are_held_out": False}}


def metadata_breakdown(data, rows, counts):
    result = []
    for axis in ("topic", "level"):
        labels = {}
        for prompt in counts:
            record = data.get("records", {}).get(prompt, {})
            value = record.get("subject", record.get("topic")) if axis == "topic" else record.get("level")
            if value is not None:
                if (not isinstance(value, (str, int, float)) or isinstance(value, bool) or
                        (isinstance(value, float) and not math.isfinite(value))):
                    raise ValueError("recorded topic/level must be a scalar label")
                labels[prompt] = value
        for label in sorted(set(labels.values()), key=repr):
            subset = {prompt: counts[prompt] for prompt, value in labels.items() if value == label}
            measured = standardized_window([g for g in rows if g["prompt_id"] in subset], subset)
            result.append({"axis": axis, "label": label, "metadata_pool_coverage": len(labels) / len(counts), **measured})
    return result


def panel_report(bundle, records, seed):
    data = read(bundle)
    input_hash = file_hash(bundle)
    expected = panel(data, input_hash, seed)
    _, _, counts = pool(data)
    policies, sources, duplicates = {}, [], 0
    for path in records:
        document = read(path)
        if document.get("result", {}).get("schema") == "srgc-n12-panel-measurement-v1":
            document = document["result"]
        if document.get("schema") != "srgc-n12-panel-measurement-v1" or document.get("panel") != expected:
            raise ValueError("panel measurement is not bound to the original fixed 20-prompt panel")
        identity = document["identity"]
        if identity.get("input_sha256") != input_hash or identity.get("seed") != seed:
            raise ValueError("panel measurement input/seed differs")
        policy = identity.get("policy_sha256")
        if not isinstance(policy, str) or len(policy) != 64:
            raise ValueError("panel measurement needs a policy weight identity")
        key = (identity.get("plan_sha256"), identity.get("measurement_sha256"), policy)
        if key in policies:
            if policies[key] != document:
                raise ValueError("the same policy has conflicting panel acquisitions; do not double-count it")
            duplicates += 1
            continue
        groups = document.get("groups", [])
        if len(groups) != 20 or {g["prompt_id"] for g in groups} != set(expected["prompt_ids"]):
            raise ValueError("panel measurement must cover the same 20 distinct prompts")
        for g in groups:
            if group_report(g)["responses"] != 8 or g.get("initial_successes") != counts[g["prompt_id"]]:
                raise ValueError("panel group response count or initial cache differs")
        if document.get("source", {}).get("stage") not in (25, 125, 250):
            raise ValueError("panel measurement requires its actual completed-update stage")
        policies[key] = document
        sources.append({"path": str(path), "sha256": file_hash(path)})
    baselines = {}
    for (plan_hash, code_hash, _), document in policies.items():
        if document["source"]["stage"] == 25:
            base_key = (plan_hash, code_hash)
            if base_key in baselines:
                raise ValueError("multiple different t25 policies; shared-prefix lineage must be resolved")
            baselines[base_key] = {g["prompt_id"]: group_report(g) for g in document["groups"]}
    rows = []
    for (plan_hash, code_hash, policy), document in policies.items():
        base = baselines.get((plan_hash, code_hash), {})
        for g in document["groups"]:
            measured = group_report(g)
            baseline = base.get(g["prompt_id"])
            rows.append({"policy_sha256": policy, "plan_sha256": plan_hash, "measurement_sha256": code_hash,
                         "arm": "prefix" if document["source"]["stage"] == 25 else document["source"]["arm"],
                         "stage": document["source"]["stage"], "prompt_id": g["prompt_id"],
                         "initial_successes": g["initial_successes"], "success_rate": measured["observed_success_fraction"],
                         "mixed": measured["outcome"] == "mixed", "mean_completion_tokens": sum(measured["response_token_lengths"]) / 8,
                         "success_rate_change_from_t25": measured["observed_success_fraction"] - baseline["observed_success_fraction"] if baseline else None,
                         "mean_length_change_from_t25": (sum(measured["response_token_lengths"]) - sum(baseline["response_token_lengths"])) / 8 if baseline else None})
    return {"schema": "srgc-n12-fixed-panel-report-v1", "panel": expected, "sources": sources, "rows": rows,
            "distinct_measured_policies": len(policies), "duplicate_policy_records_removed": duplicates,
            "response_count": 160 * len(policies), "is_held_out_evaluation": False}
