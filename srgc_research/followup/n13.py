"""N13: original candidate40/SR40 versus A-selected ON4/SR4 projected contrasts."""

import math
import re

import numpy as np

from srgc_rebuttal.srgc import cosine_scores, gradient_contrast, stream_seed, top_ids

from .common import ids, seed_value


def records(document):
    """Expose each historical check in compact exports as its own coverage row."""
    from srgc_research.stage_rollout_analysis import MAIN_RECORD

    if document.get("schema") == "srgc-n13-vectors-v1":
        return [document]
    if "datasets" in document:
        return [row for snapshot in document["datasets"].values() for row in records(snapshot)]
    if "source_results" in document:
        return [row for name, endpoint in document["source_results"].items()
                if MAIN_RECORD.fullmatch(name) and isinstance(endpoint, dict) for row in records(endpoint)]
    if "endpoints" in document:
        return [row for item in document["endpoints"] for row in records(item["endpoint"])]
    checks = document.get("checks", []) or [h for h in document.get("history", []) if h.get("d") is not None]
    if document.get("history_record", {}).get("d") is not None:
        checks = [document["history_record"]]
    if checks:
        return [{**check, "seed": document.get("seed"), "arm": document.get("arm"),
                 "identity": {k: document.get(k, document.get("identity", {}).get(k)) for k in
                              ("input_sha256", "plan_sha256", "implementation_sha256")}} for check in checks]
    if document.get("arm") in {"random", "sr", "on_policy", "switch", "prefix"}:
        return []  # An endpoint with no check is not an SR-GC observation.
    return [document]


def vectors(block, candidate_ids, sr_ids, reference_ids, dimension):
    mapping = block.get("gradients", {})
    references = block.get("reference_gradients", {})
    if set(mapping) != set(candidate_ids) | set(sr_ids) or set(references) != set(reference_ids):
        raise ValueError("gradient coverage must match the exact candidate/SR union and original reference IDs")
    def checked(value):
        raw = np.asarray(value)
        if raw.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("projected vectors must contain numeric values, not booleans or strings")
        vector = raw.astype(np.float64)
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError("projected vector must be finite and match the saved projection dimension")
        return vector
    return {i: checked(g) for i, g in mapping.items()}, np.stack([checked(references[i]) for i in reference_ids]).mean(0)


def compare(document):
    """Never reconstruct unrecorded vectors from scalar D, cosine or rewards."""
    if document.get("schema") != "srgc-n13-vectors-v1":
        value = document.get("d")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)):
            raise ValueError("recorded historical D must be finite")
        return {"status": "unmeasured", "reason": "Original projected vectors with source/ID bindings were not supplied.",
                "d40_A": None, "d4_A": None, "d40_B": None, "d4_B": None,
                "identity": document.get("identity", {}), "seed": document.get("seed"), "arm": document.get("arm"),
                "stage": document.get("checkpoint", document.get("step")), "recorded_historical_d": document.get("d")}
    identity = document.get("identity", {})
    for key in ("input_sha256", "plan_sha256", "policy_sha256", "measurement_sha256", "source_implementation_sha256"):
        if not isinstance(identity.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", identity[key]):
            raise ValueError(f"missing source/projection measurement binding: {key}")
    candidates = ids(document.get("candidate_ids"), "C40", 40)
    comparison = ids(document.get("sr_comparison_ids"), "S40", 40)
    reference = ids(document.get("reference_ids"), "original reference", 50)
    ranked = ids(document.get("cached_sr_ranked_ids"), "original cached-SR rank")
    if (not set(candidates + comparison) <= set(ranked) or set(reference) & set(ranked)):
        raise ValueError("candidate/SR/reference IDs do not match the bound pool")
    if type(document.get("stage")) is not int or document["stage"] < 0:
        raise ValueError("stage must be a completed-update checkpoint")
    seed_value(document.get("seed"))
    dimension = document.get("projection_dim")
    if type(dimension) is not int or dimension < 1:
        raise ValueError("projection_dim is required")
    blocks = document.get("blocks", {})
    if "A" not in blocks:
        return {"status": "unmeasured", "reason": "Scoring A vectors are missing.", "d40_A": None, "d4_A": None,
                "d40_B": None, "d4_B": None, "identity": identity, "stage": document["stage"], "seed": document["seed"]}
    gradient_a, reference_a = vectors(blocks["A"], candidates, comparison, reference, dimension)
    scores = cosine_scores(np.stack([gradient_a[i] for i in candidates]), reference_a)
    selected_on = list(top_ids(candidates, scores, 4, stream_seed(document["seed"] + 1000, document["stage"], "online-ties")))
    selected_sr = [i for i in ranked if i in set(candidates)][:4]
    recorded = document.get("selected_on_ids_A")
    if recorded is not None and recorded != selected_on:
        raise ValueError("recorded ON4 differs from cosine selection A with the original tie stream")
    if document.get("selected_sr_ids_A", selected_sr) != selected_sr:
        raise ValueError("recorded SR4 differs from the original cache rank restricted to C40")
    result = {"status": "measured", "identity": identity, "stage": document["stage"], "seed": document["seed"],
              "acquisition": document.get("acquisition"), "candidate_ids": candidates, "sr_comparison_ids": comparison,
              "on4_ids": selected_on, "sr4_ids": selected_sr, "selected_overlap": len(set(selected_on) & set(selected_sr)),
              "candidate_sr40_overlap": len(set(candidates) & set(comparison)),
              "d40_B": None, "d4_B": None, "sign_match_B": None, "independent_B_uses_A_selection": True,
              "actual_one_update_utility": None, "changes_live_switch_rule": False}
    for name, block in blocks.items():
        if name not in {"A", "B"}:
            raise ValueError("only independently measured scoring A/B blocks are supported")
        gradients, query = vectors(block, candidates, comparison, reference, dimension)
        d40 = gradient_contrast(candidates, comparison, gradients, query)
        d4 = gradient_contrast(selected_on, selected_sr, gradients, query)
        result.update({f"d40_{name}": d40, f"d4_{name}": d4,
                       f"sign_match_{name}": int(np.sign(d40)) == int(np.sign(d4))})
    result["d4_B_minus_A"] = result["d4_B"] - result["d4_A"] if result["d4_B"] is not None else None
    return result


def aggregate(rows):
    return {"valid_A": sum(r["status"] == "measured" for r in rows),
            "unmeasured": sum(r["status"] != "measured" for r in rows),
            "valid_B": sum(r.get("d4_B") is not None for r in rows),
            "sign_matches_A": sum(r.get("sign_match_A") is True for r in rows),
            "sign_mismatches_A": sum(r.get("sign_match_A") is False for r in rows),
            "d40_negative_d4_positive_A": sum(r.get("d40_A", 0) is not None and r.get("d4_A", 0) is not None and
                                                r.get("d40_A", 0) < 0 < r.get("d4_A", 0) for r in rows)}


def nearby_evaluation(row, analysis):
    """Only source-matched fixed held-out points; 25-update gain is association."""
    identity = row.get("identity", {})
    implementation = identity.get("source_implementation_sha256", identity.get("implementation_sha256"))
    stage, arm = row.get("stage"), identity.get("arm", row.get("arm"))
    result = {"nearby_held_out_points": [], "held_out_gain_next_25": None, "evaluation_link_is_causal": False,
              "recorded_switch_boundary": None, "evaluation_link_issue": None, "checkpoint_weight_match_verified": False}
    if stage is None or arm is None or not implementation or not identity.get("input_sha256") or not identity.get("plan_sha256"):
        return result
    matches = {key for key, cohort in analysis.cohorts.items() if cohort["seed"] == row.get("seed") and
               cohort["input_sha256"] == identity["input_sha256"] and cohort["plan_sha256"] == identity["plan_sha256"] and
               cohort["implementation_sha256"] == implementation}
    if identity.get("prefix_checkpoint_sha256"):
        matches = {key for key in matches if analysis.cohorts[key]["prefix_checkpoint_sha256"] == identity["prefix_checkpoint_sha256"]}
    relevant = {p["cohort_id"] for p in analysis.curves.values() if p["cohort_id"] in matches and
                p["kind"] == "held_out" and p["anchor_step"] is None and p["arm"] == arm}
    if len(relevant) > 1:
        result["evaluation_link_issue"] = "Multiple prefix/trajectory cohorts match; no reward curve was joined."
        return result
    points = {}
    for p in analysis.curves.values():
        if (p["cohort_id"] in matches and p["kind"] == "held_out" and p["anchor_step"] is None and
                p["step"] in (stage - 25, stage, stage + 25) and (p["arm"] == arm or (p["step"] == 25 and p["arm"] == "prefix"))):
            if p["step"] in points and points[p["step"]]["reward"] != p["reward"]:
                raise ValueError("ambiguous source-matched held-out points; resolve prefix/trajectory identities")
            points[p["step"]] = p
    result["nearby_held_out_points"] = [points[step] for step in sorted(points)]
    if stage in points and stage + 25 in points:
        result["held_out_gain_next_25"] = points[stage + 25]["reward"] - points[stage]["reward"]
    boundaries = {analysis.switches[key] for key in matches if key in analysis.switches}
    if len(boundaries) == 1:
        result["recorded_switch_boundary"] = boundaries.pop()
    return result
