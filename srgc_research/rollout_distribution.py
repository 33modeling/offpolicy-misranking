"""Read-only rollout diagnostics; endpoint counters never become raw traces.

Run ``python -m srgc_research.rollout_distribution --help``. This module does
not import a model, acquire a job lease, start training, or modify its inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("random", "sr", "on_policy", "switch")
ENDPOINT = re.compile(r"seed-(\d+)/(random|sr|on_policy|switch)-endpoint\.json")
METRICS = ("reward_pct", "zero_advantage_response_pct", "tokens_per_response")


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name, low=0, high=1):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return value


def summary(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def dataset_name(value):
    if value in ("math", "math_train"):
        return "math"
    if value == "mbpp":
        return value
    raise ValueError("dataset must be math, math_train or mbpp")


def endpoint_report(document, *, responses_per_update=None):
    dataset = dataset_name(document.get("dataset"))
    sources = document.get("source_results")
    if not isinstance(sources, dict):
        raise TypeError("endpoint export must contain source_results")
    if responses_per_update is not None:
        integer(responses_per_update, "responses_per_update", 1)
    rows, issues = [], []
    for path, record in sorted(sources.items()):
        match = ENDPOINT.fullmatch(path)
        if not match:
            continue  # Fixed schedules and same-prefix repeats are separate.
        seed, arm = int(match[1]), match[2]
        try:
            if not isinstance(record, dict):
                raise TypeError("endpoint must be an object")
            integer(record.get("seed"), "seed")
            if record["seed"] != seed or record.get("arm") != arm:
                raise ValueError("endpoint identity differs from its archive path")
            receipts = record.get("cost_receipts", {})
            if not isinstance(receipts, dict) or not isinstance(receipts.get("counts"), dict):
                raise TypeError("receipt counts are unavailable")
            counts = receipts["counts"]
            total = integer(counts.get("training.responses"), "training.responses", 1)
            zero = integer(counts.get("training.zero_advantage_responses"), "zero responses")
            tokens = integer(counts.get("training.generated_tokens"), "generated tokens", 1)
            if zero > total:
                raise ValueError("zero-advantage count exceeds the response count")
            horizon = integer(record.get("total_updates"), "total_updates")
            prefix = integer(record.get("shared_prefix_updates"), "shared_prefix_updates")
            if prefix > horizon:
                raise ValueError("shared prefix exceeds the endpoint horizon")
            expected = ((horizon - prefix) * responses_per_update
                        if responses_per_update is not None else None)
            row = {"dataset": dataset, "seed": seed, "arm": arm, "source_key": path,
                   "reward_pct": 100 * finite(record.get("reward"), "reward"),
                   "recorded_responses": total, "recorded_zero_advantage_responses": zero,
                   "recorded_generated_tokens": tokens,
                   "zero_advantage_response_pct": 100 * zero / total,
                   "tokens_per_response": tokens / total,
                   "total_updates": horizon, "shared_prefix_updates": prefix,
                   "expected_responses_assuming_fixed_workload": expected,
                   "receipt_matches_assumed_workload": total == expected if expected is not None else None,
                   "prefix_checkpoint_sha256": record.get("prefix_checkpoint_sha256"),
                   "input_sha256": record.get("input_sha256"),
                   "cost_measurement_complete": record.get("cost_measurement_complete")}
            rows.append(row)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({"source_key": path, "error": str(exc)})
    if not rows:
        raise ValueError("no valid four-arm endpoint receipts")
    aggregates = {arm: {metric: summary([r[metric] for r in rows if r["arm"] == arm])
                       for metric in METRICS} for arm in ARMS}
    paired = []
    by_seed = defaultdict(dict)
    for row in rows:
        by_seed[row["seed"]][row["arm"]] = row
    for seed, arms in sorted(by_seed.items()):
        for left, right in itertools.combinations(sorted(arms), 2):
            a, b = arms[left], arms[right]
            identities = ("prefix_checkpoint_sha256", "input_sha256", "total_updates", "shared_prefix_updates")
            matches = all(a[k] is not None and a[k] == b[k] for k in identities)
            paired.append({"seed": seed, "left": left, "right": right,
                           "reported_prefix_and_input_match": matches,
                           "difference_left_minus_right": {m: a[m] - b[m] for m in METRICS}})
    return {"schema": "srgc-endpoint-receipt-diagnostics-v1", "dataset": dataset,
            "measurement": "cumulative instrumented attempts, not deduplicated training groups",
            "responses_per_update_assumption": responses_per_update,
            "raw_trace_available": False, "stage_resolved": False,
            "all_correct_vs_all_wrong_available": False,
            "groups": None, "rows": rows, "aggregates": aggregates, "paired": paired,
            "issues": issues, "collection_errors": document.get("errors", []),
            "collection_warnings": document.get("warnings", [])}


def group_report(group):
    responses = group.get("responses")
    if not isinstance(responses, list) or len(responses) < 2:
        raise ValueError("a raw rollout group needs at least two responses")
    rewards = [finite(r.get("reward"), "binary reward") for r in responses
               if isinstance(r, dict)]
    if len(rewards) != len(responses) or any(r not in (0, 1) for r in rewards):
        raise ValueError("every rollout must contain a binary verified reward")
    successes = int(sum(rewards))
    sequences = []
    token_fields = [r.get("token_ids") for r in responses]
    if all(v is not None for v in token_fields):
        for sequence in token_fields:
            if not isinstance(sequence, list) or not sequence:
                raise ValueError("token_ids must contain response-only tokens")
            sequences.append(tuple(integer(t, "token id") for t in sequence))
    elif any(v is not None for v in token_fields):
        raise ValueError("partial token coverage within a rollout group")
    n = len(responses)
    grams = [set(itertools.pairwise(s)) for s in sequences]
    distances = [1 - len(a & b) / len(a | b) if a | b else float(sequences[i] != sequences[j])
                 for (i, a), (j, b) in itertools.combinations(enumerate(grams), 2)]
    finishes = [r.get("finish_reason") for r in responses]
    return {"responses": n, "successes": successes, "observed_success_fraction": successes / n,
            "outcome": "all_wrong" if successes == 0 else "all_correct" if successes == n else "mixed",
            "reward_variance": statistics.pvariance(rewards),
            "response_token_lengths": [len(s) for s in sequences] if sequences else None,
            "unique_token_sequence_fraction": len(set(sequences)) / n if sequences else None,
            "mean_token_bigram_jaccard_distance": statistics.mean(distances) if distances else None,
            "truncated_response_fraction": sum(v == "length" for v in finishes) / n
                if all(v is not None for v in finishes) else None}


def trace_report(document):
    if document.get("schema") != "srgc-rollout-trace-v1":
        raise ValueError("expected the documented srgc-rollout-trace-v1 schema")
    dataset = dataset_name(document.get("dataset"))
    seed = integer(document.get("seed"), "seed")
    if not isinstance(document.get("model"), str) or not document["model"]:
        raise ValueError("trace model identity is required")
    groups = document.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("trace groups must be a nonempty array")
    cells, identities = defaultdict(list), set()
    for group in groups:
        if not isinstance(group, dict) or group.get("arm") not in ARMS:
            raise ValueError("trace group arm is invalid")
        phase = group.get("phase")
        if phase not in ("training", "selection", "diagnostic"):
            raise ValueError("trace phase must identify training, selection or diagnostic")
        update = integer(group.get("update"), "update")
        draw = integer(group.get("draw", 0), "draw")
        prompt = group.get("prompt_id")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("trace prompt identity is required")
        identity = (group["arm"], phase, update, draw, prompt)
        if identity in identities:
            raise ValueError("duplicate raw group identity; failed attempts need separate exports")
        identities.add(identity)
        cells[(group["arm"], phase, update)].append((prompt, group_report(group)))
    rows = []
    for (arm, phase, update), cell in sorted(cells.items()):
        exposures = Counter(prompt for prompt, _ in cell)
        outcomes = Counter(r["outcome"] for _, r in cell)
        lengths = [n for _, r in cell for n in (r["response_token_lengths"] or [])]
        histogram = Counter(f"{r['successes']}/{r['responses']}" for _, r in cell)
        weights = [n / len(cell) for n in exposures.values()]
        rows.append({"arm": arm, "phase": phase, "update": update, "group_count": len(cell),
                     "response_count": sum(r["responses"] for _, r in cell),
                     "all_correct_groups": outcomes["all_correct"],
                     "all_wrong_groups": outcomes["all_wrong"], "mixed_groups": outcomes["mixed"],
                     "mixed_group_fraction": outcomes["mixed"] / len(cell),
                     "success_count_histogram": dict(sorted(histogram.items())),
                     "prompt_exposures": dict(sorted(exposures.items())),
                     "distinct_prompts": len(exposures),
                     "prompt_exposure_hhi": sum(p * p for p in weights),
                     "effective_prompt_count": math.exp(-sum(p * math.log(p) for p in weights)),
                     "response_token_length": summary(lengths),
                     "token_coverage_groups": sum(r["response_token_lengths"] is not None for _, r in cell),
                     "unique_token_sequence_fraction": summary([r["unique_token_sequence_fraction"] for _, r in cell
                         if r["unique_token_sequence_fraction"] is not None]),
                     "mean_token_bigram_jaccard_distance": summary([r["mean_token_bigram_jaccard_distance"] for _, r in cell
                         if r["mean_token_bigram_jaccard_distance"] is not None]),
                     "truncated_response_fraction": summary([r["truncated_response_fraction"] for _, r in cell
                         if r["truncated_response_fraction"] is not None])})
    exposures_by_arm = defaultdict(Counter)
    for (arm, phase, _update), cell in cells.items():
        exposures_by_arm[(arm, phase)].update(prompt for prompt, _ in cell)
    cumulative = []
    for (arm, phase), exposures in sorted(exposures_by_arm.items()):
        total = sum(exposures.values())
        weights = [count / total for count in exposures.values()]
        cumulative.append({"arm": arm, "phase": phase, "prompt_group_exposures": total,
                           "distinct_prompts": len(exposures), "prompt_exposures": dict(sorted(exposures.items())),
                           "prompt_exposure_hhi": sum(p * p for p in weights),
                           "effective_prompt_count": math.exp(-sum(p * math.log(p) for p in weights))})
    return {"schema": "srgc-raw-rollout-diagnostics-v1", "dataset": dataset, "seed": seed,
            "model": document["model"], "raw_trace_available": True,
            "semantic_diversity_measured": False, "rows": rows, "cumulative_prompt_coverage": cumulative}


def read_document(path):
    content = path.read_bytes()
    document = json.loads(content)
    if not isinstance(document, dict):
        raise TypeError("input document must be an object")
    return document, {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("endpoints", "traces"))
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--responses-per-update", type=int, default=None,
                        help="explicit workload assumption for endpoint receipt checks")
    parser.add_argument("--output", type=Path, required=True, help="new output file; never overwritten")
    args = parser.parse_args(argv)
    if args.kind == "traces" and args.responses_per_update is not None:
        parser.error("--responses-per-update applies only to endpoint receipts")
    try:
        reports = []
        for path in args.inputs:
            document, source = read_document(path)
            report = (endpoint_report(document, responses_per_update=args.responses_per_update)
                      if args.kind == "endpoints" else trace_report(document))
            reports.append({"source": source, **report})
        output = {"schema": "srgc-rollout-analysis-bundle-v1", "reports": reports,
                  "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        with args.output.open("x") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"{exc}\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
