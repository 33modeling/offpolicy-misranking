#!/usr/bin/env python3
"""Validate and compare within-checkpoint learning interventions, not Switch endpoints."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.srgc_stage_mechanism import ARM, HORIZON, MODES, PROTOCOL, STAGES, TOTAL_WORK, code_hash, correlation, select_sets
from srgc_rebuttal.srgc import top_ids

CONTRASTS = (("on_policy", "direction_shuffle"), ("on_policy", "random"),
             ("sr", "sr_shuffle"), ("sr", "random"), ("sr", "on_policy"), ("sr_fresh", "sr"))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_endpoint(value, expected, prefix_hash, data):
    try:
        _validate_endpoint(value, expected, prefix_hash, data)
    except (TypeError, KeyError, IndexError, AttributeError) as exc:
        raise ValueError(f"malformed mechanism endpoint: {exc}") from exc


def _validate_endpoint(value, expected, prefix_hash, data):
    fields = dict(**expected, arm=ARM, protocol=PROTOCOL, study_code_sha256=code_hash(),
                  prefix_checkpoint_sha256=prefix_hash, stages=list(STAGES), horizon=HORIZON,
                  modes=list(MODES), carrier_updates=STAGES[-1], total_work_updates=TOTAL_WORK,
                  initial_state="fresh-seeded-base-model-not-prefix")
    if any(value.get(k) != v for k, v in fields.items()):
        raise ValueError("mechanism endpoint identity/protocol/work count differs")
    if value.get("checkpoint_policy", {}).get("attention") not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError("mechanism endpoint has no valid attention identity")
    rows = value.get("rows", [])
    keys = [(r["stage"], r["mode"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != {(s, m) for s in STAGES for m in MODES}:
        raise ValueError("missing/duplicate stage intervention")
    measurements = value.get("measurements", [])
    if [m["stage"] for m in measurements] != list(STAGES):
        raise ValueError("incomplete stage measurements")
    ranks = top_ids(data["candidate_ids"], [0.] * len(data["candidate_ids"]), len(data["candidate_ids"]), expected["seed"] + 1000)
    tie_order = {i: j for j, i in enumerate(ranks)}
    chosen = {}
    for m in measurements:
        ids = m["candidate_ids"]
        if len(ids) != 40 or len(set(ids)) != 40 or not set(ids) <= set(data["candidate_ids"]):
            raise ValueError("mechanism requires 40 distinct candidate prompts")
        for block in ("A", "B"):
            b = m[block]
            for field in ("cosines", "dots", "norms"):
                if len(b[field]) != 40 or not all(finite(v) for v in b[field]):
                    raise ValueError("invalid diagnostic scores")
            if (any(abs(v) > 1 + 1e-10 for v in b["cosines"]) or any(v < 0 for v in b["norms"])
                    or not finite(b["validation_norm"]) or b["validation_norm"] < 0):
                raise ValueError("invalid diagnostic cosine or norm")
            if set(b["rewards"]) != set(ids) or set(b["success_rates"]) != set(ids):
                raise ValueError("incomplete diagnostic rewards")
            for i in ids:
                rewards = b["rewards"][i]
                if (len(rewards) != 8 or any(not finite(r) or r not in (0., 1.) for r in rewards)
                        or b["success_rates"][i] != sum(rewards) / 8):
                    raise ValueError("invalid eight-response diagnostics")
        cached = {i: statistics.mean(data["cached_rewards"][i]) for i in ids}
        sets = select_sets(ids, cached, m["A"]["success_rates"], m["A"]["cosines"],
                           seed=expected["seed"], step=m["stage"], k=4, tie_order=tie_order)
        if m["selected"] != sets or m["cached_success_rates"] != cached:
            raise ValueError("selection differs from registered A-only intervention")
        b = m["B"]
        repeat = select_sets(ids, cached, b["success_rates"], b["cosines"],
                             seed=expected["seed"], step=m["stage"], k=4, tie_order=tie_order)["on_policy"]
        corr = correlation(m["A"]["cosines"], b["cosines"])
        observed_corr = m["score_correlation"]
        if ((corr is None and observed_corr is not None) or
                (corr is not None and (not finite(observed_corr) or abs(observed_corr - corr) > 1e-10))):
            raise ValueError("incorrect diagnostic score correlation")
        overlap = len(set(sets["on_policy"]) & set(repeat)) / 4
        if not finite(m["top4_overlap_fraction"]) or m["top4_overlap_fraction"] != overlap:
            raise ValueError("incorrect diagnostic top-four overlap")
        if set(m["selected_diagnostics"]) != set(MODES):
            raise ValueError("missing selected diagnostics")
        for mode, selected in sets.items():
            indices = [ids.index(i) for i in selected]
            measured = dict(
                independent_cosine=statistics.mean(b["cosines"][j] for j in indices),
                independent_dot=statistics.mean(b["dots"][j] for j in indices),
                independent_norm=statistics.mean(b["norms"][j] for j in indices),
                current_success_rate=statistics.mean(b["success_rates"][i] for i in selected),
                current_mixed_group_fraction=statistics.mean(0 < sum(b["rewards"][i]) < 8 for i in selected),
                cache_current_absolute_gap=statistics.mean(abs(cached[i] - b["success_rates"][i]) for i in selected))
            for field, expected_value in measured.items():
                actual = m["selected_diagnostics"][mode][field]
                if not finite(actual) or not math.isclose(actual, expected_value, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"incorrect selected diagnostic: {mode}/{field}")
        chosen[m["stage"]] = sets
    baselines = {}
    for row in rows:
        if row["updates"] != HORIZON or row["selected_ids"] != chosen[row["stage"]][row["mode"]]:
            raise ValueError("branch training protocol differs")
        for field, mean_field in (("baseline_per_question", "baseline_reward"), ("per_question_reward", "reward")):
            rewards = row[field]
            if (set(rewards) != set(data["evaluation_ids"]) or not rewards or
                    any(not finite(v) or not 0 <= v <= 1 for v in rewards.values()) or
                    not finite(row[mean_field]) or abs(statistics.mean(rewards.values()) - row[mean_field]) > 1e-10):
                raise ValueError("invalid held-out branch evaluation")
        baseline = baselines.setdefault(row["stage"], row["baseline_per_question"])
        if baseline != row["baseline_per_question"]:
            raise ValueError("branches do not share the same baseline evaluation")
        if not finite(row["gain_pp"]) or abs(row["gain_pp"] - 100 * (row["reward"] - row["baseline_reward"])) > 1e-8:
            raise ValueError("incorrect branch learning gain")
    training = value.get("training_records", [])
    keys = [(r["stage"], r["mode"], r["update"]) for r in training]
    if len(keys) != len(set(keys)) or set(keys) != {(s, m, u) for s in STAGES for m in MODES for u in range(1, HORIZON + 1)}:
        raise ValueError("missing/duplicate physical branch updates")
    for r in training:
        ids = chosen[r["stage"]][r["mode"]]
        if r["train_ids"] != ids or set(r["rewards"]) != set(ids):
            raise ValueError("branch did not retain its four selected prompts")
        for rewards in r["rewards"].values():
            if len(rewards) != 8 or any(not finite(v) or v not in (0., 1.) for v in rewards):
                raise ValueError("invalid fresh training rewards")
        mixed = statistics.mean(0 < sum(r["rewards"][i]) < 8 for i in ids)
        if r["mixed_group_fraction"] != mixed:
            raise ValueError("incorrect mixed reward-group statistic")
    charges = value.get("charges")
    if not isinstance(charges, list) or not charges:
        raise ValueError("missing measured component charges")
    for charge in charges:
        if charge["stage"] not in STAGES or any(not finite(charge[k]) or charge[k] < 0 for k in ("wall_seconds", "gpu_seconds")):
            raise ValueError("invalid component costs")
    observed = Counter((c["stage"], c["component"]) for c in charges)
    expected_charges = Counter()
    for stage in STAGES:
        for component in ("diagnostic.A", "diagnostic.B", "diagnostic.ranking", "diagnostic.baseline", "carrier.restore"):
            expected_charges[stage, component] = 1
        for mode in MODES:
            expected_charges[stage, f"branch.{mode}.restore"] = 1
            expected_charges[stage, f"branch.{mode}.evaluation"] = 1
            expected_charges[stage, f"branch.{mode}.train"] = HORIZON
    if observed != expected_charges:
        raise ValueError("missing/duplicate successful-action cost receipts")


def stats(values):
    return dict(n=len(values), mean_pp=statistics.mean(values) if values else None,
                sample_sd_pp=statistics.stdev(values) if len(values) > 1 else None)


def summarize(rows):
    groups = {}
    seen = set()
    for row in rows:
        key = (row["dataset"], row["seed"])
        if key in seen:
            raise ValueError("duplicate training seed")
        seen.add(key)
        value = row.get("result")
        if value:
            key = (row["dataset"], value["implementation_sha256"], value["study_code_sha256"],
                   value["checkpoint_policy"]["attention"])
            groups.setdefault(key, []).append(row)
    result = []
    for identity, group in sorted(groups.items()):
        for left, right in CONTRASTS:
            paired_by_stage = {}
            for stage in STAGES:
                pairs = []
                for row in group:
                    arms = {r["mode"]: r for r in row["result"]["rows"] if r["stage"] == stage}
                    pairs.append(dict(seed=row["seed"], difference_pp=100 * (arms[left]["reward"] - arms[right]["reward"])))
                paired_by_stage[stage] = pairs
                result.append(dict(identity=identity, stage=stage, left=left, right=right,
                    pairs=pairs, **stats([p["difference_pp"] for p in pairs])))
            # Stage-by-selector interaction, paired WITHIN seed before averaging.
            early = {p["seed"]: p["difference_pp"] for p in paired_by_stage[STAGES[0]]}
            pairs = [dict(seed=p["seed"], difference_pp=p["difference_pp"] - early[p["seed"]])
                     for p in paired_by_stage[STAGES[-1]]]
            result.append(dict(identity=identity, stage="late-minus-early", left=left, right=right,
                               pairs=pairs, **stats([p["difference_pp"] for p in pairs])))
    return result


def collect(dataset):
    from scripts.srgc_replicate_worker import tasks_for
    from scripts.srgc_sr_refresh import _endpoint, result_identity
    from srgc_rebuttal.plan import input_path, load_plan
    from srgc_rebuttal.cost_report import measured, seed_costs
    rows, errors = [], []
    for name in (("math", "mbpp") if dataset == "all" else (dataset,)):
        try:
            tasks = tasks_for(name, "mechanism")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        for task in tasks:
            row = dict(dataset=name, seed=task.seed, output=str(task.out), result=None)
            rows.append(row)
            path = task.out / f"{ARM}-endpoint.json"
            if not path.is_file():
                continue
            try:
                plan = load_plan(task.plan)
                verified = result_identity(task.plan, plan, task.seed, recorded=True)
                value = _endpoint(path, task.plan, plan, task.seed, task.folder, ARM, verified=verified)
                row["result"] = value
                accounting = seed_costs(task.folder, input_path(task.plan, plan, task.seed), (ARM,))
                row["shared_cache_build"] = accounting["cache_build"]
                row["measured_phases"] = measured(task.out / "cost-receipts" / ARM)
                row["inclusive_study_cost"] = measured(task.out / "invocations" / ARM)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"{path}: {exc}")
    sources = {str(Path(row["output"]) / f"{ARM}-endpoint.json"): row["result"]
               for row in rows if row["result"] is not None}
    return dict(rows=rows, summaries=summarize(rows), errors=errors, source_results=sources)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("math", "mbpp", "all"), required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from scripts.srgc_result_collection import publish, print_paths
    report = collect(args.dataset)
    try:
        publish(report, "mechanism")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"COLLECTION ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print("MECHANISM: same-state 25-update GRPO interventions; study costs, not deployed-strategy totals")
        for row in report["rows"]:
            value = row["result"]
            print(f"{row['dataset']} seed={row['seed']} output={row['output']}")
            if value is None:
                print("  pending")
                continue
            for r in value["rows"]:
                charges = [c["gpu_seconds"] for c in value["charges"]
                           if c["stage"] == r["stage"] and c["component"] == f"branch.{r['mode']}.train"]
                mixed = [t["mixed_group_fraction"] for t in value["training_records"]
                         if t["stage"] == r["stage"] and t["mode"] == r["mode"]]
                print(f"  t={r['stage']:3} {r['mode']:18} reward={100*r['reward']:.3f}% "
                      f"gain={r['gain_pp']:+.3f}pp train(successful)={sum(charges)/3600:.3f} GPU-h mixed={statistics.mean(mixed):.3f}")
            for m in value["measurements"]:
                print(f"  diagnostic t={m['stage']} corr={m['score_correlation']} overlap={m['top4_overlap_fraction']:.3f}")
                for block in ('A', 'B'):
                    charge = next(c for c in value['charges'] if c['stage'] == m['stage'] and c['component'] == f'diagnostic.{block}')
                    print(f"    {block} acquisition(successful)={charge['gpu_seconds']/3600:.3f} GPU-h")
            for label in ("measured_phases", "inclusive_study_cost"):
                costs = row.get(label, {})
                print(f"  {label}: complete={costs.get('complete')} GPU-s={costs.get('total_gpu_seconds')}")
            cache = row.get("shared_cache_build")
            cache_total = cache.get("invocations", {}).get("total_gpu_seconds") if cache and cache.get("complete") else None
            print(f"  existing SR cache build (once, separate): GPU-s={cache_total}")
        for s in report["summaries"]:
            print(f"PAIRED {s['identity'][0]} t={s['stage']} {s['left']}-{s['right']}: "
                  f"n={s['n']}/5 mean={s['mean_pp']} SD={s['sample_sd_pp']} pp")
        print("A selects; independent B diagnoses only. A/B acquisition, carrier, evaluation and cache are separate costs.")
        print("Successful-action charges omit discarded attempts; full phase/invocation ledgers include measured retries.")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    print_paths(report, json_output=args.json)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
