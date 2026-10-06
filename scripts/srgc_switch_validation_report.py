#!/usr/bin/env python3
"""Seed-paired timing/rule comparisons with separately reconciled cost receipts."""

import argparse
import json
import math
from pathlib import Path
import pickle
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.srgc_replicate_worker import RULE_ARMS, TIMING_ARMS, TIMING_STEPS, tasks_for
from scripts.srgc_sr_refresh import _endpoint, result_identity
from scripts.srgc_switch_fixed import SwitchFixedEngine, fixed_step_of
from scripts.srgc_switch_rules import rule_type
from srgc_rebuttal.cost_report import seed_costs
from srgc_rebuttal.plan import input_path, load_plan

SEEDS = tuple(range(5, 10))
ARMS = ("random", "sr", "on_policy", "switch", *TIMING_ARMS, *RULE_ARMS)


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def validate_control(value, plan, folder, arm):
    """Check the executed rule/boundary, not just the filename of an endpoint."""
    if arm in RULE_ARMS:
        rule = rule_type(arm)(plan["check_interval"])
        checks = value.get("checks")
        if not isinstance(checks, list):
            raise ValueError("rule endpoint has no check history")
        switched = None
        for check in checks:
            step, d = check["step"], check["d"]
            if (type(step) is not int or not max(plan["shared_prefix_updates"], plan["first_check"])
                    <= step < plan["total_updates"] or type(d) not in (int, float)):
                raise ValueError("invalid rule check")
            if rule.observe(step, d):
                switched = step
        if switched != value.get("switched_at"):
            raise ValueError("recorded rule and transition disagree")
        stop = switched if switched is not None else plan["total_updates"] - 1
        expected = [step for step in range(max(plan["shared_prefix_updates"], plan["first_check"]), stop + 1)
                    if step % plan["check_interval"] == 0]
        if [check["step"] for check in checks] != expected:
            raise ValueError("rule checks are missing or continued after switching")
    if arm in TIMING_ARMS:
        step = fixed_step_of(arm)
        if value.get("switched_at") != step:
            raise ValueError("fixed-schedule transition differs from its named step")
        if "fixed_transition_protocol" in value:
            if (value["fixed_transition_protocol"] != SwitchFixedEngine.TRANSITION_PROTOCOL
                    or value.get("fixed_step") != step):
                raise ValueError("fixed-schedule protocol differs")
        else:
            # Old fixed200 endpoints lack protocol metadata. Certify their
            # actual boundary from the full progress history before reuse.
            progress = json.loads((folder / f"{arm}-progress.json").read_text())
            history = progress.get("history", [])
            if (progress.get("seed") != value["seed"] or progress.get("arm") != arm
                    or progress.get("step") != plan["total_updates"]
                    or [r["checkpoint"] for r in history]
                    != list(range(plan["shared_prefix_updates"], plan["total_updates"]))
                    or any(r["selector"] != ("on_policy" if r["checkpoint"] < step else "sr")
                           for r in history)):
                raise ValueError("legacy fixed-schedule boundary cannot be certified")


def measured_total(report):
    if not report or report.get("complete") is not True or not report.get("recorded_phases"):
        return None
    value = report.get("total_gpu_seconds")
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid inclusive cost receipt")
    return value


def attach_costs(row, accounting):
    cache = accounting.get("cache_build")
    cache_total = measured_total(cache.get("invocations")) if cache and cache.get("complete") is True else None
    prefix_total = measured_total(accounting["invocations"].get("prefix"))
    row["shared_costs"] = dict(cache_build_gpu_seconds=cache_total,
                               prefix_inclusive_gpu_seconds=prefix_total)
    for arm, value in row["arms"].items():
        cost = accounting["arms"][arm]
        invocation = measured_total(accounting["invocations"].get(arm)) if cost["complete"] else None
        # A cold SR-using arm incurs its cache once. Do not charge SR cache to
        # Random/On-policy, or add exclusive phase timers to inclusive timers.
        sr_cache = cache_total if arm not in ("random", "on_policy") else 0.0
        full = (invocation + prefix_total + sr_cache
                if all(v is not None for v in (invocation, prefix_total, sr_cache)) else None)
        value["cost"] = {**cost, "continuation_inclusive_gpu_seconds": invocation,
                         "protocol_cold_inclusive_gpu_seconds": full}


def comparable(arms):
    """Do not silently pair different kernels or implementation generations."""
    signatures = [(arm.get("implementation_sha256"),
                   (arm.get("checkpoint_policy") or {}).get("attention")) for arm in arms]
    return bool(signatures) and all(all(signature) for signature in signatures) and len(set(signatures)) == 1


def recorded_policy(value, folder, arm, seed, total_updates):
    from scripts.srgc_child_tuning import ATTENTION_CHOICES
    policy = value.get("checkpoint_policy")
    if policy is not None:
        if not isinstance(policy, dict) or policy.get("attention") not in ATTENTION_CHOICES:
            raise ValueError("invalid recorded attention policy")
        return policy
    # Original P0 endpoints omit attention. Read only the saved final metadata;
    # mmap keeps tensor storage on disk and weights_only rejects executable pickle.
    path = folder / f"{arm}-latest.pt"
    if not path.is_file():
        return None
    import torch
    from scripts.srgc_step_checkpoints import saved_attention
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (state.get("arm") != arm or state.get("step") != total_updates
            or state.get("config", {}).get("seed") != seed):
        raise ValueError("attention metadata checkpoint is not the completed arm")
    return {"attention": saved_attention(state), "attention_source": "final-checkpoint-metadata",
            "legacy_eager_default": "checkpoint_policy" not in state}


def paired_summary(rows, left, right):
    pairs = []
    for row in rows:
        a, b = row["arms"].get(left), row["arms"].get(right)
        if a is None or b is None or not comparable((a, b)):
            continue
        core_a = a.get("cost", {}).get("selection_training_preparation_gpu_seconds")
        core_b = b.get("cost", {}).get("selection_training_preparation_gpu_seconds")
        pairs.append(dict(seed=row["seed"], difference_pp=a["reward_percent"] - b["reward_percent"],
                          core_cost_difference_gpu_hours=(core_a - core_b) / 3600
                          if core_a is not None and core_b is not None else None))
    rewards = [p["difference_pp"] for p in pairs]
    costs = [p["core_cost_difference_gpu_hours"] for p in pairs
             if p["core_cost_difference_gpu_hours"] is not None]
    return dict(left=left, right=right, pairs=pairs, reward=stats(rewards), core_cost=stats(costs),
                wins=sum(v > 0 for v in rewards), ties=sum(v == 0 for v in rewards))


def summarize(rows):
    result = []
    groups = {}
    seen = set()
    for row in rows:
        if (row["dataset"], row["seed"]) in seen:
            raise ValueError("duplicate training seed in report")
        seen.add((row["dataset"], row["seed"]))
        reference = row["arms"].get("switch", next(iter(row["arms"].values()), {}))
        key = (row["dataset"], reference.get("implementation_sha256") or "unknown",
               (reference.get("checkpoint_policy") or {}).get("attention") or "unknown")
        groups.setdefault(key, []).append(row)
    for (dataset, implementation, attention), selected in sorted(groups.items()):
        augmented = []
        grid_rows = []
        for row in selected:
            row = {**row, "arms": dict(row["arms"])}
            grid = [row["arms"].get(arm) for arm in TIMING_ARMS]
            if all(grid) and comparable(grid):
                expected = dict(grid[0], reward_percent=statistics.mean(a["reward_percent"] for a in grid))
                costs = [a.get("cost", {}).get("selection_training_preparation_gpu_seconds") for a in grid]
                expected["cost"] = {"selection_training_preparation_gpu_seconds":
                                    statistics.mean(costs) if all(v is not None for v in costs) else None}
                row["arms"]["uniform_grid_expectation"] = expected
                grid_rows.append(row)
            augmented.append(row)
        # Secondary cross-validation, not an independent confirmatory cohort.
        # The held-out seed's rewards never choose its fixed schedule.
        folds = []
        if ({row["seed"] for row in grid_rows} == set(SEEDS)
                and comparable([row["arms"][TIMING_ARMS[0]] for row in grid_rows])):
            for row in grid_rows:
                training = [other for other in grid_rows if other["seed"] != row["seed"]]
                best = max(TIMING_ARMS, key=lambda arm: statistics.mean(
                    other["arms"][arm]["reward_percent"] for other in training))
                row["arms"]["loso_fixed"] = row["arms"][best]
                folds.append(dict(seed=row["seed"], training_seeds=[r["seed"] for r in training], arm=best))
        contrasts = [paired_summary(augmented, "switch", arm) for arm in
                     ("on_policy", "sr", *TIMING_ARMS, *RULE_ARMS, "uniform_grid_expectation", "loso_fixed")]
        result.append(dict(dataset=dataset, implementation_sha256=implementation, attention=attention,
                           comparisons=contrasts, loso_folds=folds,
                           fixed_grid_complete_seeds=[r["seed"] for r in grid_rows]))
    return result


def collect(dataset):
    rows, errors, warnings, sources = [], [], [], {}
    for name in (("math", "mbpp") if dataset == "all" else (dataset,)):
        try:
            tasks = tasks_for(name, "switch_validation")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        seen = set()
        for task in tasks:
            if task.seed in seen:
                continue
            seen.add(task.seed)
            row = dict(dataset=name, seed=task.seed, plan=str(task.plan), output=str(task.folder), arms={})
            rows.append(row)
            try:
                plan = load_plan(task.plan)
                present = {arm: task.folder / f"{arm}-endpoint.json" for arm in ARMS
                           if (task.folder / f"{arm}-endpoint.json").is_file()}
                if not present:
                    continue
                verified = result_identity(task.plan, plan, task.seed, recorded=True)
                for arm, path in present.items():
                    try:
                        captured = {}
                        value = _endpoint(path, task.plan, plan, task.seed, task.folder, arm,
                                          verified=verified, raw_records=captured)
                        raw = captured[str(path)]
                        validate_control(raw, plan, task.folder, arm)
                        try:
                            value["checkpoint_policy"] = recorded_policy(
                                value, task.folder, arm, task.seed, plan["total_updates"])
                        except (ImportError, OSError, ValueError, TypeError, KeyError, RuntimeError,
                                EOFError, pickle.UnpicklingError) as exc:
                            value["checkpoint_policy"] = None
                            warnings.append(f"{name}/seed-{task.seed}/{arm} attention: {exc}")
                        if value["checkpoint_policy"] is None:
                            warnings.append(f"{name}/seed-{task.seed}/{arm}: attention unverified; reward shown, pairing excluded")
                        value.update(check_count=len(raw.get("checks", [])),
                                     selection_count=len(raw.get("selection_steps", [])))
                        row["arms"][arm] = value
                        sources[str(path)] = raw
                    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                        errors.append(f"{path}: {exc}")
                if row["arms"]:
                    try:
                        accounting = seed_costs(task.folder, input_path(task.plan, plan, task.seed), tuple(row["arms"]))
                        attach_costs(row, accounting)
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        errors.append(f"{name}/seed-{task.seed} cost: {exc}")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"{task.key}: {exc}")
    return dict(protocol="switch-timing-rules-20261006-v1", timing_steps=TIMING_STEPS,
                expected_seeds=SEEDS, rows=rows, summaries=summarize(rows), errors=errors, warnings=warnings,
                source_results=sources)


def number(value):
    return "-" if value is None else f"{value:.3f}"


def hours(value):
    return number(value / 3600 if value is not None else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("math", "mbpp", "all"), required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from scripts.srgc_result_collection import publish, print_paths
    report = collect(args.dataset)
    try:
        publish(report, "switch_validation")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"COLLECTION ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print("SWITCH VALIDATION: same-prefix endpoints; missing/unverified costs remain '-'")
        print("GPU-h columns: selection training preparation evaluation checkpoint startup | core | inclusive | cold")
        for row in report["rows"]:
            shared = row.get("shared_costs", {})
            print(f"output={row['output']}")
            print(f"{row['dataset']} seed={row['seed']} cache_build={hours(shared.get('cache_build_gpu_seconds'))} "
                  f"prefix_inclusive={hours(shared.get('prefix_inclusive_gpu_seconds'))} GPU-h")
            for arm in ARMS:
                value = row["arms"].get(arm, {})
                cost = value.get("cost", {})
                columns = " ".join(hours(cost.get(f"{phase}_gpu_seconds")) for phase in
                                   ("selection", "training", "preparation", "evaluation", "checkpoint", "startup"))
                totals = " | ".join(hours(cost.get(key)) for key in
                                    ("selection_training_preparation_gpu_seconds", "continuation_inclusive_gpu_seconds",
                                     "protocol_cold_inclusive_gpu_seconds"))
                print(f"  {arm:20} reward={number(value.get('reward_percent'))}% "
                      f"switch={value.get('switched_at')} checks={value.get('check_count', '-')} "
                      f"refreshes={value.get('selection_count', '-')} GPU-h: {columns} | {totals}")
        for summary in report["summaries"]:
            print(f"PAIRED {summary['dataset']} implementation={summary['implementation_sha256']} "
                  f"attention={summary['attention']}")
            for comparison in summary["comparisons"]:
                reward, cost = comparison["reward"], comparison["core_cost"]
                print(f"{summary['dataset']} {comparison['left']} - {comparison['right']}: "
                      f"reward n={reward['n']}/5 mean={number(reward['mean'])} SD={number(reward['sample_sd'])} pp; "
                      f"core cost n={cost['n']}/5 mean={number(cost['mean'])} GPU-h")
        print("core=selection+training+preparation; inclusive contains phases, never add them again")
        print("cold=continuation inclusive+prefix inclusive+one SR cache (Random/On-policy exclude SR cache)")
        print("inclusive includes evaluation/checkpoint/startup inside recorded invocations, not external node admission")
        print("uniform_grid_expectation is the average of all five schedules, not another executed run")
        print("LOSO chooses on four seeds; ties prefer earlier step; secondary analysis, not independent confirmation")
        print("paired comparisons require matching implementation and recorded attention kernels")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    print_paths(report, json_output=args.json)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
