"""Read new-model costs using the frozen submitted OLMo continuation criterion.

The submitted comparison uses update timers on the retained training trajectory,
full cumulative selection, and one common cached-SR subset preparation charge.
Inclusive session costs and completed receipts from other attempts remain separate.
This reader never changes a learner, protocol, checkpoint or archived measurement.
"""

import json
import math
import statistics
from pathlib import Path

from srgc_rebuttal.plan import digest, input_path
from srgc_rebuttal.srgc import Engine

BASELINE = "3d144ee666e480766e253d06221f6e29feaef25e"
FIELDS = ("selection_gpu_seconds", "training_gpu_seconds")


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("update costs must be finite nonnegative GPU-seconds")
    return float(value)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def ranges(steps):
    """Keep missing intervals explicit without repeating hundreds of integers."""
    result = []
    for step in steps:
        if result and result[-1][1] + 1 == step:
            result[-1][1] = step
        else:
            result.append([step, step])
    return result


def unique_history(history):
    """Exact re-export copies are one update; conflicting branches are not guesses."""
    if not isinstance(history, list):
        raise TypeError("timed training history must be a list")
    seen, records, duplicates = {}, [], []
    for record in history:
        if not isinstance(record, dict):
            raise TypeError("timed update must be an object")
        step = record.get('completed_updates')
        if type(step) is not int:
            raise ValueError('timed updates must be contiguous on the retained trajectory')
        if step in seen:
            if record != seen[step]:
                raise ValueError('conflicting retained records for the same update')
            duplicates.append(record)
            continue
        seen[step] = record
        records.append(record)
    return records, duplicates


def history_costs(history, *, start, stop):
    """Retain one recorded cost per completed update; never infer missing timers."""
    history, _ = unique_history(history)
    points = []
    previous = start
    for record in history:
        if not isinstance(record, dict):
            raise TypeError("timed update must be an object")
        step = record.get("completed_updates")
        if type(step) is not int or step != previous + 1 or record.get("checkpoint") != step - 1 or step > stop:
            raise ValueError("timed updates must be contiguous on the retained trajectory")
        point = {"step": step, "checkpoint": step - 1}
        for field in FIELDS:
            point[field] = number(record[field]) if field in record else None
        if 'timing_receipt_ids' in record:
            ids = record['timing_receipt_ids']
            if not isinstance(ids, dict) or any(k not in {'selection', 'training'} or not isinstance(v, str) or not v
                                                for k, v in ids.items()):
                raise ValueError('invalid retained update timing receipt IDs')
            point['timing_receipt_ids'] = dict(ids)
        points.append(point)
        previous = step
    if previous != stop:
        raise ValueError("timed history does not reach the recorded progress step")
    return points


def checked_progress(folder, arm, plan, seed, read):
    path = folder / f"{arm}-progress.json"
    progress = read(path)
    if progress is None:
        return None
    if not isinstance(progress, dict):
        raise TypeError("timed progress must be an object")
    if (progress.get("seed") != seed or progress.get("arm") != arm or
            progress.get("sampling_protocol") != Engine.SAMPLING_PROTOCOL):
        raise ValueError("timed progress seed/arm/sampling protocol mismatch")
    step = progress.get("step")
    if type(step) is not int or not plan["shared_prefix_updates"] <= step <= plan["total_updates"]:
        raise ValueError("timed progress step is outside the frozen plan")
    points = history_costs(progress.get("history"), start=plan["shared_prefix_updates"], stop=step)
    _, duplicates = unique_history(progress['history'])
    return {"path": str(path), "step": step, "points": points, "costs": progress.get("costs", {}),
            'duplicate_history_entries': len(duplicates),
            'duplicate_history_gpu_seconds': {field: math.fsum(number(r[field]) for r in duplicates)
                                             if all(field in r for r in duplicates) else None for field in FIELDS}}


def collect(folder, arm, plan, seed, *, read=read_json):
    folder = Path(folder)
    start, stop = plan["shared_prefix_updates"], plan["total_updates"]
    progress = checked_progress(folder, arm, plan, seed, read)
    points = progress["points"] if progress else []
    # SR and Switch reuse the same initial cache: charge its SR subset preparation
    # once to both arms, as in the submitted OLMo comparison. Cache generation is separate.
    preparation = 0.0 if arm in {"random", "on_policy"} else None
    preparation_source = None
    if arm in {"sr", "switch"}:
        shared = progress if arm == "sr" else checked_progress(folder, "sr", plan, seed, read)
        if shared is not None:
            costs = shared["costs"]
            if not isinstance(costs, dict):
                raise TypeError("SR preparation costs must be an object")
            if "sr_preparation_gpu_seconds" in costs:
                preparation = number(costs["sr_preparation_gpu_seconds"])
                preparation_source = shared["path"]
    known = {field: math.fsum(p[field] for p in points if p[field] is not None) for field in FIELDS}
    recorded = {field: sum(p[field] is not None for p in points) for field in FIELDS}
    complete = len(points) == stop - start and all(n == stop - start for n in recorded.values()) and preparation is not None
    measured = bool(points) and any(recorded.values())
    values = {"selection": known[FIELDS[0]] if recorded[FIELDS[0]] else None,
              "training": known[FIELDS[1]] if recorded[FIELDS[1]] else None,
              "preparation": preparation,
              "primary": sum(known.values()) + preparation if complete else None}
    missing = {field: [step for step in range(start + 1, stop + 1)
                       if not any(p["step"] == step and p[field] is not None for p in points)] for field in FIELDS}
    from .restart_costs import reconcile

    try:
        restart = reconcile(folder, arm, points, start=start, stop=stop, read=read,
                            expected_gpus=plan.get('world_size')) if progress else {'record': 'MISSING'}
    except (OSError, ValueError, KeyError, TypeError) as error:
        # A damaged all-attempt ledger must never be used for numerical
        # subtraction. The separately validated retained history remains valid.
        restart = {'record': 'INVALID', 'error': str(error), 'removed_gpu_seconds': None}
    return {"record": "COMPLETE" if complete else "PARTIAL" if measured else "MISSING",
            "seed": seed, "arm": arm, "first_update": start + 1, "last_update": stop,
            "recorded_through_update": progress["step"] if progress else None,
            "recorded_updates": len(points), "expected_updates": stop - start,
            "gpu_seconds": values,
            "gpu_hours": {key: None if value is None else value / 3600 for key, value in values.items()},
            "known_recorded_subtotal_gpu_seconds": sum(known.values()) + (preparation or 0) if measured else None,
            "missing_update_timer_ranges": {field: ranges(steps) for field, steps in missing.items()},
            "missing_update_timer_counts": {field: len(steps) for field, steps in missing.items()},
            "source": progress["path"] if progress else None,
            "preparation_source": preparation_source, "update_timers": points,
            'restart_accounting': restart,
            'duplicate_history_entries': progress['duplicate_history_entries'] if progress else 0,
            'duplicate_history_gpu_seconds': progress['duplicate_history_gpu_seconds'] if progress else {}}


def summaries(tasks, seeds, arms):
    result = {}
    for arm in arms:
        rows = [r for r in tasks if r.get("arm") == arm]
        values = [r.get("submission_timing", {}).get("gpu_hours", {}).get("primary") for r in rows]
        complete = (len(rows) == len(seeds) and {r["seed"] for r in rows} == set(seeds) and
                    all(r.get("submission_timing", {}).get("record") == "COMPLETE" for r in rows) and
                    all(v is not None for v in values))
        result[arm] = {"planned_seeds": len(seeds), "timed_seeds": sum(v is not None for v in values),
                       "mean_gpu_hours": statistics.mean(values) if complete else None,
                       "sample_sd_gpu_hours": statistics.stdev(values) if complete and len(values) > 1 else None}
    return result


def attach(report, plan_path, plan, *, read=read_json):
    report["submission_time_basis"] = {
        "schema": "submitted-olmo-update-cost-v1", "submitted_commit": BASELINE,
        "first_update": plan["shared_prefix_updates"] + 1, "last_update": plan["total_updates"],
        "unit": "allocated GPU-hours; recorded GPU-seconds divided by 3600 once",
        "primary": "retained-update training + cumulative full selection + one common SR subset preparation",
        "excluded": ["cache creation", "shared prefix", "evaluation", "checkpointing", "startup",
                     "separate diagnostic measurements", "discarded/replayed update attempts"],
        "all_attempt_costs": "original timing/phase totals remain separate and unchanged",
        "submitted_sources": ["v6/overleaf/sections/pair_common_step_costs.tex",
                              "v6/overleaf/sections/accounting.tex",
                              "v6/evidence/2026-09-26/scoring-cost-calculation-clarification.md"],
        "scope": "matched-update continuation; not fixed-allocation or end-to-end elapsed time",
        "selection_work": "required candidate/reference gradients and rollout generation are included, not diagnostics to subtract",
        "selection_scope": "full selection timers, including scored rollouts, validation, gradients and overhead",
        "missing": "null; no scaling of candidate-only estimates or incomplete histories",
        'restart_accounting': 'one retained update per seed/arm/checkpoint; exact history copies removed; finished attempt receipts reconciled separately',
    }
    plan_path = Path(plan_path)
    for row in report["tasks"]:
        if row.get("arm") not in plan["arms"]:
            continue
        try:
            if row.get("status") == "invalid":
                raise ValueError("invalid experiment identity")
            folder = Path(report["output_root"]) / f"seed-{row['seed']}"
            marker = read(folder / "run.json")
            if marker is not None:
                expected = {"seed": row["seed"], "plan_sha256": digest(plan_path),
                            "input_sha256": digest(input_path(plan_path, plan, row["seed"]))}
                if not isinstance(marker, dict) or any(marker.get(k) != v for k, v in expected.items()):
                    raise ValueError("timed progress is not bound to this plan/input/seed")
                row["submission_timing"] = collect(folder, row["arm"], plan, row["seed"], read=read)
                restart_error = row['submission_timing'].get('restart_accounting', {}).get('error')
                if restart_error:
                    message = f"seed {row['seed']} {row['arm']} restart accounting: {restart_error}"
                    if message not in report.setdefault('errors', []):
                        report['errors'].append(message)
            else:
                # A file without a run identity must not become a measured total.
                row["submission_timing"] = collect(folder, row["arm"], plan, row["seed"], read=lambda _: None)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["submission_timing"] = {"record": "INVALID", "error": str(error),
                                        "gpu_hours": {"primary": None}}
    report["submission_time_statistics"] = summaries(report["tasks"], plan["seeds"], plan["arms"])
    return report


def render(report):
    from scripts.srgc_time_status import aligned_table

    basis = report.get("submission_time_basis")
    if not basis:
        return []
    cells = []
    for row in report.get("tasks", []):
        timing = row.get("submission_timing")
        if timing is None:
            continue
        hours = timing["gpu_hours"]
        cells.append([row["seed"], row["arm"], timing["record"],
                      f"{timing.get('recorded_updates', 0)}/{timing.get('expected_updates', '-')}",
                      *["-" if hours.get(k) is None else f"{hours[k]:.3f}"
                        for k in ("preparation", "selection", "training", "primary")]])
    if not cells:
        return []
    means = []
    for arm, values in report.get("submission_time_statistics", {}).items():
        if values["mean_gpu_hours"] is not None:
            means.append(f"{arm} {values['mean_gpu_hours']:.3f}" +
                         (f" +/- {values['sample_sd_gpu_hours']:.3f}" if values['sample_sd_gpu_hours'] is not None else '') +
                         f" (n={values['timed_seeds']})")
    from .model_result_text import restart_table

    return ["", f"SUBMITTED OLMO TIME BASIS (GPU-hours; updates {basis['first_update']}--{basis['last_update']})",
            *aligned_table(("Seed", "Arm", "Record", "Updates", "SR prep", "Select", "Train", "Total"), cells),
            *(["Submitted-basis mean +/- sample SD (GPU-h): " + " | ".join(means)] if means else []),
            "One retained timer per update; common SR preparation charged once. Original all-attempt costs remain in GPU TIMES.",
            "Missing timers/preparation keep Total unknown; incomplete subtotals do not establish a full-run time.",
            *restart_table(report),
            *[f"SUBMISSION TIME ERROR seed {r['seed']} {r['arm']}: {r['submission_timing']['error']}"
              for r in report['tasks'] if r.get('submission_timing', {}).get('error')]]
