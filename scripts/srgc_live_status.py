"""Read live optimizer progress without changing the frozen experiment runtime."""

import json
import math
from pathlib import Path
import time

from srgc_rebuttal.plan import load_plan


def read_object(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def latest_phase(directory, started):
    candidates = []
    for path in directory.glob("*.json"):
        try:
            modified = path.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        if modified / 1e9 >= started:
            candidates.append((modified, path))
    if not candidates:
        return None
    modified, path = max(candidates)
    return read_object(path), modified / 1e9


def snapshot(original, plan_path, **kwargs):
    report = original(plan_path, **kwargs)
    plan = load_plan(plan_path)
    root = Path(report["output_root"])
    for row in report["tasks"]:
        if row["arm"] == "cache":
            continue
        total = plan["shared_prefix_updates"] if row["arm"] == "prefix" else plan["total_updates"]
        row.update(total_steps=total, current_step=None, completed_steps=row["step"],
                   phase_stage=None, last_observed_step=None, progress_source=None,
                   progress_age_seconds=None)
        if row["status"] == "complete":
            row.update(step=total, completed_steps=total, progress_source="completion_receipt")
            continue
        if row["status"] == "invalid":
            continue
        receipt_path = root / ".queue/tasks" / f"{row['task']}.json"
        if not receipt_path.exists():
            continue
        try:
            receipt = read_object(receipt_path)
            started = receipt.get("started")
            if not isinstance(started, (int, float)) or not math.isfinite(started):
                continue
            scope = "shared-prefix" if row["arm"] == "prefix" else row["arm"]
            latest = latest_phase(root / f"seed-{row['seed']}" / "cost-receipts" / scope, started)
            # Old attempts can have larger step counts than the restored checkpoint.
            row.update(step=None, completed_steps=None)
            if latest is None:
                continue
            event, modified = latest
            checkpoint, phase, state = event.get("checkpoint"), event.get("phase"), event.get("state")
            if state not in {"started", "finished"}:
                raise ValueError("invalid phase state")
            if phase not in {"startup", "preparation", "selection", "training", "checkpoint_save",
                             "checkpoint_load", "evaluation"}:
                raise ValueError("invalid training phase")
            row.update(phase_stage=phase, progress_source="current_attempt_phase_receipt",
                       progress_age_seconds=max(0, time.time() - modified))
            if checkpoint is None:
                continue
            if type(checkpoint) is not int or not 0 <= checkpoint <= total:
                raise ValueError("phase checkpoint is outside the task's update range")
            completed, current = None, None
            if phase in {"selection", "training"}:
                if checkpoint == total:
                    raise ValueError("update phase starts beyond the final update")
                completed = checkpoint + int(phase == "training" and state == "finished")
                current = checkpoint + 1 if completed == checkpoint else None
            elif phase in {"checkpoint_save", "evaluation"} or (phase == "checkpoint_load" and state == "finished"):
                completed = checkpoint
            row.update(step=completed, completed_steps=completed, last_observed_step=current,
                       current_step=current if row["status"] == "running" else None)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(step=None, completed_steps=None, current_step=None)
            report["errors"].append(f"{row['task']} live progress: {type(exc).__name__}: {exc}")
    report["complete"] = report["complete"] and not report["errors"]
    return report


def render(report, *, results=False):
    lines = [f"SRGC {report['dataset']} STATUS", str(report["output_root"]),
             "  ".join(f"{key}={value}" for key, value in report["counts"].items()),
             "seed task       status                current_step completed_steps node"]
    for row in report["tasks"]:
        total = row.get("total_steps")
        def count(value):
            return "-" if value is None else f"{value}/{total}"
        lines.append(f"{row['seed']:4} {row['arm']:10} {row['status']:21} "
                     f"{count(row.get('current_step')):>12} {count(row.get('completed_steps')):>15} "
                     f"{row.get('host') or '-'}")
        if row["arm"] == "cache" and "cache_saved_prompts" in row:
            age = row["cache_last_write_age_seconds"]
            available = max(row["cache_saved_prompts"], row["cache_exported_prompts"])
            lines.append(f"     CACHE available_prompts={available}/{row['cache_total_prompts']} "
                         f"saved_prompts={row['cache_saved_prompts']} "
                         f"exported={row['cache_exported_prompts']} "
                         f"last_write_age={'none' if age is None else f'{age:.0f}s'}")
        if row.get("phase_stage"):
            lines.append(f"     phase={row['phase_stage']} phase_record_age={row['progress_age_seconds']:.0f}s")
        for progress in row.get("progress", []):
            updated = progress.get("updated")
            age = (f"{max(0, time.time() - updated):.0f}s"
                   if isinstance(updated, (int, float)) and math.isfinite(updated) else "unknown")
            lines.append(f"     observed_work={progress.get('stage')} "
                         f"prompt={progress.get('prompt', '-')} last_work_age={age}")
        if row["status"] in {"failed", "interrupted", "recoverable", "invalid"}:
            if row.get("last_observed_step") is not None:
                lines.append(f"     last_attempt_step={count(row['last_observed_step'])}")
            lines.append(f"     log={row['log']}")
    for worker in report["workers"]:
        lines.append(f"node={worker.get('host')} {worker['status']} task={worker.get('task')} "
                     f"heartbeat_age={worker['heartbeat_age_seconds']:.0f}s")
        if worker.get("error"):
            lines.append(f"  {worker['error']}")
    lines.extend(f"WARNING {warning}" for warning in report["warnings"])
    lines.extend(f"ERROR {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def main():
    from srgc_rebuttal import reports
    original_snapshot, original_render = reports.snapshot, reports.render
    reports.snapshot = lambda plan_path, **kwargs: snapshot(original_snapshot, plan_path, **kwargs)
    reports.render = render
    try:
        reports.main()
    finally:
        reports.snapshot, reports.render = original_snapshot, original_render
