"""Read live optimizer progress without changing the frozen experiment runtime."""

import json
import math
import re
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


ATTENTION = {"failed", "interrupted", "recoverable", "invalid", "attempts_exhausted"}


def cell(row):
    """Short table cell for one task."""
    status, total = row["status"], row.get("total_steps")
    if status == "complete":
        reward = row.get("reward_percent")
        return "done" if reward is None else f"done {reward:.1f}%"
    if status == "running":
        if row["arm"] == "cache" and "cache_saved_prompts" in row:
            available = max(row["cache_saved_prompts"], row["cache_exported_prompts"])
            return f"running {available}/{row['cache_total_prompts']}"
        done = row.get("completed_steps")
        return f"running {done}/{total}" if done is not None and total else "running"
    if status.startswith("waiting_for_"):
        return "waiting"
    if status == "failed":
        return f"failed x{row.get('attempt', 0)}"
    if status == "invalid":
        return "INVALID"
    return status.replace("_", " ")


def running_detail(row):
    total = row.get("total_steps")
    parts = []
    if row.get("current_step") is not None:
        parts.append(f"step {row['current_step']}/{total} in progress")
    if row.get("completed_steps") is not None:
        parts.append(f"{row['completed_steps']}/{total} done")
    if row["arm"] == "cache" and "cache_saved_prompts" in row:
        available = max(row["cache_saved_prompts"], row["cache_exported_prompts"])
        age = row["cache_last_write_age_seconds"]
        parts.append(f"{available}/{row['cache_total_prompts']} prompts cached"
                     + ("" if age is None else f" (last write {age:.0f}s ago)"))
    if row.get("phase_stage"):
        parts.append(f"{row['phase_stage'].replace('_', ' ')} ({row['progress_age_seconds']:.0f}s ago)")
    ages = [max(0, time.time() - p["updated"]) for p in row.get("progress", [])
            if isinstance(p.get("updated"), (int, float)) and math.isfinite(p["updated"])]
    if ages:
        low, high = min(ages), max(ages)
        span = f"{low:.0f}s" if high - low < 1 else f"{low:.0f}-{high:.0f}s"
        parts.append(f"gpus {len(ages)} busy (last {span})")
    if row.get("host"):
        parts.append(f"node {row['host']}")
    return " · ".join(parts) or "starting"


def last_error_line(log_path, lines=200):
    """The most recent line of a task log that looks like an error, or None."""
    try:
        tail = Path(log_path).read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return None
    for line in reversed(tail):
        stripped = line.strip()
        if re.search(r"(Error|Exception|error:|out of memory|Killed|Traceback)", stripped) and not stripped.startswith("File "):
            return stripped[:200]
    return None


def attention_detail(row):
    total = row.get("total_steps")
    parts = [row["status"].replace("_", " ")]
    error = last_error_line(row.get("log", ""))
    if error:
        parts.append(f"last error: {error}")
    if row.get("attempt"):
        parts.append(f"{row['attempt']} attempt{'s' if row['attempt'] != 1 else ''}")
    if row.get("last_observed_step") is not None:
        parts.append(f"last step {row['last_observed_step']}/{total}")
    if row.get("host"):
        parts.append(f"node {row['host']}")
    parts.append(f"log {row['log']}")
    return " · ".join(parts)


def gpu_memory_summary():
    """'0:1200MiB 1:0MiB ...' from nvidia-smi, or None when unavailable."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    cells = [f"{i.strip()}:{u.strip()}MiB" for i, u in (line.split(",", 1) for line in out.splitlines() if "," in line)]
    return " ".join(cells) or None


def this_node_lines(report, host=None, gpu_summary=gpu_memory_summary):
    """What THIS machine is doing right now, from the worker records that carry its hostname."""
    import socket
    host = host or socket.gethostname()
    mine = [w for w in report["workers"] if w.get("host") == host]
    live = [w for w in mine if w["status"] in {"preflight", "running", "idle"}]
    if live:
        worker = min(live, key=lambda w: w["heartbeat_age_seconds"])
        task = worker.get("task")
        what = {"running": f"running {task}", "idle": "idle (nothing claimable)", "preflight": "starting up"}[worker["status"]]
        row = next((r for r in report["tasks"] if r["task"] == task), None) if task else None
        if row and row.get("completed_steps") is not None and row.get("total_steps"):
            what += f" · {row['completed_steps']}/{row['total_steps']} done"
        line = f"this node ({host}): {what} · heartbeat {worker['heartbeat_age_seconds']:.0f}s ago"
    elif mine:
        worker = min(mine, key=lambda w: w["heartbeat_age_seconds"])
        age = worker["heartbeat_age_seconds"]
        when = f"{age / 60:.0f} min ago" if age >= 120 else f"{age:.0f}s ago"
        line = f"this node ({host}): NOT running · last worker {worker['status'].replace('_', ' ')} {when}"
        if worker.get("task"):
            line += f" (was on {worker['task']})"
    else:
        line = f"this node ({host}): NOT running · no worker has started here"
    lines = [line]
    summary = gpu_summary()
    if summary:
        lines.append(f"gpu memory used: {summary}")
    return lines


def render(report, *, results=False):
    tasks = report["tasks"]
    generated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(report.get("generated", time.time())))
    counts = {"done": 0, "running": 0, "waiting": 0, "attention": 0, "ready": 0}
    for row in tasks:
        status = row["status"]
        key = ("done" if status == "complete" else "running" if status == "running" else
               "waiting" if status.startswith("waiting_for_") else "attention" if status in ATTENTION else "ready")
        counts[key] += 1
    summary = " · ".join(f"{name} {count}" for name, count in counts.items() if count)
    if counts["attention"]:
        summary = summary.replace(f"attention {counts['attention']}", f"needs attention {counts['attention']}")
    lines = [f"SRGC {report['dataset']} · {generated}" + (" · STOP REQUESTED" if report.get("stop_requested") else ""),
             str(report["output_root"]), *this_node_lines(report), "", summary or "no tasks", ""]
    arms = []
    for row in tasks:
        if row["arm"] not in arms:
            arms.append(row["arm"])
    seeds = []
    for row in tasks:
        if row["seed"] not in seeds:
            seeds.append(row["seed"])
    table = {(row["seed"], row["arm"]): cell(row) for row in tasks}
    widths = {arm: max(len(arm), *(len(table.get((seed, arm), "-")) for seed in seeds)) for arm in arms}
    lines.append("seed  " + "  ".join(f"{arm:<{widths[arm]}}" for arm in arms))
    for seed in seeds:
        lines.append(f"{seed:>4}  " + "  ".join(f"{table.get((seed, arm), '-'):<{widths[arm]}}" for arm in arms))
    running = [row for row in tasks if row["status"] == "running"]
    if running:
        lines += ["", "running now:"]
        lines += [f"  {row['task']:<16} {running_detail(row)}" for row in running]
    attention = [row for row in tasks if row["status"] in ATTENTION]
    if attention:
        lines += ["", "needs attention:"]
        lines += [f"  {row['task']:<16} {attention_detail(row)}" for row in attention]
    active = [w for w in report["workers"] if w["status"] in {"preflight", "running", "idle", "heartbeat_stale"}]
    hidden = len(report["workers"]) - len(active)
    if active or hidden:
        lines += ["", "nodes:"]
        for worker in active:
            what = worker["status"].replace("_", " ") + (f" {worker['task']}" if worker.get("task") else "")
            lines.append(f"  {str(worker.get('host') or '-'):<24} {what} · heartbeat {worker['heartbeat_age_seconds']:.0f}s ago")
            if worker.get("error"):
                lines.append(f"      {worker['error']}")
        if hidden:
            lines.append(f"  ({hidden} finished or stopped node record{'s' if hidden != 1 else ''} not shown)")
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
