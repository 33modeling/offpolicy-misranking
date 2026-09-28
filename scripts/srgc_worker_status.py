"""Report worker activity independently of checkpoint-backup scans."""

from collections import Counter
import json
import re
import time

from srgc_rebuttal.plan import input_path


def activity(queue, task):
    """One readable line: attempt, update count, current phase, and how recently each GPU rank worked."""
    receipt = json.loads(queue.receipt(task).read_text())
    attempt = receipt["attempt"]
    started = receipt["started"]
    progress_dir = queue.directory / "progress" / receipt["attempt_id"]
    ages = []
    for path in sorted(progress_dir.glob("rank-*.json")):
        row = json.loads(path.read_text())
        ages.append(max(0, time.time() - row["updated"]))
    world = queue.plan.get("world_size", 4)
    if task.arm == "cache":
        phases = input_path(queue.plan_path, queue.plan, task.seed).with_suffix(".cache") / "cost-receipts"
        total = None
    else:
        scope = "shared-prefix" if task.arm == "prefix" else task.arm
        phases = queue.root / f"seed-{task.seed}" / "cost-receipts" / scope
        total = queue.plan["shared_prefix_updates"] if task.arm == "prefix" else queue.plan["total_updates"]
    recent = [(path.stat().st_mtime_ns, path) for path in phases.glob("*.json")]
    recent = [(modified, path) for modified, path in recent if modified / 1e9 >= started]
    phase, checkpoint = "child_startup", None
    if recent:
        row = json.loads(max(recent)[1].read_text())
        phase = f"{row['phase']}:{row['state']}"
        checkpoint = row.get("checkpoint")
    parts = [f"attempt={attempt}"]
    if total is not None:
        parts.append(f"update={checkpoint if checkpoint is not None else 0}/{total}")
    parts.append(f"phase={phase}")
    if ages:
        low, high = min(ages), max(ages)
        span = f"{low:.0f}s" if high - low < 1 else f"{low:.0f}-{high:.0f}s"
        parts.append(f"ranks={len(ages)}/{world} last_activity={span}")
    else:
        parts.append(f"ranks=0/{world} last_activity=-")
    return " ".join(parts)


def run_with_status(original, queue, args, environment, gpu_fds, worker_id, update,
                    *, interval=600, clock=time.monotonic):
    """Print a WORKER line only when something changed; otherwise at most one heartbeat per ``interval``."""
    last_line, last_time = None, float("-inf")

    def report(state, task=None, child_pid=None, error=None):
        nonlocal last_line, last_time
        update(state, task, child_pid, error)
        prefix = f"WORKER {state.upper()}" + (f" {task.key}" if task else "")
        try:
            if task is not None and state == "running":
                detail = activity(queue, task)
                detail += f" pid={child_pid or '-'} log={queue.directory / 'logs' / (task.key + '.log')}"
            elif state == "idle":
                rows = queue.status(max_attempts=args.max_attempts)
                counts = Counter(row["status"] for row in rows)
                detail = " ".join(f"{name}={count}" for name, count in sorted(counts.items()))
                waiting = [f"{row['task']}:{row['status']}" for row in rows if row["status"] != "complete"]
                if waiting:
                    detail += " | pending: " + ", ".join(waiting)
            else:
                detail = error or ""
            line = f"{prefix} {detail}".rstrip()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            line = f"{prefix} status_read_error={type(exc).__name__}: {exc}"
        stable = re.sub(r" last_activity=\S+", "", line)  # rank ages change every scan; not news
        now = clock()
        if stable == last_line and now - last_time < interval:
            return
        last_line, last_time = stable, now
        print(line, flush=True)

    return original(queue, args, environment, gpu_fds, worker_id, report)
