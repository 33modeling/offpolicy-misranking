"""Report worker activity independently of checkpoint-backup scans."""

from collections import Counter
import json
import time

from srgc_rebuttal.plan import input_path


def activity(queue, task):
    receipt = json.loads(queue.receipt(task).read_text())
    attempt = receipt["attempt"]
    started = receipt["started"]
    progress_dir = queue.directory / "progress" / receipt["attempt_id"]
    details = []
    for path in sorted(progress_dir.glob("rank-*.json")):
        row = json.loads(path.read_text())
        age = max(0, time.time() - row["updated"])
        details.append(f"{path.stem}={row['stage']} last_work_age={age:.0f}s")
    if task.arm == "cache":
        phases = input_path(queue.plan_path, queue.plan, task.seed).with_suffix(".cache") / "cost-receipts"
    else:
        scope = "shared-prefix" if task.arm == "prefix" else task.arm
        phases = queue.root / f"seed-{task.seed}" / "cost-receipts" / scope
    recent = [(path.stat().st_mtime_ns, path) for path in phases.glob("*.json")]
    recent = [(modified, path) for modified, path in recent if modified / 1e9 >= started]
    phase = "child_startup"
    if recent:
        row = json.loads(max(recent)[1].read_text())
        phase = f"{row['phase']}:{row['state']}"
        if row.get("checkpoint") is not None:
            phase += f" checkpoint={row['checkpoint']}"
    return f"attempt={attempt} phase={phase}" + (" " + "; ".join(details) if details else " no_work_receipt_yet")


def run_with_status(original, queue, args, environment, gpu_fds, worker_id, update,
                    *, interval=30, clock=time.monotonic):
    last_key, last_time = None, float("-inf")

    def report(state, task=None, child_pid=None, error=None):
        nonlocal last_key, last_time
        update(state, task, child_pid, error)
        key = (state, task.key if task else None, child_pid, error)
        now = clock()
        if key == last_key and now - last_time < interval:
            return
        last_key, last_time = key, now
        prefix = f"WORKER state={state} task={task.key if task else '-'} child_pid={child_pid or '-'}"
        try:
            if task is not None and state == "running":
                detail = activity(queue, task)
                detail += f" log={queue.directory / 'logs' / (task.key + '.log')}"
            elif state == "idle":
                rows = queue.status(max_attempts=args.max_attempts)
                counts = Counter(row["status"] for row in rows)
                detail = " ".join(f"{name}={count}" for name, count in sorted(counts.items()))
                waiting = [f"{row['task']}:{row['status']}" for row in rows if row["status"] != "complete"]
                detail += " tasks=" + ",".join(waiting)
            else:
                detail = error or ""
            print(f"{prefix} {detail}".rstrip(), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"{prefix} status_read_error={type(exc).__name__}: {exc}", flush=True)

    return original(queue, args, environment, gpu_fds, worker_id, report)
