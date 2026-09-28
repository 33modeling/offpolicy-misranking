"""Report worker activity independently of checkpoint-backup scans."""

from collections import Counter
import json
import re
import time

from srgc_rebuttal.plan import input_path


def activity(queue, task):
    """'update 3/25 · training · gpus 4/4 busy (last 3-9s)' for the running task."""
    receipt = json.loads(queue.receipt(task).read_text())
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
    phase, checkpoint = "starting", None
    if recent:
        row = json.loads(max(recent)[1].read_text())
        phase = row["phase"].replace("_", " ")
        checkpoint = row.get("checkpoint")
    parts = []
    if total is not None:
        parts.append(f"update {checkpoint if checkpoint is not None else 0}/{total}")
    parts.append(phase)
    if ages:
        low, high = min(ages), max(ages)
        span = f"{low:.0f}s" if high - low < 1 else f"{low:.0f}-{high:.0f}s"
        parts.append(f"gpus {len(ages)}/{world} busy (last {span})")
    else:
        parts.append(f"gpus 0/{world} reporting")
    return " · ".join(parts)


def idle_summary(rows, max_attempts):
    """'done 5 · running 3 · waiting 20 · needs attention 2: seed-6.prefix, seed-7.prefix (attempts exhausted)'."""
    counts = Counter()
    attention = []
    for row in rows:
        status = row["status"]
        if status == "complete":
            counts["done"] += 1
        elif status == "running":
            counts["running"] += 1
        elif status.startswith("waiting_for_"):
            counts["waiting"] += 1
        elif status in {"failed", "attempts_exhausted", "recoverable", "interrupted"}:
            attention.append((row["task"], status))
        else:
            counts[status] += 1
    parts = [f"{name} {counts[name]}" for name in ("done", "running", "waiting", "ready") if counts[name]]
    parts += [f"{name} {count}" for name, count in sorted(counts.items()) if name not in {"done", "running", "waiting", "ready"}]
    if attention:
        names = ", ".join(f"{task} ({status.replace('_', ' ')})" for task, status in attention)
        parts.append(f"needs attention {len(attention)}: {names}")
        if any(status == "attempts_exhausted" for _, status in attention):
            parts.append(f"restart with SRGC_MAX_ATTEMPTS>{max_attempts} to retry exhausted tasks")
    return " · ".join(parts)


def run_with_status(original, queue, args, environment, gpu_fds, worker_id, update,
                    *, interval=600, clock=time.monotonic):
    """Print a WORKER line only when something changed; otherwise at most one heartbeat per ``interval``."""
    last_line, last_time = None, float("-inf")
    announced = set()

    def report(state, task=None, child_pid=None, error=None):
        nonlocal last_line, last_time
        update(state, task, child_pid, error)
        try:
            if task is not None and state == "running":
                if task.key not in announced:
                    announced.add(task.key)
                    attempt = json.loads(queue.receipt(task).read_text()).get("attempt", "?")
                    print(f"WORKER {task.key} started · attempt {attempt} · pid {child_pid or '-'} · "
                          f"log {queue.directory / 'logs' / (task.key + '.log')}", flush=True)
                line = f"WORKER {task.key} {activity(queue, task)}"
            elif state == "idle":
                line = "WORKER idle · " + idle_summary(queue.status(max_attempts=args.max_attempts), args.max_attempts)
            else:
                line = f"WORKER {state}" + (f" {task.key}" if task else "") + (f" · {error}" if error else "")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            line = f"WORKER status read error · {type(exc).__name__}: {exc}"
        stable = re.sub(r" \(last [^)]*\)", "", line)  # rank ages change every scan; not news
        now = clock()
        if stable == last_line and now - last_time < interval:
            return
        last_line, last_time = stable, now
        print(line, flush=True)

    try:
        return original(queue, args, environment, gpu_fds, worker_id, report)
    except RuntimeError:
        explain_blocked(queue, args)
        raise


def explain_blocked(queue, args, *, lines=40):
    """When failed tasks block the queue, show each one's receipt and the tail of its log."""
    try:
        rows = queue.status(max_attempts=args.max_attempts)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"BLOCKED status_read_error={type(exc).__name__}: {exc}", flush=True)
        return
    for row in rows:
        if row["status"] not in {"failed", "attempts_exhausted"}:
            continue
        log = queue.directory / "logs" / f"{row['task']}.log"
        print(f"BLOCKED {row['task']} status={row['status']} attempts={row.get('attempt', 0)} "
              f"exit={row.get('exit_code', '-')} validation_error={row.get('validation_error') or '-'} log={log}", flush=True)
        try:
            tail = log.read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            tail = ["(log file not readable)"]
        for line in tail:
            print(f"BLOCKED {row['task']} | {line}", flush=True)
