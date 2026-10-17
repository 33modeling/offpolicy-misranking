"""Hide expired, unowned worker sessions without changing any saved evidence."""

import hashlib
import math
from pathlib import Path

from srgc_rebuttal.reports import locked

RETENTION_SECONDS = 15 * 60
_EXPIRED_STATES = {"heartbeat_stale", "failed", "stopped", "complete", "yielded"}


def prune_workers(snapshots, *, gpu_lock_roots=(), retention_seconds=RETENTION_SECONDS):
    """Keep recent sessions and every session with evidence of a live lease."""
    owned = set()
    newest_gpu_session = {}
    for report in snapshots:
        root = Path(report["output_root"])
        for task in report["tasks"]:
            if not task.get("worker_id"):
                continue
            # Probe locks even if input validation has masked the row as invalid.
            paths = [root / ".queue" / "leases" / f"{task['task']}.lock"]
            if task["arm"] != "cache":
                paths.append(root / f"seed-{task['seed']}" / f".{task['arm']}.execution.lock")
            if task["status"] == "running" or any(locked(path) for path in paths):
                owned.add(task["worker_id"])
        for worker in report["workers"]:
            age = worker.get("heartbeat_age_seconds")
            if not isinstance(age, (int, float)) or not math.isfinite(age):
                continue
            for gpu in worker.get("gpu_uuids") or []:
                if isinstance(gpu, str):
                    previous = newest_gpu_session.get(gpu)
                    if previous is None or age < previous[0]:
                        newest_gpu_session[gpu] = (age, worker.get("worker_id"))

    for report in snapshots:
        kept, hidden = [], set()
        for worker in report["workers"]:
            age = worker.get("heartbeat_age_seconds")
            expired = (worker.get("status") in _EXPIRED_STATES and
                       isinstance(age, (int, float)) and math.isfinite(age) and age > retention_seconds)
            live = worker.get("worker_id") in owned
            if expired and not live:
                # Preflight/idle children can own device leases before claiming a
                # task. Inspect existing locks; never create or remove lock files.
                live = any(locked(Path(root) / f"{hashlib.sha256(gpu.encode()).hexdigest()}.lock")
                           for gpu in worker.get("gpu_uuids") or [] if isinstance(gpu, str)
                           and (worker.get("worker_id") == newest_gpu_session[gpu][1]
                                or age <= newest_gpu_session[gpu][0])
                           for root in gpu_lock_roots)
            if expired and not live:
                hidden.add(str(worker.get("worker_id", worker.get("pid", "unknown"))))
            else:
                kept.append(worker)
        report["workers"] = kept
        report["worker_history"] = {"retention_seconds": retention_seconds,
                                    "hidden_worker_ids": sorted(hidden), "hidden_count": len(hidden)}
