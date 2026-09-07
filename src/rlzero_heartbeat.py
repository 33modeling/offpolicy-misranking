"""Publish worker liveness to shared storage for cross-node status checks."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_heartbeat(
    path: Path,
    *,
    worker: str,
    host: str,
    launcher_pid: int,
    state: str,
    started_at_ns: int,
) -> None:
    record = {
        "schema": "offpolicy-worker-heartbeat/v1",
        "worker": worker,
        "host": host,
        "launcher_pid": launcher_pid,
        "heartbeat_pid": os.getpid(),
        "state": state,
        "started_at_ns": started_at_ns,
        "heartbeat_at_ns": time.time_ns(),
    }
    # A transient NFS error must not kill the heartbeat: a dead heartbeat makes
    # a healthy worker look dead to everyone else.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        pass


def fmt_age(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    if seconds < 172800:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d"


def check_peers(
    path: Path,
    *,
    worker: str,
    stale_seconds: float,
    alerts_log: Path | None,
    last_alert: dict[str, float],
    repeat_seconds: float,
    worker_log: Path | None = None,
    terminal: Path | None = None,
) -> None:
    """Every worker watches every other worker's heartbeat file. A file that
    stops updating (node lost, launcher killed) is announced on this worker's
    terminal and in the shared ALERTS.log until the dead worker's owner records
    are gone or its heartbeat resumes. Nobody noticed two dead workers for four
    days (2026-09-02..06) because nothing said so."""
    now = time.time()
    root = path.parent
    try:
        entries = sorted(root.glob("*.json"))
    except OSError:
        return
    for entry in entries:
        if entry == path or entry.name.startswith("."):
            continue
        try:
            record = json.loads(entry.read_text(encoding="utf-8"))
            beat = int(record.get("heartbeat_at_ns", 0)) / 1e9
        except (OSError, ValueError, TypeError):
            continue
        peer = str(record.get("worker") or entry.stem)
        host = str(record.get("host") or "?")
        state = str(record.get("state") or "")
        age = now - beat
        dead = state in {"launcher-missing", "crashed"} or (state == "running" and age > stale_seconds)
        if not dead:
            last_alert.pop(peer, None)
            continue
        if now - last_alert.get(peer, 0.0) < repeat_seconds:
            continue
        last_alert[peer] = now
        if age > 86400:
            # Day-old corpse: everyone has been told for a day; remove the record
            # so it stops alarming and stops counting against expected workers.
            try:
                entry.unlink()
            except OSError:
                pass
            continue
        families = []
        queue = root.parent / ".families"
        try:
            for owner in queue.glob("*.owner.json"):
                doc = json.loads(owner.read_text(encoding="utf-8"))
                if doc.get("worker") == peer:
                    families.append(f"{doc.get('dataset')}/s{doc.get('seed')}")
        except (OSError, ValueError, TypeError):
            pass
        held = ", ".join(sorted(families)) or "no family"
        line = (
            f"[WORKER DEAD] {peer} on {host}: "
            f"{'crashed rc=' + str(record.get('exit_code', '?')) + ' ' + fmt_age(age) + ' ago' if state == 'crashed' else 'no heartbeat for ' + fmt_age(age)}"
            f"{' (launcher exited)' if state == 'launcher-missing' else ''}; it held {held}."
            f" Start a worker on {host} again: bash scripts/run_olmo3_rlzero.sh run <profile>  (seen by {worker})"
        )
        stamped = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + line + "\n"
        for target in (alerts_log, worker_log, terminal):
            if target is None:
                continue
            try:
                if target != terminal:
                    target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as stream:
                    stream.write(stamped)
            except OSError:
                pass


def report_progress(
    root: Path,
    *,
    total_points: int | None,
    worker: str,
    elapsed: float,
    not_started_grace: float,
    alerts_log: Path | None,
    worker_log: Path | None,
    terminal: Path | None,
) -> str:
    """One [progress] line from durable artifacts (training_progress.verdict).

    TRAINING/DONE go to the terminal and the worker log; NOT TRAINING, and NOT
    STARTED once the preflight grace is over, are shouted as [NOT TRAINING] and
    also appended to ALERTS.log. A worker whose heartbeat is fresh but whose
    matrix writes nothing durable must never read as a running experiment."""
    import training_progress

    stall = float(os.environ.get("OM_PROGRESS_STALL_MINUTES", "30")) * 60
    try:
        word, line, _ = training_progress.verdict(
            root, total_points=total_points, stall_seconds=stall, record_probe=True
        )
    except Exception as exc:  # a shared-volume hiccup must not kill the heartbeat
        word, line = "UNKNOWN", f"progress probe failed: {exc}"
    loud = word == "NOT TRAINING" or (word == "NOT STARTED" and elapsed > not_started_grace)
    tag = "[NOT TRAINING]" if loud else "[progress]"
    stamped = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {tag} {line}  (worker {worker})\n"
    targets = [worker_log, terminal] + ([alerts_log] if loud else [])
    for target in targets:
        if target is None:
            continue
        try:
            if target != terminal:
                target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as stream:
                stream.write(stamped)
        except OSError:
            pass
    return stamped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--peer-stale-seconds", type=float, default=300.0)
    parser.add_argument("--alerts-log", type=Path, default=None)
    parser.add_argument("--worker-log", type=Path, default=None)
    parser.add_argument("--terminal", type=Path, default=None,
                        help="tty of the launcher; alerts are written there too (empty/'not a tty' = skip)")
    parser.add_argument("--alert-repeat-seconds", type=float, default=3600.0)
    parser.add_argument("--progress-root", type=Path, default=None,
                        help="matrix root; prints a [progress] line from durable artifacts every --progress-seconds")
    parser.add_argument("--progress-seconds", type=float, default=600.0)
    parser.add_argument("--total-points", type=int, default=None)
    parser.add_argument("--not-started-grace-seconds", type=float, default=2700.0,
                        help="NOT STARTED is expected during preflight; after this it is shouted")
    args = parser.parse_args()
    terminal = args.terminal if args.terminal and str(args.terminal).startswith("/dev/") else None
    if args.launcher_pid <= 1:
        parser.error("--launcher-pid must be greater than one")
    if args.interval_seconds <= 0 or args.progress_seconds <= 0:
        parser.error("--interval-seconds and --progress-seconds must be positive")

    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)
    started_at_ns = time.time_ns()
    last_alert: dict[str, float] = {}
    next_progress = time.time() + args.progress_seconds
    while not stop_event.is_set() and process_alive(args.launcher_pid):
        write_heartbeat(
            args.path,
            worker=args.worker,
            host=args.host,
            launcher_pid=args.launcher_pid,
            state="running",
            started_at_ns=started_at_ns,
        )
        check_peers(
            args.path,
            worker=args.worker,
            stale_seconds=args.peer_stale_seconds,
            alerts_log=args.alerts_log,
            last_alert=last_alert,
            repeat_seconds=args.alert_repeat_seconds,
            worker_log=args.worker_log,
            terminal=terminal,
        )
        if args.progress_root is not None and time.time() >= next_progress:
            next_progress = time.time() + args.progress_seconds
            report_progress(
                args.progress_root,
                total_points=args.total_points,
                worker=args.worker,
                elapsed=time.time() - started_at_ns / 1e9,
                not_started_grace=args.not_started_grace_seconds,
                alerts_log=args.alerts_log,
                worker_log=args.worker_log,
                terminal=terminal,
            )
        stop_event.wait(args.interval_seconds)

    state = "stopped" if stop_event.is_set() else "launcher-missing"
    write_heartbeat(
        args.path,
        worker=args.worker,
        host=args.host,
        launcher_pid=args.launcher_pid,
        state=state,
        started_at_ns=started_at_ns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
