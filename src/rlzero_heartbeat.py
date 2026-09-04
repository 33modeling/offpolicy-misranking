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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.launcher_pid <= 1:
        parser.error("--launcher-pid must be greater than one")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)
    started_at_ns = time.time_ns()
    while not stop_event.is_set() and process_alive(args.launcher_pid):
        write_heartbeat(
            args.path,
            worker=args.worker,
            host=args.host,
            launcher_pid=args.launcher_pid,
            state="running",
            started_at_ns=started_at_ns,
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
