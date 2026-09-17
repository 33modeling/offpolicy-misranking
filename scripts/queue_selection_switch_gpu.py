#!/usr/bin/env python3
"""Run one original switch worker pass, returning peer waits to the node queue.

Only the idle-wait callback changes. Training, leases, checkpoints, scoring and
frozen scientific code stay in the original driver. Active owned work is never
interrupted to switch suites; this callback is reached only without progress.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import selection_switch_gpu as worker


def yield_to_node_queue(busy, *, last_progress, idle_timeout):
    if busy:
        print(f"[queue-yield] {len(busy)} peer-owned tasks; returning to the shared node queue", flush=True)
    return False


def run():
    original = worker.wait_for_peers
    worker.wait_for_peers = yield_to_node_queue
    try:
        return worker.main()
    finally:
        worker.wait_for_peers = original


if __name__ == "__main__":
    if sys.argv[1:2] != ["run"]:
        raise SystemExit("queue worker accepts only run; use the original driver for other commands")
    sys.exit(run())
