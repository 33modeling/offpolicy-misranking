"""Park conflicting workers before GPU admission without retrying training."""

import os
import signal
import socket
import time
import uuid

from srgc_rebuttal.plan import load_plan
from srgc_rebuttal.runtime import atomic_json, run_root
try:
    from srgc_shared_storage import RunConflict, route_plan
except ImportError:
    from scripts.srgc_shared_storage import RunConflict, route_plan


def wait_for_plan(source, *, interval=10, sleep=time.sleep, route=None, **options):
    resolve = route or route_plan
    worker_id = uuid.uuid4().hex
    receipt = None
    base = {"worker_id": worker_id, "host": socket.gethostname(), "pid": os.getpid(),
            "started": time.time(), "task": None, "child_pid": None}

    def update(state, reason):
        if receipt is not None:
            atomic_json(receipt, {**base, "status": state, "heartbeat": time.time(),
                                  "idle_reason": reason})

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    original = signal.signal(signal.SIGTERM, interrupted)
    try:
        while True:
            try:
                target = resolve(source, **options)
                update("finished", "compatible plan available")
                return target
            except RunConflict as exc:
                spec = load_plan(exc.plan)
                base["dataset"] = spec["dataset"]
                new_receipt = run_root(exc.plan, spec) / ".queue/workers" / f"{worker_id}.json"
                if receipt != new_receipt:
                    update("finished", "active plan changed")
                receipt = new_receipt
                if (receipt.parent.parent / "stop.json").exists():
                    update("stopped", "stop requested")
                    raise SystemExit(0)
                update("idle", str(exc))
                print(f"NODE idle: {exc}", flush=True)
                sleep(interval)
    except KeyboardInterrupt:
        update("stopped", "worker interrupted while idle")
        raise
    finally:
        signal.signal(signal.SIGTERM, original)
