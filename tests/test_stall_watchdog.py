import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stall_watchdog", ROOT / "scripts/_stall_watchdog.py")
watchdog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watchdog)


def phase(root, name, event_id, *, host=None, phase="train", age=0.0, state="running"):
    directory = root / "states/s3-t100/points/view-100" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "progress.json").write_text(json.dumps({"event_id": event_id, "phase": phase, "state": state,
                                                          "host": host or socket.gethostname(), "updated": time.time()}))
    log = directory / f"{phase}-0.log"
    log.write_text("[grpo] step 165/100100 reward=0.438\n")
    stamp = time.time() - age
    os.utime(log, (stamp, stamp))
    return directory


def test_silent_train_phase_is_stopped_and_host_recorded(tmp_path):
    root, faults = tmp_path / "switch", tmp_path / "node-faults"
    stalled = phase(root, "random_reduced", "evt-stalled", age=2000)
    fresh = phase(root, "gated", "evt-fresh", age=30)
    other_host = phase(root, "selection_full", "evt-other", age=5000, host="elsewhere")
    done = phase(root, "random_full", "evt-done", age=5000, state="finished")
    workers = [subprocess.Popen(["sleep", "300"], env={**os.environ, f"OM_SELECTION_COST_{eid}": "1"}, start_new_session=True)
               for eid in ("evt-stalled", "evt-fresh")]
    try:
        time.sleep(.3)
        stopped = watchdog.scan([root], faults, stall_seconds=1500)
        assert [s["event_id"] for s in stopped] == ["evt-stalled"]
        assert workers[0].wait(timeout=40) == -15
        assert workers[1].poll() is None
        record = json.loads((stalled / "stalled.json").read_text())
        assert record["silent_seconds"] >= 2000 and workers[0].pid in record["pids"]
        fault = json.loads((faults / f"{socket.gethostname()}.json").read_text())
        assert fault["event_id"] == "evt-stalled" and fault["phase"] == "train"
        assert not (fresh / "stalled.json").exists() and not (other_host / "stalled.json").exists()
        assert not (done / "stalled.json").exists()
        # A second pass finds nothing new (progress still says running, but the workers are gone).
        again = watchdog.scan([root], faults, stall_seconds=1500)
        assert [s["event_id"] for s in again] == ["evt-stalled"] and again[0]["pids"] == []
    finally:
        for w in workers:
            if w.poll() is None:
                w.kill(); w.wait(timeout=10)


def test_dry_run_and_phase_filter(tmp_path):
    root, faults = tmp_path / "switch", tmp_path / "node-faults"
    phase(root, "random_reduced", "evt-eval", phase="evaluate", age=5000)
    assert watchdog.scan([root], faults, stall_seconds=1500) == []
    stalled = phase(root, "gated", "evt-train", age=5000)
    stopped = watchdog.scan([root], faults, stall_seconds=1500, dry_run=True)
    assert len(stopped) == 1 and not (stalled / "stalled.json").exists() and not faults.exists()
