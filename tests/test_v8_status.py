"""Read-only Korean status view for V8 queues (scripts/v8_status.py)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("v8_status_under_test", REPO / "scripts/v8_status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, now):
    base = tmp_path / "v8"
    q = base / "olmo_math" / ".multi-node"
    write(q / "state.json", {"phase": "train", "failure": None})
    write(q / "jobs" / "a.json", {"state": "done", "task": {"kind": "train", "stage": "bank-trajectory"}, "attempts": []})
    write(q / "jobs" / "b.json", {"state": "running",
                                   "task": {"kind": "train", "stage": "bank-trajectory", "method_id": "on_monitored",
                                            "training_seed": 5, "run_id": "traj-5", "horizon": 275},
                                   "attempts": [{"id": "att1", "worker": "node01.gpu0123", "started_at": now - 1800}]})
    write(q / "jobs" / "c.json", {"state": "failed", "task": {"kind": "evaluate", "stage": "eval-final", "training_seed": 6},
                                   "attempts": [{"id": "att2", "error": "CUDA out of memory"}]})
    write(q / "jobs" / "d.json", {"state": "pending", "task": {"kind": "train"}, "attempts": []})
    key = hashlib.sha256(json.dumps("node01.gpu0123").encode()).hexdigest()  # srgc_v8_nodes worker heartbeat
    write(q / "workers" / (key + ".json"), {"worker": "node01.gpu0123", "at": now - 5})
    log = q / "logs" / "att1.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("Loading weights: 100%\nPROGRESS 12:00:00 stage=generation records=40\n")
    os.utime(log, (now - 40, now - 40))
    run = base / "olmo_math" / "runs" / "traj-5"
    write(run / "manifest.json", {"attempts": [{"attempt_id": "x"}]})
    (run / "attempts" / "x").mkdir(parents=True)
    (run / "attempts" / "x" / "updates.jsonl").write_text('{"completed_updates": 37}\n')
    return base


def test_snapshot_matches_run_srgc_status_layout(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    text = status.render([base], now=now, gpus=[("0", 97, 61000), ("1", 95, 60000)], host="node01")
    assert "V8 olmo_math · " in text and " · phase train" in text
    assert "this node (node01): running traj-5 · 37/275 done · log 40s ago" in text
    assert "gpu memory used: 0:61000MiB 1:60000MiB · util 0:97% 1:95%" in text
    assert "done 1 · running 1 · waiting 1 · needs attention 1" in text
    assert "seed  " in text and "on_monitored" in text and "running 37/275" in text
    assert "running now:" in text and "37/275 done · log 40s ago · node node01 · last: PROGRESS 12:00:00" in text
    assert "needs attention:" in text and "failed · last error: CUDA out of memory · 1 attempt" in text
    assert "V8 qwen_mbpp · " in text and "no tasks" in text
    assert "WARNING" not in text


def test_stall_warning_only_when_silent_and_gpus_idle(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    log = base / "olmo_math/.multi-node/logs/att1.log"
    os.utime(log, (now - 3600, now - 3600))
    assert "WARNING traj-5: no log output for 60 min" in status.render([base], now=now, gpus=[("0", 0, 500)], host="node01")
    assert "WARNING" not in status.render([base], now=now, gpus=[("0", 98, 60000)], host="node01")


def test_missing_nvidia_smi_is_reported(tmp_path):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    assert status.gpu_rows(run=missing) is None


def test_age_text():
    assert status.age_text(5) == "5s" and status.age_text(600) == "10 min" and status.age_text(None) == "?"


def test_stale_heartbeat_is_not_shown_as_confirmed_running(tmp_path):
    """The view must distinguish uncertain activity from both confirmed live work and stopped ranks."""
    now = time.time()
    base = fixture(tmp_path, now)
    key = hashlib.sha256(json.dumps("node01.gpu0123").encode()).hexdigest()
    write(base / "olmo_math" / ".multi-node" / "workers" / (key + ".json"), {"worker": "node01.gpu0123", "at": now - 900})
    text = status.render([base], now=now, gpus=[], host="node01")
    assert "running now:" not in text
    assert "worker node01.gpu0123 heartbeat age" in text and "process liveness unverified" in text
    assert "task activity unconfirmed" in text and "next worker resumes it" not in text
