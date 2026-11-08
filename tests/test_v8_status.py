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


def test_nodes_section_counts_workers_and_available_work(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    q = base / "olmo_math" / ".multi-node"
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"x")
    write(q / "jobs" / "e.json", {"state": "pending", "task": {"kind": "train", "method_id": "sr_from_50", "training_seed": 5,
                                                               "parent": {"path": str(parent)}}, "attempts": []})
    write(q / "jobs" / "f.json", {"state": "pending", "task": {"kind": "train", "method_id": "sr_from_100", "training_seed": 5,
                                                               "parent": {"path": str(tmp_path / "not-yet.pt")}}, "attempts": []})
    key = hashlib.sha256(json.dumps("node02.gpu0123").encode()).hexdigest()
    write(q / "workers" / (key + ".json"), {"worker": "node02.gpu0123", "at": now - 3, "task": None})
    key = hashlib.sha256(json.dumps("node03.gpu0123").encode()).hexdigest()
    write(q / "workers" / (key + ".json"), {"worker": "node03.gpu0123", "at": now - 4000, "task": None})
    text = status.render([base], now=now, gpus=[("0", 97, 61000)], host="node01")
    assert "nodes:" in text and "node02.gpu0123" in text and "idle (nothing claimable)" in text
    assert "(1 node record without a heartbeat in the last 3 min not shown; not proof the process stopped)" in text
    assert "ready 1" in text and "waiting" in text
    assert "workers: 2 active" in text
    assert "work available now for 2 workers (1 running + 1 ready to start)" in text


def test_unreadable_or_corrupt_records_are_reported_not_hidden(tmp_path):
    if os.geteuid() == 0:
        return  # root can read mode-000 files
    now = time.time()
    base = fixture(tmp_path, now)
    q = base / "olmo_math" / ".multi-node"
    (q / "jobs" / "zz.json").write_text('{"state": "pend')            # partial write
    (q / "state.json").chmod(0)
    try:
        text = status.render([base], now=now, gpus=[("0", 97, 61000)], host="node01")
    finally:
        (q / "state.json").chmod(0o644)
    assert "ERROR some queue records could not be read" in text
    assert "state.json" in text and "1 job record(s) unreadable or corrupt: zz.json" in text


def test_status_distinguishes_applied_controller_from_old_and_unknown_workers(tmp_path):
    now=time.time();base=fixture(tmp_path,now);q=base/'olmo_math/.multi-node'
    expected=hashlib.sha256((Path(status.__file__).parent/'srgc_v8_nodes.py').read_bytes()).hexdigest()
    for name,version in [('current.1',expected),('older.2','f'*64),('unknown.3',None)]:
        write(q/'workers'/(name+'.json'),dict(worker=name,at=now,task=None,controller_sha256=version))
    text=status.render([base],now=now,gpus=[],host='local')
    assert 'dispatch current '+expected[:12] in text
    assert 'dispatch different ffffffffffff' in text
    assert 'dispatch unverified' in text
