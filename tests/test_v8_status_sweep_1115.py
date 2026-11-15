"""V8 status numbers checked against job records in the shape the real planner writes (switching.matrix)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("v8_status_sweep", REPO / "scripts/v8_status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def beat(q, wid, age, now, task=None):
    write(q / "workers" / f"{hashlib.sha256(json.dumps(wid).encode()).hexdigest()}.json",
          {"worker": wid, "at": now - age, "task": task})


def base_with_queue(tmp_path, phase="extension"):
    base = tmp_path / "v8"
    q = base / "olmo_math" / ".multi-node"
    write(q / "state.json", {"phase": phase, "failure": None})
    return base, q


def run_at(base, rid, done, status_="running"):
    r = base / "olmo_math" / "runs" / rid
    write(r / "manifest.json", {"status": status_, "completed_updates": done, "attempts": [{"attempt_id": "x"}]})
    (r / "attempts" / "x").mkdir(parents=True, exist_ok=True)
    (r / "attempts" / "x" / "updates.jsonl").write_text(json.dumps({"completed_updates": done}) + "\n")


def extension_task(rid):
    # switching.matrix.build_stage('budget-extension'): horizon=H and parent = the H snapshot; the run goes on
    # past H until the frozen budget is spent, at most hard_cap updates.
    return {"kind": "train", "stage": "budget-extension", "method_id": "bank_sr_t025", "training_seed": 5,
            "run_id": rid, "horizon": 275, "hard_cap": 400, "budget_cap_gpu_seconds": 9000.0,
            "parent": {"kind": "snapshot", "completed_update": 275, "path": "/nowhere"}}


def test_running_budget_extension_is_not_shown_past_its_total(tmp_path):
    """A budget extension starts at H=275 and runs to at most hard_cap=400; the node table showed '310/275'."""
    now = time.time()
    base, q = base_with_queue(tmp_path)
    rid = "olmo_math.rebuttal_bank.s5.bank_sr_t025.bb22"
    write(q / "jobs" / "x.json", {"state": "running", "phase": "extension", "task": extension_task(rid),
                                  "attempts": [{"id": "xa", "worker": "h-ext.1", "started_at": now - 600}]})
    beat(q, "h-ext.1", 2, now, "child:1")
    run_at(base, rid, 310)
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    node = next(line for line in text.splitlines() if line.startswith(" ▶ h-ext"))
    assert " 310/400 " in node and "310/275" not in node, node
    stats = status.clean_block(base, "olmo_math", now, "viewer", 200)[4]
    assert stats["updates"][0] <= stats["updates"][1]


def test_extension_that_spent_its_budget_before_the_cap_is_complete_work(tmp_path):
    """completed_budget at 330 (< hard_cap) is finished work: a later failed exit is not a training failure."""
    now = time.time()
    base, q = base_with_queue(tmp_path)
    rid = "olmo_math.rebuttal_bank.s5.bank_sr_t025.bb22"
    write(q / "jobs" / "x.json", {"state": "failed", "phase": "extension", "task": extension_task(rid),
                                  "attempts": [{"id": "xa", "worker": "h-ext.1", "started_at": now - 600,
                                                "ended_at": now - 60, "exit_code": 1, "outcome": "failed"}]})
    run_at(base, rid, 330, "completed_budget")
    row = next(r for r in status.collect(base, "olmo_math", now)[1] if r["key"] == "x")
    assert row["failure_real"] is False and "the run is complete" in row["failure_cause"], row["failure_cause"]
