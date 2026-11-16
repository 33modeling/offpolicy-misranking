"""V8 status numbers checked against job records in the shape the real planner writes (switching.matrix)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
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
    node = next(line for line in text.splitlines() if line.startswith("   h-ext"))
    # no recorded cost: the budget share is unknown and said so, never a planned stop as denominator (user 11-16)
    assert " update 310, budget not read " in node and "310/" not in node and "time left unknown" in node, node
    ext = next(t for t in status.task_model(base, "olmo_math", now)["tasks"] if t["key"] == "x")
    assert ext["start"] == 275 and ext["done_u"] == 310 and ext["end"] == 400


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


def test_running_task_always_says_how_far_it_is(tmp_path):
    """'status 알수가 없어': a generation record without questions_total showed 'answering' with no total. An
    evaluation takes N from the finished evaluation of the same spec (per GPU) and says roughly how long is
    left (median of finished evaluations of that experiment x the part still to do); training shows its update."""
    now = time.time()
    base, q = base_with_queue(tmp_path, "primary_evaluation")
    rid = "olmo_math.rebuttal_bank.s5.bank_sr_t025.aa11"
    ev = {"kind": "evaluate", "training_seed": 5, "run_id": rid, "completed_update": 275, "split": "final",
          "snapshot_sha256": "ab" * 32, "evaluation_spec_hash": "cd" * 32, "task_id": "eval.x"}
    write(q / "jobs" / "e1.json", {"state": "running", "phase": "evaluation", "task": ev,
                                   "attempts": [{"id": "ea", "worker": "h-ev.1", "started_at": now - 300}]})
    for k, (start, end) in enumerate(((now - 9000, now - 7800), (now - 7000, now - 5800), (now - 5000, now - 4000))):
        write(q / "jobs" / f"d{k}.json", {"state": "done", "phase": "evaluation", "task": dict(ev, task_id=f"eval.d{k}"),
                                         "attempts": [{"id": f"da{k}", "started_at": start, "ended_at": end}]})
    write(base / "olmo_math" / "evaluations" / f"{'ef' * 8}__{('cd' * 32)[:12]}.json", {"n_questions": 2000})
    for rank in range(4):   # generation records of this attempt without questions_total
        write(q / "progress" / "ea" / f"rank-{rank}.json", {"stage": "generation", "updated": now - 5,
                                                             "question_index": 75, "response_tokens": 300})
    beat(q, "h-ev.1", 2, now, "child:1")
    train = {"kind": "train", "method_id": "bank_sr_t050", "training_seed": 6, "run_id": "t6", "horizon": 75,
             "parent": {"completed_update": 50, "path": "/nowhere"}}
    write(q / "jobs" / "t1.json", {"state": "running", "phase": "train", "task": train,
                                   "attempts": [{"id": "ta", "worker": "h-tr.1", "started_at": now - 300}]})
    run_at(base, "t6", 62)
    write(q / "progress" / "ta" / "rank-0.json", {"stage": "generation", "updated": now - 3,
                                                  "question_index": 3, "questions_total": 8})
    beat(q, "h-tr.1", 2, now, "child:1")
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=220)
    nodes = {line.split()[0]: line for line in text.split(" nodes: ")[1].splitlines() if line.startswith("   h-")}
    # 4 GPU processes at question 75 of their 500 = 300/2000 questions; 1,200 s median x 85% left = ~17 min
    # every number says what it counts (user 11-16 '87/300 몇 시간'): no bare a/b on a node line
    assert "question 300 of 2000 " in nodes["h-ev"] and " ~17m left " in nodes["h-ev"], nodes["h-ev"]
    assert "now answering question 75 of 500 on each GPU, 5s ago" in nodes["h-ev"], nodes["h-ev"]
    assert "update 62 of 75 " in nodes["h-tr"] and "answering question 3 of 8 on each GPU" in nodes["h-tr"], nodes["h-tr"]
    for line in nodes.values():
        assert not re.search(r"\b\d+/\d+\b", line), line
