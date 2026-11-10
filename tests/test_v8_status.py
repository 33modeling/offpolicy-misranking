"""Read-only Korean status view for V8 queues (scripts/v8_status.py)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
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
    assert "this node (node01): running traj-5 · 37/275 done · activity 40s ago" in text
    assert "gpu memory used: 0:61000MiB 1:60000MiB · util 0:97% 1:95%" in text
    assert "done 1 · running 1 · waiting 1 · needs attention 1" in text
    assert "seed  " in text and "on_monitored" in text and "running 37/275" in text
    assert "running now:" in text and "37/275 done · activity 40s ago · node node01 · log: PROGRESS 12:00:00" in text
    assert "needs attention:" in text and "failed · last error: CUDA out of memory · 1 attempt" in text
    assert "V8 qwen_mbpp · " in text and "no tasks" in text
    assert "WARNING" not in text


def test_stall_warning_only_when_silent_and_gpus_idle(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    log = base / "olmo_math/.multi-node/logs/att1.log"
    os.utime(log, (now - 3600, now - 3600))
    assert "WARNING traj-5: no log or progress record for 60 min" in status.render([base], now=now, gpus=[("0", 0, 500)], host="node01")
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


def test_running_task_shows_live_trainer_progress_and_activity(tmp_path):
    """The child no longer prints PROGRESS lines; status must read SRGC_PROGRESS_DIR rank files (11-08 report:
    'v8_status 제대로 업데이트 안되는 것 같아')."""
    now = time.time()
    base = fixture(tmp_path, now)
    q = base / "olmo_math" / ".multi-node"
    log = q / "logs" / "att1.log"
    os.utime(log, (now - 2400, now - 2400))                      # log silent for 40 min
    prog = q / "progress" / "att1"
    write(prog / "rank-0.json", {"stage": "generation", "updated": now - 7, "question_index": 3, "questions_total": 40,
                                 "response_tokens": 812, "max_new_tokens": 2048})
    write(prog / "rank-1.json", {"stage": "generation", "updated": now - 12})
    text = status.render([base], now=now, gpus=[("0", 0, 500)], host="node01")
    assert "activity 7s ago" in text
    assert "now: Writing model answers for the current question | question 3/40 on this GPU" in text
    assert "2 GPU processes, last record 7s-12s ago" in text
    assert "WARNING" not in text                                  # progress is fresh even though the log is silent


def test_progress_falls_back_to_this_hosts_latest_launch_directory(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    launch = base / ".node-progress" / "node01" / "launch-abc"
    write(launch / "rank-0.json", {"stage": "checkpoint_save", "updated": now - 3})
    text = status.render([base], now=now, gpus=[("0", 97, 60000)], host="node01")
    assert "now: Saving model, optimizer and RNG state · last record 3s ago" in text


def test_stall_warning_needs_both_log_and_progress_silent(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    q = base / "olmo_math" / ".multi-node"
    os.utime(q / "logs" / "att1.log", (now - 3600, now - 3600))
    write(q / "progress" / "att1" / "rank-0.json", {"stage": "generation", "updated": now - 3500})
    assert "WARNING traj-5: no log or progress record for 58 min" in status.render([base], now=now, gpus=[("0", 0, 1)], host="node01")



def test_default_screen_is_the_run_srgc_status_renderer(tmp_path, monkeypatch):
    """`run_v8_nodes.sh status` must look like `run_srgc.sh <dataset> status` (11-08: 'status 다시 만들어')."""
    monkeypatch.setenv("V8_CODE_ROOT", str(REPO))
    now = time.time()
    base = fixture(tmp_path, now)
    q = base / "olmo_math" / ".multi-node"
    write(q / "progress" / "att1" / "rank-0.json", {"stage": "generation", "updated": now - 5})
    write(q / "progress" / "att1" / "rank-1.json", {"stage": "generation", "updated": now - 9})
    text = status.render([base], now=now, host="node01", style="srgc")
    assert "SRGC V8 olmo_math · " in text and " · phase train\n" in text
    olmo = text.split("SRGC V8 olmo_mbpp")[0]   # the OLMo MATH block ends where the next experiment (OLMo MBPP) starts
    assert olmo.index("seed  ") < olmo.index("running now:") < olmo.index("SRGC V8 olmo_math · ") < olmo.index("done 1 · running 1")  # summary at the bottom
    assert olmo.rstrip().splitlines()[-1].startswith("workers: ")
    assert "done 1 · running 1 · waiting 1 · needs attention 1" in text
    assert "running 37/275" in text
    assert "running now:" in text and "step 38/275 in progress · 37/275 done · generation (5s ago) · gpus 2 busy (last 5-9s) · node node01" in text
    assert "needs attention:" in text and "nodes:" in text and "node01" in text and "heartbeat" in text
    assert "SRGC V8 qwen_mbpp" in text and "no queue yet" in text
    assert "workers: 1 active" in text


def test_falls_back_to_simple_view_without_original_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "original_renderer", lambda: None)
    now = time.time()
    base = fixture(tmp_path, now)
    text = status.render([base], now=now, gpus=[("0", 97, 61000)], host="node01", style="srgc")
    assert text.startswith("(original run_srgc.sh status renderer not found")
    assert "V8 olmo_math · " in text



def test_simple_view_also_puts_summary_at_the_bottom(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    text = status.render([base], now=now, gpus=[("0", 97, 61000)], host="node01", cohorts=("olmo_math",))
    assert text.index("seed  ") < text.index("V8 olmo_math · ") < text.index("done 1 · running 1")
    assert text.rstrip().splitlines()[-2].startswith("done 1 · running 1")
    assert text.rstrip().splitlines()[-1].startswith("workers: ")



def failure_notes(base, cohort, now):
    """Per job key: the failure note text (at / why / next) that status keeps for every failed task, also when the
    task retries by itself and is therefore not drawn under needs attention (user 11-10)."""
    rows = status.collect(base, cohort, now)[1]
    return {r['key']: f"{r.get('failure_at', '')}\n      why:  {r.get('failure_cause', '')}\n      next: {r.get('failure_next', '')}"
            for r in rows if r['status'] == 'failed'}


def test_clean_view_is_compact_with_summary_last(tmp_path):
    """'status 좀 보기 좋게 정리하라고': short columns, one symbol per cell, summary at the bottom."""
    now = time.time()
    base = fixture(tmp_path, now)
    q = base / "olmo_math" / ".multi-node"
    write(q / "progress" / "att1" / "rank-0.json", {"stage": "generation", "updated": now - 5,
                                                   "question_index": 3, "questions_total": 40, "response_tokens": 812})
    text = status.render_clean([base], now=now, gpus=[("0", 97, 61000)], host="node01", width=160)
    lines = text.rstrip().splitlines()
    norm = lambda line: ' '.join(line.split())
    assert lines[0].startswith("OLMo MATH · training ─")
    assert "train  On" in text and "▶37" in text       # at update 37 (user: "training 쪽도 그냥 275로")
    top, bottom = text.split("━━ V8 status · ")
    assert "needs attention" not in top
    summary = next(norm(line) for line in bottom.splitlines() if line.startswith(" OLMo MATH "))
    assert summary.startswith("OLMo MATH training ░░░░░░░░░░░░ 1% 2% (1/55-60) 0% (0/50-55) 0% (0/≥55) ") and summary.endswith("37/8,125+")
    assert "cells to fill: OLMo MATH ≥160 (train 55-60 + ext 50-55 + eval ≥55)" in bottom
    header = next(norm(line) for line in bottom.splitlines() if norm(line).startswith("phase progress"))
    assert header == "phase progress train ext eval run wait fail updates"
    assert norm(next(line for line in bottom.splitlines() if line.startswith(" Qwen MBPP"))) == "Qwen MBPP not started"
    # the OOM evaluation retries by itself: one auto-retry line, no needs attention (user 11-10)
    assert "needs attention" not in bottom and " auto-retry, nothing to do: OLMo MATH 1 failed task(s)" in bottom
    assert " nodes (1 host)  1 running" in bottom and "DEAD" not in bottom
    row = next(line for line in bottom.splitlines() if line.startswith(" ▶ node01"))
    for cell in (" running ", " OLMo ", " s5 On ", " 37/275 ", "answering 3/40, 812 tok (5s ago)"):
        assert cell in row, (cell, row)
    assert lines[-1] == " this node  node01 · GPU 97% · ▶ s5 On (OLMo MATH)"
    assert all(len(line) <= 160 for line in lines)


def worker_beat(q, wid, age, task=None, now=None):
    import hashlib
    write(q / "workers" / f"{hashlib.sha256(json.dumps(wid).encode()).hexdigest()}.json",
          {"worker": wid, "at": (now or time.time()) - age, "task": task})


def test_node_table_one_row_per_worker_with_what_it_does(tmp_path):
    """'어느 노드가 뭐 하는지 그게 중요하다고': every worker (host.pid) with experiment, task, updates, time, step."""
    import fcntl, os
    now = time.time()
    base = fixture(tmp_path, now)
    q = base / "olmo_math" / ".multi-node"
    mq = base / "qwen_mbpp" / ".multi-node"
    for job in (q / "jobs").glob("*.json"):
        record = json.loads(job.read_text())
        for attempt in record.get("attempts", []):
            if attempt.get("worker") == "node01.gpu0123":
                attempt["worker"] = "h100-a01.48213"
        write(job, record)
    for f in (q / "workers").glob("*.json"):
        f.unlink()
    worker_beat(q, "h100-a01.48213", 4, "child:1", now)
    def job(queue, key, arm, seed, wid, started, kind="train"):
        write(queue / "jobs" / f"{key}.json", {"state": "running", "phase": "train", "task": {"kind": kind, "method_id": arm,
              "training_seed": seed, "run_id": key, "horizon": 275}, "attempts": [{"id": key + "a", "worker": wid, "started_at": now - started}]})
    worker_beat(q, "h100-a02.51007", 6, "child:2", now); job(q, "j2", "bank_sr_t025", 6, "h100-a02.51007", 3 * 3600 + 120)
    write(q / "progress" / "j2a" / "rank-0.json", {"stage": "gradient_scoring", "updated": now - 20})
    worker_beat(q, "h100-a03.6120", 2460, None, now); job(q, "j3", "bank_sr_t050", 6, "h100-a03.6120", 5 * 3600)
    worker_beat(q, "h100-a04.77", 700, None, now); job(q, "j4", "bank_sr_t100", 7, "h100-a04.77", 3600)
    (q / "leases").mkdir(exist_ok=True); (q / "leases" / "j3.lock").touch()
    fd = os.open(q / "leases" / "j4.lock", os.O_CREAT | os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX)
    (mq / "jobs").mkdir(parents=True, exist_ok=True)
    for g in ("0-1-2-3", "4-5-6-7"):
        (mq / "admission" / f"h100-b10-{g}-1a2b3c4d").mkdir(parents=True)
    (q / "admission" / "h100-a02-0-1-2-3-deadbeef").mkdir(parents=True)
    worker_beat(mq, "h100-b10.7001", 3, "child:3", now); job(mq, "k1", "on_monitored", 7, "h100-b10.7001", 40 * 60)
    worker_beat(mq, "h100-b10.7002", 9, "runtime_initialization", now)
    worker_beat(q, "h100-b11.300", 5, None, now)
    worker_beat(q, "h100-b12.301", 900, None, now)
    try:
        text = status.render_clean([base], now=now, gpus=[("0", 97, 61000)], host="h100-a01", width=200)
    finally:
        os.close(fd)
    bottom = text.split("V8 status · ")[1].splitlines()
    head = next(line for line in bottom if line.startswith(" nodes ("))
    # a03's task lock is free, so the next free node resumes s6 SR50: a03 is just gone (user: '새노드 할당되면 알아서');
    # a04 still holds its lock while silent, so nobody else can take that task: the only node shown as a problem.
    assert head.startswith(" nodes (5 hosts)  3 running · 1 starting · 1 idle  ✗ 1 hung: h100-a04"), head
    assert "(2 gone not shown; --all-nodes)" in bottom[bottom.index(head) + 1]
    rows = [line for line in bottom if line[:3] in (" ▶ ", " ✗ ", " · ", "   ") and " h100-" in line[:12]]
    assert [r.split()[1] if r[1] in "▶✗·" else r.split()[0] for r in rows] == \
        ["h100-a04", "h100-a01", "h100-a02", "h100-b10", "h100-b10", "h100-b11"]   # hung, running, starting, idle
    by = {}
    for r in rows:
        by.setdefault(r.split()[1] if r[1] in "▶✗·" else r.split()[0], []).append(r)
    a2 = by["h100-a02"][0]
    assert a2.startswith(" ▶ ") and " 0-3 " in a2 and " OLMo " in a2 and " s6 SR25 " in a2 and " 3h02m " in a2 \
        and "scoring questions (20s ago)" in a2
    assert "h100-a03" not in by and "DEAD" not in text
    assert by["h100-a04"][0].startswith(" ✗ ") and " hung " in by["h100-a04"][0] and "lock still held" in by["h100-a04"][0]
    assert "no other node can take that task" in text
    b10 = by["h100-b10"]
    assert " 0-3 or 4-7 " in b10[0] and " Qwen " in b10[0] and " s7 On " in b10[0] and " 40m " in b10[0]
    assert " starting " in b10[1] and "loading runtime / checking node" in b10[1]
    assert " idle " in by["h100-b11"][0] and "idle while 1 task is ready" in by["h100-b11"][0]   # a03's task, free to take over
    assert "h100-b12" not in by and not any(" gone " in r for r in rows)   # 'gone 보기 싫어': gone workers are only counted
    everything = status.render_clean([base], now=now, gpus=[("0", 97, 61000)], host="h100-a01", width=200, all_nodes=True)
    assert " h100-b12 " in everything and " gone " in everything
    assert not (q / "leases" / "j2.lock").exists()   # the probe never creates lease files


def test_clean_view_finished_nodes_are_not_dead(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    q = base / "olmo_math" / ".multi-node"
    write(q / "state.json", {**json.loads((q / "state.json").read_text()), "phase": "complete"})
    for name in list((q / "jobs").glob("*.json")):
        job = json.loads(name.read_text()); job["state"] = "done"; write(name, job)
    for f in (q / "workers").glob("*.json"):
        f.unlink()
    for wid in ("node0.11", "node1.12"):
        worker_beat(q, wid, 4000, None, now)
    bottom = status.render_clean([base], now=now, gpus=[], host="node09", width=160).split("V8 status · ")[1]
    assert "DEAD" not in bottom and "(2 finished not shown; --all-nodes)" in bottom and "node0 " not in bottom


def test_short_arm_names():
    assert status.short_arm("on_monitored") == "On"
    assert status.short_arm("bank_sr_t025") == "SR25" and status.short_arm("bank_uniform_t050") == "U50"



def two_cohort_base(tmp_path, now, olmo_phase="train"):
    base = tmp_path / "v8"
    for cohort, phase in (("olmo_math", olmo_phase), ("qwen_mbpp", "train")):
        write(base / cohort / ".multi-node" / "state.json", {"phase": phase, "failure": None})
        (base / cohort / ".multi-node" / "jobs").mkdir(parents=True, exist_ok=True)
    return base, base / "olmo_math" / ".multi-node", base / "qwen_mbpp" / ".multi-node"


def node_rows(text):
    bottom = text.split("V8 status · ")[1].splitlines()
    return [line for line in bottom if line[:3] in (" ▶ ", " ✗ ", " · ", "   ") and " h-" in line[:10]]


def test_removed_node_disappears_within_90_seconds(tmp_path):
    """'idle 아닌데 자꾸 idle 이래 이미 삭제한 노드인데': a node without a heartbeat for >90 s is not idle.
    (Not 30 s: another node's heartbeat file can be read up to ~60 s stale on a shared file system.)"""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    for queue in (q, m):
        worker_beat(queue, "h-alive.1", 2, None, now)
        worker_beat(queue, "h-removed.2", 100, None, now)
        worker_beat(queue, "h-slowread.3", 70, None, now)   # stale read of a live node: still alive
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    rows = node_rows(text)
    assert len(rows) == 2 and " h-alive " in rows[0] and " idle " in rows[0] and " h-slowread " in rows[1]
    assert "(1 gone not shown; --all-nodes)" in text


def test_busy_in_other_cohort_is_not_idle(tmp_path):
    """Fresh heartbeat in Qwen but silent in OLMo during OLMo's CPU stage / task scan: busy, not idle."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now, olmo_phase="freeze")
    worker_beat(m, "h-cpu.1", 2, None, now); worker_beat(q, "h-cpu.1", 150, None, now)
    log = q / "logs" / "cpu-freeze-abc.log"; log.parent.mkdir(parents=True); log.write_text("x\n"); os.utime(log, (now - 2, now - 2))
    rows = node_rows(status.render_clean([base], now=now, gpus=[], host="viewer", width=200))
    assert len(rows) == 1 and rows[0].startswith(" · ") and " busy " in rows[0] and " OLMo " in rows[0] \
        and "CPU stage budget freeze, log 2s ago" in rows[0]
    base2, q2, m2 = two_cohort_base(tmp_path / "b", now)
    worker_beat(m2, "h-scan.1", 2, None, now); worker_beat(q2, "h-scan.1", 150, None, now)
    worker_beat(m2, "h-short.2", 2, None, now); worker_beat(q2, "h-short.2", 60, None, now)   # short scan / stale read
    rows = node_rows(status.render_clean([base2], now=now, gpus=[], host="viewer", width=200))
    assert " busy " in rows[0] and " h-scan " in rows[0] and "checking/claiming tasks (2 min)" in rows[0]
    assert " idle " in rows[1] and " h-short " in rows[1]
    base3, q3, m3 = two_cohort_base(tmp_path / "c", now)
    worker_beat(m3, "h-gave.1", 2, None, now); worker_beat(q3, "h-gave.1", 1000, None, now)
    rows = node_rows(status.render_clean([base3], now=now, gpus=[], host="viewer", width=200))
    assert " idle " in rows[0] and "not serving OLMo MATH (gave up? see its console)" in rows[0]


def test_assigned_task_uses_the_lease_lock_not_heartbeat_age(tmp_path):
    import fcntl
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    def job(key, wid, started=600):
        write(q / "jobs" / f"{key}.json", {"state": "running", "phase": "train", "task": {"kind": "train", "method_id": "bank_sr_t025",
              "training_seed": 5, "run_id": key, "horizon": 275}, "attempts": [{"id": key + "a", "worker": wid, "started_at": now - started}]})
    (q / "leases").mkdir()
    job("held", "h-load.1"); worker_beat(q, "h-load.1", 240, None, now)          # model load: silent but lock held
    job("free40", "h-blip.2"); worker_beat(q, "h-blip.2", 120, None, now); (q / "leases" / "free40.lock").touch()
    job("free90", "h-gone.3"); worker_beat(q, "h-gone.3", 200, None, now); (q / "leases" / "free90.lock").touch()
    fd = os.open(q / "leases" / "held.lock", os.O_CREAT | os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        text = status.render_clean([base], now=now, gpus=[], host="viewer", width=220)
        stats = status.clean_block(base, "olmo_math", now, "viewer", 220)[4]   # while the lock is still held
    finally:
        os.close(fd)
    by = {next(w for w in r.split() if w.startswith("h-")): r for r in node_rows(text)}
    assert " running " in by["h-load"] and "no heartbeat 4 min, task lock held" in by["h-load"]
    assert " running " in by["h-blip"] and not by["h-blip"].startswith(" ✗ ")   # 'free' lock + 2 min: still running
    # 'free' lock + over 3 min: the next free node takes the task over (srgc_v8_nodes claim), so it counts as ready
    # and its old node is not listed (user: '새노드 할당되면 알아서 거기에 시작하면 되지 왜 표시를 해놓는거냐')
    assert "h-gone" not in by and "DEAD" not in text and "h-gone" not in text.split("V8 status · ")[1].split(" nodes (")[0]
    assert stats["running"] == 2 and stats["dead"] == 0 and stats["ready"] >= 1


def test_idle_with_ready_work_says_so(tmp_path):
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    parent = tmp_path / "parent.pt"; parent.write_bytes(b"x")
    write(q / "jobs" / "r.json", {"state": "pending", "phase": "train", "task": {"kind": "train", "method_id": "bank_sr_t050",
          "training_seed": 5, "run_id": "r", "horizon": 275, "parent": {"path": str(parent)}}, "attempts": []})
    worker_beat(q, "h-idle.1", 1, None, now); worker_beat(m, "h-idle.1", 1, None, now)
    rows = node_rows(status.render_clean([base], now=now, gpus=[], host="viewer", width=220))
    assert " idle " in rows[0] and "idle while 1 task is ready" in rows[0]


def test_one_scan_per_cohort_per_refresh_and_fast_watch(tmp_path, monkeypatch):
    now = time.time()
    base = fixture(tmp_path, now)
    calls = []
    original = status._collect
    monkeypatch.setattr(status, "_collect", lambda b, c, n: calls.append(c) or original(b, c, n))
    status.render_clean([base], now=now, gpus=[], host="viewer", width=160)
    assert calls == ["olmo_math"]   # once per refresh (no qwen queue in this fixture)
    assert status._COLLECT_CACHE is None
    import inspect
    assert "const=3.0" in inspect.getsource(status.main)



def test_odd_records_never_crash_the_screen(tmp_path):
    """Harness (v1.0.16, with the original checkout): an int worker id crashed the node table and a run
    manifest attempt without attempt_id raised KeyError. Both must render; a broken record is listed."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    write(q / "jobs" / "a1.json", {"state": "running", "task": {"kind": "train", "run_id": "r1", "training_seed": 5,
          "method_id": "on_monitored", "horizon": 275}, "attempts": [{"id": "x", "worker": 123, "started_at": now - 60}]})
    write(base / "olmo_math" / "runs" / "r1" / "manifest.json", {"attempts": [{"no_attempt_id": True}]})
    write(q / "jobs" / "a2.json", {"state": "pending", "task": ["not", "a", "dict"], "attempts": "nope"})
    write(q / "jobs" / "a3.json", {"state": "running", "task": {"kind": "train", "run_id": "r3"}, "attempts": [None]})
    write(base / "olmo_math" / "runs" / "r3" / "manifest.json", {"attempts": "broken"})
    # v1.0.17 harness: a failed/interrupted attempt whose times are strings crashed the attention sort.
    write(q / "jobs" / "a4.json", {"state": "failed", "task": {"kind": "train", "run_id": "r4", "training_seed": 6},
          "attempts": [{"id": "y", "started_at": "2026-11-08T10:00", "ended_at": "later"}]})
    write(q / "jobs" / "a5.json", {"state": "interrupted", "task": {"kind": "train", "run_id": "r5"},
          "attempts": [{"id": "z", "started_at": True}]})
    import fcntl
    (q / "leases").mkdir(); (q / "leases" / "a1.lock").touch()
    fd = os.open(q / "leases" / "a1.lock", os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX)   # held: the int-id worker is listed
    try:
        for style in ("clean", "simple", "srgc"):
            text = status.render([base], now=now, gpus=[], host="viewer", style=style)
            assert "Traceback" not in text and "OLMo" in text or "olmo_math" in text
        clean = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    finally:
        os.close(fd)
    assert " 123 " in clean and "DEAD" not in clean



def test_healthy_nodes_never_get_an_x_and_attention_is_at_the_bottom(tmp_path):
    """'status 이상해 정상인게 엑스표시나와', 'needs attention 이거 업데이트가 안되는거 같음'."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    (q / "leases").mkdir()
    for i in range(4):   # running tasks whose lock looks free from this node, heartbeat 5-150 s old
        key = f"t{i}"
        write(q / "jobs" / f"{key}.json", {"state": "running", "phase": "train", "task": {"kind": "train",
              "method_id": "bank_sr_t025", "training_seed": 5 + i, "run_id": key, "horizon": 275},
              "attempts": [{"id": key + "a", "worker": f"h-ok{i}.1", "started_at": now - 900}]})
        (q / "leases" / f"{key}.lock").touch()
        worker_beat(q, f"h-ok{i}.1", (5, 60, 120, 150)[i], f"child:{i}", now)
    write(q / "jobs" / "f.json", {"state": "failed", "phase": "train", "task": {"kind": "train", "method_id": "bank_sr_t050",
          "training_seed": 9, "run_id": "f"}, "attempts": [{"id": "fa", "started_at": now - 7300, "ended_at": now - 7200,
          "error": "task exit=1"}]})
    write(q / "jobs" / "i.json", {"state": "interrupted", "phase": "train", "task": {"kind": "train", "method_id": "on_monitored",
          "training_seed": 8, "run_id": "i"}, "attempts": [{"id": "ia", "started_at": now - 400, "ended_at": now - 300}]})
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    rows = node_rows(text)
    assert len(rows) == 4 and all(r.startswith(" ▶ ") and " running " in r for r in rows), rows
    stats = status.clean_block(base, "olmo_math", now, "viewer", 200)[4]
    assert stats["running"] == 4 and stats["dead"] == 0 and "on dead node" not in text and "DEAD" not in text
    top, bottom = text.split("V8 status · ")
    # its first failure retries by itself, so nothing needs a person (user 11-10: 'needs attention 은 노드가
    # 날아갔는데도 왜 남아있는거냐. 현재 노드 정보나 뿌려')
    assert "needs attention" not in text and " auto-retry, nothing to do: OLMo MATH 1 failed task(s)" in bottom
    # interrupted reruns on the next free node by itself: only its ⏸ cell and the intr count show it
    assert "⏸ OLMo MATH s8 On" not in bottom
    assert stats["failed"] == 1 and stats["interrupted"] == 1



def test_failed_tasks_say_why_and_whether_it_is_a_real_failure(tmp_path):
    """'중요한거 왜 failed 이냐니까', '진짜 fail 맞아?', 'fail인거 note나 똑바로 표시해'."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    def fail(key, seed, arm, attempt, log=None, manifest=None, done=None, recovery=None, policy="new-v10"):
        job = {"state": "failed", "phase": "train", "task": {"kind": "train", "method_id": arm, "training_seed": seed,
               "run_id": key, "horizon": 275}, "attempts": [dict({"id": key + "a", "worker": f"h-{key}.1",
               "started_at": now - 9000, "ended_at": now - 7200, "controller_policy": policy}, **attempt)]}
        if recovery:
            job["recovery_history"] = recovery
        write(q / "jobs" / f"{key}.json", job)
        if log:
            (q / "logs").mkdir(exist_ok=True); (q / "logs" / f"{key}a.log").write_text(log)
        if manifest or done is not None:
            write(base / "olmo_math" / "runs" / key / "manifest.json", dict({"attempts": [{"attempt_id": "x"}]}, **(manifest or {})))
            (base / "olmo_math" / "runs" / key / "attempts" / "x").mkdir(parents=True)
            (base / "olmo_math" / "runs" / key / "attempts" / "x" / "updates.jsonl").write_text(json.dumps({"completed_updates": done}) + "\n")
    generic = {"exit_code": 1, "outcome": "failed", "error": "task exit=1; output verified=False"}
    fail("kill", 5, "bank_sr_t025", {"exit_code": "9", "outcome": "failed", "error": "task exit=9; output verified=False"},
         log="Loading weights: 100%\n", done=120)
    fail("oom", 6, "bank_sr_t050", generic, done=88,
         log="[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB.\n"
             "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: cluster_child.py FAILED\n")
    fail("fin", 7, "on_monitored", generic, log="NCCL WARN\n", manifest={"status": "completed_horizon", "completed_updates": 275}, done=275)
    fail("rank", 8, "bank_uniform_t050", dict(generic, child_failure={"failures": [{"rank": 2, "message": "ValueError: bank hash differs"}]}), done=0)
    fail("stall", 9, "bank_sr_t025", {"exit_code": 124, "outcome": "failed", "error": "task exit=124; output verified=False"}, done=201)
    fail("old", 4, "bank_sr_t050", generic, log="RuntimeError: old bug\n", done=10, policy="old-v9")
    fail("rfile", 3, "bank_sr_t025", generic, log="no error text here\n", done=5)
    write(q / "progress" / "rfilea" / "rank-3-error.json", {"rank": 3, "error": {"message": {"message":
          "Traceback (most recent call last):\n  ...\nRuntimeError: NCCL communicator was aborted on rank 3"}}})
    worker_beat(q, "h-new.1", 2, None, now)
    import hashlib
    write(q / "workers" / (hashlib.sha256(json.dumps("h-new.1").encode()).hexdigest() + ".json"),
          {"worker": "h-new.1", "at": now - 2, "task": None, "dispatch_policy": "new-v10"})
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    bottom = text.split("━━ V8 status · ")[1]
    notes = failure_notes(base, "olmo_math", now)
    names = {"  ! OLMo MATH s5 SR25 marked failed ": "kill", "  ✗ OLMo MATH s6 SR50 FAILED ": "oom",
             "  ! OLMo MATH s7 On marked failed ": "fin", "  ✗ OLMo MATH s8 U50 FAILED ": "rank",
             "  ✗ OLMo MATH s9 SR25 FAILED ": "stall", "  ✗ OLMo MATH s4 SR50 FAILED ": "old", "  ✗ OLMo MATH s3 SR25 FAILED ": "rfile"}
    def block(head):
        return notes[names[head]]
    # every one of them retries by itself (or is already complete): nothing under needs attention
    assert "needs attention" not in bottom and " auto-retry, nothing to do: OLMo MATH 7 failed task(s)" in bottom
    k = block("  ! OLMo MATH s5 SR25 marked failed ")
    assert "at update 120/275" in k and "NOT a training error: killed from outside (SIGKILL, exit 9)" in k
    assert "check dmesg on h-kill first" in k and "not retried yet: a node started with the current bundle retries" in k
    o = block("  ✗ OLMo MATH s6 SR50 FAILED ")
    assert "why:  GPU out of memory: torch.OutOfMemoryError: CUDA out of memory" in o and "ChildFailedError" not in o.split("next:")[0]
    f = block("  ! OLMo MATH s7 On marked failed ")
    assert "NOT a training failure: the run is complete (275/275 updates)" in f and "work is already complete" in f
    assert "rank 2: ValueError: bank hash differs" in block("  ✗ OLMo MATH s8 U50 FAILED ")
    assert "stall watchdog: no training progress for 30 min" in block("  ✗ OLMo MATH s9 SR25 FAILED ")
    assert "failed under older code: a node started with the current bundle retries it automatically" in block("  ✗ OLMo MATH s4 SR50 FAILED ")
    assert "rank 3: RuntimeError: NCCL communicator was aborted on rank 3" in block("  ✗ OLMo MATH s3 SR25 FAILED ")
    stats = status.clean_block(base, "olmo_math", now, "viewer", 200)[4]
    assert stats["failed"] == 5 and stats["marked"] == 2
    table = text.split("━━ V8 status")[0]
    lines = table.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(" train "))
    grid = "\n".join(line for line in lines[start + 1:] if line.strip()[:1].isdigit())   # seed rows of the tables
    assert "!" in grid and "✗" in grid



def test_failure_next_step_follows_the_default_retry_rule(tmp_path):
    """v11 (9c1e31c) retries failed tasks by default; the note must say so, with the attempt count."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    write(q / "state.json", {"phase": "train", "failure": None, "retry_failed_requested_at": now - 100,
                             "automatic_retry_policy": {"policy": "v11", "max_failed_attempts": 3, "retry_delay": 60.0}})
    def job(key, seed, attempts):
        write(q / "jobs" / f"{key}.json", {"state": "failed", "phase": "train", "task": {"kind": "train", "method_id": "bank_sr_t025",
              "training_seed": seed, "run_id": key, "horizon": 275}, "attempts": attempts})
    a = lambda ended: {"id": "x", "exit_code": 1, "outcome": "failed", "started_at": now - 900, "ended_at": now - ended}
    job("soon", 5, [a(20)])
    job("ready", 6, [a(30), a(300)])
    job("spent", 7, [a(900), a(600), a(300)])
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    notes = failure_notes(base, "olmo_math", now)
    assert "retried automatically: attempt 2/3 starts in 40s" in notes["soon"]
    assert "retried automatically: attempt 3/3 is waiting for the next free node" in notes["ready"]
    assert "BLOCKED: failure cap reached (3 recorded; limit 3); no more automatic retries; failed checkpoints/logs retained; repair required" in notes["spent"]
    bottom = text.split("━━ V8 status · ")[1]
    # one line per blocked task; no why/next/log blocks of old nodes (user 11-10 '현재 노드 정보나 뿌려')
    assert " needs attention (1)" in bottom and "  ✗ OLMo MATH s7 SR25: stopped after 3 failures · last: exit 1 without an error message" in bottom
    assert " s5 SR25" not in bottom.split(" needs attention")[1] and "      why:" not in bottom and "      log:" not in bottom
    assert " auto-retry, nothing to do: OLMo MATH 2 failed task(s)" in bottom



def test_summary_numbers_add_up_and_updates_count_each_job_once(tmp_path, monkeypatch):
    """'status 집계가 뭔가 이상해 숫자가 안맞아': columns must sum to the task total, train + eval = total, and a
    branch or extension adds only horizon - start updates (it resumes from its parent's saved update)."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    def job(key, state, kind, arm, seed, horizon=None, start=None, phase="train", attempts=None):
        task = {"kind": kind, "method_id": arm, "training_seed": seed, "run_id": key}
        if horizon: task["horizon"] = horizon
        if start is not None: task["parent"] = {"completed_update": str(start), "path": "/nowhere"}
        write(q / "jobs" / f"{key}.json", {"state": state, "phase": phase, "task": task, "attempts": attempts or []})
    job("on", "done", "train", "on_monitored", 5, horizon=75, start=25)          # adds 50
    job("b25", "done", "train", "bank_sr_t025", 5, horizon=75, start=25)        # adds 50
    job("b50", "pending", "train", "bank_sr_t050", 5, horizon=75, start=50)     # plans 25
    job("ext", "pending", "train", "bank_sr_t025", 5, horizon=120, start=75, phase="extension")   # plans 45
    job("ghost", "running", "train", "bank_uniform_t050", 5, horizon=75, start=50, attempts=[{"id": "g"}])  # no worker
    job("odd", "weird", "train", "on_monitored", 6, horizon=75)
    for i in range(3):
        job(f"ev{i}", "done" if i == 0 else "pending", "evaluate", "evaluate", 5, phase="evaluation")
    monkeypatch.setattr(status, "runner_config", lambda cohort: None)   # exact arithmetic of the jobs themselves
    stats = status.clean_block(base, "olmo_math", now, "viewer", 200)[4]
    assert stats["total"] == 9 and stats["train"] == (2, 5) and stats["ext"] == (0, 1) and stats["eval"] == (1, 3)
    assert stats["done"] + stats["running"] + stats["ready"] + stats["waiting"] + stats["failed"] + stats["marked"] + \
        stats["interrupted"] + stats["dead"] + stats["other"] == stats["total"]
    assert stats["updates"] == (100, 50 + 50 + 25 + 45 + 25 + 75)   # was 75*5 + 120 = 495 before (each job from 0)
    assert stats["dead"] == 0 and stats["other"] == 1 and stats["running"] == 0 and stats["ready"] >= 1   # ghost: next node resumes it
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=220)
    assert "marked running but no worker is recorded for it" not in text and "unknown job state 'weird'" in text
    head = next(line for line in text.splitlines() if line.split()[:2] == ["phase", "progress"])
    assert " stuck " not in head and " other " in head and " train " in head and " eval " in head



def test_cells_to_fill_are_planned_totals_before_and_after_each_phase_is_sealed(tmp_path, monkeypatch):
    """'status에 전부 몇칸을 채워야 하는지 표기해': later phases' jobs do not exist yet, so the totals come from the
    plan: per seed 1 On + one SR branch per grid point (+ at most one Uniform), one extension per non-On run,
    at least one final evaluation per run; exact once plans/<phase>.json is sealed."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    cfg = {"seeds": {"training": [5, 6]}, "bank": {"core_grid": [50], "extended_grid": [25]},
           "training": {"matched_update_endpoint": 75, "prefix_updates": 25}}
    monkeypatch.setattr(status, "runner_config", lambda cohort: cfg)
    write(q / "jobs" / "on5.json", {"state": "done", "phase": "train", "task": {"kind": "train", "method_id": "on_monitored",
          "training_seed": 5, "run_id": "on5", "horizon": 75, "parent": {"completed_update": "25"}}, "attempts": []})
    stats = status.clean_block(base, "olmo_math", now, "viewer", 200)[4]
    plan = stats["planned"]
    assert plan["train"] == (6, 8, "range") and plan["ext"] == (4, 6, "range") and plan["eval"] == (6, None, "at least")
    assert stats["pct"] == round(100 * 1 / 16)          # 1 done of at least 6 + 4 + 6 cells
    assert stats["updates"] == (50, 2 * ((75 - 25) + (75 - 25) + (75 - 50)))
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    assert "cells to fill: OLMo MATH ≥16 (train 6-8 + ext 4-6 + eval ≥6)" in text
    assert "exact when each phase is planned" in text
    for queue in (q, m):
        for phase in ("train", "extension", "evaluation"):
            write(queue / "plans" / f"{phase}.json", {"sealed": True, "tasks": []})
    for i in range(2):
        write(q / "jobs" / f"x{i}.json", {"state": "pending", "phase": "extension", "task": {"kind": "train",
              "method_id": "bank_sr_t050", "training_seed": 5, "run_id": f"x{i}", "horizon": 100}, "attempts": []})
    write(q / "jobs" / "e.json", {"state": "pending", "phase": "evaluation", "task": {"kind": "evaluate", "training_seed": 5}, "attempts": []})
    stats = status.clean_block(base, "olmo_math", now, "viewer", 200)[4]
    assert stats["planned"]["train"] == (1, 1, "exact") and stats["planned"]["ext"] == (2, 2, "exact")
    assert stats["planned"]["eval"] == (1, 1, "exact")
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    assert "cells to fill: OLMo MATH 4 (train 1 + ext 2 + eval 1)" in text and "exact when each phase" not in text



def test_seed_table_draws_every_planned_branch_before_its_job_exists(tmp_path, monkeypatch):
    """'SR25 부터 결국 250 까지 있는거 아니야', '왜 아예 표시를 안하냐고': all planned seeds and SR25..SR250 + U columns
    are drawn from the start; cells without a job yet are '·' (not created yet); after sealing, the jobs decide."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    cfg = {"seeds": {"training": [5, 6, 7, 8, 9]}, "bank": {"core_grid": [50, 100, 150, 200], "extended_grid": [25, 75, 125, 175, 225, 250]},
           "training": {"matched_update_endpoint": 275, "prefix_updates": 25}}
    monkeypatch.setattr(status, "runner_config", lambda cohort: cfg)
    write(q / "jobs" / "on5.json", {"state": "running", "phase": "train", "task": {"kind": "train", "method_id": "on_monitored",
          "training_seed": 5, "run_id": "on5", "horizon": 275, "parent": {"completed_update": "25"}},
          "attempts": [{"id": "a", "worker": "h-1.1", "started_at": now - 60}]})
    worker_beat(q, "h-1.1", 2, "child:1", now)
    write(q / "jobs" / "u6.json", {"state": "done", "phase": "train", "task": {"kind": "train", "method_id": "bank_uniform_t150",
          "training_seed": 6, "run_id": "u6", "horizon": 275, "parent": {"completed_update": "150"}}, "attempts": []})
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    block = text.split("Qwen MBPP ·")[0]
    lines = block.splitlines()
    i_train = next(i for i, line in enumerate(lines) if line.startswith(" train"))
    i_later = next(i for i, line in enumerate(lines) if line.startswith(" ext "))
    assert lines[i_train].split() == ["train", "On", "SR25", "SR50", "SR75", "SR100", "SR125", "SR150", "SR175", "SR200", "SR225",
                                      "SR250", "U"]
    rows = {line.split()[0]: line.split() for line in lines[i_train + 1:i_later]}
    assert sorted(rows) == ["5", "6", "7", "8", "9"]
    assert rows["5"][1] == "▶" and rows["5"][2:12] == ["·"] * 10 and rows["9"][1:] == ["·"] * 12
    assert rows["6"][12] == "✓" and "U = Uniform at the seed's switch point: s6 @150" in block
    later = {line.split()[0]: line.split() for line in lines[i_later + 1:i_later + 6]}
    assert later["9"] == ["9", "none"] + ["·"] * 11      # ext table: On is never extended; nothing else exists yet
    i_eval = next(i for i, line in enumerate(lines) if line.startswith(" eval "))
    assert {line.split()[0]: line.split() for line in lines[i_eval + 1:i_eval + 6]}["9"] == ["9"] + ["·"] * 12
    for phase in ("train", "extension", "evaluation"):
        write(q / "plans" / f"{phase}.json", {"sealed": True, "tasks": []})
    sealed = status.render_clean([base], now=now, gpus=[], host="viewer", width=200).split("Qwen MBPP ·")[0]
    assert next(line for line in sealed.splitlines() if line.startswith(" train")).split()[:3] == ["train", "On", "U"]



def test_default_screen_lists_all_four_experiments(tmp_path):
    """'v8 status 에 olmo mbpp qwen math 추가해', 'status 순서 맞춰': model, then task."""
    now = time.time()
    base = tmp_path / "v8"
    for cohort in ("olmo_mbpp", "qwen_math"):
        write(base / cohort / ".multi-node" / "state.json", {"phase": "train", "failure": None})
        (base / cohort / ".multi-node" / "jobs").mkdir(parents=True)
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    assert status.COHORTS == ("olmo_math", "olmo_mbpp", "qwen_math", "qwen_mbpp")
    summary = [line.split("  ")[0].strip() for line in text.split("━━ V8 status · ")[1].splitlines()
               if line.startswith((" OLMo", " Qwen"))]
    assert summary == ["OLMo MATH", "OLMo MBPP", "Qwen MATH", "Qwen MBPP"]
    assert "OLMo MBPP · training" in text and "Qwen MATH · training" in text



def test_extension_and_evaluation_cells_follow_each_task_through_its_states(tmp_path):
    """'status ext eval 도 노드 표시해 현황', 'ext eval 칸만 만들고 제대로 업데이트 하고 잇는지 체크': per-branch ext and eval
    cells change with the jobs (pending -> running -> done); a running evaluation is '▶', not the scored run's
    update count; evaluation jobs are named by the run they score in the node table."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    R = "olmo_math.rebuttal_bank.s5."
    def job(key, state, task, phase, worker=None):
        attempts = [{"id": key + "a", "worker": worker, "started_at": now - 600}] if worker else []
        write(q / "jobs" / f"{key}.json", {"state": state, "phase": phase, "task": task, "attempts": attempts})
    def run(rid, done):
        write(base / "olmo_math" / "runs" / rid / "manifest.json", {"attempts": [{"attempt_id": "x"}]})
        (base / "olmo_math" / "runs" / rid / "attempts" / "x").mkdir(parents=True, exist_ok=True)
        (base / "olmo_math" / "runs" / rid / "attempts" / "x" / "updates.jsonl").write_text(json.dumps({"completed_updates": done}) + "\n")
    def cells(text, table):
        lines = text.split("Qwen MBPP ·")[0].splitlines()
        i = next(k for k, line in enumerate(lines) if line.startswith(f" {table} "))
        header = lines[i].split()
        row = next(line for line in lines[i + 1:] if line.split()[:1] == ["5"]).split()
        return dict(zip(header[1:], row[1:]))
    train = {"kind": "train", "method_id": "bank_sr_t025", "training_seed": 5, "run_id": R + "bank_sr_t025.aa11", "horizon": 275}
    extension = {"kind": "train", "method_id": "bank_sr_t025", "training_seed": 5, "run_id": R + "bank_sr_t025.bb22", "horizon": 400,
                 "parent": {"completed_update": "275"}}
    e275 = {"kind": "evaluate", "training_seed": 5, "run_id": R + "bank_sr_t025.aa11", "completed_update": 275}
    e400 = {"kind": "evaluate", "training_seed": 5, "run_id": R + "bank_sr_t025.bb22", "completed_update": 400}
    job("t25", "done", train, "train"); run(R + "bank_sr_t025.aa11", 275)
    for wid in ("h-ext.1", "h-ev.1"):
        worker_beat(q, wid, 2, "child:1", now)
    job("x25", "pending", extension, "extension"); job("e1", "pending", e275, "evaluation")
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=220)
    assert cells(text, "ext")["SR25"] == "·" and cells(text, "eval")["SR25"] == "·" and cells(text, "ext")["On"] == "none"
    job("x25", "running", extension, "extension", "h-ext.1"); run(R + "bank_sr_t025.bb22", 310)
    job("e1", "running", e275, "evaluation", "h-ev.1"); run(R + "bank_sr_t025.aa11", 275)
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=220)
    assert cells(text, "ext")["SR25"] == "▶310" and cells(text, "eval")["SR25"] == "▶275"   # updates: run at 310, eval of update 275
    node = {line.split()[1]: line for line in text.splitlines() if line.startswith(" ▶ h-")}
    assert " s5 SR25 ext " in node["h-ext"] and " 310/400 " in node["h-ext"] and " s5 eval SR25@275 " in node["h-ev"]
    job("x25", "done", extension, "extension"); run(R + "bank_sr_t025.bb22", 400)
    job("e1", "done", e275, "evaluation"); job("e2", "running", e400, "evaluation", "h-ev.1")
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=220)
    assert cells(text, "ext")["SR25"] == "✓" and cells(text, "eval")["SR25"] == "▶400"   # 275 done, the one at update 400 running
    assert any(" s5 eval SR25 ext@400 " in line for line in text.splitlines() if line.startswith(" ▶ h-ev"))
    job("e2", "done", e400, "evaluation")
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=220)
    assert cells(text, "eval")["SR25"] == "✓"
    stats = status.clean_block(base, "olmo_math", now, "viewer", 220)[4]
    assert stats["ext"] == (1, 1) and stats["eval"] == (2, 2) and stats["running"] == 0

def test_native_crash_says_where_the_rank_crashed(tmp_path):
    """'exit -11 torchrun 세그폴트 에러 고쳐볼래': the why line names the crashing function, file and line from the
    rank's own fault dump (same text as the CPU torchrun SIGSEGV evidence), and the dump path stays on one line."""
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    dump_path = q / "progress" / "na" / "native-fault-rank-1-64b9a24193bf3b4ad0f74e88.log"
    dump = ('V8 native fault context {"schema": 1, "rank": 1}\nFatal Python error: Segmentation fault\n\n'
            'Current thread 0x0000787a06d7b080 (most recent call first):\n'
            '  File "/opt/venv/lib/python3.12/site-packages/torch/nn/functional.py", line 2551 in embedding\n'
            '  File "/srv/code/srgc_rebuttal/torch_backend.py", line 412 in generate\n'
            '  File "/srv/code/srgc_rebuttal/run_experiment.py", line 301 in evaluate\n'
            '  File "/srv/code/scripts/v8_task_entry.py", line 336 in main\n\n'
            'Thread 0x0000787a01ffe640 (most recent call first):\n  File "/usr/lib/python3.12/threading.py", line 355 in wait\n')
    dump_path.parent.mkdir(parents=True); dump_path.write_text(dump)
    failure = {"rank": 1, "pid": 4242, "signal": 11, "message": "SIGSEGV (exit -11): native crash; last stage=answering",
               "native_fault_log": str(dump_path), "native_python_stack": dump, "last_progress": {"pid": 4242, "stage": "answering"}}
    write(q / "jobs" / "n.json", {"state": "failed", "phase": "evaluation", "task": {"kind": "evaluate", "training_seed": 6,
          "method_id": "on_monitored", "run_id": "n", "completed_update": 275},
          "attempts": [{"id": "na", "worker": "h-n.1", "started_at": now - 900, "ended_at": now - 600, "exit_code": -11,
                        "outcome": "failed", "error": "task exit=-11; output verified=False",
                        "child_failure": {"failures": [failure, {"rank": 0, "message": "Signal 15 (SIGTERM) received"}]}}]})
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=120)
    auto = re.sub(r"\n {3}", " ", text.split("V8 status · ")[1])
    assert "needs attention" not in auto
    assert ("auto-retry, nothing to do: OLMo MATH 1 failed task(s) restart from their checkpoints on the next free node (✗ cells); "
            "latest SIGSEGV (exit -11) crashed in embedding (torch/nn/functional.py:2551) ← generate") in auto
    # the same task after a second crash on another node: blocked, so it is listed with why and the dump path
    job = json.loads((q / "jobs" / "n.json").read_text())
    job["attempts"].insert(0, dict(job["attempts"][0], id="na0", host="h-m", worker="h-m.1", child_failure=None))
    job["attempts"][1].update(host="h-n")
    write(q / "jobs" / "n.json", job)
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=120)
    bottom = text.split("V8 status · ")[1]
    assert " needs attention (1)" in bottom
    joined = re.sub(r"\n {4}(?=\S)", " ", bottom)          # an item may wrap onto one indented line
    line = next(x for x in joined.splitlines() if x.startswith("  ✗ OLMo MATH s6 eval"))
    assert ": stopped after 2 failures · last: SIGSEGV (exit -11) in embedding (torch/nn/functional.py:2551) ← generate" in line, line
    assert "BLOCKED: SIGSEGV (exit -11) on 2 different nodes (h-m, h-n)" in failure_notes(base, "olmo_math", now)["n"]
    return
    why = next(line for line in bottom.splitlines() if line.startswith("      why:  "))
    assert why.startswith("      why:  rank 1 SIGSEGV (exit -11), native crash during answering; crashed in embedding"), why
    joined = re.sub(r"\n {12}", " ", bottom)
    assert ("crashed in embedding (torch/nn/functional.py:2551) ← generate (srgc_rebuttal/torch_backend.py:412) ← "
            "evaluate (srgc_rebuttal/run_experiment.py:301)") in joined, joined
    assert "SIGTERM" not in why                       # the peers torchrun stopped are not the cause
    assert f"      stack: {dump_path}" in bottom.splitlines()        # whole path on one line, copyable
    no_frame = status.native_story([{"rank": 3, "signal": "11", "native_python_stack":
        "Fatal Python error: Segmentation fault\n\nCurrent thread 0x1 (most recent call first):\n  <no Python frame>\n"}])[0]
    assert "no Python frame (inside a native library" in no_frame
    assert status.native_story([{"rank": 0, "message": "ValueError: x"}]) == (None, None)
