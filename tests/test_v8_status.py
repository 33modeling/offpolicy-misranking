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
    olmo = text.split("SRGC V8 qwen_mbpp")[0]
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
    assert "seed  On" in text and "▶37" in text
    top, bottom = text.split("━━ V8 status · ")
    assert "needs attention" not in top
    summary = next(norm(line) for line in bottom.splitlines() if line.startswith(" OLMo MATH "))
    assert summary.startswith("OLMo MATH training") and " 25% 1/4 " in summary and summary.endswith("37/275")
    header = next(norm(line) for line in bottom.splitlines() if norm(line).startswith("phase progress"))
    assert header == "phase progress done run wait fail updates"
    assert norm(next(line for line in bottom.splitlines() if line.startswith(" Qwen MBPP"))) == "Qwen MBPP not started"
    assert " needs attention (1)" in bottom and "  ✗ OLMo MATH s6 eval eval-final failed" in bottom
    assert " nodes (1 host)  1 running  0 DEAD" in bottom
    row = next(line for line in bottom.splitlines() if line.startswith(" ▶ node01"))
    for cell in (" running ", " OLMo ", " s5 On ", " 37/275 ", "answering 3/40, 812 tok (5s ago)"):
        assert cell in row, (cell, row)
    assert lines[-1] == " this node  node01 · GPU 97% · ▶ s5 On (OLMo)"
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
    assert head.startswith(" nodes (6 hosts)  3 running · 1 starting · 1 idle  ✗ 2 DEAD: h100-a03, h100-a04")
    assert "(1 gone not shown; --all-nodes)" in bottom[bottom.index(head) + 1]
    rows = [line for line in bottom if line[:3] in (" ▶ ", " ✗ ", " · ", "   ") and " h100-" in line[:12]]
    assert [r.split()[1] if r[1] in "▶✗·" else r.split()[0] for r in rows] == \
        ["h100-a01", "h100-a02", "h100-b10", "h100-b10", "h100-b11", "h100-a03", "h100-a04"]   # running, starting, idle, DEAD
    by = {}
    for r in rows:
        by.setdefault(r.split()[1] if r[1] in "▶✗·" else r.split()[0], []).append(r)
    a2 = by["h100-a02"][0]
    assert a2.startswith(" ▶ ") and " 0-3 " in a2 and " OLMo " in a2 and " s6 SR25 " in a2 and " 3h02m " in a2 \
        and "scoring questions (20s ago)" in a2
    assert by["h100-a03"][0].startswith(" ✗ ") and "down or hung → restart it" in by["h100-a03"][0]
    assert "lock still held" in by["h100-a04"][0]
    b10 = by["h100-b10"]
    assert " 0-3 or 4-7 " in b10[0] and " Qwen " in b10[0] and " s7 On " in b10[0] and " 40m " in b10[0]
    assert " starting " in b10[1] and "loading runtime / checking node" in b10[1]
    assert " idle " in by["h100-b11"][0] and "waiting for work" in by["h100-b11"][0]
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
    assert "0 DEAD" in bottom and "(2 finished not shown; --all-nodes)" in bottom and "node0 " not in bottom


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
    assert " running " in by["h-blip"] and not by["h-blip"].startswith(" ✗ ")   # 'free' lock + 2 min: not DEAD
    assert by["h-gone"].startswith(" ✗ ") and " DEAD " in by["h-gone"] and " 3 min " in by["h-gone"] and "down or hung" in by["h-gone"]
    assert stats["running"] == 2 and stats["dead"] == 1


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
    (q / "leases").mkdir(); (q / "leases" / "a1.lock").touch()
    for style in ("clean", "simple", "srgc"):
        text = status.render([base], now=now, gpus=[], host="viewer", style=style)
        assert "Traceback" not in text and "OLMo" in text or "olmo_math" in text
    clean = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    assert " 123 " in clean and " DEAD " in clean



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
    assert stats["running"] == 4 and stats["dead"] == 0 and "on dead node" not in text and "  0 DEAD" in text
    top, bottom = text.split("V8 status · ")
    assert "needs attention" not in top and " needs attention (2)" in bottom
    lines = bottom.splitlines()
    i = lines.index(" needs attention (2)")
    assert lines[i + 1].startswith("  ✗ OLMo MATH s9 SR50 failed ") and "(attempt 1): task exit=1" in lines[i + 1]
    assert lines[i + 2].startswith("  ⏸ OLMo MATH s8 On interrupted 5 min ago: reruns automatically")
    assert stats["failed"] == 1 and stats["interrupted"] == 1
