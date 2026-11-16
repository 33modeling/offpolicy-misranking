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






def test_missing_nvidia_smi_is_reported(tmp_path):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    assert status.gpu_rows(run=missing) is None


def test_age_text():
    assert status.age_text(5) == "5s" and status.age_text(600) == "10 min" and status.age_text(None) == "?"

























def failure_notes(base, cohort, now):
    """Per job key: the failure note text (at / why / next) that status keeps for every failed task, also when the
    task retries by itself and is therefore not drawn under needs attention (user 11-10)."""
    rows = status.collect(base, cohort, now)[1]
    return {r['key']: f"{r.get('failure_at', '')}\n      why:  {r.get('failure_cause', '')}\n      next: {r.get('failure_next', '')}"
            for r in rows if r['status'] == 'failed'}




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
    bottom = text.split(" nodes: ")[1].splitlines()
    head = " nodes: " + bottom[0]
    # a03's task lock is free, so the next free node resumes s6 SR50: a03 is just gone (user: '새노드 할당되면 알아서');
    # a04 still holds its lock while silent, so nobody else can take that task: the only node shown as a problem.
    assert head.startswith(" nodes: 5 live — 3 working, 1 starting, 1 idle — why: ") and head.endswith("· 1 HUNG: h100-a04"), head
    assert any("(2 gone not shown; --all-nodes)" in line for line in bottom)
    rows = [line for line in bottom if line.startswith("   h100-")]
    assert [r.split()[0] for r in rows] == \
        ["h100-a04", "h100-a01", "h100-a02", "h100-b10", "h100-b10", "h100-b11"]   # hung, running, starting, idle
    by = {}
    for r in rows:
        by.setdefault(r.split()[0], []).append(r)
    a2 = by["h100-a02"][0]
    assert " working " in a2 and " GPU 0-3 " in a2 and " OLMo MATH " in a2 and " s6 SR25 training " in a2 \
        and "now scoring questions, 20s ago" in a2 and " update ? of 275 " in a2   # no update recorded yet: '?', not the start
    assert "h100-a03" not in by and "DEAD" not in text
    assert " HUNG " in by["h100-a04"][0] and "lock still held" in by["h100-a04"][0]
    assert "no other node can take that task" in text
    b10 = by["h100-b10"]
    assert " 0-3 or 4-7 " in b10[0] and " Qwen MBPP " in b10[0] and " s7 On training " in b10[0]
    assert " starting " in b10[1] and "loading runtime / checking node" in b10[1]
    assert " idle " in by["h100-b11"][0] and "idle although 2 tasks could start" in by["h100-b11"][0]   # a03's task, free to take over
    assert "h100-b12" not in by and not any(" gone " in r for r in rows)   # 'gone 보기 싫어': gone workers are only counted
    # the running tasks are named the same way in the not-finished list, with how far they are and the time left
    listed = text.split(" OLMo MATH: not finished ")[1].split("\n\n")[0]
    assert "s6 SR25 training" in listed and "on h100-a02" in listed and "~10h25m left" in listed
    everything = status.render_clean([base], now=now, gpus=[("0", 97, 61000)], host="h100-a01", width=200, all_nodes=True)
    assert "   h100-b12 " in everything and " gone " in everything
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
    bottom = text.split(" nodes: ")[1].splitlines() if " nodes: " in text else []
    return [line for line in bottom if line.startswith("   h-")]


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
    assert len(rows) == 1 and " busy " in rows[0] and " OLMo MATH " in rows[0] \
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
        model = status.task_model(base, "olmo_math", now)   # while the lock is still held
    finally:
        os.close(fd)
    by = {next(w for w in r.split() if w.startswith("h-")): r for r in node_rows(text)}
    assert " working " in by["h-load"] and "no heartbeat 4 min, task lock held" in by["h-load"]
    assert " working " in by["h-blip"] and " HUNG " not in by["h-blip"]   # 'free' lock + 2 min: still running
    # 'free' lock + over 3 min: the next free node takes the task over (srgc_v8_nodes claim), so it counts as ready
    # and its old node is not listed (user: '새노드 할당되면 알아서 거기에 시작하면 되지 왜 표시를 해놓는거냐')
    assert "h-gone" not in by and "DEAD" not in text and "h-gone" not in text.split(" nodes: ")[0]
    states = status.model_counts(model)[1]
    assert states["running"] == 2 and states["retrying"] == 1 and states["person"] == 0   # h-gone's task: next node resumes it


def test_idle_with_ready_work_says_so(tmp_path):
    now = time.time()
    base, q, m = two_cohort_base(tmp_path, now)
    parent = tmp_path / "parent.pt"; parent.write_bytes(b"x")
    write(q / "jobs" / "r.json", {"state": "pending", "phase": "train", "task": {"kind": "train", "method_id": "bank_sr_t050",
          "training_seed": 5, "run_id": "r", "horizon": 275, "parent": {"path": str(parent)}}, "attempts": []})
    worker_beat(q, "h-idle.1", 1, None, now); worker_beat(m, "h-idle.1", 1, None, now)
    rows = node_rows(status.render_clean([base], now=now, gpus=[], host="viewer", width=220))
    assert " idle " in rows[0] and "idle although 1 task could start" in rows[0]


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
    assert len(rows) == 4 and all(" working " in r and " HUNG " not in r for r in rows), rows
    states = status.model_counts(status.task_model(base, "olmo_math", now))[1]
    assert states["running"] == 4 and states["person"] == 0 and "DEAD" not in text
    # its first failure retries by itself, so nothing needs a person (user 11-10: 'needs attention 은 노드가
    # 날아갔는데도 왜 남아있는거냐. 현재 노드 정보나 뿌려')
    assert "STOPPED — needs you" not in text and "   will restart by itself on the next free node, from its last checkpoint (2):" in text
    assert "s9 SR50 training: failed 0 of 4 so far (task exit=1)" in text and "s8 On training: was interrupted" in text   # old record without exit code: the dispatcher counts 0 and "s8 On (interrupted)" in bottom
    assert states["retrying"] == 2



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
    bottom = text
    notes = failure_notes(base, "olmo_math", now)
    names = {"  ! OLMo MATH s5 SR25 marked failed ": "kill", "  ✗ OLMo MATH s6 SR50 FAILED ": "oom",
             "  ! OLMo MATH s7 On marked failed ": "fin", "  ✗ OLMo MATH s8 U50 FAILED ": "rank",
             "  ✗ OLMo MATH s9 SR25 FAILED ": "stall", "  ✗ OLMo MATH s4 SR50 FAILED ": "old", "  ✗ OLMo MATH s3 SR25 FAILED ": "rfile"}
    def block(head):
        return notes[names[head]]
    # every one of them retries by itself (or is already complete): nothing under needs attention
    # every one of them retries by itself, and the run that is complete counts as done
    assert "STOPPED — needs you" not in bottom and "   will restart by itself on the next free node, from its last checkpoint (6):" in bottom
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
    model = status.task_model(base, "olmo_math", now)
    assert {t["key"]: t["state"] for t in model["tasks"] if not t["planned"]} == \
        {"kill": "retrying", "oom": "retrying", "fin": "done", "rank": "retrying", "stall": "retrying", "old": "retrying", "rfile": "retrying"}
    grid = "\n".join(line for line in text.split(" by seed and branch\n")[1].split("   columns:")[0].splitlines()
                     if line.split()[:1] and line.split()[0].startswith("s"))   # seed rows of the grid
    assert "retry" in grid and "FAIL" not in grid and "!" not in grid



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
    stopped = text.split("   STOPPED — needs you (1):\n")[1].split("\n   will restart")[0]
    # one line per blocked task; no why/next/log blocks of old nodes (user 11-10 '현재 노드 정보나 뿌려')
    assert "     s7 SR25 training: failed 3 times, no more automatic retries · last error: exit 1" in stopped
    assert status.RETRY_COMMAND in stopped
    assert " s5 SR25" not in stopped and "      why:" not in text and "      log:" not in text
    assert "s5 SR25 training: failed 1 of 3 so far (exit 1)" in text and "s6 SR25 training: failed 2 of 3 so far (exit 1)" in text












def test_default_screen_lists_all_four_experiments(tmp_path):
    """'v8 status 에 olmo mbpp qwen math 추가해', 'status 순서 맞춰': model, then task."""
    now = time.time()
    base = tmp_path / "v8"
    for cohort in ("olmo_mbpp", "qwen_math"):
        write(base / cohort / ".multi-node" / "state.json", {"phase": "train", "failure": None})
        (base / cohort / ".multi-node" / "jobs").mkdir(parents=True)
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=200)
    assert status.COHORTS == ("olmo_math", "olmo_mbpp", "qwen_math", "qwen_mbpp")
    summary = [line.split(" · ")[0].strip() for line in text.split(" experiments\n")[1].split("\n\n")[0].splitlines()
               if line.startswith(("   OLMo", "   Qwen"))]
    assert summary == ["OLMo MATH", "OLMo MBPP", "Qwen MATH", "Qwen MBPP"]
    assert "OLMo MBPP · stage: training" in text and "Qwen MATH · stage: training" in text




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
    auto = re.sub(r"\s+", " ", text)   # wrapped lines joined
    assert "STOPPED — needs you" not in auto
    assert ("s6 ? eval at 275: failed 1 of 4 so far (SIGSEGV (exit -11) in embedding (torch/nn/functional.py:2551) ← "
            "generate (srgc_rebuttal/torch_backend.py:412))") in auto, auto
    # crashed on three different nodes (TASK_NATIVE_HOST_LIMIT 3, 0f877d7): no node retries it, a person must look
    job = json.loads((q / "jobs" / "n.json").read_text())
    for i, host in enumerate(("h-m", "h-o")):
        job["attempts"].insert(0, dict(job["attempts"][-1], id=f"na{i}", host=host, worker=f"{host}.1", child_failure=None))
    job["attempts"][-1].update(host="h-n")
    write(q / "jobs" / "n.json", job)
    text = status.render_clean([base], now=now, gpus=[], host="viewer", width=120)
    assert "   STOPPED — needs you (1):" in text
    joined = re.sub(r"\n {7,}(?=\S)", " ", text)          # an item may wrap onto indented lines
    line = next(x for x in joined.splitlines() if x.startswith("     s6 ? eval at 275: "))
    assert "failed 3 times on 3 nodes (h-o, h-m, h-n), no more automatic retries" in line and \
        "SIGSEGV (exit -11) in embedding (torch/nn/functional.py:2551)" in line, line
    note = failure_notes(base, "olmo_math", now)["n"]
    assert "BLOCKED: SIGSEGV (exit -11) on 3 different nodes (h-o, h-m, h-n)" in note
    assert "why:  rank 1 SIGSEGV (exit -11), native crash during answering; crashed in embedding" in note
    assert ("crashed in embedding (torch/nn/functional.py:2551) ← generate (srgc_rebuttal/torch_backend.py:412) ← "
            "evaluate (srgc_rebuttal/run_experiment.py:301)") in note
    no_frame = status.native_story([{"rank": 3, "signal": "11", "native_python_stack":
        "Fatal Python error: Segmentation fault\n\nCurrent thread 0x1 (most recent call first):\n  <no Python frame>\n"}])[0]
    assert "no Python frame (inside a native library" in no_frame
    assert status.native_story([{"rank": 0, "message": "ValueError: x"}]) == (None, None)




def test_native_crash_stack_at_the_end_of_the_task_log_is_found(tmp_path):
    """No rank dump file: the torchrun task log holds the faulthandler stack at its end, after >64 KB of output."""
    log = tmp_path / "task.log"
    log.write_text("x" * 200000 + "\nFatal Python error: Segmentation fault\n\nCurrent thread 0x1 (most recent call first):\n"
                   '  File "/env/site-packages/vllm/worker.py", line 12 in step\n'
                   '  File "/code/v8/experiments/switching/trainer.py", line 99 in rollout\n\nExtension modules: x\n')
    cause, _ = status.native_story([{"rank": None, "signal": 11, "error_file": str(log)}])
    assert "step (vllm/worker.py:12)" in cause and "no fault dump" not in cause, cause






