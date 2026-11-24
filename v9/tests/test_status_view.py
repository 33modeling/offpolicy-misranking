"""`sh v9/run.sh status`: running tasks from worker records, failures first, waiting reasons per cohort."""
import json
import os
import socket
import time

from v9lib import status_view
from v9lib.common import item


def fake_plan(running_tid):
    def plan(stage):
        if stage == "freeze-tau":
            return [item("freeze-tau", "olmo_math", "tau", "done", "tau=75 basis=budget")]
        if stage == "collect":
            return [item("collect", "all", "files", "ready", "not written yet")]
        if stage == "x1":
            return [item("x1", "olmo_math", "s5.staged", "done"),
                    item("x1", "qwen_mbpp", "s5.staged", "blocked", "waiting for V8 eval-final"),
                    item("x1", "qwen_mbpp", "s6.staged", "blocked", "waiting for V8 eval-final"),
                    item("x1", "llama_math", "s5.on.train", "ready", task={"run_id": running_tid}),
                    item("x1", "llama_math", "s6.on.train", "ready", task={"run_id": "llama-s6"}),
                    item("x1", "llama_mbpp", "s5.on.train", "failed", "failed 2 times, no more automatic retries · last error: "
                         "exit code 1", task={"run_id": "llama-mbpp-s5"})]
        return []
    return plan


def render(plan):
    lines = []
    status_view.render(plan, lambda t: t["run_id"], out=lines.append)
    return "\n".join(lines)


def test_status_follows_the_v8_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    log = tmp_path / "worker.log"
    log.write_text("[v9] start\nupdate 120/275 reward 0.41\n")
    d = status_view.running_dir();d.mkdir(parents=True)
    (d / "llama-s5.json").write_text(json.dumps(dict(task_id="llama-s5", host=socket.gethostname(), pid=os.getpid(),
                                                     started_at=time.time() - 3700, log=str(log))))
    text = render(fake_plan("llama-s5"))
    order = ["V9 status · ", "this node (", " time left", " experiments", "X1 Staged vs SR vs Alignment at 275: not finished",
             "by seed and arm", " nodes: 1 live — 1 working"]
    at = [text.index(x) for x in order]
    assert at == sorted(at), at
    assert "tau · frozen · tau=75" in text
    assert "X1 Staged vs SR vs Alignment at 275 · 16% of the work done · 1 node working on it" in text
    assert "Llama MATH  done 0 of 2 · 1 running · 1 can start" in text
    assert "STOPPED — needs you (1):" in text and "Llama MBPP s5.on.train: failed 2 times" in text
    assert "→ fix the cause, then run `sh v9/run.sh retry-failed` on one node" in text
    assert "running now (1):" in text and "Llama MATH s5.on.train · on " in text and "for 1h01m" in text
    assert "update 120/275 reward 0.41" in text
    assert "can start now (1): the next free node takes Llama MATH s6.on.train" in text
    assert "waiting for V8 eval-final (2): Qwen MBPP s5.staged, Qwen MBPP s6.staged" in text
    assert "Llama MATH X1 by seed and arm" in text and "▶" in text and "FAIL" in text


def test_dead_worker_record_is_not_counted_as_running(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    d = status_view.running_dir();d.mkdir(parents=True)
    (d / "llama-s5.json").write_text(json.dumps(dict(task_id="llama-s5", host=socket.gethostname(), pid=2**22 + 12345,
                                                     started_at=time.time() - 9000, log=str(tmp_path / "gone.log"))))
    text = render(fake_plan("llama-s5"))
    assert "this node (" in text and "no V9 task here" in text
    assert "possibly stopped: Llama MATH s5.on.train" in text and "can start now (2)" in text
    assert "nodes: 0 live" in text


def test_tau_not_frozen_says_what_happens(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    def plan(stage):
        if stage == "freeze-tau":
            return [item("freeze-tau", "olmo_math", "tau", "ready", "olmo_math bank complete; run freeze-tau")]
        return [item(stage, "olmo_math", "all", "blocked", "tau is not frozen yet (stage freeze-tau runs first)")]
    text = render(plan)
    assert "tau · NOT FROZEN yet · the first `sh v9/run.sh` freezes it" in text


def test_split_key():
    assert status_view.split_key("s5.on.train") == ("s5", "on", "train")
    assert status_view.split_key("s5.switch050@B") == ("s5", "switch050@B", "")
    assert status_view.split_key("all") == ("", "all", "")
    assert status_view.split_key("s10.prefix") == ("s10", "prefix", "")


def _worker_log(root, host, pid, lines):
    d = root / "logs";d.mkdir(parents=True, exist_ok=True)
    p = d / f"worker-{host}-20261123T210000-{pid}.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_working_node_is_found_from_its_worker_log_without_a_record(tmp_path, monkeypatch):
    root = tmp_path / "v9";monkeypatch.setenv("V9_ROOT", str(root))
    started = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 3700))
    _worker_log(root, "run371000-v9-s01-1123-21-00-00", 812, [
        "[v9] worker start host=run371000-v9-s01-1123-21-00-00 stages=freeze-tau,x1",
        f"[v9] {started} run371000-v9-s01-1123-21-00-00 stage=x1 cohort=llama_math task=llama-s5",
        "[13:01:00] Llama / MATH | update 100/275 | reward 0.40",
        "[14:02:11] Llama / MATH | update 120/275 | reward 0.41"])
    _worker_log(root, "run371001-v9-s02", 90, [
        "[v9] worker start host=run371001-v9-s02 stages=freeze-tau,x1",
        f"[v9] {started} run371001-v9-s02 stage=x1 cohort=llama_math task=llama-s6",
        "[v9] x1: task llama-s6 FAILED rc=1 (attempt kept; rerunning resumes)"])
    _worker_log(root, "run371002-v9-s03", 91, [
        "[v9] worker start host=run371002-v9-s03 stages=freeze-tau,x1", "[v9] nothing more is ready on this node"])
    text = render(fake_plan("llama-s5"))
    assert "Llama MATH s5.on.train · on run371000-v9-s01-1123-21-00-00 · for 1h01m · update 120 of 275" in text
    assert "nodes: 2 live — 1 working, 1 idle" in text
    assert "run371000-v9-s01-1123-21-00-00     working  X1  Llama MATH s5.on.train" in text
    assert "run371001-v9-s02" in text and "run371002-v9-s03" not in text


def test_old_silent_log_is_not_a_live_node(tmp_path, monkeypatch):
    root = tmp_path / "v9";monkeypatch.setenv("V9_ROOT", str(root))
    p = _worker_log(root, "run1", 5, ["[v9] 2026-11-23T01:00:00 run1 stage=x1 cohort=llama_math task=llama-s5"])
    old = time.time() - status_view.STALE_SECONDS - 60;os.utime(p, (old, old))
    text = render(fake_plan("llama-s5"))
    assert "running now (1)" not in text and "nodes: 0 live" in text


def _beat(root, host, pid, **fields):
    d = root / ".state" / "workers";d.mkdir(parents=True, exist_ok=True)
    (d / f"{host}-{pid}.json").write_text(json.dumps(dict(host=host, pid=pid, at=time.time(), **fields)))


def test_nodes_come_from_heartbeats_with_why_idle(tmp_path, monkeypatch):
    root = tmp_path / "v9"
    monkeypatch.setenv("V9_ROOT", str(root))
    log = tmp_path / "w.log";log.write_text("update 50/275\n")
    _beat(root, "node-a", 11, state="working", task_id="llama-s5", stage="x1", cohort="llama_math",
          task_started=time.time() - 600, log=str(log))
    _beat(root, "node-b", 12, state="waiting", doing="2 task(s) wait for V8 results; V8 workers are live on qwen_mbpp")
    old = root / ".state" / "workers" / "node-c-13.json"
    old.write_text(json.dumps(dict(host="node-c", pid=13, at=time.time() - 3600, state="working", task_id="x")))
    text = render(fake_plan("llama-s5"))
    assert " nodes: 2 live — 1 working, 1 idle — why: 2 task(s) wait for V8 results" in text
    assert "node-a" in text and "working" in text and "Llama MATH s5.on.train" in text
    assert "node-b" in text and "waiting  2 task(s) wait for V8 results" in text
    assert "node-c" not in text                      # heartbeat older than 5 min: not live
    assert "screen counts disagree" not in text


def test_time_left_from_finished_task_times(tmp_path, monkeypatch):
    root = tmp_path / "v9"
    monkeypatch.setenv("V9_ROOT", str(root))
    d = root / ".state" / "done-times";d.mkdir(parents=True)
    for n, sec in enumerate((3600, 7200, 5400)):
        (d / f"t{n}.json").write_text(json.dumps(dict(stage="x1", cohort="llama_math", seconds=sec)))
    text = render(fake_plan("nothing-running"))
    # x1 not done: llama s5, s6 ready (2 x median 1.5 h) + qwen 2 blocked + 1 failed at the stage median
    assert "X1 Staged vs SR vs Alignment at 275: about 7h30m left, done around" in text
    assert "All experiments: about 7h30m left" in text


def test_time_left_unknown_without_measured_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    text = render(fake_plan("nothing-running"))
    assert "unknown — no finished V9 task of this stage measured yet" in text
    assert "All experiments: unknown until X1 can start and has a measured task time" in text


def test_screen_counts_check_reports_disagreement():
    items = [dict(stage="x1", state="done"), dict(stage="x1", state="ready")]
    assert status_view.check_screen({"x1": items}, items, [], {"x1": (2, {"done": 1, "ready": 1})}) == []
    bad = status_view.check_screen({"x1": items}, items, [], {"x1": (3, {"done": 2})})
    assert any("total 3 on screen, 2 tasks" in b for b in bad) and any("done 2 on screen, 1 tasks" in b for b in bad)


def test_stage_that_only_waits_has_no_time(tmp_path, monkeypatch):
    root = tmp_path / "v9"
    monkeypatch.setenv("V9_ROOT", str(root))
    d = root / ".state" / "done-times";d.mkdir(parents=True)
    (d / "t.json").write_text(json.dumps(dict(stage="x1", cohort="qwen_mbpp", seconds=600)))
    plan = lambda stage: [item("x1", "qwen_mbpp", "s5.staged", "blocked", "waiting for V8 eval-final")] if stage == "x1" else []
    text = render(plan)
    assert "X1 Staged vs SR vs Alignment at 275: unknown — no task of this stage can start now (1 wait for V8 results)" in text
    assert "0m left" not in text


def test_ready_task_that_crashed_before_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    note = ("failed 1 time(s); the next free node retries it from its checkpoint (attempt 2 of 2) · last error: exit code "
            "139: a GPU process crashed with SIGSEGV; torchrun then stopped the other ranks")

    def plan(stage):
        if stage == "x3":
            return [item("x3", "olmo_math", "s9.alignment.train", "ready", task={"run_id": "h-s9"}, retry_note=note)]
        return fake_plan("none")(stage)
    text = render(plan)
    assert "failed before, retried by itself (1):" in text
    assert "OLMo MATH s9.alignment.train: failed 1 time(s)" in text and "SIGSEGV" in text
