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
