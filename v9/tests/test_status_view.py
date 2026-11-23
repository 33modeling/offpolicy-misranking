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
                    item("x1", "llama_mbpp", "s5.on.train", "failed", "failed 2 times (last: exit code 1); fix the cause, "
                         "then `sh v9/run.sh retry-failed`", task={"run_id": "llama-mbpp-s5"})]
        return []
    return plan


def test_status_shows_running_failed_waiting_and_next(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    log = tmp_path / "worker.log"
    log.write_text("[v9] start\nupdate 120/275 reward 0.41\n")
    d = status_view.running_dir();d.mkdir(parents=True)
    (d / "llama-s5.json").write_text(json.dumps(dict(task_id="llama-s5", host=socket.gethostname(), pid=os.getpid(),
                                                     started_at=time.time() - 3700, log=str(log))))
    lines = []
    status_view.render(fake_plan("llama-s5"), lambda t: t["run_id"], out=lines.append)
    text = "\n".join(lines)
    assert "tau: frozen, tau=75" in text
    assert "1 running" in text and "1 can start now" in text and "2 waiting" in text and "1 FAILED" in text
    assert "running now (1):" in text and "x1 llama_math s5.on.train" in text and "for 1h01m" in text
    assert "update 120/275 reward 0.41" in text
    assert "FAILED — needs you (1):" in text and "retry-failed" in text
    assert "waiting (2): waiting for V8 eval-final [s5.staged, s6.staged]" in text
    assert "next: a free node takes x1 llama_math s6.on.train" in text


def test_dead_worker_record_is_not_counted_as_running(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    d = status_view.running_dir();d.mkdir(parents=True)
    (d / "llama-s5.json").write_text(json.dumps(dict(task_id="llama-s5", host=socket.gethostname(), pid=2**22 + 12345,
                                                     started_at=time.time() - 9000, log=str(tmp_path / "gone.log"))))
    lines = []
    status_view.render(fake_plan("llama-s5"), lambda t: t["run_id"], out=lines.append)
    text = "\n".join(lines)
    assert "running now: nothing" in text and "possibly stopped: x1 llama_math s5.on.train" in text
    assert "2 can start now" in text


def test_tau_not_frozen_says_what_happens(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    def plan(stage):
        if stage == "freeze-tau":
            return [item("freeze-tau", "olmo_math", "tau", "ready", "olmo_math bank complete; run freeze-tau")]
        return [item(stage, "olmo_math", "all", "blocked", "tau is not frozen yet (stage freeze-tau runs first)")]
    lines = []
    status_view.render(plan, lambda t: t["run_id"], out=lines.append)
    text = "\n".join(lines)
    assert "tau: NOT FROZEN yet" in text and "the first `sh v9/run.sh` freezes it" in text
