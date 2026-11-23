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
             "by seed and arm", " nodes: 1 working on V9"]
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
    assert "nodes: 0 working on V9" in text


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
