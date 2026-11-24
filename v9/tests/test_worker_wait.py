"""V9 worker: waits while work can still open up, restarts stopped tasks by itself when nothing else is left."""
import json
import time

from v9lib import orchestrate
from v9lib.common import item


def test_waits_while_another_node_works(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"));monkeypatch.setenv("V8_ROOT", str(tmp_path / "v8"))
    d = tmp_path / "v9" / ".state" / "workers";d.mkdir(parents=True)
    (d / "other-1.json").write_text(json.dumps(dict(host="other", pid=1, at=time.time(), state="working", task_id="t")))
    blocked = [item("x2", "olmo_math", "s5.sr@B", "blocked", "waiting for x1")]
    assert "other node(s) run V9 tasks: other" in orchestrate.wait_reason(blocked)
    (d / "other-1.json").write_text(json.dumps(dict(host="other", pid=1, at=time.time() - 900, state="working")))
    assert orchestrate.wait_reason(blocked) is None            # stale heartbeat: nothing can open up
    assert orchestrate.wait_reason([item("x1", "olmo_math", "s5", "done")]) is None


def test_waits_for_live_v8_results(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"));monkeypatch.setenv("V8_ROOT", str(tmp_path / "v8"))
    blocked = [item("x1", "qwen_mbpp", "s5.staged", "blocked", "waiting for V8 eval-final")]
    assert orchestrate.wait_reason(blocked) is None
    w = tmp_path / "v8" / "qwen_mbpp" / ".multi-node" / "workers";w.mkdir(parents=True)
    (w / "n.json").write_text("{}")
    assert "V8 workers are live on qwen_mbpp" in orchestrate.wait_reason(blocked)


def test_auto_retry_at_most_three_times(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    monkeypatch.setattr(orchestrate, "AUTO_RETRIES", 3)
    for _ in range(2):
        orchestrate.record_failure("t1", "exit code 1")
    it = item("x1", "llama_math", "s5.on.train", "failed", "x", task={"run_id": "t1"})
    for n in range(3):
        assert orchestrate.auto_retry([it]) == ["t1"]
        assert orchestrate.failures("t1")["count"] == 0
        for _ in range(2):
            orchestrate.record_failure("t1", "exit code 1")
    assert orchestrate.auto_retry([it]) == []
    f = orchestrate.failures("t1")
    assert len(f["auto_retries"]) == 3 and f["count"] == 2 and len(f["errors"]) == 8


def test_heartbeat_file_written_and_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"))
    b = orchestrate.Beat();b.start()
    b.set(state="working", task_id="t9", stage="x3", cohort="olmo_math", task_started=time.time())
    rec = json.loads(b.path.read_text())
    assert rec["state"] == "working" and rec["task_id"] == "t9" and rec["at"] > time.time() - 5
    b.set(state="waiting", doing="why")
    assert "task_id" not in json.loads(b.path.read_text())
    b.close()
    assert not b.path.exists()


def test_random_tasks_run_after_every_other_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("V9_ROOT", str(tmp_path / "v9"));monkeypatch.setenv("V8_ROOT", str(tmp_path / "v8"))
    done, order = set(), []
    plans = {"x1": [("x1.rand", True), ("x1.staged", False)], "x2": [("x2.a", False)]}

    def plan(stage):
        return [item(stage, "olmo_math", tid, "done" if tid in done else "ready", task={"run_id": tid},
                     **({"late": True} if late else {})) for tid, late in plans.get(stage, [])]

    def launch(stage, it):
        order.append(it["task"]["run_id"]);done.add(it["task"]["run_id"])
        return 0
    monkeypatch.setattr(orchestrate, "plan", plan);monkeypatch.setattr(orchestrate, "launch", launch)
    orchestrate.work(["x1", "x2"])
    assert order == ["x1.staged", "x2.a", "x1.rand"]


def test_rank_crash_is_a_failure_even_when_rank0_wrote_75():
    from v9lib import orchestrate as o
    line = "  traceback : Signal 11 (SIGSEGV) received by PID 620590"
    crashes = [o.RANK_SIGNAL.search(line).group(2)]
    assert crashes == ["SIGSEGV"]
    rc, why = o.classify_exit(1, 75, crashes, stopping=False)
    assert rc == 139 and "SIGSEGV" in why
    # the worker itself was asked to stop: stays resumable
    assert o.classify_exit(1, 75, crashes, stopping=True) == (75, None)
    # pod shutdown (SIGTERM from outside) stays resumable
    assert o.classify_exit(1, 75, ["SIGTERM"], stopping=False) == (75, None)
    # OOM killer is a failure
    assert o.classify_exit(1, None, ["SIGKILL"], stopping=False)[0] == 137
    assert o.classify_exit(0, None, [], stopping=False) == (0, None)
    assert o.classify_exit(1, None, [], stopping=False) == (1, None)
    o.CRASH["why"] = why
    assert o.failure_text(139).startswith("exit code 139: a GPU process crashed with SIGSEGV")
    o.CRASH["why"] = None
