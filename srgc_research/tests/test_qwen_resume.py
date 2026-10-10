"""Resume-first shared claims: actual DAGs, stale attempts and parallel nodes."""

import importlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.srgc_seed_order import seed_first
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.runtime import atomic_json, lease
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs
from srgc_research.dispatch import qwen_run
from srgc_research.dispatch.qwen_resume import ResumeFirst, resume_first_worker


@pytest.fixture
def queues(tmp_path):
    result = []
    for name in ("math", "mbpp"):
        folder = tmp_path / name
        folder.mkdir()
        path = write_inputs(folder)
        if name == "mbpp":
            spec = json.loads(path.read_text())
            spec.update(dataset="mbpp", verifier="srgc_rebuttal.verifiers:code_reward")
            atomic_json(path, spec)
        queue = TaskQueue(path)
        queue.bind()
        queue.tasks = seed_first(queue.tasks, queue.plan["seeds"])
        result.append(queue)
    return result


def abandoned(queue, task, *, attempt=3, state="running"):
    if task.arm in {"on_policy", "switch", "sr", "random"}:
        finish_fake(queue, Task(task.seed, "prefix"))
    atomic_json(queue.receipt(task), {"task": task.key, "status": state, "attempt": attempt,
        "retry_attempt": attempt, "attempt_id": f"old-{task.key}", "started": time.time() - 50,
        "finished": time.time() - 50})


def test_resume_beats_new_earlier_seed_and_bypasses_abandoned_attempt_limit(queues):
    queue = queues[0]
    old = Task(9, "sr")
    abandoned(queue, old, attempt=99)
    manager = ResumeFirst(queues)
    before = {p: p.read_bytes() for p in queue.plan_path.parent.glob("*.json")}
    with manager.claim(queue) as task:
        assert task == old
        row = json.loads(queue.receipt(task).read_text())
        assert row["attempt"] == 100 and row["retry_attempt"] == 1 and row["resume_first"]
        assert queue.claim_fds
    assert {p: p.read_bytes() for p in before} == before
    history = json.loads((queue.directory / "attempts" / f"old-{old.key}.json").read_text())
    assert history["status"] == "abandoned" and history["attempt"] == 99


def test_mbpp_resume_blocks_new_math_until_all_existing_work_finishes(queues):
    math, mbpp = queues
    old = Task(8, "switch")
    abandoned(mbpp, old)
    manager = ResumeFirst(queues)
    with manager.claim(math) as task:
        assert task is None
    with manager.claim(mbpp) as task:
        assert task == old
        # A second node must not open a fresh seed while the old job is owned.
        second = ResumeFirst(queues)
        with second.claim(math) as other:
            assert other is None
        finish_fake(mbpp, task)
        assert mbpp.finish(task, 0) == 0
    with manager.claim(math) as task:
        assert task == Task(5, "prefix")
        assert not json.loads(math.receipt(task).read_text())["resume_first"]


def test_resume_backlog_keeps_other_already_running_tasks_before_fresh_work(queues):
    math, mbpp = queues
    orphan, running = Task(8, "prefix"), Task(7, "prefix")
    abandoned(math, orphan)
    abandoned(mbpp, running)
    manager = ResumeFirst(queues)
    with lease(mbpp.directory / "leases" / f"{running.key}.lock"):
        with manager.claim(math) as task:
            assert task == orphan
            finish_fake(math, task)
            math.finish(task, 0)
        with manager.claim(math) as task:
            assert task is None
        finish_fake(mbpp, running)
    with manager.claim(math) as task:
        assert task == Task(5, "prefix")


def test_clean_queues_still_allow_multiple_nodes_to_claim_new_work_in_parallel(queues):
    manager = ResumeFirst(queues)
    with manager.claim(queues[0]) as first:
        assert first == Task(5, "prefix")
        with ResumeFirst(queues).claim(queues[0]) as second:
            assert second == Task(6, "prefix")


def test_partial_checkpoint_without_a_receipt_is_resumed_first(queues):
    queue = queues[0]
    checkpoint = queue.root / "seed-9/prefix-latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"must resume this checkpoint without editing it")
    with ResumeFirst(queues).claim(queue) as task:
        assert task == Task(9, "prefix")
        assert checkpoint.read_bytes() == b"must resume this checkpoint without editing it"


def test_active_manual_execution_is_not_duplicated(queues):
    queue = queues[0]
    task = Task(8, "prefix")
    abandoned(queue, task)
    with lease(queue.root / "seed-8/.prefix.execution.lock"):
        manager = ResumeFirst(queues)
        # There is no unowned resume backlog; unrelated free-node work is allowed.
        with manager.claim(queue) as picked:
            assert picked != task


def test_real_failed_attempts_keep_their_limit_and_block_new_work(queues):
    queue = queues[0]
    task = Task(9, "prefix")
    abandoned(queue, task, attempt=100)
    for i in range(3):
        atomic_json(queue.directory / "attempts" / f"failure-{i}.json",
                    {"task": task.key, "status": "failed", "attempt_id": f"failure-{i}"})
    with pytest.raises(RuntimeError, match="fresh tasks remain blocked"), ResumeFirst(queues).claim(queue):
        pass
    assert not queue.receipt(Task(5, "prefix")).exists()


def test_abandoned_history_does_not_consume_real_failure_budget(queues):
    queue = queues[0]
    task = Task(9, "prefix")
    abandoned(queue, task, attempt=100)
    for i in range(8):
        atomic_json(queue.directory / "attempts" / f"abandoned-{i}.json",
                    {"task": task.key, "status": "abandoned", "attempt_id": f"abandoned-{i}"})
    atomic_json(queue.directory / "attempts/actual-failure.json",
                {"task": task.key, "status": "failed", "attempt_id": "actual-failure"})
    with ResumeFirst(queues).claim(queue) as picked:
        assert picked == task
        assert json.loads(queue.receipt(task).read_text())["retry_attempt"] == 2


def test_failed_old_task_waits_for_cooldown_without_opening_fresh_work(queues):
    queue = queues[0]
    task = Task(9, "prefix")
    abandoned(queue, task, state="failed", attempt=1)
    row = json.loads(queue.receipt(task).read_text())
    row["finished"] = time.time()
    atomic_json(queue.receipt(task), row)
    with ResumeFirst(queues).claim(queue, retry_failed=True) as picked:
        assert picked is None
    with ResumeFirst(queues).claim(queue, retry_failed=True, retry_delay=0) as picked:
        assert picked == task


def test_missing_prerequisite_is_restored_for_same_existing_seed(queues):
    queue = queues[0]
    task = Task(9, "sr")
    abandoned(queue, task)
    (queue.root / "seed-9/prefix-ready.json").unlink()
    with ResumeFirst(queues).claim(queue) as picked:
        assert picked == Task(9, "prefix")


def test_completed_receipt_without_endpoint_does_not_silently_restart(queues):
    queue = queues[0]
    abandoned(queue, Task(9, "sr"), state="complete")
    with pytest.raises(ValueError, match="lost its completion evidence"), ResumeFirst(queues).claim(queue):
        pass


def parallel_worker(paths, barrier, jobs):
    queues = [TaskQueue(Path(path)) for path in paths]
    for queue in queues:
        queue.tasks = seed_first(queue.tasks, queue.plan["seeds"])
    manager = ResumeFirst(queues)
    barrier.wait(10)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        for queue in queues:
            with manager.claim(queue) as task:
                if task is not None:
                    started = time.monotonic()
                    is_resume = json.loads(queue.receipt(task).read_text())["resume_first"]
                    time.sleep(.1)
                    finish_fake(queue, task)
                    assert queue.finish(task, 0) == 0
                    jobs.put((queue.plan["dataset"], task.key, is_resume, started, time.monotonic()))
                    break
        if all(all(queue.complete(task) for task in queue.tasks) for queue in queues):
            return
        time.sleep(.01)
    raise AssertionError("parallel queue did not drain")


def test_two_nodes_finish_all_resumes_before_new_tasks_and_do_not_duplicate(queues):
    for queue, seed in ((queues[0], 8), (queues[0], 9), (queues[1], 8), (queues[1], 9)):
        abandoned(queue, Task(seed, "prefix"))
    ctx = multiprocessing.get_context("fork")
    barrier, jobs = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=parallel_worker, args=([str(q.plan_path) for q in queues], barrier, jobs))
                 for _ in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(25)
            assert process.exitcode == 0
        rows = [jobs.get(timeout=2) for _ in range(50)]
        assert len({row[:2] for row in rows}) == 50
        resumed = [row for row in rows if row[2]]
        fresh = [row for row in rows if not row[2]]
        assert len(resumed) == 4 and len(fresh) == 46
        assert min(row[3] for row in fresh) >= max(row[4] for row in resumed)
        events = sorted([(r[3], 1) for r in resumed] + [(r[4], -1) for r in resumed])
        active, peak = 0, 0
        for _, delta in events:
            active += delta
            peak = max(peak, active)
        assert peak == 2
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def test_claim_is_inherited_by_child_after_parent_is_killed(queues, tmp_path):
    queue = queues[0]
    task = Task(9, "prefix")
    abandoned(queue, task)
    ready = tmp_path / "child-ready"
    def worker():
        manager = ResumeFirst(queues)
        with manager.claim(queue) as picked:
            assert picked == task
            child = subprocess.Popen([sys.executable, "-c",
                "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(20)",
                str(ready)], pass_fds=queue.claim_fds)
            child.wait()
    ctx = multiprocessing.get_context("fork")
    parent = ctx.Process(target=worker)
    parent.start()
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        child_pid = int(ready.read_text())
        parent.kill()
        parent.join(5)
        manager = ResumeFirst(queues)
        with manager.claim(queue) as picked:
            assert picked is None
        os.kill(child_pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while manager.owned(queue, task) and time.monotonic() < deadline:
            time.sleep(.01)
        with manager.claim(queue) as picked:
            assert picked == task
    finally:
        if parent.is_alive():
            parent.kill()
            parent.join()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_existing_worker_drain_gets_resume_claim_and_restores_hooks(queues, monkeypatch):
    monkeypatch.syspath_prepend(str(qwen_run.REPO / "scripts"))
    worker = importlib.import_module("srgc_qwen35_worker")
    from scripts import srgc_process_guard as guard
    original_claim, original_drain, owners = queues[0].claim, worker.drain, guard.OWNER_MARKERS
    abandoned(queues[0], Task(9, "prefix"))
    def drain(queues, args, *rest):
        assert args.retry_failed
        assert "srgc_research.dispatch.qwen_run" in guard.OWNER_MARKERS
        with queues[0].claim() as task:
            assert task == Task(9, "prefix")
    with patch.object(worker, "drain", drain), resume_first_worker():
        worker.drain(queues, SimpleNamespace(retry_failed=False), {}, (), "test", lambda *_: None)
    assert queues[0].claim == original_claim and worker.drain == original_drain and guard.OWNER_MARKERS == owners


def test_real_worker_drain_finishes_mbpp_resume_and_retries_it_before_any_new_math(queues, monkeypatch):
    monkeypatch.syspath_prepend(str(qwen_run.REPO / "scripts"))
    worker = importlib.import_module("srgc_qwen35_worker")
    from srgc_rebuttal import cluster
    old = Task(9, "sr")
    abandoned(queues[1], old, attempt=99)
    options = SimpleNamespace(retry_failed=False, max_attempts=3, retry_delay=0,
        heartbeat_seconds=.01, stall_seconds=30, poll_seconds=.01)
    jobs = []
    def child(command, *args, **kwargs):
        queue, task = command
        jobs.append((queue.plan["dataset"], task.key))
        if len(jobs) == 1:
            return 9
        finish_fake(queue, task)
        return 0
    with patch.object(cluster, "gpu_identity"), \
            patch.object(cluster, "task_command", side_effect=lambda q, t: (q, t)), \
            patch.object(cluster, "run_child", side_effect=child), \
            patch.object(cluster, "publish_reports"), resume_first_worker():
        worker.drain(queues, options, {}, (), "test", lambda *_args, **_kw: None)
    assert jobs[:2] == [("mbpp", old.key), ("mbpp", old.key)]
    assert len(jobs) == 50
    assert all(all(queue.complete(task) for task in queue.tasks) for queue in queues)
    assert not options.retry_failed


def test_math_only_node_respects_shared_mbpp_resume_gate(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(qwen_run.REPO / "scripts"))
    from scripts import srgc_qwen35 as qwen
    from srgc_rebuttal import cluster_queue
    from srgc_research.tests.test_qwen_status import ChatTokenizer
    root = tmp_path / "qwen"
    paths = {name: qwen.prepare(name, qwen_run.REPO / "srgc_rebuttal/experiments" / filename,
                               root, ChatTokenizer()) for name, filename in (
        ("math", "additional_seeds.json"), ("mbpp", "mbpp_seeds.json"))}
    with qwen.runtime_adapter():
        math, mbpp = [cluster_queue.TaskQueue(paths[name]) for name in ("math", "mbpp")]
        for queue in (math, mbpp):
            queue.bind()
        abandoned(mbpp, Task(9, "cache"))
        manager = ResumeFirst([math])
        assert len(manager.queues) == 2
        with manager.claim(math) as task:
            assert task is None


def test_finished_endpoint_still_holding_lease_blocks_new_work(queues):
    math, mbpp = queues
    old = Task(9, "prefix")
    abandoned(mbpp, old)
    manager = ResumeFirst(queues)
    with manager.claim(mbpp) as task:
        finish_fake(mbpp, task)
        mbpp.finish(task, 0)
        with ResumeFirst(queues).claim(math) as other:
            assert other is None
    with manager.claim(math) as other:
        assert other == Task(5, "prefix")


def test_startup_redirect_keeps_resume_policy_after_exec_without_changing_preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(qwen_run.REPO / "scripts"))
    start = importlib.import_module("srgc_qwen35_start")
    with patch.object(start, "prepare_missing") as prepare, patch("os.execv") as execute:
        qwen_run.main(["all"])
    assert prepare.call_count == 1
    assert execute.call_args.args[1][1:5] == ["-m", "srgc_research.dispatch.qwen_run", "all", "run"]


def test_explicit_run_enters_existing_diagnostics_with_hooks(monkeypatch):
    monkeypatch.syspath_prepend(str(qwen_run.REPO / "scripts"))
    diagnostics = importlib.import_module("srgc_qwen35_diagnostics")
    worker = importlib.import_module("srgc_qwen35_worker")
    original = worker.drain
    def entry():
        assert worker.drain is not original
        assert sys.argv[1:] == ["math", "run", "--root", "/existing/run"]
    with patch.object(diagnostics, "main", side_effect=entry):
        qwen_run.main(["math", "run", "--root", "/existing/run"])
    assert worker.drain is original


def test_new_dispatch_does_not_change_pinned_runtime_hashes():
    from scripts.srgc_qwen35 import ADAPTER_FILES
    from srgc_research.storage import runtime_files
    assert not any(name.startswith("srgc_research/dispatch/") for name in runtime_files())
    assert not any(name.startswith("srgc_research/dispatch/") for name in ADAPTER_FILES)


@pytest.mark.parametrize("value", [[], {"schema": "wrong"}, {"schema": "qwen-resume-first-v1", "plans": {}, "tasks": "wrong"}])
def test_invalid_resume_metadata_is_preserved_and_blocks_fresh_work(queues, value):
    manager = ResumeFirst(queues)
    atomic_json(manager.marker, value)
    before = manager.marker.read_bytes()
    with pytest.raises((ValueError, TypeError)), manager.claim(queues[0]):
        pass
    assert manager.marker.read_bytes() == before
    assert not queues[0].receipt(Task(5, "prefix")).exists()
