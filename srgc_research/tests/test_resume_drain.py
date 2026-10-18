"""Real queue evidence: sibling recovery, ownership, retries and stop handling."""

import importlib
import json
import multiprocessing
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.plan import input_path
from srgc_rebuttal.runtime import Busy, atomic_json, lease
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs
from srgc_research.dispatch import qwen_run, resume_drain
from srgc_research.dispatch.ready_dispatch import (
    GemmaResumeFirst,
    LlamaResumeFirst,
    ResumeFirst,
)
from srgc_research.tests.test_qwen_resume import abandoned


@pytest.fixture(params=("qwen", "llama", "gemma"))
def model(request, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(qwen_run.REPO / "scripts"))
    choices = {
        "qwen": ("srgc_qwen35_worker", ResumeFirst, "qwen35-9b-{dataset}.json", "SRGC_QWEN_PLANS"),
        "llama": ("srgc_research.dispatch.llama31.worker", LlamaResumeFirst, "llama31-8b-{dataset}.json", "SRGC_LLAMA_PLANS"),
        "gemma": ("srgc_research.dispatch.gemma4.worker", GemmaResumeFirst, "gemma4-12b-pt-{dataset}.json", "SRGC_GEMMA_PLANS"),
    }
    module, factory, pattern, env_key = choices[request.param]
    worker = importlib.import_module(module)
    parent = tmp_path / "experiments"
    parent.mkdir()
    queues = []
    for dataset in ("math", "mbpp"):
        folder = tmp_path / dataset
        folder.mkdir()
        source = write_inputs(folder)
        plan = json.loads(source.read_text())
        plan.update(dataset="math_train" if dataset == "math" else "mbpp", output_root=str(folder / "runs"),
                    input_pattern=str(folder / "inputs-{seed}.json"))
        if dataset == "mbpp":
            plan["verifier"] = "srgc_rebuttal.verifiers:code_reward"
        path = parent / pattern.format(dataset=dataset)
        atomic_json(path, plan)
        queue = TaskQueue(path)
        queue.bind()
        queues.append(queue)
    queues[0].tasks = [Task(5, "prefix")]
    options = SimpleNamespace(max_attempts=3, retry_delay=0, heartbeat_seconds=.01,
                              stall_seconds=30, poll_seconds=.01, retry_failed=False)
    jobs, updates = [], []

    def run(child=None, selected=None, worker_id="test"):
        def complete(command, log, environment, **kwargs):
            queue, task = command
            jobs.append(("math" if queue.plan["dataset"] == "math_train" else "mbpp", task.key))
            assert queue.claim_fds
            assert set(json.loads(environment[env_key])) == {str(q.plan_path) for q in queues}
            kwargs["heartbeat"](123)
            if child:
                return child(queue, task, log, **kwargs)
            finish_fake(queue, task)
            return 0
        with patch.object(cluster, "gpu_identity"), \
                patch.object(cluster, "task_command", side_effect=lambda q, t: (q, t)), \
                patch.object(cluster, "run_child", side_effect=complete), \
                patch.object(cluster, "publish_reports"), \
                resume_drain.resume_worker(worker, factory, pattern=pattern,
                                          env_key=env_key, label=request.param.upper()):
            worker.drain(selected or [queues[0]], options, {}, (), worker_id,
                         lambda state, **details: updates.append((state, details)))
    return SimpleNamespace(queues=queues, run=run, jobs=jobs, updates=updates,
                           factory=factory, worker=worker, options=options,
                           pattern=pattern, env_key=env_key)


def run_worker(model, monkeypatch, tmp_path, *, child=None, admission=None, selected=None, probe=None):
    from srgc_rebuttal import cluster_queue

    queues = {q.plan_path: q for q in model.queues}
    monkeypatch.setattr(cluster_queue, "TaskQueue", lambda path: queues[path.resolve()])

    def complete(command, log, environment, **kwargs):
        queue, task = command
        model.jobs.append(("math" if queue.plan["dataset"] == "math_train" else "mbpp", task.key))
        if child:
            return child(queue, task, **kwargs)
        finish_fake(queue, task)
        return 0

    with patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", ("a", "b", "c", "d")), side_effect=probe), \
            patch.object(cluster, "task_command", side_effect=lambda q, t: (q, t)), \
            patch.object(cluster, "run_child", side_effect=complete), \
            patch.object(cluster, "publish_reports") as publish, \
            patch.object(model.worker, "runtime_signature", return_value={"test": "runtime"}), \
            resume_drain.resume_worker(model.worker, model.factory, pattern=model.pattern,
                                      env_key=model.env_key, label="TEST"):
        model.worker.worker([q.plan_path for q in selected or [model.queues[0]]], model.options, tmp_path, tmp_path,
                            admission or (lambda *a, **kw: {"passed": True}))
    return publish


def test_worker_checks_sibling_backlog_even_when_requested_dataset_is_complete(model, monkeypatch, tmp_path):
    finish_fake(model.queues[0], Task(5, "prefix"))
    old = Task(9, "sr")
    abandoned(model.queues[1], old)
    run_worker(model, monkeypatch, tmp_path)
    assert model.jobs == [("mbpp", old.key)]
    assert model.queues[1].complete(old)


def test_completed_worker_does_not_request_gpu_or_admission(model, monkeypatch, tmp_path):
    finish_fake(model.queues[0], Task(5, "prefix"))
    monkeypatch.setattr(cluster, "child_environment", lambda: pytest.fail("completed worker requested GPU setup"))
    run_worker(model, monkeypatch, tmp_path,
               admission=lambda *a, **kw: pytest.fail("completed worker requested admission"))
    assert model.jobs == []


def test_worker_binds_sibling_runtime_before_admission(model, monkeypatch, tmp_path):
    atomic_json(model.queues[1].directory / "runtime.json", {"incompatible": "runtime"})
    monkeypatch.setattr(cluster, "device_leases", lambda *a: pytest.fail("incompatible runtime requested GPUs"))
    with pytest.raises(ValueError, match="runtime differs"):
        run_worker(model, monkeypatch, tmp_path,
                   admission=lambda *a, **kw: pytest.fail("incompatible cohort admitted"))
    assert not (model.queues[0].directory / "runtime.json").exists()


@pytest.mark.parametrize("signal_number", (signal.SIGTERM, signal.SIGKILL))
def test_worker_propagates_child_interruption_and_keeps_stopped_receipt(model, monkeypatch, tmp_path, signal_number):
    with pytest.raises(SystemExit) as stopped:
        run_worker(model, monkeypatch, tmp_path, child=lambda *a, **kw: -signal_number)
    assert stopped.value.code == 128 + signal_number
    receipts = list((model.queues[0].directory / "workers").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "stopped"


def test_dataset_only_worker_finishes_math_before_sibling_resume(model):
    math, mbpp = model.queues
    old = Task(9, "sr")
    abandoned(mbpp, old, attempt=99)
    prefix = mbpp.root / "seed-9/prefix.pt"
    before = prefix.read_bytes()
    model.run()
    assert model.jobs == [("math", "seed-5.prefix"), ("mbpp", old.key)]
    assert mbpp.complete(old) and math.complete(Task(5, "prefix"))
    assert prefix.read_bytes() == before
    assert not mbpp.receipt(Task(5, "prefix")).exists()
    assert model.updates[-1][0] == "complete"
    row = json.loads((mbpp.directory / "workers/test.json").read_text())
    assert row["status"] == "complete"


def test_error_prerequisite_recovers_before_an_independent_interrupted_task(model):
    math = model.queues[0]
    interrupted, failed = Task(8, "sr"), Task(9, "switch")
    abandoned(math, interrupted, state="interrupted")
    abandoned(math, failed, state="failed", attempt=99)
    (math.root / "seed-9/prefix-ready.json").unlink()
    math.tasks = [interrupted, Task(9, "prefix"), failed]
    with model.factory(model.queues).claim(math, retry_failed=True, retry_delay=0) as task:
        assert task == Task(9, "prefix")


def test_launch_exception_retries_saved_task_without_abandoning_worker(model):
    math = model.queues[0]
    old, fresh = Task(9, "sr"), Task(5, "prefix")
    abandoned(math, old, state="failed", attempt=99)
    math.tasks = [fresh, old]

    def child(queue, task, log, **kwargs):
        if len(model.jobs) == 1:
            raise OSError("temporary process launch failure")
        finish_fake(queue, task)
        return 0

    model.run(child)
    assert model.jobs == [("math", old.key), ("math", old.key), ("math", fresh.key)]
    history = [json.loads(p.read_text()) for p in (math.directory / "attempts").glob("*.json")]
    assert any(row["status"] == "failed" for row in history)


def test_sibling_real_failure_is_retried_after_math_finishes(model):
    old = Task(9, "sr")
    abandoned(model.queues[1], old)

    def child(queue, task, log, **kwargs):
        if queue.plan["dataset"] == "mbpp" and sum(name == "mbpp" for name, _ in model.jobs) == 1:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("RuntimeError: real failed attempt\n")
            return 9
        finish_fake(queue, task)
        return 0
    model.run(child)
    assert model.jobs == [("math", "seed-5.prefix"), ("mbpp", old.key), ("mbpp", old.key)]
    history = [json.loads(path.read_text()) for path in (model.queues[1].directory / "attempts").glob("*.json")]
    assert any(row["status"] == "failed" and row["exit_code"] == 9 for row in history)


def test_repeated_child_failure_does_not_starve_other_old_errors_in_same_model(model):
    queue = model.queues[0]
    failed, other, fresh = Task(5, "on_policy"), Task(5, "sr"), Task(6, "prefix")
    queue.tasks = [failed, other, fresh]
    abandoned(queue, failed, attempt=40, state="failed")
    abandoned(queue, other, attempt=4, state="failed")
    checkpoint = queue.root / "seed-5/on_policy-latest.pt"
    checkpoint.write_bytes(b"unchanged optimizer checkpoint")

    def child(queue, task, log, **kwargs):
        if task == failed and sum(key == failed.key for _, key in model.jobs) == 1:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("[rank2]: RuntimeError: actual failed training child\n"
                           "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:\n")
            return 1
        finish_fake(queue, task)
        return 0

    model.run(child)
    assert model.jobs == [("math", task.key) for task in (failed, other, failed, fresh)]
    assert checkpoint.read_bytes() == b"unchanged optimizer checkpoint"
    assert json.loads(queue.receipt(failed).read_text())["attempt"] == 42


def test_failed_training_rank_returns_to_model_chain_after_releasing_task_lease(model, monkeypatch):
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    queue = model.queues[0]
    task, fresh = Task(5, "on_policy"), Task(6, "prefix")
    queue.tasks = [task, fresh]
    abandoned(queue, task, attempt=40, state="failed")

    def fail(queue, task, log, **kwargs):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("[rank2]: RuntimeError: original rank failure\n"
                       "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:\n")
        return 1

    with patch.object(cluster, "gpu_identity"), \
            patch.object(cluster, "task_command", side_effect=lambda q, t: (q, t)), \
            patch.object(cluster, "run_child", side_effect=lambda command, log, env, **kw: fail(*command, log)), \
            patch.object(cluster, "publish_reports"):
        result = resume_drain.drain(model.worker, model.factory, model.pattern, model.env_key,
            "TEST", [queue], model.options, {}, (), "test", lambda *a, **kw: None)
    assert result == 1
    assert json.loads(queue.receipt(task).read_text())["attempt"] == 41
    assert not queue.receipt(fresh).exists()
    with lease(queue.directory / "leases" / f"{task.key}.lock"):
        pass
    # Same saved task is immediately eligible on the next recovery turn.
    with model.factory(model.queues).claim(queue, retry_failed=True, retry_delay=0) as selected:
        assert selected == task


@pytest.mark.parametrize("archived_history", (False, True))
def test_mbpp_error_x4_is_recovered_after_all_math_work(model, archived_history):
    math, mbpp = model.queues
    interrupted, failed = Task(8, "sr"), Task(9, "switch")
    math.tasks = [interrupted, Task(5, "prefix")]
    abandoned(math, interrupted, attempt=99)
    abandoned(mbpp, failed, attempt=4, state="failed")
    checkpoint = mbpp.root / "seed-9/switch-latest.pt"
    checkpoint.write_bytes(b"saved optimizer and selector state")
    old = json.loads(mbpp.receipt(failed).read_text())
    if archived_history:
        for attempt in range(1, 5):
            attempt_id = old["attempt_id"] if attempt == 4 else f"earlier-{failed.key}-{attempt}"
            atomic_json(mbpp.directory / "attempts" / f"{attempt_id}.json",
                        {**old, "attempt": attempt, "retry_attempt": attempt, "attempt_id": attempt_id})
    archives = {p: p.read_bytes() for p in (mbpp.directory / "attempts").glob("*.json")}
    observed = []

    def complete(queue, task, log, **kwargs):
        observed.append(json.loads(queue.receipt(task).read_text()))
        finish_fake(queue, task)
        return 0

    model.run(complete)
    assert model.jobs == [("math", interrupted.key), ("math", "seed-5.prefix"), ("mbpp", failed.key)]
    assert observed[-1]["attempt"] == 5
    assert observed[-1]["resume_first"]
    assert checkpoint.read_bytes() == b"saved optimizer and selector state"
    assert json.loads((mbpp.directory / "attempts" / f"{old['attempt_id']}.json").read_text()) == old
    assert all(path.read_bytes() == data for path, data in archives.items())
    assert model.updates[-1][0] == "complete"


def test_failed_arm_beats_an_earlier_interrupted_arm_in_the_same_dataset(model):
    queue = model.queues[0]
    interrupted, failed = Task(9, "on_policy"), Task(8, "sr")
    queue.tasks = [interrupted, failed]
    abandoned(queue, interrupted, attempt=99)
    abandoned(queue, failed, attempt=4, state="failed")
    model.run()
    assert model.jobs == [("math", failed.key), ("math", interrupted.key)]


def test_old_error_continues_past_three_failures_and_finishes(model):
    task = Task(9, "sr")
    queue = model.queues[1]
    abandoned(queue, task, attempt=4, state="failed")

    def fail(queue, task, log, **kwargs):
        if queue.plan["dataset"] == "mbpp" and sum(name == "mbpp" for name, _ in model.jobs) <= 5:
            return 9
        finish_fake(queue, task)
        return 0

    model.run(fail)
    assert model.jobs == [("math", "seed-5.prefix")] + [("mbpp", task.key)] * 6
    receipt = json.loads(queue.receipt(task).read_text())
    assert receipt["attempt"] == 10 and receipt["status"] == "complete"
    budget = model.factory(model.queues).root / ".dispatch" / f"{model.factory.retry_revision}.json"
    assert not budget.exists()
    for _ in range(2):
        model.run(fail)
    assert len(model.jobs) == 7
    history = [json.loads(p.read_text()) for p in (queue.directory / "attempts").glob("*.json")]
    assert sorted(row["attempt"] for row in history if row["status"] == "failed") == [4, 5, 6, 7, 8, 9]


def test_chain_startup_recovers_error_x4_before_gpu_admission_can_skip_model(model, monkeypatch, tmp_path):
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    failed = Task(9, "sr")
    abandoned(model.queues[1], failed, attempt=4, state="failed")
    run_worker(model, monkeypatch, tmp_path)
    assert model.jobs == [("math", "seed-5.prefix"), ("mbpp", failed.key)]
    assert model.queues[1].complete(failed)


def test_corrupt_inputs_are_refused_before_renewing_old_error_budget(model):
    queue = model.queues[1]
    task = Task(9, "sr")
    abandoned(queue, task, attempt=4, state="failed")
    before = queue.receipt(task).read_bytes()
    path = input_path(queue.plan_path, queue.plan, task.seed)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen inputs changed"):
        model.run()
    assert not model.jobs
    assert queue.receipt(task).read_bytes() == before
    assert not (model.factory(model.queues).root / ".dispatch" / f"{model.factory.retry_revision}.json").exists()


def test_active_sibling_owner_does_not_idle_a_worker_with_ready_work(model, capsys):
    old = Task(9, "sr")
    mbpp = model.queues[1]
    abandoned(mbpp, old)
    assert model.factory(model.queues).pending()
    with lease(mbpp.directory / "leases" / f"{old.key}.lock"), \
            patch.object(resume_drain.time, "sleep", side_effect=KeyboardInterrupt), \
            pytest.raises(KeyboardInterrupt):
        model.run()
    assert model.jobs == [("math", "seed-5.prefix")]
    assert "owned=1" in model.updates[-1][1]["idle_reason"]
    assert "mbpp:seed-9.sr" in capsys.readouterr().out
    assert model.queues[0].complete(Task(5, "prefix"))
    assert json.loads(mbpp.receipt(old).read_text())["attempt"] == 3


def test_ready_arm_runs_while_another_arm_in_the_same_dataset_is_owned(model):
    math = model.queues[0]
    running, ready = Task(5, "on_policy"), Task(5, "sr")
    math.tasks = [running, ready]
    abandoned(math, running)
    model.factory(model.queues).pending()
    with lease(math.directory / "leases" / f"{running.key}.lock"), \
            patch.object(resume_drain.time, "sleep", side_effect=KeyboardInterrupt), \
            pytest.raises(KeyboardInterrupt):
        model.run()
    assert model.jobs == [("math", ready.key)]
    assert math.complete(ready) and not math.complete(running)
    assert json.loads(math.receipt(running).read_text())["attempt"] == 3


def test_active_prefix_still_blocks_its_own_arms_but_not_other_seeds(model):
    math = model.queues[0]
    running, independent = Task(5, "prefix"), Task(6, "prefix")
    math.tasks = [Task(5, "on_policy"), Task(5, "sr"), independent, running]
    abandoned(math, running)
    model.factory(model.queues).pending()
    with lease(math.directory / "leases" / f"{running.key}.lock"), \
            patch.object(resume_drain.time, "sleep", side_effect=KeyboardInterrupt), \
            pytest.raises(KeyboardInterrupt):
        model.run()
    assert model.jobs == [("math", independent.key)]
    assert not math.receipt(Task(5, "on_policy")).exists()
    assert not math.receipt(Task(5, "sr")).exists()


def test_two_idle_nodes_claim_distinct_ready_jobs_while_a_resume_is_owned(model):
    math, mbpp = model.queues
    math.tasks = [Task(5, "prefix"), Task(6, "prefix")]
    old = Task(9, "sr")
    abandoned(mbpp, old)
    model.factory(model.queues).pending()
    context = multiprocessing.get_context("fork")
    ready, results = context.Barrier(2), context.Queue()

    def claim_one():
        ready.wait(10)
        with model.factory(model.queues).claim(math, retry_failed=True, retry_delay=0) as task:
            results.put(task.key if task else None)
            if task:
                time.sleep(.1)
                finish_fake(math, task)
                math.finish(task, 0)

    nodes = [context.Process(target=claim_one) for _ in range(2)]
    with lease(mbpp.directory / "leases" / f"{old.key}.lock"):
        try:
            for node in nodes:
                node.start()
            for node in nodes:
                node.join(10)
                assert node.exitcode == 0
            jobs = [results.get(timeout=2) for _ in nodes]
            assert sorted(jobs) == ["seed-5.prefix", "seed-6.prefix"]
            assert not results.qsize()
        finally:
            for node in nodes:
                if node.is_alive():
                    node.kill()
                    node.join()
    assert json.loads(mbpp.receipt(old).read_text())["attempt"] == 3


@pytest.mark.parametrize("code", (129, 130, 137, 143, -1, -2, -9, -15))
def test_interruption_keeps_checkpoint_and_refunds_attempt_without_retrying(model, code):
    finish_fake(model.queues[0], Task(5, "prefix"))
    old = Task(9, "sr")
    mbpp = model.queues[1]
    abandoned(mbpp, old)
    checkpoint = mbpp.root / "seed-9/sr-latest.pt"
    checkpoint.write_bytes(b"saved optimizer checkpoint")
    model.run(lambda *a, **kw: code)
    assert model.jobs == [("mbpp", old.key)]
    row = json.loads(mbpp.receipt(old).read_text())
    assert row["status"] == "interrupted" and row["retry_attempt"] == 0
    assert row["exit_code"] == (128 - code if code < 0 else code)
    assert checkpoint.read_bytes() == b"saved optimizer checkpoint"
    assert model.updates[-1][0] == "stopped"


def test_input_validation_failure_during_interruption_is_never_reported_as_stopped(model):
    finish_fake(model.queues[0], Task(5, "prefix"))
    old = Task(9, "sr")
    mbpp = model.queues[1]
    abandoned(mbpp, old)

    def corrupt(queue, task, log, **kwargs):
        path = input_path(queue.plan_path, queue.plan, task.seed)
        path.write_bytes(path.read_bytes() + b"\n")
        return 143

    with pytest.raises(ValueError, match="frozen inputs changed"):
        model.run(corrupt)
    row = json.loads(mbpp.receipt(old).read_text())
    assert row["status"] == "failed" and row["retry_attempt"] == 1
    assert "frozen inputs changed" in row["validation_error"]
    assert all(state != "stopped" for state, _ in model.updates)


def test_repeated_kills_do_not_exhaust_retries_and_next_launch_resumes(model):
    finish_fake(model.queues[0], Task(5, "prefix"))
    task = Task(9, "sr")
    queue = model.queues[1]
    abandoned(queue, task)
    checkpoint = queue.root / "seed-9/sr-latest.pt"
    checkpoint.write_bytes(b"last fully saved update")
    for _ in range(5):
        model.run(lambda *a, **kw: -signal.SIGKILL)
        row = json.loads(queue.receipt(task).read_text())
        assert row["status"] == "interrupted" and row["retry_attempt"] == 0
        assert row["resume_first"]
        assert checkpoint.read_bytes() == b"last fully saved update"
    model.run()
    assert queue.complete(task)
    assert model.jobs[:6] == [("mbpp", task.key)] * 6
    assert not any(json.loads(p.read_text())["status"] == "failed"
                   for p in (queue.directory / "attempts").glob("*.json"))


def test_torchrun_rank_sigkill_and_explicit_oom_both_resume_without_attempt_cap(model):
    finish_fake(model.queues[0], Task(5, "prefix"))
    task = Task(9, "sr")
    queue = model.queues[1]
    abandoned(queue, task)

    def killed(queue, task, log, **kwargs):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            handle.write("exitcode : -9\nSignal 9 (SIGKILL) received\n")
        return 1

    model.run(killed)
    assert json.loads(queue.receipt(task).read_text())["status"] == "interrupted"
    assert model.updates[-1][0] == "stopped"

    def oom(queue, task, log, **kwargs):
        if queue.plan["dataset"] != "mbpp":
            finish_fake(queue, task)
            return 0
        with log.open("a") as handle:
            handle.write("CUDA out of memory\nexitcode : -9\n")
        if sum(name == "mbpp" for name, _ in model.jobs) > 5:
            finish_fake(queue, task)
            return 0
        return 137

    model.run(oom)
    row = json.loads(queue.receipt(task).read_text())
    assert row["status"] == "complete"
    assert sum(json.loads(p.read_text())["status"] == "failed"
               for p in (queue.directory / "attempts").glob("*.json")) == 4


def test_repeated_worker_errors_are_retried_until_the_saved_task_finishes(model):
    old = Task(9, "sr")
    abandoned(model.queues[1], old)
    def child(queue, task, log, **kwargs):
        if queue.plan["dataset"] == "mbpp" and sum(name == "mbpp" for name, _ in model.jobs) <= 4:
            return 9
        finish_fake(queue, task)
        return 0
    model.run(child)
    assert model.jobs == [("math", "seed-5.prefix")] + [("mbpp", old.key)] * 5
    assert model.queues[0].complete(Task(5, "prefix"))


def test_different_sibling_snapshot_is_rejected_before_any_launch(model):
    mbpp = model.queues[1]
    plan = json.loads(mbpp.plan_path.read_text())
    plan["model_snapshot_sha256"] = "different"
    # Bind a disjoint protocol just as a separately prepared sibling would do.
    mbpp.directory.joinpath("protocol.json").unlink()
    atomic_json(mbpp.plan_path, plan)
    atomic_json(model.queues[0].directory / "runtime.json", {"test": "runtime"})
    with pytest.raises(ValueError, match="different local model snapshots"):
        model.run()
    assert model.jobs == []
    assert not (mbpp.directory / "protocol.json").exists()
    assert not (mbpp.directory / "runtime.json").exists()


def test_heartbeat_callbacks_keep_their_original_dataset_and_task(model):
    abandoned(model.queues[1], Task(9, "sr"))
    callbacks = []

    def child(queue, task, log, **kwargs):
        callbacks.append(kwargs["heartbeat"])
        finish_fake(queue, task)
        return 0
    model.run(child)
    callbacks[1](321)
    assert model.updates[-1][1]["active"].plan["dataset"] == "mbpp"
    assert model.updates[-1][1]["task"] == Task(9, "sr")


def test_two_dataset_only_nodes_share_sibling_backlog_without_duplicate_jobs(model):
    mbpp = model.queues[1]
    for task in (Task(8, "sr"), Task(9, "switch")):
        abandoned(mbpp, task)
    context = multiprocessing.get_context("fork")
    ready, results = context.Event(), context.Queue()

    def worker(number):
        ready.wait(10)
        def child(queue, task, log, **kwargs):
            results.put(("math" if queue.plan["dataset"] == "math_train" else "mbpp", task.key))
            time.sleep(.15)
            finish_fake(queue, task)
            return 0
        model.run(child, worker_id=f"node-{number}")

    nodes = [context.Process(target=worker, args=(i,)) for i in range(2)]
    try:
        for node in nodes:
            node.start()
        ready.set()
        for node in nodes:
            node.join(15)
            assert node.exitcode == 0
        jobs = [results.get(timeout=2) for _ in range(3)]
        assert sorted(jobs) == sorted((("mbpp", "seed-8.sr"), ("mbpp", "seed-9.switch"),
                                      ("math", "seed-5.prefix")))
        assert results.empty()
    finally:
        for node in nodes:
            if node.is_alive():
                node.kill()
                node.join()


def test_cooldown_does_not_block_independent_ready_work(model, capsys):
    old = Task(9, "sr")
    mbpp = model.queues[1]
    abandoned(mbpp, old, state="failed", attempt=1)
    row = json.loads(mbpp.receipt(old).read_text())
    atomic_json(mbpp.receipt(old), {**row, "finished": time.time()})
    model.options.retry_delay = 120
    with patch.object(resume_drain.time, "sleep", side_effect=KeyboardInterrupt), \
            pytest.raises(KeyboardInterrupt):
        model.run()
    assert model.jobs == [("math", "seed-5.prefix")]
    assert "retry in" in capsys.readouterr().out


def test_cooling_interrupted_task_keeps_its_lease_free_and_blocks_fresh_work(model):
    math = model.queues[0]
    old, ready = Task(9, "sr"), Task(5, "prefix")
    math.tasks = [old, ready]
    abandoned(math, old, state="failed", attempt=1)
    atomic_json(math.receipt(old), {**json.loads(math.receipt(old).read_text()), "finished": time.time()})
    model.options.retry_delay = 120

    with patch.object(resume_drain.time, "sleep", side_effect=KeyboardInterrupt), \
            pytest.raises(KeyboardInterrupt):
        model.run()
    assert model.jobs == []
    assert not math.locked(old)
    assert not math.receipt(ready).exists()


def test_stopped_sibling_backlog_does_not_block_requested_ready_work(model):
    math, mbpp = model.queues
    old = Task(9, "sr")
    abandoned(mbpp, old)
    before = mbpp.receipt(old).read_bytes()
    with patch.object(cluster, "stop_requested", side_effect=lambda queue, **kw: queue.plan_path == mbpp.plan_path), \
            patch.object(resume_drain.time, "sleep", side_effect=AssertionError("ready work was left idle")):
        model.run()
    assert model.jobs == [("math", "seed-5.prefix")]
    assert math.complete(Task(5, "prefix"))
    assert mbpp.receipt(old).read_bytes() == before
    assert model.updates[-1][0] == "complete"


def test_public_model_entry_installs_and_restores_operational_drain(model):
    from srgc_research.dispatch import gemma_run, llama_run
    from srgc_research.dispatch.gemma4 import cli as gemma_cli
    from srgc_research.dispatch.gemma4 import entry as gemma_entry
    from srgc_research.dispatch.llama31 import cli as llama_cli

    worker = model.worker
    original_drain = worker.drain
    original_worker = worker.worker

    def invoke_worker(*unused):
        # CLI imports this function locally before entering its resume context.
        # Install it in the outer entry before that import can capture a stale
        # worker with the premature completion check.
        assert worker.worker is not original_worker
        if worker.__name__.endswith("gemma4.worker"):
            with gemma_cli.resume_first_worker():
                worker.drain([], model.options, {}, (), "test", lambda *_: None)
        elif worker.__name__.endswith("llama31.worker"):
            with llama_cli.resume_first_worker():
                worker.drain([], model.options, {}, (), "test", lambda *_: None)
        else:
            worker.drain([], model.options, {}, (), "test", lambda *_: None)
        return 0

    with patch.object(resume_drain, "drain", return_value=None) as actual:
        if worker.__name__.endswith("gemma4.worker"):
            with patch.object(gemma_entry, "main", side_effect=invoke_worker):
                assert gemma_run.main(["all"]) == 0
        elif worker.__name__.endswith("llama31.worker"):
            with patch.object(llama_cli, "main", side_effect=invoke_worker):
                assert llama_run.main(["all"]) == 0
        else:
            diagnostics = importlib.import_module("srgc_qwen35_diagnostics")
            with patch.object(diagnostics, "main", side_effect=invoke_worker):
                assert qwen_run.main(["all", "run"]) == 0
    actual.assert_called_once()
    assert actual.call_args.args[:2] == (worker, model.factory)
    assert worker.drain is original_drain
    assert worker.worker is original_worker


def test_qwen_module_propagates_results_failure_to_the_shell():
    script = (
        "import runpy,sys; from srgc_research.dispatch import model_results; "
        "model_results.main=lambda args: 7; "
        "sys.argv=['qwen','all','results']; "
        "runpy.run_module('srgc_research.dispatch.qwen_run',run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=qwen_run.REPO,
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 7, result.stderr


def test_qwen_transition_waits_for_delayed_gpu_release_and_restores_hooks(monkeypatch, capsys):
    from srgc_rebuttal.runtime import Busy
    from srgc_research.dispatch import model_launch

    monkeypatch.syspath_prepend(str(qwen_run.REPO / "scripts"))
    diagnostics = importlib.import_module("srgc_qwen35_diagnostics")
    expected = ("0,1,2,3", ("a", "b", "c", "d"))
    def run():
        assert cluster.gpu_identity() == expected
        assert cluster.gpu_identity() == expected
        return 0
    with patch.object(cluster, "gpu_identity", side_effect=[expected, Busy("CUDA release pending"), expected]) as identity, \
            patch.object(diagnostics, "main", side_effect=run), \
            patch.object(model_launch.time, "sleep") as sleep:
        assert qwen_run.main(["all", "run"]) == 0
        assert cluster.gpu_identity is identity
    sleep.assert_called_once_with(1)
    assert "QWEN GPU memory release complete" in capsys.readouterr().out


def test_model_chain_yields_peer_owned_backlog_after_finishing_ready_work(model, monkeypatch):
    from srgc_research.dispatch.model_chain import YIELD
    monkeypatch.setenv('SRGC_MODEL_CHAIN_CHILD', '1')
    old = Task(9, 'sr')
    sibling = model.queues[1]
    abandoned(sibling, old)
    before = sibling.receipt(old).read_bytes()
    assert model.factory(model.queues).pending()
    with lease(sibling.directory / 'leases' / f'{old.key}.lock'), \
            patch.object(resume_drain.time, 'sleep', side_effect=AssertionError('model must hand off, not sleep')), \
            pytest.raises(SystemExit) as stopped:
        model.run()
    assert stopped.value.code == YIELD
    assert model.jobs == [('math', 'seed-5.prefix')]
    assert sibling.receipt(old).read_bytes() == before
    assert model.updates[-1][0] == 'yielded'


def test_model_chain_owned_work_yields_before_gpu_admission(model, monkeypatch, tmp_path):
    from srgc_research.dispatch.model_chain import YIELD
    monkeypatch.setenv('SRGC_MODEL_CHAIN_CHILD', '1')
    queue = model.queues[0]
    with lease(queue.directory / 'leases' / 'seed-5.prefix.lock'), \
            patch.object(cluster, 'gpu_identity', side_effect=AssertionError('must not request GPUs')), \
            pytest.raises(SystemExit) as stopped:
        run_worker(model, monkeypatch, tmp_path,
                   admission=lambda *a, **kw: pytest.fail('owned work requested admission'))
    assert stopped.value.code == YIELD
    assert model.jobs == []
    rows = list((queue.directory / 'workers').glob('*.json'))
    assert json.loads(rows[0].read_text())['status'] == 'yielded'


def test_model_chain_keeps_retrying_error_tasks_without_an_attempt_limit(model,monkeypatch):
    monkeypatch.setenv('SRGC_MODEL_CHAIN_CHILD','1')
    old = Task(9, "sr")
    queue = model.queues[1]
    abandoned(queue, old)

    def child(queue, task, log, **kwargs):
        if queue.plan["dataset"] == "mbpp" and sum(name == "mbpp" for name, _ in model.jobs) <= 4:
            return 9
        finish_fake(queue, task)
        return 0

    # Each failure releases this worker's leases and returns to model_chain.
    # Subsequent recovery turns keep retrying the saved task without a cap.
    for turn in range(5):
        model.run(child)
        assert sum(name == "mbpp" for name, _ in model.jobs) == turn + 1
    assert model.jobs == [("math", "seed-5.prefix")] + [("mbpp", old.key)] * 5
    assert queue.complete(old)
    assert json.loads(queue.receipt(old).read_text())["attempt"] == 8


def test_temporary_gpu_query_failure_yields_and_next_worker_resumes_old_task_first(model, monkeypatch, tmp_path):
    from srgc_research.dispatch.model_chain import BUSY

    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    queue = model.queues[0]
    old, fresh = Task(9, "sr"), Task(5, "prefix")
    queue.tasks = [fresh, old]
    abandoned(queue, old, attempt=99, state="failed")
    before = queue.receipt(old).read_bytes()
    error = subprocess.CalledProcessError(9, ["nvidia-smi", "--query-gpu=uuid"])
    with pytest.raises(SystemExit) as stopped:
        run_worker(model, monkeypatch, tmp_path, probe=error)
    assert stopped.value.code == BUSY and not model.jobs
    assert queue.receipt(old).read_bytes() == before
    run_worker(model, monkeypatch, tmp_path)
    assert model.jobs == [("math", old.key), ("math", fresh.key)]


@pytest.mark.parametrize('blocked_at', ('gpu_probe', 'guard_wait'))
def test_busy_gpu_does_not_record_an_unstarted_retry_as_a_training_error(model, monkeypatch, tmp_path, blocked_at):
    from srgc_research.dispatch.model_chain import BUSY

    monkeypatch.setenv('SRGC_MODEL_CHAIN_CHILD', '1')
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'math')
    queue = model.queues[0]
    old, fresh = Task(5, 'on_policy'), Task(6, 'prefix')
    queue.tasks = [fresh, old]
    abandoned(queue, old, state='failed', attempt=40)
    probe = [('0,1,2,3', ('a', 'b', 'c', 'd')), Busy('previous CUDA context is still releasing')] if blocked_at == 'gpu_probe' else None
    blocked_children = []
    def unavailable(*args, **kwargs):
        blocked_children.append(True)
        assert len(blocked_children) == 1, 'busy guard entered a repeated failure loop'
        return BUSY if blocked_at == 'guard_wait' else 0
    with pytest.raises(SystemExit) as yielded:
        run_worker(model, monkeypatch, tmp_path, probe=probe,
                   child=unavailable)
    assert yielded.value.code == BUSY
    record = json.loads(queue.receipt(old).read_text())
    assert record['status'] == 'interrupted', 'an attempt without a started GPU task was recorded as failed'
    attempts = [json.loads(path.read_text()) for path in (queue.directory / 'attempts').glob('*.json')]
    assert len([row for row in attempts if row['status'] == 'failed']) == 1
    assert not queue.receipt(fresh).exists()
    model.jobs.clear()
    run_worker(model, monkeypatch, tmp_path)
    assert model.jobs == [('math', old.key), ('math', fresh.key)]


def test_single_model_retry_waits_for_gpu_release_then_resumes_saved_arm(model, monkeypatch, tmp_path):
    monkeypatch.delenv('SRGC_MODEL_CHAIN_CHILD', raising=False)
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'math')
    queue = model.queues[0]
    old, fresh = Task(5, 'on_policy'), Task(6, 'prefix')
    queue.tasks = [fresh, old]
    abandoned(queue, old, state='failed', attempt=40)
    free = ('0,1,2,3', ('a', 'b', 'c', 'd'))
    run_worker(model, monkeypatch, tmp_path, probe=[free, Busy('CUDA teardown pending'), free, free])
    assert model.jobs == [('math', old.key), ('math', fresh.key)]
    attempts = [json.loads(path.read_text()) for path in (queue.directory / 'attempts').glob('*.json')]
    assert len([row for row in attempts if row['status'] == 'failed']) == 1
