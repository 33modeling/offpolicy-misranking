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
from srgc_rebuttal.runtime import atomic_json, lease
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
        plan.update(dataset=dataset, output_root=str(folder / "runs"),
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
            jobs.append((queue.plan["dataset"], task.key))
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


def run_worker(model, monkeypatch, tmp_path, *, child=None, admission=None):
    from srgc_rebuttal import cluster_queue

    queues = {q.plan_path: q for q in model.queues}
    monkeypatch.setattr(cluster_queue, "TaskQueue", lambda path: queues[path.resolve()])

    def complete(command, log, environment, **kwargs):
        queue, task = command
        model.jobs.append((queue.plan["dataset"], task.key))
        if child:
            return child(queue, task, **kwargs)
        finish_fake(queue, task)
        return 0

    with patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", ("a", "b", "c", "d"))), \
            patch.object(cluster, "task_command", side_effect=lambda q, t: (q, t)), \
            patch.object(cluster, "run_child", side_effect=complete), \
            patch.object(cluster, "publish_reports") as publish, \
            patch.object(model.worker, "runtime_signature", return_value={"test": "runtime"}), \
            resume_drain.resume_worker(model.worker, model.factory, pattern=model.pattern,
                                      env_key=model.env_key, label="TEST"):
        model.worker.worker([model.queues[0].plan_path], model.options, tmp_path, tmp_path,
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


def test_dataset_only_worker_actually_finishes_sibling_resume_before_new_work(model):
    math, mbpp = model.queues
    old = Task(9, "sr")
    abandoned(mbpp, old, attempt=99)
    prefix = mbpp.root / "seed-9/prefix.pt"
    before = prefix.read_bytes()
    model.run()
    assert model.jobs == [("mbpp", old.key), ("math", "seed-5.prefix")]
    assert mbpp.complete(old) and math.complete(Task(5, "prefix"))
    assert prefix.read_bytes() == before
    assert not mbpp.receipt(Task(5, "prefix")).exists()
    assert model.updates[-1][0] == "complete"
    row = json.loads((mbpp.directory / "workers/test.json").read_text())
    assert row["status"] == "complete"


def test_sibling_real_failure_is_retried_before_any_fresh_requested_job(model):
    old = Task(9, "sr")
    abandoned(model.queues[1], old)

    def child(queue, task, log, **kwargs):
        if len(model.jobs) == 1:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("RuntimeError: real failed attempt\n")
            return 9
        finish_fake(queue, task)
        return 0
    model.run(child)
    assert model.jobs == [("mbpp", old.key), ("mbpp", old.key), ("math", "seed-5.prefix")]
    history = [json.loads(path.read_text()) for path in (model.queues[1].directory / "attempts").glob("*.json")]
    assert any(row["status"] == "failed" and row["exit_code"] == 9 for row in history)


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


@pytest.mark.parametrize("code", (130, 137, 143, -2, -9, -15))
def test_interruption_keeps_checkpoint_and_refunds_attempt_without_retrying(model, code):
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


def test_torchrun_rank_sigkill_resumes_but_explicit_oom_uses_failure_budget(model):
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
        return 137

    with pytest.raises(RuntimeError, match="failure limit|bounded retries"):
        model.run(oom)
    row = json.loads(queue.receipt(task).read_text())
    assert row["status"] == "failed" and row["retry_attempt"] == 3


def test_retry_budget_is_bounded_without_blocking_independent_ready_work(model):
    old = Task(9, "sr")
    abandoned(model.queues[1], old)
    def child(queue, task, log, **kwargs):
        if queue.plan["dataset"] == "mbpp":
            return 9
        finish_fake(queue, task)
        return 0
    with pytest.raises(RuntimeError, match="failure limit|bounded retries"):
        model.run(child)
    assert model.jobs == [("mbpp", old.key)] * 3 + [("math", "seed-5.prefix")]
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
    callbacks[0](321)
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
            results.put((queue.plan["dataset"], task.key))
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


def test_cooling_task_lease_is_free_while_an_independent_job_runs(model):
    math = model.queues[0]
    old, ready = Task(9, "sr"), Task(5, "prefix")
    math.tasks = [old, ready]
    abandoned(math, old, state="failed", attempt=1)
    atomic_json(math.receipt(old), {**json.loads(math.receipt(old).read_text()), "finished": time.time()})
    model.options.retry_delay = 120

    def child(queue, task, log, **kwargs):
        assert task == ready
        assert not math.locked(old), "skipped task lease must not be held for the next job's lifetime"
        finish_fake(queue, task)
        return 0

    with patch.object(resume_drain.time, "sleep", side_effect=KeyboardInterrupt), \
            pytest.raises(KeyboardInterrupt):
        model.run(child)
    assert model.jobs == [("math", ready.key)]


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
