"""Real queue evidence: sibling recovery, ownership, retries and stop handling."""

import importlib
import json
import multiprocessing
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.runtime import atomic_json, lease
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs
from srgc_research.dispatch import qwen_run, resume_drain
from srgc_research.dispatch.model_resume import GemmaResumeFirst, LlamaResumeFirst
from srgc_research.dispatch.qwen_resume import ResumeFirst
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
                           factory=factory, worker=worker, options=options)


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


def test_active_sibling_owner_is_preserved_and_idle_reason_is_visible(model, capsys):
    old = Task(9, "sr")
    mbpp = model.queues[1]
    abandoned(mbpp, old)
    assert model.factory(model.queues).pending()
    with lease(mbpp.directory / "leases" / f"{old.key}.lock"), \
            patch.object(resume_drain.time, "sleep", side_effect=KeyboardInterrupt), \
            pytest.raises(KeyboardInterrupt):
        model.run()
    assert model.jobs == []
    assert "owned=1" in model.updates[-1][1]["idle_reason"]
    assert "mbpp:seed-9.sr" in capsys.readouterr().out
    assert not model.queues[0].receipt(Task(5, "prefix")).exists()


@pytest.mark.parametrize("code", (130, 143))
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
    assert checkpoint.read_bytes() == b"saved optimizer checkpoint"
    assert model.updates[-1][0] == "stopped"


def test_retry_budget_remains_bounded_when_sibling_fails_repeatedly(model):
    old = Task(9, "sr")
    abandoned(model.queues[1], old)
    with pytest.raises(RuntimeError, match="failure limit|bounded retries"):
        model.run(lambda *a, **kw: 9)
    assert model.jobs == [("mbpp", old.key)] * 3
    assert not model.queues[0].receipt(Task(5, "prefix")).exists()


def test_different_sibling_snapshot_is_rejected_before_any_launch(model):
    mbpp = model.queues[1]
    plan = json.loads(mbpp.plan_path.read_text())
    plan["model_snapshot_sha256"] = "different"
    # Bind a disjoint protocol just as a separately prepared sibling would do.
    mbpp.directory.joinpath("protocol.json").unlink()
    atomic_json(mbpp.plan_path, plan)
    with pytest.raises(ValueError, match="different local model snapshots"):
        model.run()
    assert model.jobs == []


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


def test_cooldown_is_visible_without_starting_new_work(model, capsys):
    old = Task(9, "sr")
    mbpp = model.queues[1]
    abandoned(mbpp, old, state="failed", attempt=1)
    row = json.loads(mbpp.receipt(old).read_text())
    atomic_json(mbpp.receipt(old), {**row, "finished": time.time()})
    model.options.retry_delay = 120
    with patch.object(resume_drain.time, "sleep", side_effect=KeyboardInterrupt), \
            pytest.raises(KeyboardInterrupt):
        model.run()
    assert model.jobs == []
    assert "retry in" in capsys.readouterr().out


def test_public_model_entry_installs_and_restores_operational_drain(model):
    from srgc_research.dispatch import gemma_run, llama_run
    from srgc_research.dispatch.gemma4 import cli as gemma_cli
    from srgc_research.dispatch.gemma4 import entry as gemma_entry
    from srgc_research.dispatch.llama31 import cli as llama_cli

    worker = model.worker
    original_drain = worker.drain

    def invoke_worker(*unused):
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
