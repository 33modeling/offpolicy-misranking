"""Dataset barriers must survive peer ownership, failures and phase handoff."""

import json
from unittest.mock import patch

import pytest

from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import Task
from srgc_rebuttal.runtime import lease
from srgc_rebuttal.tests.test_cluster import finish_fake
from srgc_research.dispatch.model_chain import PAUSED, YIELD
from srgc_research.tests.test_qwen_resume import abandoned
from srgc_research.tests.test_resume_drain import (
    model as model,  # noqa: PLC0414 -- Exercise all three public dispatchers.
)
from srgc_research.tests.test_resume_drain import run_worker


def workloads(model):
    math, mbpp = model.queues
    math.tasks = [Task(5, "prefix"), Task(5, "on_policy"), Task(5, "sr")]
    mbpp.tasks = [Task(9, "switch")]
    abandoned(mbpp, mbpp.tasks[0], attempt=4, state="failed")
    return math, mbpp


def test_math_phase_finishes_all_arms_without_running_existing_mbpp_errors(model, monkeypatch):
    math, mbpp = workloads(model)
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    checkpoint = mbpp.root / "seed-9/switch-latest.pt"
    checkpoint.write_bytes(b"saved MBPP training state")
    model.run(selected=model.queues)
    assert model.jobs == [("math", task.key) for task in math.tasks]
    assert all(math.complete(task) for task in math.tasks)
    assert not mbpp.complete(mbpp.tasks[0])
    row = json.loads(mbpp.receipt(mbpp.tasks[0]).read_text())
    assert row["status"] == "failed" and row["attempt"] == 4
    assert checkpoint.read_bytes() == b"saved MBPP training state"
    assert model.updates[-1][0] == "complete"


def test_dataset_handoff_ignores_old_retry_budget_and_resumes_mbpp_error(model, monkeypatch):
    math, mbpp = workloads(model)
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    model.run(selected=model.queues)
    budget = model.factory(model.queues).root / ".dispatch" / f"{model.factory.retry_revision}.json"
    budget.parent.mkdir(parents=True, exist_ok=True)
    budget.write_bytes(b"obsolete exhausted retry budget")
    before = budget.read_bytes()
    monkeypatch.setenv("SRGC_DATASET_PHASE", "mbpp")
    model.run(selected=model.queues)
    assert model.jobs == [("math", task.key) for task in math.tasks] + [("mbpp", "seed-9.switch")]
    assert budget.read_bytes() == before
    row = json.loads(mbpp.receipt(mbpp.tasks[0]).read_text())
    assert row["status"] == "complete" and row["attempt"] == 5


@pytest.mark.parametrize("endpoint_written", (False, True))
def test_math_execution_lease_blocks_mbpp_even_after_endpoint_write(model, endpoint_written):
    math, mbpp = model.queues
    task = math.tasks[0]
    abandoned(mbpp, Task(9, "sr"), attempt=4, state="failed")
    manager = model.factory(model.queues)
    with lease(manager.execution_lock(math, task)):
        if endpoint_written:
            finish_fake(math, task)
        with manager.claim(mbpp, retry_failed=True, retry_delay=0) as selected:
            assert selected is None
        assert not mbpp.locked(Task(9, "sr"))
    finish_fake(math, task)
    with manager.claim(mbpp, retry_failed=True, retry_delay=0) as selected:
        assert selected == Task(9, "sr")


def test_owned_completed_math_yields_before_admission_and_keeps_mbpp_blocked(model, monkeypatch, tmp_path):
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    math, mbpp = model.queues
    abandoned(mbpp, Task(9, "sr"))
    task = math.tasks[0]
    finish_fake(math, task)
    with lease(math.directory / "leases" / f"{task.key}.lock"), pytest.raises(SystemExit) as stopped:
        run_worker(model, monkeypatch, tmp_path,
                   admission=lambda *a, **kw: pytest.fail("completed but owned MATH requested GPUs"))
    assert stopped.value.code == YIELD
    assert not model.jobs


def test_math_errors_are_retried_past_three_attempts_without_falling_through_to_mbpp(model, monkeypatch):
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    math, mbpp = model.queues
    failed = Task(5, "sr")
    math.tasks = [failed]
    abandoned(math, failed, attempt=4, state="failed")
    abandoned(mbpp, Task(9, "switch"), attempt=4, state="failed")
    def child(queue, task, log, **kwargs):
        if len(model.jobs) <= 5:
            return 1
        finish_fake(queue, task)
        return 0
    model.run(child, selected=model.queues)
    assert model.jobs == [("math", failed.key)] * 6
    assert json.loads(mbpp.receipt(Task(9, "switch")).read_text())["attempt"] == 4


def test_paused_math_is_not_reported_complete_and_does_not_admit_mbpp(model, monkeypatch, tmp_path):
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    with patch.object(cluster, "stop_requested", side_effect=lambda q, **kw: q.plan["dataset"] == "math_train"), \
            pytest.raises(SystemExit) as stopped:
        run_worker(model, monkeypatch, tmp_path,
                   admission=lambda *a, **kw: pytest.fail("paused MATH admitted a child"))
    assert stopped.value.code == PAUSED
    assert not model.jobs
    receipt = next((model.queues[0].directory / "workers").glob("*.json"))
    assert json.loads(receipt.read_text())["status"] == "stopped"


def test_math_retry_delay_yields_without_using_mbpp_as_fallback(model, monkeypatch, tmp_path):
    import time

    from srgc_rebuttal.runtime import atomic_json

    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    math, mbpp = model.queues
    task = math.tasks[0]
    abandoned(math, task, attempt=4, state="failed")
    atomic_json(math.receipt(task), {**json.loads(math.receipt(task).read_text()), "finished": time.time()})
    abandoned(mbpp, Task(9, "sr"), attempt=4, state="failed")
    model.options.retry_delay = 120
    with pytest.raises(SystemExit) as stopped:
        run_worker(model, monkeypatch, tmp_path,
                   admission=lambda *a, **kw: pytest.fail("cooling MATH admitted a child"))
    assert stopped.value.code == YIELD
    assert not model.jobs


def test_invalid_dataset_phase_is_refused_before_launch(model, monkeypatch):
    monkeypatch.setenv("SRGC_DATASET_PHASE", "typo")
    with pytest.raises(ValueError, match="SRGC_DATASET_PHASE"):
        model.run()
    assert not model.jobs


def test_explicit_mbpp_start_can_finish_existing_math_before_its_own_phase(model, monkeypatch, tmp_path):
    math, mbpp = workloads(model)
    full_math = type(math)(math.plan_path)
    expected = [("math", task.key) for task in full_math.tasks if not full_math.complete(task)]
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    run_worker(model, monkeypatch, tmp_path, selected=[mbpp])
    assert model.jobs == expected
    assert all(full_math.complete(task) for task in full_math.tasks)
    assert json.loads(mbpp.receipt(mbpp.tasks[0]).read_text())["attempt"] == 4


def test_old_math_error_resumes_before_an_independent_fresh_seed_at_startup(model, monkeypatch, tmp_path):
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    math = model.queues[0]
    failed, ready = Task(5, "sr"), Task(6, "prefix")
    math.tasks = [failed, ready]
    abandoned(math, failed, attempt=4, state="failed")
    for _ in range(3):
        with model.factory(model.queues).claim(math, retry_failed=True, retry_delay=0) as task:
            assert task == failed
            math.finish(task, 1)
    run_worker(model, monkeypatch, tmp_path)
    assert model.jobs == [("math", failed.key), ("math", ready.key)]
    assert math.complete(ready)


def test_resume_only_finishes_interrupted_math_without_opening_a_fresh_seed(model, monkeypatch):
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    monkeypatch.setenv("SRGC_RESUME_ONLY", "1")
    math = model.queues[0]
    old, fresh = Task(9, "sr"), Task(5, "prefix")
    math.tasks = [fresh, old]
    abandoned(math, old, attempt=99, state="failed")
    model.options.max_attempts = 1
    model.run(selected=model.queues)
    assert model.jobs == [("math", old.key)]
    assert math.complete(old) and not math.complete(fresh)
    assert not math.receipt(fresh).exists()


def test_empty_recovery_pass_returns_before_gpu_admission_without_marking_new_work_complete(model, monkeypatch, tmp_path):
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    monkeypatch.setenv("SRGC_RESUME_ONLY", "1")
    run_worker(model, monkeypatch, tmp_path,
               admission=lambda *a, **kw: pytest.fail("empty recovery pass requested GPUs"))
    assert not model.jobs
    assert not model.queues[0].complete(Task(5, "prefix"))
