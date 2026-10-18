"""Dispatch actual adapter-prepared MATH plans, retaining their frozen labels."""

import importlib
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srgc_rebuttal import cluster, cluster_queue
from srgc_rebuttal.cluster_queue import Task
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json
from srgc_research.dispatch import ready_dispatch, resume_drain
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- Use each real model adapter's preparation.
)

FACTORIES = {'qwen35': ready_dispatch.ResumeFirst, 'llama31': ready_dispatch.LlamaResumeFirst,
             'gemma4': ready_dispatch.GemmaResumeFirst}
WORKERS = {'qwen35': 'scripts.srgc_qwen35_worker',
           'llama31': 'srgc_research.dispatch.llama31.worker',
           'gemma4': 'srgc_research.dispatch.gemma4.worker'}


def failed_cache(queue):
    task = Task(5, 'cache')
    atomic_json(queue.receipt(task), {'task': task.key, 'status': 'failed', 'attempt': 99,
                'attempt_id': 'historical-error', 'finished': time.time() - 120})
    return task


def test_actual_prepared_math_plan_is_selected_and_its_error_is_claimed(cohort, monkeypatch):
    original = {name: path.read_bytes() for name, path in cohort.plans.items()}
    assert json.loads(original['math'])['dataset'] == 'math_train'
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'math')
    with cohort.adapter.runtime_adapter():
        math, mbpp = [cluster_queue.TaskQueue(cohort.plans[name]) for name in ('math', 'mbpp')]
        for queue in (math, mbpp):
            queue.bind()
        task = failed_cache(math)
        manager = FACTORIES[cohort.model]([math, mbpp])
        assert resume_drain.phase_queues([math, mbpp]) == [math]
        pending = manager.pending()
        assert resume_drain.phase_pending(manager, pending) == {(str(math.plan_path), task.key)}
        with manager.claim(math, retry_failed=True, retry_delay=0) as selected:
            assert selected == task
    assert all(path.read_bytes() == original[name] for name, path in cohort.plans.items())


def test_actual_math_plan_prevents_mbpp_from_skipping_unfinished_math(cohort, monkeypatch):
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'mbpp')
    with cohort.adapter.runtime_adapter():
        math, mbpp = [cluster_queue.TaskQueue(cohort.plans[name]) for name in ('math', 'mbpp')]
        failed_cache(mbpp)
        manager = FACTORIES[cohort.model]([math, mbpp])
        assert not manager.dataset_allowed(math)
        assert not manager.dataset_allowed(mbpp)
        with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
            assert task is None


def test_worker_restarts_actual_prepared_math_error_without_requesting_mbpp(cohort, monkeypatch):
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'math')
    monkeypatch.setenv('SRGC_MODEL_CHAIN_CHILD', '1')
    monkeypatch.setenv('SRGC_RESUME_ONLY', '1')
    worker = importlib.import_module(WORKERS[cohort.model])
    args = SimpleNamespace(retry_delay=0, heartbeat_seconds=.01, stall_seconds=30, poll_seconds=.01)
    commands = []
    prefix = {'qwen35': 'qwen35-9b', 'llama31': 'llama31-8b', 'gemma4': 'gemma4-12b-pt'}[cohort.model]
    env_key = {'qwen35': 'SRGC_QWEN_PLANS', 'llama31': 'SRGC_LLAMA_PLANS', 'gemma4': 'SRGC_GEMMA_PLANS'}[cohort.model]

    def child(command, log, environment, **kwargs):
        commands.append(command)
        assert command[command.index('--plan') + 1] == str(cohort.plans['math'])
        assert command[command.index('--stage') + 1] == 'cache'
        assert command[command.index('--cache-seed') + 1] == '5'
        assert command[command.index('--bundle') + 1] == str(input_path(
            cohort.plans['math'], load_plan(cohort.plans['math']), 5))
        return -9

    original = {name: path.read_bytes() for name, path in cohort.plans.items()}
    with cohort.adapter.runtime_adapter(), \
            patch.object(worker, 'runtime_signature', return_value={'cpu_test': True}), \
            patch.object(cluster, 'gpu_identity', return_value=('0,1,2,3', ('a', 'b', 'c', 'd'))), \
            patch.object(cluster, 'run_child', side_effect=child):
        queue = cluster_queue.TaskQueue(cohort.plans['math'])
        task = failed_cache(queue)
        for expected_attempt in (100, 101):
            with pytest.raises(SystemExit) as stopped:
                resume_drain.worker_loop(worker, FACTORIES[cohort.model], prefix + '-{dataset}.json',
                    env_key, cohort.model, list(cohort.plans.values()), args, cohort.root.parent.parent,
                    cohort.root.parent, lambda *a, **kw: {'cpu_test': True})
            assert stopped.value.code == 137
            record = json.loads(queue.receipt(task).read_text())
            assert record['status'] == 'interrupted' and record['attempt'] == expected_attempt
        assert len(commands) == 2
        archived = json.loads((queue.directory / 'attempts/historical-error.json').read_text())
        assert archived['status'] == 'failed' and archived['attempt'] == 99
    assert all(path.read_bytes() == original[name] for name, path in cohort.plans.items())
