"""Finish both initial caches before MATH training, including on idle peers."""

import os
from unittest.mock import patch

import pytest

from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_research.dispatch import gemma_text_run as run
from srgc_research.dispatch.gemma4_text import entry
from srgc_research.dispatch.gemma4_text.resume import GemmaTextResumeFirst
from srgc_research.dispatch.model_chain import YIELD
from srgc_research.tests.test_model_results import make_cohort


def test_both_cache_pass_precedes_math_and_restores_environment(monkeypatch):
    monkeypatch.setenv('SRGC_GEMMA_CACHE_FIRST', '1')
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'math')
    monkeypatch.delenv('SRGC_CACHE_ONLY', raising=False)
    stages = []
    with patch.object(entry, 'main', side_effect=lambda args: stages.append(os.environ.get('SRGC_CACHE_ONLY')) or 0):
        assert run.main(['all']) == 0
    assert stages == ['1', None]
    assert 'SRGC_CACHE_ONLY' not in os.environ


def test_peer_owned_cache_blocks_math_training(monkeypatch):
    monkeypatch.setenv('SRGC_GEMMA_CACHE_FIRST', '1')
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'math')
    monkeypatch.delenv('SRGC_CACHE_ONLY', raising=False)
    with patch.object(entry, 'main', side_effect=SystemExit(YIELD)) as called, pytest.raises(SystemExit) as error:
        run.main(['all'])
    assert error.value.code == YIELD and called.call_count == 1
    assert 'SRGC_CACHE_ONLY' not in os.environ


@pytest.fixture
def queues(tmp_path, monkeypatch):
    cohort = make_cohort('gemma4_text', tmp_path)
    monkeypatch.setenv('GROUP_VOLUME', cohort.env['GROUP_VOLUME'])
    monkeypatch.setenv('OM_WORK', cohort.env['OM_WORK'])
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'math')
    monkeypatch.setenv('SRGC_CACHE_ONLY', '1')
    monkeypatch.setenv('SRGC_RESUME_ONLY', '1')
    with cohort.adapter.runtime_adapter():
        items = [TaskQueue(path) for path in cohort.plans.values()]
        for queue in items:
            queue.bind()
        yield cohort, items


def test_idle_math_worker_claims_mbpp_cache_despite_resume_only(queues):
    _, items = queues
    manager = GemmaTextResumeFirst(items)
    assert not all(items[0].complete(task) for task in items[0].tasks)
    with manager.claim(items[1], retry_failed=True, retry_delay=0) as task:
        assert task == Task(5, 'cache')
        peer = GemmaTextResumeFirst([TaskQueue(queue.plan_path) for queue in items])
        with peer.claim(peer.queues[str(items[1].plan_path.resolve())], retry_failed=True, retry_delay=0) as other:
            assert other == Task(6, 'cache')


def test_cache_pass_never_claims_prefix_or_training_even_if_ready(queues, monkeypatch):
    _, items = queues
    manager = GemmaTextResumeFirst(items)
    items[1].tasks = [Task(5, 'prefix'), Task(5, 'on_policy')]
    monkeypatch.setattr(TaskQueue, 'ready', lambda *args: True)
    with manager.claim(items[1], retry_failed=True, retry_delay=0) as task:
        assert task is None


def test_cache_pass_honors_mbpp_stop_and_preserves_protocol(queues):
    from srgc_rebuttal.runtime import atomic_json

    cohort, items = queues
    original = {path: path.read_bytes() for path in cohort.plans.values()}
    before = cohort.adapter.adapter_digest()
    atomic_json(items[1].directory / 'stop.json', {'immediate': True})
    with GemmaTextResumeFirst(items).claim(items[1], retry_failed=True, retry_delay=0) as task:
        assert task is None
    assert {path: path.read_bytes() for path in original} == original
    assert cohort.adapter.adapter_digest() == before


def test_cooling_math_cache_does_not_leave_mbpp_cache_idle(queues):
    import time

    from srgc_rebuttal.runtime import atomic_json

    _, items = queues
    task = Task(5, 'cache')
    atomic_json(items[0].receipt(task), {'task': task.key, 'status': 'failed',
                'attempt': 1, 'finished': time.time(), 'attempt_id': 'failed-cache'})
    with GemmaTextResumeFirst(items).claim(items[1], retry_failed=True, retry_delay=60) as selected:
        assert selected == Task(5, 'cache')
