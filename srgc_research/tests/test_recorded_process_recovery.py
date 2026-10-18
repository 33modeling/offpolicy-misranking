"""A failed recorded rank must release the real execution lease before resume."""

import importlib
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from scripts import srgc_process_guard as guard
from srgc_rebuttal import cluster_queue
from srgc_rebuttal.build_cache import write_cost_summary
from srgc_rebuttal.cluster_queue import Task
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json
from srgc_research.dispatch import resume_drain
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- Real saved plans for all model families.
)
from srgc_research.tests.test_model_results import finish
from srgc_research.tests.test_prepared_dispatch import FACTORIES, WORKERS


def test_recording_guard_uses_one_module_and_preserves_live_rank_owners(cohort):
    worker = importlib.import_module(WORKERS[cohort.model])
    marker = ('srgc_qwen35_reward_curve_rank.py' if cohort.model == 'qwen35' else
              f'srgc_research.dispatch.reward_curve_rank --model {cohort.model}')
    legacy = object()
    owner = 'run_srgc_qwen35.py'
    table = {900001: (1, 1, f'python {owner} all'),
             900002: (900001, 1, f'python {marker} --task on_policy'),
             900003: (1, 1, f'python {marker} --task on_policy'),
             900004: (1, 2, f'python {marker} --task on_policy')}
    with patch.dict(sys.modules, {'srgc_process_guard': legacy}), \
            patch.object(guard.os, 'getuid', return_value=1), \
            patch.object(guard, 'TARGET_MARKERS', ()), \
            patch.object(guard, 'process_guard', side_effect=lambda *a: nullcontext()):
        with resume_drain.resume_worker(worker, FACTORIES[cohort.model], pattern='unused',
                    env_key='unused', label='test', result_model=cohort.model):
            assert importlib.import_module('srgc_process_guard') is guard
            with guard.process_guard(cohort.plans['math']):
                assert guard.orphan_pids(table=table) == [900003]
        assert sys.modules['srgc_process_guard'] is legacy
        assert guard.TARGET_MARKERS == ()


@pytest.mark.parametrize('arm', ('on_policy', 'switch', 'sr', 'random', 'prefix'))
def test_failed_rank_releases_execution_lock_and_same_arm_resumes(cohort, arm, monkeypatch):
    folder = finish(cohort, 'math', arm='random' if arm != 'random' else 'sr')
    plan = load_plan(cohort.plans['math'])
    bundle = input_path(cohort.plans['math'], plan, 5)
    write_cost_summary(bundle, json.loads(bundle.read_text()))
    if arm == 'prefix':
        (folder / 'prefix-ready.json').unlink()
    checkpoint = folder / f'{arm}-latest.pt'
    checkpoint.write_bytes(b'existing optimizer checkpoint must remain unchanged')
    worker = importlib.import_module(WORKERS[cohort.model])
    task = Task(5, arm)
    marker = ('srgc_qwen35_reward_curve_rank.py' if cohort.model == 'qwen35'
              else 'srgc_research.dispatch.reward_curve_rank')
    native = {'qwen35': 'srgc_qwen35_rank.py', 'llama31': 'srgc_research.dispatch.llama31.rank',
              'gemma4': 'srgc_research.dispatch.gemma4.rank'}[cohort.model]
    original_orphans = guard.orphan_pids
    monkeypatch.setenv('SRGC_DATASET_PHASE', 'math')
    with cohort.adapter.runtime_adapter():
        queue = cluster_queue.TaskQueue(cohort.plans['math'])
        queue.bind()
        atomic_json(queue.receipt(task), {'task': task.key, 'status': 'failed', 'attempt': 40,
                    'attempt_id': 'old-error', 'finished': time.time() - 120})
        manager = FACTORIES[cohort.model]([queue])
        lock = manager.execution_lock(queue, task)
        code = ('import fcntl,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); '
                'p.parent.mkdir(parents=True,exist_ok=True); f=p.open("a+"); '
                'fcntl.flock(f,fcntl.LOCK_EX); print("READY",flush=True); time.sleep(60)')
        child = subprocess.Popen([sys.executable, '-u', '-c', code, str(lock), marker,
            '--model', cohort.model, '--plan', str(queue.plan_path), '--seed', '5', '--task', arm],
            start_new_session=True, stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == 'READY'
            assert manager.owned(queue, task)
            # Match the frozen CLI's native-only target list, and isolate this
            # real-process test from any other same-user test subprocesses.
            with patch.object(guard, 'TARGET_MARKERS', (native,)), \
                    patch.object(guard, 'gpu_memory_summary', return_value=None), \
                    patch.object(guard, 'orphan_pids', side_effect=lambda *a, **kw:
                        [pid for pid in original_orphans(*a, **kw) if pid == child.pid]), \
                    resume_drain.resume_worker(worker, FACTORIES[cohort.model], pattern='unused',
                        env_key='unused', label='test', result_model=cohort.model), \
                    guard.process_guard(queue.plan_path):
                assert not manager.owned(queue, task), 'failed rank still owns the execution lock'
                with manager.claim(queue, retry_failed=True, retry_delay=0) as selected:
                    assert selected == task
                    assert json.loads(queue.receipt(task).read_text())['attempt'] == 41
            assert checkpoint.read_bytes() == b'existing optimizer checkpoint must remain unchanged'
            assert child.wait(timeout=5) < 0
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdout.close()
