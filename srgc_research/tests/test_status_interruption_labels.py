"""Interrupted nodes must not turn lifetime starts into an escalating ERROR count."""

import time

import pytest

from srgc_rebuttal.runtime import atomic_json
from srgc_research.dispatch import qwen_status as status
from srgc_research.tests.test_qwen_status import prepared as prepared, files  # noqa: F401


@pytest.mark.parametrize('attempt', [22, 30])
@pytest.mark.parametrize('model', ['Qwen3.5-9B', 'Llama-3.1-8B-Instruct', 'Gemma-4-12B-PT'])
def test_lifetime_launches_are_never_reported_as_error_counts(prepared, attempt, model):
    root = prepared.root / 'runs/math/.queue'
    atomic_json(root / 'tasks/seed-7.cache.json', {'task': 'seed-7.cache', 'status': 'failed',
                                                'attempt': attempt, 'exit_code': 1})
    log = root / 'logs/seed-7.cache.log'
    log.parent.mkdir(parents=True)
    log.write_text('torch.OutOfMemoryError: CUDA out of memory\n')
    text = status.render(prepared.read('math'), prepared.root, model=model)
    assert 'ERROR x' not in text
    assert 'CUDA out of memory' in text
    assert 'ERROR 1' in text


@pytest.mark.parametrize('code', [-1, 129, -2, 130, -15, 143])
def test_legacy_signal_receipts_are_read_only_resume_evidence(prepared, code):
    root = prepared.root / 'runs/math/.queue'
    atomic_json(root / 'tasks/seed-7.cache.json', {'task': 'seed-7.cache', 'status': 'failed',
                                                'attempt': 30, 'exit_code': code})
    # This is historical text, not evidence that the signal was a new OOM.
    log = root / 'logs/seed-7.cache.log'
    log.parent.mkdir(parents=True)
    log.write_text('old attempt: torch.OutOfMemoryError: CUDA out of memory\n')
    before = files(prepared.root)
    report = prepared.read('math')
    row = next(row for row in report[0]['tasks'] if row['task'] == 'seed-7.cache')
    assert row['status'] == 'interrupted'
    assert row['receipt_status'] == 'failed' and row['attempt'] == 30
    text = status.render(report, prepared.root)
    assert 'RESUME 1 | ERROR 0' in text
    assert 'RESUME PENDING' in text and '\nATTENTION\n' not in text
    assert 'OutOfMemoryError' not in text and 'ERROR x' not in text
    assert files(prepared.root) == before


def test_input_validation_failure_is_not_disguised_as_a_node_interruption(prepared):
    root = prepared.root / 'runs/math/.queue'
    atomic_json(root / 'tasks/seed-7.cache.json', {'task': 'seed-7.cache', 'status': 'failed',
          'attempt': 30, 'exit_code': 143, 'validation_error': 'frozen inputs changed'})
    report = prepared.read('math')
    row = next(row for row in report[0]['tasks'] if row['task'] == 'seed-7.cache')
    assert row['status'] == 'failed'
    text = status.render(report, prepared.root)
    assert 'ERROR 1' in text and 'frozen inputs changed' in text


def test_lost_worker_lease_is_pending_recovery_not_an_error(prepared):
    root = prepared.root / 'runs/math/.queue'
    atomic_json(root / 'tasks/seed-7.cache.json', {'task': 'seed-7.cache', 'status': 'running',
                                                'attempt': 22, 'host': 'disconnected-node'})
    text = status.render(prepared.read('math'), prepared.root)
    assert 'RESUME 1 | ERROR 0' in text
    assert 'RESUME PENDING' in text and '\nATTENTION\n' not in text
    assert 'disconnected-node' in text


@pytest.mark.parametrize('code', [-9, 137])
def test_ambiguous_kill_with_oom_is_not_hidden_as_preemption(prepared, code):
    root = prepared.root / 'runs/math/.queue'
    atomic_json(root / 'tasks/seed-7.cache.json', {'task': 'seed-7.cache', 'status': 'failed',
                                                'attempt': 30, 'exit_code': code})
    log = root / 'logs/seed-7.cache.log'
    log.parent.mkdir(parents=True)
    log.write_text('torch.OutOfMemoryError: CUDA out of memory\n')
    report = prepared.read('math')
    row = next(row for row in report[0]['tasks'] if row['task'] == 'seed-7.cache')
    assert row['status'] == 'failed'
    text = status.render(report, prepared.root)
    assert 'ERROR 1' in text and 'CUDA out of memory' in text


@pytest.mark.parametrize('error', ['SystemExit: 143', 'SystemExit: 129', 'KeyboardInterrupt: '])
def test_a_normally_stopped_worker_does_not_leave_an_error_alarm(prepared, error):
    root = prepared.root / 'runs/math/.queue'
    atomic_json(root / 'workers/stopped.json', {'worker_id': 'stopped', 'host': 'departed-node',
                'heartbeat': time.time(), 'status': 'stopped', 'task': None, 'error': error})
    text = status.render(prepared.read('math'), prepared.root)
    assert 'departed-node' in text and 'STOPPED' in text
    assert '\nATTENTION\n' not in text
