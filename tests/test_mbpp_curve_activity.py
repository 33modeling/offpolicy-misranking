"""Curve ownership remains visible across server clock skew without fake heartbeats."""

import fcntl
import pytest

from test_selection_switch_status import core, point, published, rule, running, status
import mbpp_status


def curve(root, offset, location='selection_reduced/curve'):
    core.atomic_json(root / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
    directory = point(root) / location
    running(directory, 'remote-curve-node', now=10000 + offset, phase='curve')
    lock = directory / '.cost.lock'
    lock.touch()
    return directory, lock


@pytest.mark.parametrize('offset', [-3600, 3600])
@pytest.mark.parametrize('location', ['selection_reduced/curve', 'curve-parent', 'selection_reduced'])
def test_live_curve_with_clock_skew_is_not_lost_or_claimed_as_fresh(tmp_path, offset, location):
    directory, lock = curve(tmp_path, offset, location)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000, local_gpus=False)
        task = next(t for t in data['tasks'] if t['directory'] == str(directory.relative_to(tmp_path)))
        assert task['status'] == 'RUNNING' and task['owner_active']
        assert task['heartbeat_fresh'] is False
        assert data['active_nodes'] == 1
        assert next(n for n in data['nodes'] if n['host'] == 'remote-curve-node')['state'] == 'RUN'
        assert mbpp_status.active(task)
        assert '잠금 유지' in mbpp_status.remark(task)
        assert '실행 신호 끊김' not in mbpp_status.remark(task)
        nodes = mbpp_status.node_assignments({'suites': [data]})
        assert next(n for n in nodes if n['host'] == 'remote-curve-node')['assignments']
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0


def test_finished_receipt_wins_even_while_meter_lock_remains_held(tmp_path):
    directory, lock = curve(tmp_path, -3600)
    progress = core.read(directory / 'progress.json')
    progress.update(ledger='reporting', gpus=4, gpu_type='test')
    core.atomic_json(directory / 'progress.json', progress)
    core.atomic_json(directory / 'cost-events' / f'{progress["event_id"]}.json', {
        **progress, 'state': 'finished', 'exit_code': 0, 'seconds': 100,
        'allocated_gpu_seconds': 400, 'time': 7000})
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0


@pytest.mark.parametrize('state', ['failed', 'finished'])
def test_terminal_phase_not_resurrected_by_a_lock(tmp_path, state):
    directory, lock = curve(tmp_path, -3600)
    progress = core.read(directory / 'progress.json')
    core.atomic_json(directory / 'progress.json', {**progress, 'state': state})
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0


def test_missing_lock_is_not_created_and_unrelated_experiment_unchanged(tmp_path):
    directory, lock = curve(tmp_path, -3600)
    lock.unlink()
    assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0
    assert not lock.exists()
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'math500'})
    lock.touch()
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status.snapshot(tmp_path, now=10000, local_gpus=False)['active_nodes'] == 0


def test_published_result_keeps_completion_state_and_live_owner(tmp_path):
    directory, lock = curve(tmp_path, -3600, 'selection_reduced')
    published(directory)
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000, local_gpus=False)
        task = next(t for t in data['tasks'] if t['directory'] == str(directory.relative_to(tmp_path)))
        assert task['status'] == 'EVAL' and task['training_published'] and task['owner_active']
        assert not task['heartbeat_fresh']
        assert next(n for n in data['nodes'] if n['host'] == 'remote-curve-node')['state'] == 'RUN'
