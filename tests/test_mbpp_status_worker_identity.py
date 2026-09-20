"""Keep writer identities through the read-only MBPP task projection."""

import fcntl

import pytest

from test_selection_switch_status import completed_prefix, core, point, rule, running, status
from _status_execution import current_tasks, execution_tasks
import mbpp_status


def live(root, relative, worker, event):
    directory = root / relative
    running(directory, 'same-host', now=10000, phase='curve')
    path = directory / 'progress.json'
    core.atomic_json(path, {**core.read(path), 'worker_id': worker, 'event_id': event})
    return directory


@pytest.mark.parametrize('worker', [None, 'worker-a'])
def test_snapshot_preserves_exact_meter_and_optional_worker_identity(tmp_path, worker):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    branch = str((point(tmp_path) / 'selection_reduced').relative_to(tmp_path))
    live(tmp_path, branch, worker, 'shared-event')
    live(tmp_path, branch + '/curve/step-25', worker, 'shared-event')
    suite = status.snapshot(tmp_path, now=10000, local_gpus=False)
    active = [task for task in suite['tasks'] if task.get('heartbeat_fresh')]
    assert len(active) == 2
    assert all(task.get('event_id') == 'shared-event' for task in active)
    assert all(task.get('worker_id') == worker for task in active)
    assert len(current_tasks(execution_tasks(active))) == 1


def test_same_host_different_workers_are_not_merged_as_parent_and_child(tmp_path):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    branch = str((point(tmp_path) / 'selection_reduced').relative_to(tmp_path))
    live(tmp_path, branch, 'worker-a', 'event-a')
    live(tmp_path, branch + '/curve/step-25', 'worker-b', 'event-b')
    suite = status.snapshot(tmp_path, now=10000, local_gpus=False)
    assert len(mbpp_status.current_work(suite)) == 2
    nodes = [node for node in mbpp_status.node_assignments({'suites': [suite]}) if node['assignments']]
    assert {node['worker_id'] for node in nodes} == {'worker-a', 'worker-b'}


def test_skewed_heartbeat_cannot_claim_changed_worker_identity(tmp_path, monkeypatch):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    directory = live(tmp_path, 'states/s0-t25/points/view-25/selection_reduced', 'worker-a', 'event-a')
    path = directory / 'progress.json'
    core.atomic_json(path, {**core.read(path), 'updated': 6000})
    lock = directory / '.cost.lock'
    lock.touch()
    original = core.read
    reads = []

    def changed(candidate):
        value = original(candidate)
        if candidate == path:
            reads.append(1)
            if len(reads) > 1:
                value['worker_id'] = 'worker-b'
        return value

    monkeypatch.setattr(core, 'read', changed)
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite = status.snapshot(tmp_path, now=10000, local_gpus=False)
    assert len(reads) >= 2
    assert not any(task.get('owner_active') for task in suite['tasks'])


@pytest.mark.parametrize('completed,claimed', [((), 25), ((25,), 50), ((25, 50), 100), ((25, 50, 100), None)])
def test_prefix_lease_before_heartbeat_exposes_only_first_unfinished_segment(tmp_path, completed, claimed):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    for step in completed:
        completed_prefix(tmp_path, step=step)
    prefix = tmp_path / 'prefixes/seed-0'
    prefix.mkdir(parents=True, exist_ok=True)
    lock = prefix / '.prefix.lock'
    lock.touch()
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite = status.snapshot(tmp_path, now=10000, local_gpus=False)
        tasks = [task for task in suite['tasks'] if task.get('task_lease_held')]
        assert [task['step'] for task in tasks] == ([] if claimed is None else [claimed])
        assert all(task['kind'] == 'prefix' and not task.get('host') for task in tasks)
        assert len(mbpp_status.current_work(suite)) == len(tasks)
    assert not any(task.get('task_lease_held') for task in status.snapshot(
        tmp_path, now=10000, local_gpus=False)['tasks'])
