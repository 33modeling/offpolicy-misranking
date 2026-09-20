"""Imported heartbeat bytes are history, unless this repair has a live meter."""

import fcntl
import hashlib

import pytest

from test_mbpp_real_result_status import real_schema_root
from test_selection_switch_status import core, point, running, status
import mbpp_status
from mbpp_repair import RERUN_BRANCHES, DEPENDENT_BRANCHES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def copied_running(tmp_path):
    real_schema_root(tmp_path)
    for relative in (*RERUN_BRANCHES, *DEPENDENT_BRANCHES):
        for name in ('result.json', 'result.sha256.json', 'curve.json'):
            (tmp_path / relative / name).unlink()
    directory = point(tmp_path) / 'random_reduced'
    running(directory, 'original-node', now=10000, phase='curve', pid=321)
    progress_path = directory / 'progress.json'
    progress = core.read(progress_path)
    core.atomic_json(progress_path, {**progress, 'ledger': 'reporting', 'gpus': 4, 'gpu_type': 'test'})
    core.atomic_json(tmp_path / 'repair.json', {
        'schema': 'mbpp-repair/v1',
        'source_switch_sha256': digest(tmp_path / 'switch.json'),
        'snapshot_files': {str(progress_path.relative_to(tmp_path)): digest(progress_path)}})
    return tmp_path, directory


def snapshot(root):
    return status.snapshot(root, now=10000, local_gpus=False)


def test_imported_recent_heartbeat_does_not_reduce_37_saved_completions(copied_running):
    root, _directory = copied_running
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in root.rglob('*') if path.is_file()}
    suite = snapshot(root)
    assert mbpp_status.counts(suite)['done'] == 37
    assert mbpp_status.counts(suite)['states']['RUN'] == 0
    assert mbpp_status.current_work(suite) == []
    assert suite['active_nodes'] == 0
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('field,value', [('host', 'repair-node'), ('pid', 456),
                                        ('updated', 10000), ('event_id', 'new-event')])
def test_new_repair_progress_remains_visible(copied_running, field, value):
    root, directory = copied_running
    progress = core.read(directory / 'progress.json')
    core.atomic_json(directory / 'progress.json', {**progress, field: value})
    suite = snapshot(root)
    assert mbpp_status.counts(suite)['states']['RUN'] == 1
    assert len(mbpp_status.current_work(suite)) == 1


def test_held_repair_meter_is_visible_even_before_heartbeat_changes(copied_running):
    root, directory = copied_running
    lock = directory / '.cost.lock'
    lock.touch()
    with lock.open('r+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite = snapshot(root)
        assert mbpp_status.counts(suite)['states']['RUN'] == 1
    assert mbpp_status.counts(snapshot(root))['states']['RUN'] == 0


@pytest.mark.parametrize('held', [False, True])
def test_exact_finished_receipt_still_wins_over_imported_running_record(copied_running, held):
    root, directory = copied_running
    progress = core.read(directory / 'progress.json')
    core.atomic_json(directory / 'cost-events' / f'{progress["event_id"]}.json', {
        **progress, 'state': 'finished', 'exit_code': 0, 'seconds': 100,
        'allocated_gpu_seconds': 400, 'time': 9999})
    lock = directory / '.cost.lock'
    lock.touch()
    with lock.open('r+b') as handle:
        if held:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite = snapshot(root)
        assert mbpp_status.counts(suite)['done'] == 37
        assert not mbpp_status.current_work(suite)


@pytest.mark.parametrize('problem', ['missing', 'wrong-schema', 'wrong-manifest'])
def test_unbound_repair_marker_cannot_hide_current_work(copied_running, problem):
    root, _directory = copied_running
    path = root / 'repair.json'
    if problem == 'missing':
        path.unlink()
    else:
        repair = core.read(path)
        repair['schema' if problem == 'wrong-schema' else 'source_switch_sha256'] = 'wrong'
        core.atomic_json(path, repair)
    assert mbpp_status.counts(snapshot(root))['states']['RUN'] == 1
