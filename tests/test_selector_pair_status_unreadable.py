"""Damaged progress metadata cannot hide another Pair worker."""

import os

import pytest

from test_selector_pair_status import branch, core, prepared, status


def evidence(root):
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            result[str(path)] = ('link', os.readlink(path), path.lstat().st_mtime_ns)
        elif path.is_file():
            result[str(path)] = ('file', path.read_bytes(), path.stat().st_mtime_ns)
    return result


@pytest.mark.parametrize('damage', ['symlink_loop', 'invalid_utf8', 'directory', 'scan_error'])
def test_unreadable_progress_preserves_other_live_worker(tmp_path, monkeypatch, damage):
    prepared(tmp_path)
    broken = branch(tmp_path) / 'progress.json'
    broken.parent.mkdir(parents=True)
    if damage == 'symlink_loop':
        broken.symlink_to(broken.name)
    elif damage == 'invalid_utf8':
        broken.write_bytes(b'\xff')
    elif damage == 'directory':
        broken.mkdir()
    else:
        def fail(_):
            raise OSError('unreadable legacy scan path')
        monkeypatch.setattr(status.gpu, 'pair_progress', fail)
    good = tmp_path / 'branches/cached/states/s1-t50/points/view-50/selection_reduced'
    core.atomic_json(good / 'progress.json', {
        'host': 'shared-host', 'worker_id': 'independent-worker', 'event_id': 'live-event',
        'state': 'running', 'phase': 'train', 'updated': 995, 'seconds': 5, 'timeout': 100})
    before = evidence(tmp_path)
    data = status.snapshot(tmp_path, now=1000)
    task = next(row for row in data['tasks'] if row['seed'] == 1 and row['step'] == 50
                and row['name'] == 'cached')
    assert task['status'] == 'RUN'
    assert task['worker_id'] == 'independent-worker'
    assert any(node['current'] and node['worker_id'] == 'independent-worker' for node in data['nodes'])
    assert len(data['tasks']) == 42
    assert status.display.counts(status.dashboard_data(data)['suites'][0])['states']['RUN'] == 1
    assert 'shared-host' in status.render(data, width=160)
    assert evidence(tmp_path) == before
