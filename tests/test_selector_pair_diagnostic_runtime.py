"""Expose current runtime receipts and CPU barriers without mutating a run."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_selector_pair_diagnostic import diagnostic, snapshot


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


@pytest.mark.parametrize('name', ['.pair-barrier.lock', '.pair-runtime.lock',
                                  'branches/cached/.fit.lock', 'gate-fit/.task.lock'])
def test_global_barrier_lease_is_probed_readonly(diagnostic, tmp_path, name):
    lock = tmp_path / name
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(b'preserve the owner and lock inode')
    lock.chmod(0o444)
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(tmp_path)
        text = diagnostic.queue_report(tmp_path, uncapped=True)
        assert f'GLOBAL_LOCK {name} exclusive-holder' in text
        assert snapshot(tmp_path) == before
        with lock.open('rb') as probe, pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert f'GLOBAL_LOCK {name} free-at-probe' in diagnostic.queue_report(tmp_path)


def test_uncapped_raw_receipts_and_distinct_same_host_workers_keep_hashes(diagnostic, tmp_path):
    paths = [record(tmp_path / 'pair.json', {'protocol_id': 'current-protocol'}),
             record(tmp_path / 'branches/cached/mbpp-branch-quarantine-runtime.json',
                    {'frozen_sha256': 'previous', 'runtime_sha256': 'actual-current-revision'}),
             record(tmp_path / 'pair-recollection-runtime.json', {'runtime_sha256': 'pair-revision'}),
             record(tmp_path / 'development/s0-t25/result.json', {'published': True})]
    for worker in ('worker-one', 'worker-two'):
        paths.append(record(tmp_path / f'queue-workers/{worker}.json', {
            'worker': worker, 'host': 'same-host', 'pid': 123, 'state': 'RUN',
            'task': 'development/s0-t25/cached/selection_reduced', 'updated': 1}))
    before = snapshot(tmp_path)
    text = diagnostic.queue_report(tmp_path, uncapped=True)
    for path in paths:
        raw = path.read_bytes()
        assert f'FILE {path.relative_to(tmp_path)} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}' in text
        assert raw.decode() in text
    assert 'QUEUE_RECORDS found=2 exported=2 omitted=0' in text
    assert 'ordering is not a liveness test' in text
    assert snapshot(tmp_path) == before
    ordinary = diagnostic.queue_report(tmp_path)
    assert 'actual-current-revision' not in ordinary
    assert len(ordinary.encode()) <= diagnostic.QUEUE_REPORT_BYTES


def test_runtime_external_symlink_loop_and_oversize_stay_bounded(diagnostic, tmp_path):
    root = tmp_path / 'pair'
    root.mkdir()
    external = record(tmp_path / 'external.json', {'secret': 'EXTERNAL_NOT_EXPORTED'})
    (root / 'pair.json').symlink_to(external)
    (root / 'loop-runtime.json').symlink_to('loop-runtime.json')
    record(root / 'large-runtime.json', {'value': 'x' * 70000})
    os.mkfifo(root / 'pipe-runtime.json')
    (root / 'queue-workers').symlink_to(tmp_path, target_is_directory=True)
    before = snapshot(tmp_path)
    text = diagnostic.queue_report(root, uncapped=True)
    assert 'UNREADABLE pair.json: ValueError' in text and 'EXTERNAL_NOT_EXPORTED' not in text
    assert 'UNREADABLE loop-runtime.json: RuntimeError' in text
    assert 'UNREADABLE large-runtime.json: ValueError: metadata read limit exceeded' in text
    assert 'UNREADABLE pipe-runtime.json: ValueError: metadata is not a regular file' in text
    assert 'UNREADABLE queue-workers: ValueError' in text
    assert len(text.encode()) < 100000
    assert snapshot(tmp_path) == before


def test_queue_record_count_is_bounded_and_omission_explicit(diagnostic, tmp_path):
    for worker in range(260):
        record(tmp_path / f'queue-workers/worker-{worker}.json', {'worker': worker})
    before = snapshot(tmp_path)
    text = diagnostic.queue_report(tmp_path, uncapped=True)
    assert 'QUEUE_RECORDS found=260 exported=256 omitted=4' in text
    assert text.count('FILE queue-workers/') == 256
    assert snapshot(tmp_path) == before


def test_combined_bash_includes_runtime_and_barrier_in_one_txt(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    work, home = tmp_path / 'work', tmp_path / 'home'
    home.mkdir()
    root = work / 'runs/selector-pair-v1'
    receipt = record(root / 'branches/on_policy/mbpp-branch-quarantine-runtime.json',
                     {'runtime_sha256': 'CURRENT_RECEIPT_FOR_COMPARISON'})
    record(root / 'queue-workers/current.json', {'worker': 'unique-current-worker',
           'state': 'WAIT', 'task': 'development/s0-t25', 'host': 'same-host'})
    lock = root / '.pair-barrier.lock'
    lock.write_bytes(b'original barrier')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('MBPP_', 'SWITCH_MBPP_')) and key != 'PAIR_ROOT'}
    before = snapshot(work)
    with lock.open('rb') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = subprocess.run(['bash', 'scripts/check_mbpp_pair.sh'], cwd=repo,
            env={**env, 'HOME': str(home), 'OM_WORK': str(work), 'SWITCH_PYTHON': sys.executable},
            capture_output=True, text=True, timeout=20)
        assert process.returncode == 0, process.stdout + process.stderr
    outputs = list(home.rglob('*.txt'))
    assert len(outputs) == 1
    text = outputs[0].read_text()
    assert 'GLOBAL_LOCK .pair-barrier.lock exclusive-holder' in text
    assert receipt.read_text() in text and 'unique-current-worker' in text
    assert snapshot(work) == before
