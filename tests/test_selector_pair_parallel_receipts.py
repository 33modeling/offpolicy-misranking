"""Schedule amendment admission and byte-preserving predecessor migration."""

import fcntl
import sys

import pytest

import queue_selector_pair_gpu as adapter
from test_selector_pair_curve_guard import frozen_protocol


def saved(path):
    return path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns


def predecessor(root, protocol):
    guard = {**adapter.guard_receipt(root, protocol),
             'guard_sha256': adapter.PRE_PARALLEL_GUARD_SHA256}
    adapter.worker.core.atomic_json(root / adapter.RECEIPT, guard)
    recovery = adapter.recovery_receipt(root, protocol)
    recovery['runtime_code_hashes']['queue_selector_pair_gpu.py'] = adapter.PRE_PARALLEL_GUARD_SHA256
    adapter.worker.core.atomic_json(root / adapter.COST_RECEIPT, recovery)


def test_parallel_amendment_preserves_both_predecessor_receipts(tmp_path):
    protocol = frozen_protocol(tmp_path)
    predecessor(tmp_path, protocol)
    before = {path: saved(path) for path in tmp_path.glob('*.json')}
    adapter.validate_receipts(tmp_path, protocol)
    assert not (tmp_path / adapter.parallel.RECEIPT).exists()
    adapter.bind_receipt(tmp_path, protocol)
    adapter.bind_recovery_receipt(tmp_path, protocol)
    adapter.bind_parallel_receipt(tmp_path, protocol)
    assert all(saved(path) == value for path, value in before.items())
    amendment = tmp_path / adapter.parallel.RECEIPT
    assert adapter.worker.core.read(amendment) == adapter.parallel.receipt_value(tmp_path, protocol)
    published = saved(amendment)
    adapter.bind_parallel_receipt(tmp_path, protocol)
    assert saved(amendment) == published


@pytest.mark.parametrize('name', ['selector_pair_cost_recovery.py', 'recover_selection_switch_cost.py',
                                '_recovery_owners.py', 'mbpp_storage_audit.py'])
def test_predecessor_exception_never_accepts_unreviewed_helper_changes(tmp_path, name):
    protocol = frozen_protocol(tmp_path)
    predecessor(tmp_path, protocol)
    path = tmp_path / adapter.COST_RECEIPT
    changed = adapter.worker.core.read(path)
    changed['runtime_code_hashes'][name] = 'unreviewed'
    adapter.worker.core.atomic_json(path, changed)
    before = saved(path)
    with pytest.raises(ValueError, match='frozen contract changed'):
        adapter.validate_receipts(tmp_path, protocol)
    assert saved(path) == before
    assert not (tmp_path / adapter.parallel.RECEIPT).exists()


def test_amendment_publication_holds_barrier_and_runtime_leases(tmp_path, monkeypatch):
    protocol = frozen_protocol(tmp_path)
    original = adapter.worker.base.bind
    published = []

    def bind(path, value):
        for name in ('.pair-barrier.lock', '.pair-runtime.lock'):
            with (tmp_path / name).open('rb') as handle, pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        published.append(path.name)
        return original(path, value)

    monkeypatch.setattr(adapter.worker.base, 'bind', bind)
    adapter.bind_parallel_receipt(tmp_path, protocol)
    assert published == [adapter.parallel.RECEIPT]


@pytest.mark.parametrize('kind', ['altered', 'symlink'])
def test_bad_schedule_receipt_blocks_recovery_admission_and_dispatch(tmp_path, monkeypatch, kind):
    protocol = frozen_protocol(tmp_path)
    path = tmp_path / adapter.parallel.RECEIPT
    adapter.worker.core.atomic_json(path, {'unreviewed': True})
    if kind == 'symlink':
        target = tmp_path / 'original-receipt.json'
        path.rename(target)
        path.symlink_to(target)
    before = saved(path)
    calls = []
    monkeypatch.setattr(sys, 'argv', [adapter.__file__, 'run', '--root', str(tmp_path)])
    monkeypatch.setattr(adapter.cost_recovery, 'recover', lambda *a: calls.append('recovery'))
    monkeypatch.setattr(adapter.worker, 'admit_node', lambda *a: calls.append('admit'))
    monkeypatch.setattr(adapter.worker, 'main', lambda: adapter.worker.admit_node(tmp_path, protocol))
    with pytest.raises((ValueError, RuntimeError)):
        adapter.run()
    assert calls == [] and saved(path) == before


def test_existing_unfrozen_test_work_cannot_be_retroactively_authorized(tmp_path):
    protocol = frozen_protocol(tmp_path)
    path = (tmp_path / 'branches/cached/states/s3-t25/points/view-25'
            / 'selection_full/cost.jsonl')
    path.parent.mkdir(parents=True)
    path.write_text('{"existing":"cost history"}\n')
    before = saved(path)
    with pytest.raises((ValueError, RuntimeError)):
        adapter.bind_receipt(tmp_path, protocol)
    assert saved(path) == before
    assert not (tmp_path / adapter.RECEIPT).exists()
    assert not (tmp_path / adapter.parallel.RECEIPT).exists()
