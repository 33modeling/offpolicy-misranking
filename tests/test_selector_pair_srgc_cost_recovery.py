"""Interrupted SR-GC measurements retain charges, projections and live workers."""
import pytest

import selector_pair_srgc as srgc
import selector_pair_srgc_cost_recovery as recovery
from test_selector_pair_cost_recovery import case, snapshot, base, core, worker


@pytest.fixture
def measurement(case):
    case.protocol['gpu_type'] = 'H100'
    core.atomic_json(case.root / 'pair.json', case.protocol)
    srgc.activate(case.root, case.protocol)
    directory = case.root / 'sr-gc/s4-t100'
    start = {'event_id': 'interrupted', 'state': 'started', 'phase': 'sr-gc-candidate-b',
             'ledger': 'deployment', 'gpus': 4, 'gpu_type': 'H100',
             'host': 'remote-stopped-worker', 'time': 100.}
    base.journal(directory / 'cost.jsonl', start)
    core.atomic_json(directory / 'progress.json', {**start, 'state': 'running', 'seconds': 12., 'updated': 112.})
    core.atomic_json(directory / 'candidate-b-0.done.json', {'preserve': 'completed shard'})
    return case, directory, start


def test_unclosed_srgc_cost_recovers_once_without_erasing_work(measurement):
    case, directory, _ = measurement
    before = {p: p.read_bytes() for p in case.root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='unknown cost cannot be treated as zero'):
        base.spent(directory)
    row, = recovery.recover(case.root, case.protocol, now=5000)
    assert row['status'] == 'recovered' and row['seconds'] == 72.
    assert row['allocated_gpu_seconds'] == 288.
    assert row['evidence']['duration_is_estimate'] and not row['evidence']['directly_measured']
    assert base.spent(directory) == 288.
    for path, original in before.items():
        if path == directory / 'cost.jsonl':
            assert path.read_bytes().startswith(original)
        else:
            assert path.read_bytes() == original
    closed = snapshot(directory / 'cost.jsonl')
    assert recovery.recover(case.root, case.protocol, now=6000) == []
    assert snapshot(directory / 'cost.jsonl') == closed


@pytest.mark.parametrize('lock', ['.decision.lock', '.cost.lock', '.candidate-b-0.lock',
                                 'ranking/.validation-1.lock', '../../.pair-barrier.lock'])
def test_live_decision_meter_shard_and_publisher_are_untouched(measurement, lock):
    case, directory, _ = measurement
    before = snapshot(directory / 'cost.jsonl')
    with base.lease((directory / lock).resolve()):
        row, = recovery.recover(case.root, case.protocol, now=5000)
    assert row['status'] == 'active'
    assert snapshot(directory / 'cost.jsonl') == before


def test_recent_remote_owner_is_not_declared_stopped(measurement):
    case, directory, _ = measurement
    before = snapshot(directory / 'cost.jsonl')
    row, = recovery.recover(case.root, case.protocol, now=122)
    assert row['status'] == 'skipped' and 'termination unconfirmed' in row['reason']
    assert snapshot(directory / 'cost.jsonl') == before


def test_exact_finish_receipt_is_replayed_without_estimation(measurement):
    case, directory, start = measurement
    core.atomic_json(directory / 'cost-events/interrupted.json', {
        **start, 'state': 'finished', 'time': 115., 'seconds': 15., 'allocated_gpu_seconds': 60., 'exit_code': 0})
    row, = recovery.recover(case.root, case.protocol, now=122)
    assert row['status'] == 'recovered' and row['seconds'] == 15.
    assert row['evidence']['kind'] == 'atomic_finish_receipt'
    assert 'duration_is_estimate' not in row['evidence']
    assert base.spent(directory) == 60.


@pytest.mark.parametrize('publication', ['decision.json', '../../test-decisions.json'])
def test_frozen_decision_costs_cannot_be_rewritten(measurement, publication):
    case, directory, _ = measurement
    path = (directory / publication).resolve()
    core.atomic_json(path, {'already': 'published'})
    before = snapshot(directory / 'cost.jsonl')
    row, = recovery.recover(case.root, case.protocol, now=5000)
    assert row['status'] == 'blocked' and 'sealed by a published decision' in row['reason']
    assert snapshot(directory / 'cost.jsonl') == before


@pytest.mark.parametrize('name', ['cost.jsonl', 'progress.json', '.decision.lock', '.candidate-b-0.lock',
                                'ranking/.candidate-0.lock', 'cost-events/interrupted.json'])
def test_symlinked_evidence_and_locks_are_never_followed(measurement, name):
    case, directory, _ = measurement
    path = directory / name
    target = case.root / 'outside'
    if path.exists():
        path.rename(target)
    else:
        target.write_text('{}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)
    before = snapshot(target)
    row, = recovery.recover(case.root, case.protocol, now=5000)
    assert row['status'] == 'blocked'
    assert snapshot(target) == before


def test_confirmed_local_stop_recovers_without_remote_grace_period(measurement, monkeypatch):
    case, directory, _ = measurement
    policy = recovery.pair_recovery.recovery_module()
    monkeypatch.setattr(policy, 'local_event_owner', lambda *a: 'stopped')
    monkeypatch.setattr(recovery.pair_recovery, 'recovery_module', lambda: policy)
    row, = recovery.recover(case.root, case.protocol, now=122)
    assert row['status'] == 'recovered'
    assert row['evidence']['kind'] == 'confirmed_local_stop_last_evidence'


def test_live_event_owner_never_recovers_even_after_long_silence(measurement, monkeypatch):
    case, directory, _ = measurement
    policy = recovery.pair_recovery.recovery_module()
    monkeypatch.setattr(policy, 'local_event_owner', lambda *a: 'live')
    monkeypatch.setattr(recovery.pair_recovery, 'recovery_module', lambda: policy)
    before = snapshot(directory / 'cost.jsonl')
    row, = recovery.recover(case.root, case.protocol, now=5000)
    assert row['status'] == 'active'
    assert snapshot(directory / 'cost.jsonl') == before


def test_unknown_allocation_or_overlapping_event_is_preserved(measurement):
    case, directory, start = measurement
    base.journal(directory / 'cost.jsonl', {**start, 'event_id': 'other', 'time': 200.})
    before = snapshot(directory / 'cost.jsonl')
    row, = recovery.recover(case.root, case.protocol, now=5000)
    assert row['status'] == 'blocked' and 'overlapping' in row['reason']
    assert snapshot(directory / 'cost.jsonl') == before
