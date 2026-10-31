"""Hand-calculated allocation cases; no CUDA jobs or edited source measurements."""
import json

import pytest

from scripts import srgc_gpu_accounting as accounting
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- fixture registration
)
from srgc_research.tests.test_model_results import finish, saved
from srgc_research.tests.test_model_status import read_status


def receipt(identity='one', *, seconds=40., phase='training', checkpoint=25, state='finished'):
    row = {'id': identity, 'phase': phase, 'checkpoint': checkpoint, 'gpu_count': 4, 'state': state}
    if state == 'finished':
        row.update(gpu_seconds=seconds, wall_seconds=seconds / 4)
    return row


def put(directory, name, row):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(row))
    return path


def bytes_snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_one_operation_copied_twice_still_costs_four_gpus_times_ten_seconds(tmp_path):
    put(tmp_path, 'original.json', receipt())
    put(tmp_path, 'export-copy.json', receipt())
    before = bytes_snapshot(tmp_path)
    costs = accounting.ReceiptLedger(tmp_path).totals()
    assert costs['known_gpu_seconds']['training_gpu_seconds'] == 40.
    assert costs['total_gpu_seconds'] == 40.
    assert costs['recorded_phases'] == 1
    assert costs['duplicate_receipt_copies'] == 1
    assert costs['duplicate_copy_gpu_seconds_excluded'] == 40.
    assert bytes_snapshot(tmp_path) == before


@pytest.mark.parametrize('start_first', [True, False])
def test_start_and_finish_copies_name_one_finished_operation(tmp_path, start_first):
    put(tmp_path, 'a.json' if start_first else 'z.json', receipt(state='started'))
    put(tmp_path, 'z.json' if start_first else 'a.json', receipt())
    costs = accounting.ReceiptLedger(tmp_path).totals()
    assert costs['complete'] and costs['total_gpu_seconds'] == 40.
    assert costs['unfinished_phases'] == []
    assert costs['duplicate_receipt_copies'] == 0


def test_real_replay_with_a_new_id_at_the_same_checkpoint_is_still_charged(tmp_path):
    put(tmp_path, 'original.json', receipt('first-attempt'))
    put(tmp_path, 'replayed.json', receipt('real-second-attempt'))
    costs = accounting.ReceiptLedger(tmp_path).totals()
    assert costs['total_gpu_seconds'] == 80.
    assert costs['duplicate_receipt_copies'] == 0


@pytest.mark.parametrize('field,value', [('gpu_seconds', 80.), ('phase', 'selection'),
                                        ('checkpoint', 26), ('gpu_count', 8)])
def test_conflicting_evidence_under_one_id_is_an_error_not_a_guessed_sum(tmp_path, field, value):
    first = receipt()
    second = {**first, field: value}
    if field == 'gpu_seconds':
        second['wall_seconds'] = value / 4
    if field == 'gpu_count':
        second['wall_seconds'] = second['gpu_seconds'] / value
    put(tmp_path, 'a.json', first)
    put(tmp_path, 'b.json', second)
    with pytest.raises(ValueError):
        accounting.ReceiptLedger(tmp_path).totals()


def test_inclusive_session_is_not_added_to_its_contained_phase(tmp_path):
    put(tmp_path, 'phase.json', receipt(seconds=40.))
    put(tmp_path, 'session.json', receipt('session', phase='session', seconds=80.))
    costs = accounting.ReceiptLedger(tmp_path).totals()
    assert costs['total_gpu_seconds'] == 40.
    assert costs['inclusive_session_gpu_seconds'] == 80.
    assert sum(costs['known_gpu_seconds'].values()) == 40.


def test_unfinished_duration_never_comes_from_file_age(tmp_path):
    put(tmp_path, 'past.json', receipt())
    put(tmp_path, 'pending.json', receipt('pending', checkpoint=26, state='started'))
    costs = accounting.ReceiptLedger(tmp_path).totals()
    assert not costs['complete'] and costs['total_gpu_seconds'] is None
    assert costs['known_completed_gpu_seconds'] == 40.
    assert costs['unfinished_phases'] == [{'id': 'pending', 'phase': 'training', 'checkpoint': 26}]


def test_measured_zero_session_is_distinct_from_absent_measurements(tmp_path):
    put(tmp_path, 'zero.json', receipt(phase='session', seconds=0.))
    assert accounting.measured(tmp_path)['total_gpu_seconds'] == 0.
    empty = tmp_path / 'empty'
    assert accounting.measured(empty)['total_gpu_seconds'] is None
    assert not accounting.measured(empty)['complete']
    assert accounting.ReceiptLedger(empty).totals()['total_gpu_seconds'] is None


@pytest.mark.parametrize('value', [True, -1., float('inf'), float('nan'), '40', 10 ** 400])
def test_invalid_recorded_seconds_are_rejected(tmp_path, value):
    put(tmp_path, 'bad.json', {**receipt(), 'gpu_seconds': value})
    with pytest.raises(ValueError):
        accounting.ReceiptLedger(tmp_path).totals()


def test_world_size_cannot_be_multiplied_twice(tmp_path):
    put(tmp_path, 'bad.json', {**receipt(), 'gpu_seconds': 160.})
    with pytest.raises(ValueError, match='allocated GPUs'):
        accounting.ReceiptLedger(tmp_path).totals()


def test_rank_measurements_are_summed_once_under_the_phase_allocation(tmp_path):
    row = receipt()
    row['rank_timings'] = [{'local_gpu_count': 1, 'wall_seconds': seconds} for seconds in (9., 10., 8., 10.)]
    put(tmp_path, 'phase.json', row)
    assert accounting.ReceiptLedger(tmp_path).totals()['total_gpu_seconds'] == 40.
    row['rank_timings'][0]['local_gpu_count'] = 4
    put(tmp_path, 'phase.json', row)
    with pytest.raises(ValueError, match='rank GPU allocations'):
        accounting.ReceiptLedger(tmp_path).totals()


def test_overlapping_or_nonreconciling_stages_are_rejected(tmp_path):
    row = receipt()
    row['stages'] = {'generation': {'gpu_seconds': 30.}, 'forward': {'gpu_seconds': 30.}}
    put(tmp_path, 'phase.json', row)
    with pytest.raises(ValueError, match='stage costs'):
        accounting.ReceiptLedger(tmp_path).totals()


def seeded(folder, bundle):
    bundle.write_text('{}')
    put(folder / 'cost-receipts/shared-prefix', 'prefix.json', receipt('prefix', seconds=40.))
    put(folder / 'invocations/prefix', 'session.json', receipt('prefix-session', phase='session', seconds=48.))
    put(folder / 'cost-receipts/on_policy', 'train.json', receipt('arm-training', seconds=80.))
    put(folder / 'invocations/on_policy', 'session.json', receipt('arm-session', phase='session', seconds=100.))
    put(folder, 'on_policy-endpoint.json', {'cost_measurement_complete': True})
    from srgc_rebuttal.plan import digest
    put(bundle.with_suffix('.cache'), 'cost-summary.json', {
        'complete': True, 'bundle_sha256': digest(bundle), 'invocations': {'complete': True, 'total_gpu_seconds': 20.}})


def test_seed_total_charges_prefix_and_cache_once_without_adding_phase_views(tmp_path):
    folder, bundle = tmp_path / 'seed-5', tmp_path / 'inputs.json'
    seeded(folder, bundle)
    before = bytes_snapshot(tmp_path)
    result = accounting.seed_costs(folder, bundle, ['on_policy'])
    totals = result['experiment_accounting']
    assert totals['complete']
    assert totals['phase_gpu_seconds'] == 120.
    assert totals['inclusive_invocation_gpu_seconds'] == 148.
    assert totals['experiment_including_cache_gpu_seconds'] == 168.
    assert totals['queue_wait_gpu_seconds'] is None
    assert bytes_snapshot(tmp_path) == before


def test_copied_prefix_and_arm_receipts_do_not_inflate_any_seed_total(tmp_path):
    folder, bundle = tmp_path / 'seed-5', tmp_path / 'inputs.json'
    seeded(folder, bundle)
    for scope, name in [('shared-prefix', 'prefix.json'), ('on_policy', 'train.json')]:
        path = folder / 'cost-receipts' / scope / name
        path.with_name('copy.json').write_bytes(path.read_bytes())
    result = accounting.seed_costs(folder, bundle, ['on_policy'])
    assert result['arms']['on_policy']['training_gpu_seconds'] == 80.
    assert result['experiment_accounting']['experiment_including_cache_gpu_seconds'] == 168.
    assert result['experiment_accounting']['duplicate_copy_gpu_seconds_excluded'] == 120.


def test_mismatched_bundle_identity_is_never_relabelled_as_valid_time(tmp_path):
    folder, bundle = tmp_path / 'seed-5', tmp_path / 'inputs.json'
    seeded(folder, bundle)
    bundle.write_text('{"other":true}')
    with pytest.raises(ValueError, match='different input bundle'):
        accounting.seed_costs(folder, bundle, ['on_policy'])


def test_finished_phases_cannot_exceed_their_finished_allocation(tmp_path):
    folder, bundle = tmp_path / 'seed-5', tmp_path / 'inputs.json'
    seeded(folder, bundle)
    put(folder / 'invocations/on_policy', 'session.json', receipt('arm-session', phase='session', seconds=40.))
    with pytest.raises(ValueError, match='inclusive invocation'):
        accounting.seed_costs(folder, bundle, ['on_policy'])


def test_public_model_status_deduplicates_real_receipt_ids_and_preserves_original_records(cohort, capsys):
    folder = finish(cohort, "math")
    endpoint = folder / "on_policy-endpoint.json"
    document = json.loads(endpoint.read_text())
    document["cost_measurement_complete"] = True
    endpoint.write_text(json.dumps(document))
    phase = folder / "cost-receipts/on_policy"
    put(phase, "phase.json", receipt())
    put(phase, "copied.json", receipt())
    put(folder / "invocations/on_policy", "session.json", receipt("session", phase="session", seconds=48.))
    before = saved(cohort)
    code, report = read_status(cohort, capsys)
    assert code == 0
    row = next(row for row in report["datasets"][0]["tasks"] if row["task"] == "seed-5.on_policy")
    assert row["timing"]["gpu_seconds"]["training"] == 40.
    assert row["timing"]["gpu_hours"]["training"] == 40. / 3600
    assert row["timing"]["duplicate_receipt_copies"] == 1
    assert row["timing"]["duplicate_copy_gpu_seconds_excluded"] == 40.
    assert saved(cohort) == before


def test_live_partial_cache_samples_are_not_added_to_inclusive_allocations(tmp_path):
    folder, bundle = tmp_path / 'seed-5', tmp_path / 'inputs.json'
    seeded(folder, bundle)
    put(bundle.with_suffix('.cache') / 'live-costs', 'rank-0.json',
        {'local_gpu_seconds': 999999., 'state': 'partial', 'additive_to_phase_totals': False})
    result = accounting.seed_costs(folder, bundle, ['on_policy'])
    assert result['experiment_accounting']['experiment_including_cache_gpu_seconds'] == 168.


def test_a_finished_zero_phase_is_known_even_before_the_endpoint_is_complete(tmp_path):
    from scripts.srgc_time_status import from_ledger

    put(tmp_path, 'zero.json', receipt(seconds=0.))
    measured = accounting.ReceiptLedger(tmp_path).totals()
    displayed = from_ledger(measured, completed=False)
    assert displayed['record'] == 'PARTIAL'
    assert displayed['gpu_seconds']['training'] == displayed['gpu_seconds']['phases'] == 0.


@pytest.mark.parametrize("wrong_scope", ["invocations/on_policy", "cost-receipts/on_policy"])
def test_receipt_scope_cannot_confuse_inclusive_and_contained_costs(tmp_path, wrong_scope):
    folder, bundle = tmp_path / "seed-5", tmp_path / "inputs.json"
    seeded(folder, bundle)
    phase = "training" if wrong_scope.startswith("invocations") else "session"
    put(folder / wrong_scope, "wrong-scope.json", receipt("wrong-scope", phase=phase))
    with pytest.raises(ValueError, match="wrong scope"):
        accounting.seed_costs(folder, bundle, ["on_policy"])


def test_aggregate_overflow_is_invalid_evidence_instead_of_infinite_gpu_hours(tmp_path):
    put(tmp_path, "a.json", receipt("a", seconds=1e308))
    put(tmp_path, "b.json", receipt("b", seconds=1e308))
    with pytest.raises(ValueError, match="finite numeric range"):
        accounting.ReceiptLedger(tmp_path).totals()


@pytest.mark.parametrize("phase", ["training", "startup", "checkpoint_save"])
@pytest.mark.parametrize("seconds", [0.0, 40.0])
def test_partial_phase_distinguishes_measured_zero_from_no_receipt(tmp_path, phase, seconds):
    row = {"id": "one", "phase": phase, "checkpoint": 25, "gpu_count": 4,
           "state": "finished", "wall_seconds": seconds / 4, "gpu_seconds": seconds}
    (tmp_path / "finished.json").write_text(json.dumps(row))
    (tmp_path / "waiting.json").write_text(json.dumps({
        "id": "unfinished", "phase": "selection", "checkpoint": 25,
        "gpu_count": 4, "state": "started"}))
    from scripts.srgc_time_status import from_ledger

    shown = from_ledger(accounting.ReceiptLedger(tmp_path).totals(), completed=False)
    costs = shown["gpu_seconds"]
    assert shown["record"] == "PARTIAL"
    assert costs["selection"] is None
    assert costs["training"] == (seconds if phase == "training" else None)
    assert costs["startup"] == (seconds if phase == "startup" else None)
    assert costs["checkpoint"] == (seconds if phase == "checkpoint_save" else None)
    assert costs["primary"] == (seconds if phase == "training" else None)
    assert costs["phases"] == seconds
    assert shown["gpu_hours"]["phases"] == seconds / 3600


@pytest.mark.parametrize("startup", [0.0, 40.0])
def test_partial_legacy_summary_zero_without_phase_identity_stays_unknown(startup):
    legacy = {"recorded_phases": 2, "complete": False,
              "known_gpu_seconds": {"selection_gpu_seconds": 0.0, "training_gpu_seconds": 0.0,
                                    "startup_gpu_seconds": startup},
              "unfinished_phases": [{"id": "pending", "phase": "selection"}]}
    from scripts.srgc_time_status import from_ledger

    shown = from_ledger(legacy, completed=False)
    assert shown["gpu_seconds"]["selection"] is None
    assert shown["gpu_seconds"]["training"] is None
    assert shown["gpu_seconds"]["startup"] == (startup or None)
