"""Interrupted/replayed work never charges a retained update twice."""

import copy
import json
from unittest.mock import patch

import pytest

from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_research.dispatch import model_results
from srgc_research.dispatch.model_result_text import render
from srgc_research.dispatch.restart_costs import reconcile
from srgc_research.dispatch.submission_time import collect, read_json
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- pytest fixture registration
)
from srgc_research.tests.test_model_results import finish, read_result, saved
from srgc_research.tests.test_submission_time import PLAN, progress


def event(folder, arm, identity, checkpoint, seconds, phase='training', *, state='finished'):
    row = {'id': identity, 'phase': phase, 'checkpoint': checkpoint, 'gpu_count': 4,
           'state': state}
    ledger = PhaseLedger(folder / 'cost-receipts' / arm)
    ledger.record({**row, 'state': 'started'})
    if state == 'finished':
        row.update(gpu_seconds=seconds, wall_seconds=seconds / 4)
        ledger.record(row)
    return row


def fixture(folder, arm='random', *, step=27):
    path = progress(folder, arm, step=step)
    points = json.loads(path.read_text())['history']
    for row in points:
        for phase in ('selection', 'training'):
            seconds = row[phase + '_gpu_seconds']
            if seconds:
                event(folder, arm, f'{phase}-{row["checkpoint"]}', row['checkpoint'], seconds, phase)
    return points


def test_restarts_at_same_checkpoint_subtract_finished_replays_once(tmp_path):
    points = fixture(tmp_path)
    for name in ('discarded-a', 'discarded-b', 'discarded-c'):
        event(tmp_path, 'random', name, 25, 8.)
    # A copied export of the same durable receipt is also not a new operation.
    folder = tmp_path / 'cost-receipts/random'
    (folder / 'copied-retained.json').write_bytes((folder / 'training-25.json').read_bytes())
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    audit = reconcile(tmp_path, 'random', [dict(step=p['completed_updates'], **p) for p in points],
                      start=25, stop=27, read=read_json, expected_gpus=4)
    assert audit['record'] == 'COMPLETE'
    assert audit['recorded_before_gpu_seconds'] == 36.
    assert audit['removed_gpu_seconds'] == 28.
    assert audit['retained_gpu_seconds'] == 8.
    assert audit['duplicate_receipt_copies'] == 1
    assert len(audit['source_receipts']) == 6
    assert all(row['gpu_seconds'] == row['wall_seconds'] * row['gpu_count'] for row in audit['source_receipts'])
    assert len(audit['discarded_completed_attempts']) == 3
    assert before == {p: p.read_bytes() for p in before}


def test_unclosed_attempt_and_missing_receipt_never_become_zero_or_a_full_subtraction(tmp_path):
    points = fixture(tmp_path)
    event(tmp_path, 'random', 'killed', 25, None, state='started')
    event(tmp_path, 'random', 'closed-replay', 25, 12.)
    kwargs = {'start': 25, 'stop': 27, 'read': read_json}
    measured = [dict(step=p['completed_updates'], **p) for p in points]
    audit = reconcile(tmp_path, 'random', measured, **kwargs)
    assert audit['record'] == 'PARTIAL' and audit['removal_is_lower_bound']
    assert audit['removed_gpu_seconds'] == 12. and audit['unfinished_receipts'] == 1
    (tmp_path / 'cost-receipts/random/training-26.json').unlink()
    audit = reconcile(tmp_path, 'random', measured, **kwargs)
    assert audit['removed_gpu_seconds'] is None and audit['retained_gpu_seconds'] is None
    assert audit['unmatched_retained_updates'][0]['step'] == 27


def test_receipt_gpu_conversion_is_verified_and_never_divided_by_world_size(tmp_path):
    points = fixture(tmp_path)
    measured = [dict(step=p['completed_updates'], **p) for p in points]
    row_path = tmp_path / 'cost-receipts/random/training-25.json'
    row = read_json(row_path)
    row['gpu_seconds'] *= 4
    row_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='wall time'):
        reconcile(tmp_path, 'random', measured, start=25, stop=27, read=read_json)


def test_exact_history_copies_are_removed_without_changing_rewards(tmp_path):
    path = progress(tmp_path, 'random')
    document = read_json(path)
    document['history'] *= 4
    path.write_text(json.dumps(document))
    corrected = collect(tmp_path, 'random', PLAN, 5)
    assert corrected['record'] == 'COMPLETE'
    assert corrected['recorded_updates'] == 250
    assert corrected['gpu_seconds']['primary'] == 1000.
    assert corrected['duplicate_history_entries'] == 750
    assert corrected['duplicate_history_gpu_seconds']['training_gpu_seconds'] == 3000.
    from srgc_research.dispatch.reward_curves import training_points
    rewards = training_points(document['history'], start=25, stop=275)
    assert len(rewards) == 250 and all(row['reward'] == .5 for row in rewards)


@pytest.mark.parametrize('arm', ['random', 'sr', 'on_policy', 'switch'])
@pytest.mark.parametrize('dataset', ['math', 'mbpp'])
def test_all_three_models_and_four_arms_export_after_restart_subtraction(cohort, dataset, arm):
    folder = finish(cohort, dataset, arm=arm)
    if arm == 'switch':
        progress(folder, 'sr')
    fixture(folder, arm, step=275)
    event(folder, arm, 'discarded-before-restart', 25, 64.)
    before = saved(cohort)
    with patch.dict('os.environ', cohort.env):
        assert model_results.export(cohort.model, dataset, environment=cohort.env, text=True) == 0
    result = read_result(cohort)
    report = result['datasets'][dataset]
    task = next(r for r in report['tasks'] if r['seed'] == 5 and r['arm'] == arm)
    measured = task['submission_timing']
    kept = 1400. if arm in {'on_policy', 'switch'} else 1000.
    assert measured['gpu_seconds']['primary'] == kept + (2. if arm in {'sr', 'switch'} else 0.)
    audit = measured['restart_accounting']
    assert audit['recorded_before_gpu_seconds'] == kept + 64.
    assert audit['removed_gpu_seconds'] == 64.
    assert audit['retained_gpu_seconds'] == kept
    assert audit['record'] == 'COMPLETE'
    assert task['reward_percent'] == 50.
    assert 'RESTART TIME CORRECTION' in render(result)
    assert saved(cohort) == before


def test_explicit_retained_receipt_id_is_required_to_match(tmp_path):
    points = fixture(tmp_path)
    measured = [dict(step=p['completed_updates'], **p) for p in points]
    measured[0]['timing_receipt_ids'] = {'training': 'wrong-attempt'}
    original = copy.deepcopy(measured)
    with pytest.raises(ValueError, match='receipt ID'):
        reconcile(tmp_path, 'random', measured, start=25, stop=27, read=read_json)
    assert measured == original


def test_conflicting_receipt_copies_do_not_trigger_estimated_subtraction(tmp_path):
    points = fixture(tmp_path)
    measured = [dict(step=p['completed_updates'], **p) for p in points]
    directory = tmp_path / 'cost-receipts/random'
    copied = read_json(directory / 'training-25.json')
    copied.update(gpu_seconds=8., wall_seconds=2.)
    (directory / 'conflicting-copy.json').write_text(json.dumps(copied))
    with pytest.raises(ValueError, match='conflicting completed copies'):
        reconcile(tmp_path, 'random', measured, start=25, stop=27, read=read_json)


def test_receipt_allocation_must_match_the_recorded_plan(tmp_path):
    points = fixture(tmp_path)
    measured = [dict(step=p['completed_updates'], **p) for p in points]
    with pytest.raises(ValueError, match='experiment plan'):
        reconcile(tmp_path, 'random', measured, start=25, stop=27, read=read_json, expected_gpus=16)
