"""Match submitted OLMo timers without changing archived costs or learner identity."""

import copy
import json
from unittest.mock import patch

import pytest

from srgc_rebuttal.srgc import Engine
from srgc_research.dispatch import model_results
from srgc_research.dispatch.model_result_transfer import records
from srgc_research.dispatch.submission_time import (
    BASELINE,
    collect,
    history_costs,
    render,
    summaries,
)
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- pytest fixture registration
)
from srgc_research.tests.test_model_results import finish, read_result, saved
from srgc_research.tests.test_model_status import read_status

PLAN = {"shared_prefix_updates": 25, "total_updates": 275}


def progress(folder, arm, *, seed=5, step=275, preparation=2.):
    history = [{"checkpoint": update - 1, "completed_updates": update,
                "selection_gpu_seconds": 40. if (update - 1) % 25 == 0 and arm in {"on_policy", "switch"} else 0.,
                "training_gpu_seconds": 4., "selector": "on_policy" if arm == "on_policy" else arm,
                "metrics": {"sample_reward": .5}} for update in range(26, step + 1)]
    document = {"seed": seed, "arm": arm, "step": step, "sampling_protocol": Engine.SAMPLING_PROTOCOL,
                "costs": {"sr_preparation_gpu_seconds": preparation}, "history": history}
    path = folder / f"{arm}-progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def test_retained_update_timers_exclude_other_attempts_and_convert_gpu_seconds_once(tmp_path):
    progress(tmp_path, 'on_policy')
    receipts = tmp_path / 'cost-receipts/on_policy'
    receipts.mkdir(parents=True)
    # The submitted-time reader must not include a duplicate finished update,
    # an inclusive invocation, checkpoint/evaluation costs, or multiply by four again.
    (receipts / 'duplicate.json').write_text(json.dumps({'phase': 'training', 'gpu_seconds': 36000}))
    (tmp_path / 'session.json').write_text(json.dumps({'gpu_seconds': 72000}))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    value = collect(tmp_path, 'on_policy', PLAN, 5)
    assert value['record'] == 'COMPLETE'
    assert value['gpu_seconds'] == {'training': 1000., 'selection': 400., 'preparation': 0., 'primary': 1400.}
    assert value['gpu_hours']['primary'] == 1400 / 3600
    assert len(value['update_timers']) == 250
    assert value['update_timers'][0]['step'] == 26
    assert value['update_timers'][-1]['step'] == 275
    assert all(not missing for missing in value['missing_update_timer_ranges'].values())
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


def test_sr_and_switch_charge_the_same_recorded_subset_preparation_once(tmp_path):
    progress(tmp_path, 'sr', preparation=1.25)
    progress(tmp_path, 'switch', preparation=80.)
    sr, switch = (collect(tmp_path, arm, PLAN, 5) for arm in ('sr', 'switch'))
    assert sr['gpu_seconds']['preparation'] == switch['gpu_seconds']['preparation'] == 1.25
    assert sr['gpu_seconds']['primary'] == 1001.25
    assert switch['gpu_seconds']['primary'] == 1401.25
    assert sr['preparation_source'] == switch['preparation_source']


def test_missing_common_sr_preparation_does_not_use_a_different_switch_charge(tmp_path):
    progress(tmp_path, 'switch', preparation=80.)
    value = collect(tmp_path, 'switch', PLAN, 5)
    assert value['record'] == 'PARTIAL'
    assert value['gpu_seconds']['preparation'] is None
    assert value['gpu_seconds']['primary'] is None


def test_incomplete_and_old_histories_do_not_extrapolate_missing_costs(tmp_path):
    path = progress(tmp_path, 'on_policy', step=50)
    doc = json.loads(path.read_text())
    del doc['history'][0]['training_gpu_seconds']
    path.write_text(json.dumps(doc))
    value = collect(tmp_path, 'on_policy', PLAN, 5)
    assert value['record'] == 'PARTIAL'
    assert value['recorded_through_update'] == 50 and value['recorded_updates'] == 25
    assert value['gpu_hours']['primary'] is None
    assert value['missing_update_timer_ranges']['training_gpu_seconds'] == [[26, 26], [51, 275]]
    assert value['missing_update_timer_counts']['training_gpu_seconds'] == 226
    assert value['known_recorded_subtotal_gpu_seconds'] == 40 + 24 * 4


@pytest.mark.parametrize('field,bad', [('training_gpu_seconds', -1), ('selection_gpu_seconds', float('inf')),
                                    ('training_gpu_seconds', True)])
def test_invalid_times_are_rejected(tmp_path, field, bad):
    path = progress(tmp_path, 'random')
    doc = json.loads(path.read_text())
    doc['history'][0][field] = bad
    with pytest.raises(ValueError, match='finite nonnegative'):
        history_costs(doc['history'], start=25, stop=275)


def test_exact_duplicate_history_is_counted_once_and_gaps_or_conflicts_fail(tmp_path):
    path = progress(tmp_path, 'random')
    original = json.loads(path.read_text())['history']
    assert history_costs([original[0], *original, *original], start=25, stop=275) == history_costs(original, start=25, stop=275)
    with pytest.raises(ValueError, match='contiguous'):
        history_costs(original[1:], start=25, stop=275)
    with pytest.raises(ValueError, match='conflicting'):
        history_costs([*original, {**original[0], 'training_gpu_seconds': 999.}], start=25, stop=275)


@pytest.mark.parametrize('dataset', ('math', 'mbpp'))
def test_three_models_status_and_compact_results_keep_two_time_views_without_writes(cohort, capsys, dataset):
    folder = finish(cohort, dataset)
    progress(folder, 'on_policy')
    before = saved(cohort)
    code, status = read_status(cohort, capsys)
    assert code == 0
    report = status['datasets'][('math', 'mbpp').index(dataset)]
    assert report['submission_time_basis']['submitted_commit'] == BASELINE
    assert report['submission_time_basis']['first_update'] == 26
    row = next(r for r in report['tasks'] if r['task'] == 'seed-5.on_policy')
    assert row['submission_timing']['gpu_seconds']['primary'] == 1400.
    assert row['timing']['record'] == 'MISSING'
    text = '\n'.join(render(report))
    assert 'SUBMITTED OLMO TIME BASIS' in text and '26--275' in text
    with patch.dict('os.environ', cohort.env):
        assert model_results.export(cohort.model, environment=cohort.env) == 0
    result = read_result(cohort)['datasets'][dataset]
    exported = next(r for r in result['tasks'] if r['task'] == 'seed-5.on_policy')
    exported['submission_timing']['update_timers'] = records(exported['submission_timing']['update_timers'])
    assert exported['submission_timing'] == row['submission_timing']
    assert exported['timing'] == {'record': row['timing']['record']} == {'record': 'MISSING'}
    assert result['submission_time_basis'] == report['submission_time_basis']
    assert 'recorded_execution_settings' in result
    assert saved(cohort) == before


def test_unbound_or_different_identity_progress_cannot_supply_a_comparison(cohort, capsys):
    folder = finish(cohort, 'math')
    progress(folder, 'on_policy')
    marker_path = folder / 'run.json'
    marker = json.loads(marker_path.read_text())
    marker['input_sha256'] = '0' * 64
    marker_path.write_text(json.dumps(marker))
    _, status = read_status(cohort, capsys)
    row = next(r for r in status['datasets'][0]['tasks'] if r['task'] == 'seed-5.on_policy')
    assert row['submission_timing']['record'] == 'INVALID'
    assert row['submission_timing']['gpu_hours']['primary'] is None


def test_missing_or_invalid_cost_does_not_overwrite_the_original_phase_total():
    report = {'submission_time_basis': {'first_update': 26, 'last_update': 275}, 'tasks': [
        {'seed': 5, 'arm': 'random', 'timing': {'gpu_hours': {'primary': 100.}},
         'submission_timing': {'record': 'MISSING', 'gpu_hours': {'primary': None}}}]}
    before = copy.deepcopy(report)
    assert 'MISSING' in '\n'.join(render(report))
    assert report == before


def test_means_require_all_planned_seeds_and_keep_sample_sd():
    rows = [{'seed': seed, 'arm': 'on_policy', 'submission_timing': {
        'record': 'COMPLETE', 'gpu_hours': {'primary': float(seed)}}} for seed in range(5, 10)]
    stats = summaries(rows, list(range(5, 10)), ['on_policy'])['on_policy']
    assert stats['mean_gpu_hours'] == 7.
    assert stats['sample_sd_gpu_hours'] == pytest.approx(2.5 ** .5)
    rows[0]['submission_timing']['record'] = 'PARTIAL'
    rows[0]['submission_timing']['gpu_hours']['primary'] = None
    assert summaries(rows, list(range(5, 10)), ['on_policy'])['on_policy']['mean_gpu_hours'] is None
