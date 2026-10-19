"""The public results command prints short scientific tables ready to copy."""

import copy
import os
from unittest.mock import patch

import pytest

from srgc_research.dispatch.model_result_text import ARMS, render
from srgc_research.dispatch.model_results import main
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 -- pytest fixture registration
)
from srgc_research.tests.test_model_results import finish, read_result, saved
from srgc_research.tests.test_submission_time import progress


def result():
    datasets = {}
    for dataset, reward in (('math', 20.), ('mbpp', 70.)):
        datasets[dataset] = {
            'plan': {'seeds': [5, 6, 7, 8, 9]},
            'submission_time_basis': {'first_update': 26, 'last_update': 275},
            'tasks': [{'seed': seed, 'arm': arm, 'status': 'complete', 'reward_percent': reward,
                       'switched_at': 50 if arm == 'switch' else None,
                       'timing': {'gpu_hours': {'primary': 999.}},
                       'submission_timing': {'record': 'COMPLETE', 'gpu_hours': {'primary': 1.25, 'selection': .25},
                                             'missing_update_timer_counts': {'selection_gpu_seconds': 0}}}
                      for seed in range(5, 10) for arm in ARMS],
            'initial_candidate_accuracy': [{'seed': 5, 'success_percent': 50.5}],
        }
    return {'model': 'test-model', 'datasets': datasets, 'errors': []}


def test_two_dataset_tables_keep_rewards_and_submitted_costs_separate():
    data = result()
    before = copy.deepcopy(data)
    text = render(data)
    math, mbpp = text.split('MBPP FINAL REWARD (%)')
    assert 'MATH FINAL REWARD (%)' in math and '20.000' in math and '70.000' not in math
    assert '70.000' in mbpp and '20.000' not in mbpp
    assert 'Switch after' in text and 'Mean' in text and 'SD' in text
    assert 'submitted OLMo basis; updates 26--275' in text
    assert '1.250' in text and '999.000' not in text
    assert len(text.splitlines()) < 55
    assert '{' not in text and '.json' not in text and 'RESULT:' not in text
    assert data == before


@pytest.mark.parametrize('failure', ['invalid', 'missing', 'partial'])
def test_unmeasured_or_invalid_costs_are_not_printed_as_full_totals(failure):
    data = result()
    row = data['datasets']['math']['tasks'][0]
    if failure == 'invalid':
        row['status'] = 'invalid'
    else:
        row['submission_timing']['record'] = failure.upper()
        row['submission_timing']['missing_update_timer_counts']['selection_gpu_seconds'] = 1
    # This sentinel must not become either a finished cost or the mean.
    row['submission_timing']['gpu_hours'] = {'primary': 123.456, 'selection': 123.456}
    assert '123.456' not in render(data)


def test_public_results_is_copyable_without_opening_any_result_file(cohort, capsys):
    folder = finish(cohort, 'math')
    progress(folder, 'on_policy')
    before = saved(cohort)
    with patch.dict(os.environ, cohort.env, clear=True):
        assert main([cohort.model, 'all', 'results', '--root', str(cohort.root)]) == 0
    output = capsys.readouterr().out
    assert output.startswith('MODEL: ')
    assert 'MATH FINAL REWARD (%)' in output and 'MBPP FINAL REWARD (%)' in output
    assert 'MATH GPU TIME' in output and '0.389' in output
    assert '.json' not in output and 'SIZE:' not in output and 'RESULT:' not in output
    assert str(cohort.root) not in output
    assert saved(cohort) == before
    # Keep the original machine-readable evidence as an internal artifact.
    artifact = read_result(cohort)
    task = next(r for r in artifact['datasets']['math']['tasks'] if r['task'] == 'seed-5.on_policy')
    assert task['submission_timing']['gpu_seconds']['primary'] == 1400.
