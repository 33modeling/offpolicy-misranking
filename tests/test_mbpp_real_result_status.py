"""Status uses the Switch writer's schema, not the generic Net-Gain schema."""

import hashlib

import pytest

from test_selection_switch_status import core, point, completed_prefix, status
import mbpp_status

WRITER_SCHEMA = 'offpolicy-selected-prefix-switch/v1'


def real_schema_root(root):
    core.atomic_json(root / 'switch.json', {
        'schema': WRITER_SCHEMA, 'dataset': 'mbpp', 'gate': 'convergence'})
    for seed in range(5):
        for step in (25, 50, 100):
            completed_prefix(root, seed, step)
            arms = status.rule.DEV_ARMS if seed < 3 else status.rule.TEST_ARMS
            for arm in arms:
                directory = point(root, seed, step) / arm
                core.atomic_json(directory / 'result.json', {'schema': WRITER_SCHEMA, 'complete': True})
                digest = hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest()
                core.atomic_json(directory / 'result.sha256.json', {'sha256': digest})
                core.atomic_json(directory / 'curve.json', {
                    'schema': WRITER_SCHEMA, 'result_sha256': digest, 'points': {'25': {'reward': .1}}})


def test_real_writer_schema_counts_all_48_in_overview_and_full_rows(tmp_path):
    real_schema_root(tmp_path)
    suite = status.snapshot(tmp_path, now=10000, local_gpus=False)
    counts = mbpp_status.counts(suite)
    assert suite['branch_counts'] == {'DONE': 48}
    assert (suite['development_done'], suite['test_done']) == (18, 30)
    assert (counts['planned'], counts['done'], counts['remaining']) == (48, 48, 0)
    output = mbpp_status.render({'updated': 10000, 'suites': [suite]}, width=180, all_tasks=True)
    assert '완료 확인 48개' in output
    assert len([line for line in output.splitlines() if line.startswith('DONE states/')]) == 48


def test_generic_net_gain_schema_is_not_a_switch_completion(tmp_path):
    real_schema_root(tmp_path)
    directory = point(tmp_path) / 'selection_reduced'
    core.atomic_json(directory / 'result.json', {'schema': 'offpolicy-net-gain-gate/v3-1', 'complete': True})
    digest = hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest()
    core.atomic_json(directory / 'result.sha256.json', {'sha256': digest})
    suite = status.snapshot(tmp_path, now=10000, local_gpus=False)
    assert suite['branch_counts'] == {'DONE': 47, 'INVALID': 1}
    assert mbpp_status.counts(suite)['done'] == 47


@pytest.mark.parametrize('nested', [False, True])
def test_active_or_lease_only_phase_prevents_48_of_48_completion(tmp_path, nested):
    real_schema_root(tmp_path)
    suite = status.snapshot(tmp_path, now=10000, local_gpus=False)
    directory = str((point(tmp_path) / 'selection_reduced').relative_to(tmp_path))
    phase = {'seed': 0, 'step': 25, 'kind': 'phase', 'arm': 'selection_reduced/curve',
             'status': 'WAIT', 'directory': directory + ('/curve' if nested else ''),
             'task_lease_held': True, 'host': 'unconfirmed-old-host'}
    suite['tasks'].append(phase)
    counts = mbpp_status.counts(suite)
    assert counts['done'] == 47 and counts['states']['RUN'] == 1
    output = mbpp_status.render({'updated': 10000, 'suites': [suite]}, width=180, all_tasks=True)
    assert '완료 확인 47개' in output
    assert f'RUN {directory}\n' in output
