"""Export all 42 measured endpoints without recertifying original budgets."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import selector_pair_results as results
from test_selector_pair_results import branch_fixture, put_metadata
from test_selector_pair_finish_saved import experiment, fake_gpu, snapshot


def seal_result(target, result):
    put_metadata(target / 'result.json', result)
    put_metadata(target / 'result.sha256.json', {'sha256': results.saved_final.digest(target / 'result.json')})


def final_fixture(root, output, seed):
    step, arm = results.saved_final.TARGETS[seed]
    source = results.saved_final.directory(root, seed, step, 'on_policy', arm)
    stop = 457 if seed == 1 else 256
    contract = {'config': {'seed': seed, 'drift': step}, 'evaluation': {'val': ['q0', 'q1']}, 'eval_k': 8}
    put_metadata(source.parent / 'contract.json', contract)
    put_metadata(source / 'policy/policy_train.json', {'completed_steps': stop})
    planned = [
        {'step': step, 'k': 8, 'final': False, 'adapter': '/original/parent', 'hashes': {}},
        {'step': stop, 'k': 8, 'final': True, 'adapter': str(source / 'policy'),
         'hashes': {'policy_train.json': results.saved_final.digest(source / 'policy/policy_train.json')}},
    ]
    target = output / f'seed-{seed}'
    plan = {'schema': results.saved_final.FINAL_SCHEMA, 'source_root': str(root), 'directory': str(source),
            'seed': seed, 'start_step': step, 'completed_steps': stop, 'arm': arm,
            'canonical_complete': False, 'points': planned,
            'inputs': {str(source.parent / 'contract.json'): results.saved_final.digest(source.parent / 'contract.json')},
            'original_costs': {'.': {'gpu_seconds': 90000}}}
    put_metadata(target / 'plan.json', plan)
    result = {'schema': results.saved_final.FINAL_SCHEMA, 'seed': seed, 'start_step': step,
              'completed_steps': stop, 'plan_sha256': results.saved_final.digest(target / 'plan.json'),
              'evaluation_complete': True, 'canonical_complete': False, 'training_performed': False,
              'points': [{'step': p['step'], 'k': p['k'], 'final': p['final'],
                          'rewards': {'0': .25, '1': .75}} for p in planned],
              'original_costs': plan['original_costs'], 'new_evaluation_cost': {'gpu_seconds': 100}}
    seal_result(target, result)
    return source, target, result


@pytest.fixture
def all_results(tmp_path):
    root, output = tmp_path / 'selector-pair-v1', tmp_path / 'selector-pair-final-eval-v1'
    for _, _, branches in results.COMPLETION_GROUPS:
        for selector, arm, seed, step in branches:
            if selector == 'on_policy' and results.saved_final.TARGETS.get(seed) == (step, arm):
                continue
            branch_fixture(root, selector='adaptive-cached' if selector == 'adaptive' else selector,
                           seed=seed, step=step, arm=arm)
    final_fixture(root, output, 1)
    final_fixture(root, output, 4)
    put_metadata(root / 'development/s0-t25/result.json', {'existing_paired_result': True})
    return root, output


def export(monkeypatch, root, path, *args):
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(root), '--out', str(path), *args])
    results.main()
    return json.loads(path.read_text().split('DATA_JSON\n', 1)[1])


def test_exports_40_plus_two_once_without_waiting_for_strict_report(all_results, monkeypatch, tmp_path):
    root, output = all_results
    before = snapshot(root), snapshot(output)
    monkeypatch.setattr(results, 'run_report', lambda *a: pytest.fail('must not wait for matched-budget report'))
    target = tmp_path / 'results.txt'
    data = export(monkeypatch, root, target)
    rows = data['branch_measurements']
    assert len(rows) == data['branch_completion']['endpoints'] == 42
    assert len({(r['selector_branch'], r['arm'], r['seed'], r['prefix_updates']) for r in rows}) == 42
    assert data['branch_results_complete'] and data['branch_completion']['complete']
    assert data['branch_completion']['supplemental_endpoints'] == 2
    assert data['paired_validation']['status'] == 'not_run_supplemental_evaluations'
    assert data['export_exit_code'] == 0 and not data['complete'] and data['rows'] == []
    assert data['missing_states'] is None and data['paired_status'] == 'unverified'
    supplemental = [r for r in rows if r.get('result_source') == 'saved_final_evaluation']
    assert [(r['seed'], r['updates']) for r in supplemental] == [(1, 407), (4, 156)]
    assert all(r['mean_reward'] == .5 and r['question_rewards'] == {'0': .25, '1': .75}
               and not r['canonical_complete'] and not r['eligible_for_paired_comparison'] for r in supplemental)
    assert all(r['original_costs']['.']['gpu_seconds'] == 90000
               and r['new_evaluation_cost']['gpu_seconds'] == 100 for r in supplemental)
    assert 'total endpoints 42/42' in target.read_text()
    assert 'Includes 2 saved-final evaluations' in target.read_text()
    assert (snapshot(root), snapshot(output)) == before


@pytest.mark.parametrize('entry', ['scripts/run_selector_pair_results.sh', 'scripts/run_selector_pair.sh'])
def test_bash_results_need_no_gpu_lock_or_training(all_results, tmp_path, entry):
    root, output = all_results
    target = tmp_path / 'results.txt'
    args = ['bash', entry, *(['results'] if entry.endswith('/run_selector_pair.sh') else []), '--out', str(target)]
    process = subprocess.run(args, cwd=Path(results.__file__).parents[1],
        env={**os.environ, 'PAIR_ROOT': str(root), 'PAIR_PYTHON': sys.executable,
             'OM_WORK': str(tmp_path / 'work'), 'OM_LOCAL_LOCK_DIR': '/unwritable/no-gpu-lock'},
        capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert '[pair-results] endpoints 42/42; saved-final evaluations 2' in process.stdout
    assert '[node]' not in process.stdout and 'nvidia-smi' not in process.stderr


def test_deployed_exporter_also_exports_42_without_gpu_admission(all_results, tmp_path):
    import selector_pair_deploy as deploy
    from test_selector_pair_deploy import git

    root, output = all_results
    checkout = tmp_path / 'checkout'
    git(tmp_path, 'clone', '--shared', '--no-checkout', '-q', str(Path(results.__file__).parents[1]), str(checkout))
    runtime = deploy.stage_runtime(checkout)
    target = tmp_path / 'deployed-results.txt'
    process = subprocess.run(['bash', str(runtime / 'scripts/run_selector_pair_results.sh'), '--out', str(target)],
        env={**os.environ, 'PAIR_ROOT': str(root), 'PAIR_PYTHON': sys.executable,
             'OM_WORK': str(tmp_path / 'work'), 'OM_LOCAL_LOCK_DIR': '/unwritable/no-gpu-lock'},
        capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    data = json.loads(target.read_text().split('DATA_JSON\n', 1)[1])
    assert data['branch_completion']['endpoints'] == 42 and data['exporter']['version'].endswith('/v9')
    assert '[node]' not in process.stdout


@pytest.mark.parametrize('damage', ['seal', 'plan', 'source', 'policy', 'seed', 'step', 'nan', 'questions',
                                   'final', 'k', 'points', 'schema', 'outside', 'costs'])
def test_damaged_new_result_is_not_counted_and_does_not_hide_the_other_one(all_results, damage):
    root, output = all_results
    source = results.saved_final.directory(root, 1, 50, 'on_policy', 'selection_reduced')
    target = output / 'seed-1'
    result = json.loads((target / 'result.json').read_text())
    if damage == 'seal':
        put_metadata(target / 'result.sha256.json', {'sha256': 'wrong'})
    elif damage == 'plan':
        put_metadata(target / 'plan.json', {})
    elif damage == 'source':
        put_metadata(source.parent / 'contract.json', {})
    elif damage == 'policy':
        put_metadata(source / 'policy/policy_train.json', {'completed_steps': 458})
    elif damage == 'outside':
        (target / 'result.json').unlink()
        (target / 'result.json').symlink_to('/etc/passwd')
    else:
        if damage == 'seed': result['seed'] = 4
        elif damage == 'step': result['completed_steps'] = 100
        elif damage == 'nan': result['points'][-1]['rewards']['0'] = float('nan')
        elif damage == 'questions': result['points'][-1]['rewards'] = {'unknown': .5}
        elif damage == 'final': result['points'][-1]['final'] = False
        elif damage == 'k': result['points'][-1]['k'] = 1
        elif damage == 'points': result['points'] = []
        elif damage == 'costs': result['original_costs'] = {'.': {'gpu_seconds': 0}}
        else: result['schema'] = 'wrong'
        seal_result(target, result)
    data = results.final_evaluation_measurements(root, output)
    assert len(data['errors']) == 1 and [r['seed'] for r in data['rows']] == [4]
    canonical, _ = results.saved_branch_measurements(root)
    assert results.branch_completion(results.merge_final_evaluations(canonical, data['rows']))['endpoints'] == 41


def test_canonical_result_has_precedence_without_duplicate_branch(all_results):
    root, output = all_results
    source = results.saved_final.directory(root, 1, 50, 'on_policy', 'selection_reduced')
    endpoint = {'schema': results.RESULT_SCHEMA, 'complete': True, 'completed_steps': 457,
                'rewards': {'0': .4, '1': .6}}
    seal_result(source, endpoint)
    canonical, _ = results.saved_branch_measurements(root)
    data = results.final_evaluation_measurements(root, output)
    merged = results.merge_final_evaluations(canonical, data['rows'])
    assert len(merged) == 42
    row, = [r for r in merged if r['seed'] == 1 and r['prefix_updates'] == 50 and r['selector_branch'] == 'on_policy']
    assert row['status'] == 'saved_branch_measurement'
    assert results.branch_completion(merged)['supplemental_endpoints'] == 1


def test_missing_new_result_keeps_41_and_reports_pending(all_results):
    root, output = all_results
    (output / 'seed-4/result.json').unlink()
    data = results.final_evaluation_measurements(root, output)
    assert not data['errors'] and data['pending'][0]['seed'] == 4
    rows, _ = results.saved_branch_measurements(root)
    completion = results.branch_completion(results.merge_final_evaluations(rows, data['rows']))
    assert completion['endpoints'] == 41 and not completion['complete']


def test_custom_output_root_is_supported(all_results, tmp_path, monkeypatch):
    root, output = all_results
    custom = tmp_path / 'different-final-output'
    output.rename(custom)
    data = export(monkeypatch, root, tmp_path / 'results.txt', '--final-eval-root', str(custom))
    assert data['branch_completion']['endpoints'] == 42


def test_explicit_strict_check_keeps_all_42_even_when_budget_validation_fails(all_results, tmp_path, monkeypatch):
    root, _ = all_results
    monkeypatch.setattr(results, 'run_report', lambda *a: SimpleNamespace(
        returncode=7, stdout='', stderr='original allocation exceeded'))
    target = tmp_path / 'results.txt'
    with pytest.raises(SystemExit) as error:
        export(monkeypatch, root, target, '--validate-pairs')
    assert error.value.code == 7
    data = json.loads(target.read_text().split('DATA_JSON\n', 1)[1])
    assert data['branch_results_complete'] and len(data['branch_measurements']) == 42
    assert data['paired_validation']['status'] == 'failed' and not data['complete']


def test_environment_output_override_is_supported(all_results, tmp_path, monkeypatch):
    root, output = all_results
    custom = tmp_path / 'custom-output'
    output.rename(custom)
    monkeypatch.setenv('PAIR_FINAL_EVAL_ROOT', str(custom))
    data = export(monkeypatch, root, tmp_path / 'results.txt')
    assert data['branch_completion']['endpoints'] == 42


def test_real_finisher_output_imports_without_reading_weights(experiment, fake_gpu, monkeypatch):
    import selector_pair_finish_saved as finish
    root, output, source, seed, c = experiment
    monkeypatch.setattr(results.saved_final, 'TARGETS', {seed: (100, 'random_full')})
    # The small finisher fixture validates its contract through a stub; supply
    # the same source contract bytes for the independent read-only exporter.
    put_metadata(source.parent / 'contract.json', c)
    choice = finish.core.read(source / 'decision.json')
    choice['binding']['contract_sha256'] = finish.base.digest(source.parent / 'contract.json')
    put_metadata(source / 'decision.json', choice)
    finish.finish(root, output, seed, list('0123'))
    before = snapshot(root), snapshot(output)
    data = results.final_evaluation_measurements(root, output)
    assert not data['errors'] and data['rows'][0]['mean_reward'] == .5
    assert data['rows'][0]['updates'] == 50 and len(data['rows'][0]['curve_points']) == 3
    assert (snapshot(root), snapshot(output)) == before
