"""Repair exports must preserve original measurements without pooling run costs."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def results(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('mbpp_repair_results')
    monkeypatch.setattr(module.switch_results, 'report', lambda root: 'auxiliary report')
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint(root, relative, reward=.5):
    step = int(relative.split('/')[1].split('-t')[1])
    directory = root / relative
    digest = save(directory / 'result.json', {
        'schema': 'offpolicy-selected-prefix-switch/v1',
        'complete': True, 'completed_steps': step + 10, 'rewards': {'q0': reward, 'q1': reward}})
    save(directory / 'result.sha256.json', {'sha256': digest})
    save(directory / 'curve.json', {'schema': 'offpolicy-selected-prefix-switch/v1', 'result_sha256': digest,
        'points': {str(step+10): {'updates': 10, 'reward': reward, 'final': True}}})


def ledger(root, relative, seconds, *, finished=True):
    path = root / relative / 'cost.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    start = {'event_id': 'event', 'phase': 'train', 'ledger': 'deployment', 'state': 'started'}
    rows = [start]
    if finished:
        rows.append({**start, 'state': 'finished', 'allocated_gpu_seconds': seconds, 'exit_code': 0})
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def freeze(root, meta):
    meta['snapshot_files'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file() and path.name != 'repair.json'}
    save(root / 'repair.json', meta)


def prepared(tmp_path):
    source, root = tmp_path / 'source', tmp_path / 'repair'
    digest = save(source / 'switch.json', {'dataset': 'mbpp'})
    save(root / 'switch.json', {'dataset': 'mbpp'})
    non_gated, gated = [], []
    for seed in range(5):
        for step in (25, 50, 100):
            arms = ['selection_reduced', 'random_reduced']
            if seed >= 3:
                arms += ['selection_full', 'random_full', 'gated']
            for arm in arms:
                relative = f'states/s{seed}-t{step}/points/view-{step}/{arm}'
                (gated if arm == 'gated' else non_gated).append(relative)
    meta = {'schema': 'mbpp-repair/v1', 'source_root': str(source), 'source_switch_sha256': digest,
            'reused_branches': non_gated[5:], 'rerun_branches': non_gated[:5], 'dependent_branches': gated}
    for relative in meta['reused_branches']:
        endpoint(root, relative)
        ledger(root, relative, 77)
        ledger(source, relative, 77)
    for relative in meta['rerun_branches']:
        ledger(root, 'original-attempts/' + relative, 123)
        ledger(source, relative, 999)
    freeze(root, meta)
    return root, source, meta


def test_partial_export_reuses_37_and_never_pools_failed_original_and_new_cost(results, tmp_path):
    root, source, meta = prepared(tmp_path)
    retry = meta['rerun_branches'][0]
    endpoint(root, retry, .75)
    ledger(root, retry, 20)
    data = results.export(root)
    assert len(data['branches']) == 48
    assert data['measured'] == {'reused_branches': 37, 'rerun_branches': 1, 'dependent_branches': 0}
    row = next(row for row in data['branches'] if row['path'] == retry)
    assert row['original_source_cost']['total'] == 123
    assert row['new_repair_cost']['total'] == 20
    assert row['measurement']['mean_reward'] == .75
    assert row['measurement']['question_rewards'] == {'q0': .75, 'q1': .75}
    reused = next(row for row in data['branches'] if row['origin'] == 'reused_branches')
    assert reused['original_source_cost']['total'] == 77 and reused['new_repair_cost'] is None
    assert not data['complete'] and not data['endpoint_coverage_complete']
    assert all(row['measurement']['mean_reward'] is None for row in data['branches']
               if row['origin'] == 'dependent_branches')


def test_real_results_bash_writes_one_partial_txt_without_touching_run(results, tmp_path):
    root, source, _ = prepared(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    before = {path: path.read_bytes() for directory in (root, source)
              for path in directory.rglob('*') if path.is_file()}
    env = {**os.environ, 'HOME': str(tmp_path), 'MBPP_REPAIR_ROOT': str(root),
           'MBPP_REPAIR_SOURCE': str(source), 'SWITCH_PYTHON': sys.executable,
           'OM_WORK': str(tmp_path / 'work')}
    env.pop('PYTHONPATH', None)
    result = subprocess.run(['bash', 'scripts/run_mbpp_repair.sh', 'results'],
                            cwd=repo, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    output, = tmp_path.glob('*.txt')
    data = json.loads(output.read_text().split('DATA_JSON\n', 1)[1])
    assert output.name == 'mbpp-repair-results.txt' and output.stat().st_size <= 1024*1024
    assert data['measured']['reused_branches'] == 37 and not data['complete']
    assert before == {path: path.read_bytes() for directory in (root, source)
                      for path in directory.rglob('*') if path.is_file()}


@pytest.mark.parametrize('filename', ['result.json', 'curve.json'])
@pytest.mark.parametrize('schema', [None, 'offpolicy-net-gain-gate/v3-1'])
def test_wrong_schema_is_not_accepted_even_with_matching_seal(results, tmp_path, filename, schema):
    relative = 'states/s0-t25/points/view-25/selection_reduced'
    endpoint(tmp_path, relative)
    path = tmp_path / relative / filename
    value = json.loads(path.read_text())
    value.pop('schema')
    if schema is not None:
        value['schema'] = schema
    digest = save(path, value)
    if filename == 'result.json':
        save(path.with_name('result.sha256.json'), {'sha256': digest})
    row = results.measurement(tmp_path, relative, 25)
    assert row['issues'] and not row['curve_complete']
    assert row['mean_reward'] == (.5 if filename == 'curve.json' else None)


def test_frozen_original_cost_remains_available_without_source_root(results, tmp_path):
    root, source, meta = prepared(tmp_path)
    source.rename(tmp_path / 'source-offline')
    data = results.export(root)
    assert data['measured']['reused_branches'] == 37
    assert all(row['original_source_cost']['total'] == 123 for row in data['branches']
               if row['origin'] == 'rerun_branches')
    assert data['errors'] and data['source_manifest_matches_snapshot'] is None


@pytest.mark.parametrize('filename', ['result.json', 'result.sha256.json'])
def test_changed_reused_endpoint_is_not_counted_as_new_or_valid(results, tmp_path, filename):
    root, _, meta = prepared(tmp_path)
    path = root / meta['reused_branches'][0] / filename
    path.write_text(path.read_text() + '\n')
    data = results.export(root)
    assert data['measured']['reused_branches'] == 36
    row = next(row for row in data['branches'] if row['path'] == meta['reused_branches'][0])
    assert row['measurement']['mean_reward'] is None and row['measurement']['issues']
    assert row['new_repair_cost'] is None


def test_modified_frozen_cost_is_not_reported_as_known_original_cost(results, tmp_path):
    root, _, meta = prepared(tmp_path)
    retry = meta['rerun_branches'][0]
    ledger(root, 'original-attempts/' + retry, 1000)
    row = next(row for row in results.export(root)['branches'] if row['path'] == retry)
    assert row['original_source_cost']['total'] is None
    assert row['original_source_cost']['finished_events_subtotal'] is None


@pytest.mark.parametrize('damage', ['missing-seal', 'partial', 'nan', 'outside-root'])
def test_damaged_new_endpoint_does_not_hide_other_partial_results(results, tmp_path, damage):
    root, _, meta = prepared(tmp_path)
    retry = meta['rerun_branches'][0]
    endpoint(root, retry)
    path = root / retry / 'result.json'
    if damage == 'missing-seal':
        path.with_name('result.sha256.json').unlink()
    elif damage == 'outside-root':
        other = tmp_path / 'external.json'
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    else:
        value = json.loads(path.read_text())
        if damage == 'partial':
            value['complete'] = False
        else:
            value['rewards']['q0'] = float('nan')
        digest = save(path, value)
        save(path.with_name('result.sha256.json'), {'sha256': digest})
    data = results.export(root)
    assert data['measured']['reused_branches'] == 37
    assert data['measured']['rerun_branches'] == 0


def test_open_new_ledger_has_no_invented_zero_total(results, tmp_path):
    root, _, meta = prepared(tmp_path)
    retry = meta['rerun_branches'][0]
    ledger(root, retry, 0, finished=False)
    row = next(row for row in results.export(root)['branches'] if row['path'] == retry)
    assert row['new_repair_cost']['total'] is None
    assert row['new_repair_cost']['finished_events_subtotal'] is None


def test_report_failure_still_writes_one_bounded_txt_without_mutating_sources(results, tmp_path, monkeypatch):
    root, _, _ = prepared(tmp_path)
    output = tmp_path / 'mbpp-repair-results.txt'
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    def failed(root):
        raise ValueError('incomplete arm: before')
    monkeypatch.setattr(results.switch_results, 'report', failed)
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(root), '--out', str(output)])
    results.main()
    data = json.loads(output.read_text().split('DATA_JSON\n', 1)[1])
    assert data['auxiliary_report']['status'] == 'failed'
    assert data['measured']['reused_branches'] == 37
    assert len(data['branches']) == 48 and output.stat().st_size <= 1024 * 1024
    assert list(tmp_path.glob('*.txt')) == [output]
    assert not list(tmp_path.glob('*.tmp.*'))
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize('damage', ['overlap', 'escape', 'missing', 'source-self'])
def test_invalid_repair_inventory_writes_diagnostic_not_scientific_fiction(results, tmp_path, monkeypatch, damage):
    root, _, meta = prepared(tmp_path)
    if damage == 'overlap':
        meta['rerun_branches'][0] = meta['reused_branches'][0]
    elif damage == 'escape':
        meta['rerun_branches'][0] = '../escape'
    elif damage == 'missing':
        meta['reused_branches'].pop()
    else:
        meta['source_root'] = str(root)
    save(root / 'repair.json', meta)
    output = tmp_path / 'results.txt'
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(root), '--out', str(output)])
    with pytest.raises(SystemExit):
        results.main()
    data = json.loads(output.read_text().split('DATA_JSON\n', 1)[1])
    assert data['branches'] == [] and data['errors'] and not data['complete']


def test_endpoint_only_all_48_does_not_claim_final_curve_completion(results, tmp_path):
    root, _, meta = prepared(tmp_path)
    for relative in meta['rerun_branches'] + meta['dependent_branches']:
        endpoint(root, relative)
    (root / meta['rerun_branches'][0] / 'curve.json').unlink()
    data = results.export(root)
    assert data['endpoint_coverage_complete'] and not data['complete']


def test_oversized_question_identifiers_keep_endpoint_rows_in_one_bounded_file(results, tmp_path, monkeypatch):
    root, _, meta = prepared(tmp_path)
    relative = meta['rerun_branches'][0]
    directory = root / relative
    step = int(relative.split('/')[1].split('-t')[1])
    digest = save(directory / 'result.json', {'schema': 'offpolicy-selected-prefix-switch/v1',
                                              'complete': True, 'completed_steps': step + 1,
                                              'rewards': {'q' * (1024 * 1024): .5}})
    save(directory / 'result.sha256.json', {'sha256': digest})
    output = tmp_path / 'results.txt'
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(root), '--out', str(output)])
    with pytest.raises(SystemExit):
        results.main()
    data = json.loads(output.read_text().split('DATA_JSON\n', 1)[1])
    row = next(row for row in data['branches'] if row['path'] == relative)
    assert row['measurement']['mean_reward'] == .5
    assert row['measurement']['question_rewards'] is None
    assert row['measurement']['question_rewards_omitted_for_size'] == 1
    assert len(data['branches']) == 48 and output.stat().st_size <= 1024 * 1024


def test_missing_repair_root_writes_diagnostic_without_creating_experiment(results, tmp_path, monkeypatch):
    root = tmp_path / 'absent'
    output = tmp_path / 'results.txt'
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(root), '--out', str(output)])
    with pytest.raises(SystemExit):
        results.main()
    assert output.is_file() and not root.exists()
