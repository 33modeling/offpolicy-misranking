"""One damaged arm must not hide valid measurements from an upload."""

import hashlib
import importlib
import json
import sys

import pytest

from test_switch_results import ROOT, branch, core, result_text, rule


@pytest.mark.parametrize('damage', ['null', 'list', 'string', 'empty-rewards', 'list-rewards',
                                  'bad-key', 'nan', 'bool', 'wrong-schema', 'partial', 'unsealed', 'bad-seal'])
def test_unverified_endpoint_is_retained_but_excluded_from_measured_contrast(tmp_path, monkeypatch, damage):
    core.atomic_json(tmp_path / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    branch(tmp_path, 's0-t25', 'random_reduced', rewards=[.5, .75], updates=10)
    bad = branch(tmp_path, 's0-t25', 'selection_reduced', rewards=[1., 1.], updates=10)
    path = bad / 'result.json'
    value = core.read(path)
    if damage in ('null', 'list', 'string'):
        value = {'null': None, 'list': [1], 'string': 'unfinished'}[damage]
    elif damage == 'empty-rewards':
        value['rewards'] = {}
    elif damage == 'list-rewards':
        value['rewards'] = [.5]
    elif damage == 'bad-key':
        value['rewards'] = {'not-a-question': .5}
    elif damage == 'nan':
        value['rewards'] = {'0': float('nan')}
    elif damage == 'bool':
        value['rewards'] = {'0': True}
    elif damage == 'wrong-schema':
        value['schema'] = 'other-experiment'
    elif damage == 'partial':
        value['complete'] = False
    if damage == 'nan':
        path.write_text(json.dumps(value))
    else:
        core.atomic_json(path, value)
    seal = bad / 'result.sha256.json'
    core.atomic_json(seal, {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if damage == 'unsealed':
        seal.unlink()
    elif damage == 'bad-seal':
        core.atomic_json(seal, {'sha256': 'wrong'})
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    text = result_text(tmp_path)
    assert 'random_reduced     reward= 62.50' in text
    assert 'selection_reduced  reward=  none' in text
    assert 'RAW_ENDPOINT unverified' in text
    assert 'selection_reduced-random_reduced=' not in text
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    results = importlib.import_module('switch_results')
    compact = results.compact_branch(bad, 25, 'selection_reduced')
    assert compact['reported_reward'] is None
    assert compact['result'] != 'sealed-metadata'


@pytest.mark.parametrize('filename,value', [
    ('decision.json', ['invalid']), ('execution.json', 'invalid'), ('failure.json', ['invalid']),
    ('policy/budget_stop.json', {'completed_steps': None}), ('curve.json', {'points': {'0': None}}),
    ('curve.json', {'points': []}), ('curve.json', ['invalid']),
])
def test_malformed_optional_arm_metadata_does_not_hide_endpoint(tmp_path, filename, value):
    directory = branch(tmp_path, 's0-t25', 'random_reduced', rewards=[.5], updates=10)
    core.atomic_json(directory / filename, value)
    assert 'random_reduced     reward= 50.00' in result_text(tmp_path)


@pytest.mark.parametrize('filename,value', [
    ('model.json', {'model': None}), ('model.json', {'ridge': {'coef': None}}),
    ('development-report.json', {'rows': [None]}), ('test-report.json', {'rows': None}),
])
def test_partial_root_metadata_does_not_hide_measured_branch(tmp_path, filename, value):
    branch(tmp_path, 's0-t25', 'random_reduced', rewards=[.5], updates=10)
    core.atomic_json(tmp_path / filename, value)
    assert 'random_reduced     reward= 50.00' in result_text(tmp_path)


def test_combined_txt_retains_valid_arm_when_sibling_json_is_null(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    exporter = importlib.import_module('mbpp_paper_results')
    root = tmp_path / 'suite'
    core.atomic_json(root / 'switch.json', {'schema': rule.SCHEMA, 'dataset': 'mbpp'})
    branch(root, 's0-t25', 'random_reduced', rewards=[.5], updates=10)
    broken = branch(root, 's0-t25', 'selection_reduced', rewards=[1.], updates=10)
    core.atomic_json(broken / 'result.json', None)
    output = tmp_path / 'mbpp-results.txt'
    monkeypatch.setattr(sys, 'argv', ['mbpp_paper_results.py', '--root', str(root), '--out', str(output)])
    exporter.main()
    text = output.read_text()
    assert 'random_reduced     reward= 50.00' in text
    assert 'RAW_ENDPOINT unverified null' in text
    assert json.loads(text.split('DATA_JSON\n')[1])['suites'][0]['status'] == 'exported'


def test_large_invalid_metadata_is_bounded_without_losing_good_rewards(tmp_path):
    branch(tmp_path, 's0-t25', 'random_reduced', rewards=[.5], updates=10)
    broken = branch(tmp_path, 's0-t25', 'selection_reduced', rewards=[1.], updates=10)
    for name in ('result.json', 'curve.json'):
        core.atomic_json(broken / name, {'invalid': 'x' * 2_000_000})
    core.atomic_json(tmp_path / 'development-report.json', {'rows': ['x' * 2_000_000]})
    text = result_text(tmp_path)
    assert 'random_reduced     reward= 50.00' in text
    assert text.count('[truncated; not a measured value]') == 3
    assert 'sha256=' in text and 'bytes=' in text
    assert len(text.encode()) < 20_000


def test_malformed_state_directory_does_not_block_valid_sibling(tmp_path):
    branch(tmp_path, 's0-t25', 'random_reduced', rewards=[.5], updates=10)
    (tmp_path / 'states/sbroken-t25').mkdir()
    text = result_text(tmp_path)
    assert 'random_reduced     reward= 50.00' in text
    assert 'UNVERIFIED STATE skipped malformed directory name: sbroken-t25' in text


def test_symlink_loop_endpoint_does_not_hide_saved_sibling(tmp_path):
    branch(tmp_path, 's0-t25', 'random_reduced', rewards=[.5], updates=10)
    broken = branch(tmp_path, 's0-t25', 'selection_reduced', rewards=[1.], updates=10)
    (broken / 'result.json').unlink()
    (broken / 'result.json').symlink_to('result.json')
    text = result_text(tmp_path)
    assert 'random_reduced     reward= 50.00' in text
    assert 'ENDPOINT unreadable:' in text


@pytest.mark.parametrize('authority', [False, True])
def test_multiple_points_require_authoritative_suite_binding(tmp_path, authority):
    directory = branch(tmp_path, 's0-t25', 'random_reduced', rewards=[.5], updates=10)
    point = directory.parent
    core.atomic_json(point / 'contract.json', {'identity': 'selected'})
    (point.parent / 'view-old').mkdir()
    if authority:
        core.atomic_json(point.parent.parent / 'suite.json', {
            'schema': 'offpolicy-selection-gate-gpu/one-shot-v1',
            'points': [{'name': point.name, 'sha256': hashlib.sha256((point / 'contract.json').read_bytes()).hexdigest()}]})
    text = result_text(tmp_path)
    if authority:
        assert 'random_reduced     reward= 50.00' in text
    else:
        assert 'random_reduced     reward=  none' in text
        assert 'state point unresolved' in text
