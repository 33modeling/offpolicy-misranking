"""Partial exports retain independently certified arms without relaxing contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_rloo_report import fixture, measured_policy, reporting


def matrix_point(tmp_path):
    _, prepared, _ = fixture(tmp_path)
    out = tmp_path / 'matrix/math500-d0/s0'
    out.parent.mkdir(parents=True)
    prepared.rename(out)
    return out


def evidence(root):
    return {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('damaged', [
    'fresh_r/evaluation/shard-0.jsonl',
    'fresh_r/evaluation/shard-0.done.json',
    'fresh_r/policy/optimizer.pt',
])
def test_damaged_arm_does_not_hide_certified_sibling(tmp_path, damaged):
    out = matrix_point(tmp_path)
    measured_policy(out, 'passrate_beta', .5)
    measured_policy(out, 'fresh_r', .75)
    (out / damaged).write_text('damaged')
    before = evidence(tmp_path)
    with pytest.raises((ValueError, OSError)):
        reporting.point_report(out)
    point = reporting.report(out.parent.parent)['points'][0]
    assert point['status'] == 'invalid' and point['invalid_arms'] == ['fresh_r']
    assert len(point['rows']) == 1
    assert point['rows'][0]['arm'] == 'passrate_beta'
    assert point['rows'][0]['mean_reward'] == .5
    bad = next(item for item in point['evaluations'] if item['arm'] == 'fresh_r')
    assert bad['status'] == 'invalid' and bad['error']
    assert bad['observed_mean_reward'] is None and bad['prompt_rewards'] == {}
    assert bad['completed_shards'] == [] and bad['missing_shards'] == list(range(4))
    assert evidence(tmp_path) == before


def test_invalid_reference_never_produces_comparison(tmp_path):
    out = matrix_point(tmp_path)
    for arm, reward in [('random', .25), ('passrate_beta', .5), ('fresh_r', .75)]:
        measured_policy(out, arm, reward)
    (out / 'passrate_beta/evaluation/shard-3.jsonl').write_text('damaged')
    point = reporting.report(out.parent.parent)['points'][0]
    assert point['invalid_arms'] == ['passrate_beta']
    assert {row['arm'] for row in point['rows']} == {'random', 'fresh_r'}
    for row in point['rows']:
        assert 'vs_passrate_beta' not in row and 'vs_before' not in row
        assert row['missing_references'] == ['before', 'passrate_beta']
    fresh = next(row for row in point['rows'] if row['arm'] == 'fresh_r')
    assert fresh['vs_random'] == {'mean': .5, 'lower': .5, 'upper': .5}


@pytest.mark.parametrize('damage', ['schema', 'scientific_code', 'shared_inputs'])
def test_partial_export_never_relaxes_shared_contract(tmp_path, damage):
    out = matrix_point(tmp_path)
    measured_policy(out, 'passrate_beta', .5)
    contract = reporting.experiment.ed.read(out / 'experiment.json')
    if damage == 'schema':
        contract['schema'] = 'invalid'
        reporting.experiment.ed.atomic_json(out / 'experiment.json', contract)
    elif damage == 'scientific_code':
        contract['code_hashes']['src/train_policy_rloo.py'] = '0' * 64
        reporting.experiment.ed.atomic_json(out / 'experiment.json', contract)
    else:
        (out / 'subsets/subset-fresh_r.json').write_text('damaged')
    with pytest.raises(ValueError):
        reporting.point_report(out, allow_partial=True)
    point = reporting.report(out.parent.parent)['points'][0]
    assert point['status'] == 'invalid' and point['rows'] == []
    assert point['error'] and 'evaluations' not in point


def test_bash_results_writes_one_bounded_txt_despite_invalid_arm(tmp_path):
    out = matrix_point(tmp_path)
    measured_policy(out, 'passrate_beta', .5)
    measured_policy(out, 'fresh_r', .75)
    (out / 'fresh_r/evaluation/shard-0.jsonl').write_text('damaged')
    (out / 'passrate_beta/cost.jsonl').write_text(json.dumps({'note': 'x' * 2_000_000}) + '\n')
    before = evidence(out.parent.parent)
    process = subprocess.run(['bash', 'scripts/run_rloo.sh', 'results'],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=30,
        env={**os.environ, 'HOME': str(tmp_path), 'RLOO_ROOT': str(out.parent.parent),
             'RLOO_PYTHON': sys.executable, 'CUDA_VISIBLE_DEVICES': ''})
    assert process.returncode == 1, process.stderr
    exports = list(tmp_path.glob('rloo-results*.txt'))
    assert len(exports) == 1 and exports[0].stat().st_size < 1_900_000
    text = exports[0].read_text()
    assert 'TXT saved:' in process.stdout and '0\t0\tinvalid\tpassrate_beta\t0.5\t' in text
    point = json.loads(text.split('DATA_JSON\n', 1)[1])['points'][0]
    assert point['invalid_arms'] == ['fresh_r'] and point['rows'][0]['mean_reward'] == .5
    cached = next(item for item in point['evaluations'] if item['arm'] == 'passrate_beta')
    assert cached['cost_ledger']['status'] == 'omitted_size_limit'
    assert evidence(out.parent.parent) == before
    assert not (out / 'results.json').exists()
