"""No-GPU contracts for the six-point MBPP calibration follow-up."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import mbpp_offpolicy_followup as followup

REPO = Path(__file__).resolve().parents[1]
TAG = 'test-matrix'


@pytest.fixture
def matrix(tmp_path):
    root = tmp_path / 'matrix'
    for seed, drift, run in followup.points(root, TAG):
        run.mkdir(parents=True)
        (run / 'DONE').write_text('done\n')
        for name in followup.INPUTS:
            (run / name).write_text('{}')
        (run / 'run_config.json').write_text(json.dumps(dict(
            dataset='mbpp', seed=seed, drift=drift, prompt_format='olmo_rlzero_math')))
        (run / 'scores_oracle.json').write_text(json.dumps({str(i): {'score': 0} for i in range(512)}))
        (run / 'rollouts_behavior_train.jsonl').write_text(''.join(
            json.dumps(dict(prompt_idx=i, rollout_idx=j, reward=j % 2)) + '\n'
            for i in range(512) for j in range(8)))
        if drift:
            adapter = run / f'policy_step_{drift}'
            adapter.mkdir()
            (adapter / 'adapter_config.json').write_text('{}')
            (adapter / 'adapter_model.safetensors').write_bytes(b'test adapter')
    return root


def completed(run):
    config = followup.read_json(run / 'run_config.json')
    scores = {est: {str(i): {'a': i / 1024, 'b': (i % 231) / 1024} for i in range(512)}
              for est in followup.ESTIMATORS}
    (run / 'scores_stale_splithalf.json').write_text(json.dumps(scores))
    (run / 'scores_stale_splithalf.protocol.json').write_text(json.dumps({
        'schema': 'offpolicy-stale-splithalf/v1', 'prompts': 512, 'shards': 4,
        'parameters': {'proj_dim': 4096, 'grad_layers': 4, 'clip_cap': 10., 'micro_batch': 2,
                       'adapter': str(run / 'policy_step_400') if config['drift'] else None},
        'full_score_check': {est: {'prompts': 16, 'max_abs_difference': 0} for est in followup.ESTIMATORS}}))


def test_fixed_scope_and_read_only_status(matrix):
    before = sorted(str(p) for p in matrix.rglob('*'))
    data = followup.collect(matrix, TAG)
    assert data['completed_points'] == 0
    assert len(data['points']) == 6
    assert sorted(str(p) for p in matrix.rglob('*')) == before
    followup.prepare(matrix, TAG)
    assert {r['status'] for r in followup.collect(matrix, TAG)['points']} == {'READY'}


def test_rerun_prepare_is_idempotent(matrix):
    followup.prepare(matrix, TAG)
    paths = [run / followup.BINDING for _, _, run in followup.points(matrix, TAG)]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    followup.prepare(matrix, TAG)
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == before


@pytest.mark.parametrize('problem', ['wrong_dataset', 'missing_adapter', 'not_done', 'missing_validation',
                                    'missing_response', 'duplicate_response'])
def test_preflight_blocks_before_creating_bindings(matrix, problem):
    run = followup.points(matrix, TAG)[-1][2]
    if problem == 'wrong_dataset':
        config = followup.read_json(run / 'run_config.json')
        config['dataset'] = 'math500'
        (run / 'run_config.json').write_text(json.dumps(config))
    elif problem == 'missing_adapter':
        (run / 'policy_step_400/adapter_model.safetensors').unlink()
    elif problem == 'not_done':
        (run / 'DONE').unlink()
    elif problem == 'missing_validation':
        (run / 'val_groups.pt').unlink()
    else:
        path = run / 'rollouts_behavior_train.jsonl'
        lines = path.read_text().splitlines(keepends=True)
        path.write_text(''.join(lines[:-1] if problem == 'missing_response' else lines + [lines[0]]))
    with pytest.raises((ValueError, OSError)):
        followup.prepare(matrix, TAG)
    assert not list(matrix.rglob(followup.BINDING))


def test_existing_unbound_scores_preserved(matrix):
    run = followup.points(matrix, TAG)[0][2]
    completed(run)
    before = (run / 'scores_stale_splithalf.json').read_bytes()
    with pytest.raises(ValueError, match='unbound'):
        followup.prepare(matrix, TAG)
    assert (run / 'scores_stale_splithalf.json').read_bytes() == before


def test_changed_inputs_not_reused(matrix):
    followup.prepare(matrix, TAG)
    run = followup.points(matrix, TAG)[0][2]
    completed(run)
    (run / 'val_groups.pt').write_bytes(b'changed')
    row = followup.collect(matrix, TAG)['points'][0]
    assert row['status'] == 'BLOCKED' and 'inputs changed' in row['error']


def test_active_point_never_claimed_complete(matrix):
    followup.prepare(matrix, TAG)
    run = followup.points(matrix, TAG)[0][2]
    completed(run)
    with (run / '.stale-splithalf.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert followup.collect(matrix, TAG)['points'][0]['status'] == 'RUN'
    assert followup.collect(matrix, TAG)['points'][0]['status'] == 'DONE'


def test_fifo_lock_does_not_hang_status(matrix):
    followup.prepare(matrix, TAG)
    run = followup.points(matrix, TAG)[0][2]
    os.mkfifo(run / '.stale-splithalf.lock')
    record = followup.collect(matrix, TAG)['points'][0]
    assert record['status'] == 'BLOCKED' and 'regular lock' in record['error']


@pytest.mark.parametrize('problem', ['missing_protocol', 'nan', 'missing_candidate', 'missing_check', 'wrong_count'])
def test_invalid_scores_are_not_done(matrix, problem):
    followup.prepare(matrix, TAG)
    run = followup.points(matrix, TAG)[0][2]
    completed(run)
    path = run / ('scores_stale_splithalf.json' if problem in ('nan', 'missing_candidate')
                  else 'scores_stale_splithalf.protocol.json')
    data = followup.read_json(path)
    if problem == 'missing_protocol':
        path.unlink()
    else:
        if problem == 'nan':
            data['g11']['0']['a'] = float('nan')
        elif problem == 'missing_candidate':
            del data['g00']['1']
        elif problem == 'missing_check':
            data['full_score_check'] = {}
        else:
            data['prompts'] = 400
        path.write_text(json.dumps(data))
    assert followup.collect(matrix, TAG)['points'][0]['status'] == 'BLOCKED'


def test_partial_export_keeps_good_data_and_overwrites_stale_file(matrix, tmp_path):
    followup.prepare(matrix, TAG)
    completed(followup.points(matrix, TAG)[0][2])
    data = followup.collect(matrix, TAG, include_scores=True)
    assert data['completed_points'] == 1
    assert len(data['points'][0]['rows']) == 4
    assert data['points'][0]['rows'][0]['n'] == 512
    assert data['points'][0]['rows'][0]['k'] == 51
    target = tmp_path / 'results.txt'
    target.write_text('old content')
    followup.export(data, target)
    raw = json.loads(target.read_text().split('DATA_JSON\n')[1])
    assert raw['completed_points'] == 1
    assert len(raw['points'][0]['half_scores']['g11']) == 512


def test_real_bash_completed_run_and_results_need_no_gpu(matrix, tmp_path):
    followup.prepare(matrix, TAG)
    for _, _, run in followup.points(matrix, TAG):
        completed(run)
    env = dict(os.environ, HOME=str(tmp_path), OM_WORK=str(tmp_path / 'work'),
               OM_OLMO3_ROOT=str(matrix), OM_OLMO3_MODEL_TAG=TAG,
               VENV_DIR=str(Path(sys.executable).parent.parent), CUDA_VISIBLE_DEVICES='')
    for mode in ('status', 'results', 'run'):
        result = subprocess.run(['bash', 'scripts/run_mbpp_offpolicy.sh', mode], cwd=REPO,
                                env=env, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert '6/6 complete' in result.stdout
    exported = json.loads((tmp_path / 'mbpp-offpolicy-results.txt').read_text().split('DATA_JSON\n')[1])
    assert sum(len(p['rows']) for p in exported['points']) == 24


def test_bash_missing_sources_writes_fresh_partial_export(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path), OM_WORK=str(tmp_path / 'work'),
               OM_OLMO3_ROOT=str(tmp_path / 'absent'), OM_OLMO3_MODEL_TAG=TAG,
               VENV_DIR=str(Path(sys.executable).parent.parent))
    target = tmp_path / 'mbpp-offpolicy-results.txt'
    target.write_text('stale')
    result = subprocess.run(['bash', 'scripts/run_mbpp_offpolicy.sh', 'results'], cwd=REPO,
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 1
    assert '0/6 complete' in target.read_text() and 'stale' not in target.read_text()


def test_bash_syntax_and_no_cross_experiment_cleanup():
    for name in ('run_mbpp_offpolicy.sh', 'run_stale_splithalf.sh'):
        subprocess.run(['bash', '-n', str(REPO / 'scripts' / name)], check=True)
    source = (REPO / 'scripts/run_stale_splithalf.sh').read_text()
    assert 'e5_cleanup_previous' not in source
    assert 'export E5_FORCE=0' in source
    assert '8>&- 9>&-' not in source


def launcher_fixture(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    (repo / 'src').mkdir()
    shutil.copy(REPO / 'scripts/run_stale_splithalf.sh', repo / 'scripts/run_stale_splithalf.sh')
    (repo / 'scripts/setup_env.sh').write_text('export PYTHONPATH="$PWD/src"\n')
    (repo / 'scripts/_lease.sh').write_text('lease_note() { :; }\n')
    (repo / 'scripts/_e5_node.sh').write_text(
        'e5_acquire_node() { exec 8>"$TEST_NODE_LOCK"; flock -n 8 || return 75; }\n')
    (repo / 'src/stale_splithalf.py').write_text(
        'import argparse,json,os,pathlib\n'
        'p=argparse.ArgumentParser();p.add_argument("--run");p.add_argument("--shard");'
        'p.add_argument("--shards");p.add_argument("--check-full");'
        'p.add_argument("--merge",action="store_true");a=p.parse_args()\n'
        'r=pathlib.Path(a.run)\n'
        'if a.merge: (r/"scores_stale_splithalf.json").write_text("{}")\n'
        'else:\n'
        ' os.fstat(8);os.fstat(9)\n'
        ' (r/("worker-"+a.shard+".json")).write_text(json.dumps(dict('
        'gpu=os.environ["CUDA_VISIBLE_DEVICES"],format=os.environ["OM_PROMPT_FORMAT"])))\n')
    root = tmp_path / 'matrix'
    run = root / 'family-mbpp-s0' / f'{TAG}-s0-mbpp-d0'
    run.mkdir(parents=True)
    (run / 'DONE').write_text('done')
    (run / 'run_config.json').write_text(json.dumps(dict(attn='eager', lora_targets='q_proj', prompt_format='source-format')))
    env = dict(os.environ, OM_WORK=str(tmp_path), OM_OLMO3_ROOT=str(root), OM_OLMO3_MODEL_TAG=TAG,
               VENV_DIR=str(Path(sys.executable).parent.parent), E5_SEEDS='0', CUDA_VISIBLE_DEVICES='3,2,1,0',
               TEST_NODE_LOCK=str(tmp_path / 'node.lock'))
    return repo, run, env


def test_real_scoring_launcher_routes_mbpp_and_keeps_worker_locks(tmp_path):
    repo, run, env = launcher_fixture(tmp_path)
    result = subprocess.run(['bash', 'scripts/run_stale_splithalf.sh', 'mbpp', 'd0'],
                            cwd=repo, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / 'scores_stale_splithalf.json').is_file()
    workers = [json.loads((run / f'worker-{i}.json').read_text()) for i in range(4)]
    assert [r['gpu'] for r in workers] == ['3', '2', '1', '0']
    assert {r['format'] for r in workers} == {'source-format'}


def test_real_duplicate_launch_preserves_node_owner(tmp_path):
    repo, run, env = launcher_fixture(tmp_path)
    with Path(env['TEST_NODE_LOCK']).open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        result = subprocess.run(['bash', 'scripts/run_stale_splithalf.sh', 'mbpp', 'd0'],
                                cwd=repo, env=env, text=True, capture_output=True, timeout=10)
        assert result.returncode == 75, result.stdout + result.stderr
        assert not list(run.glob('worker-*.json'))
