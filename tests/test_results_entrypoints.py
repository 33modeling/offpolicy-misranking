"""Actual Bash results entrypoints stay CPU-only and replace one partial TXT."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from test_selector_pair_results import branch_fixture
from test_switch_results import branch
from test_rloo_experiment import fixture as rloo_fixture
from test_rloo_report import measured_policy

ROOT = Path(__file__).resolve().parents[1]


def source_bytes(work):
    return {str(path.relative_to(work)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in work.rglob('*') if path.is_file()}


@pytest.mark.parametrize('kind', ['pair', 'mbpp', 'rloo'])
@pytest.mark.parametrize('entrypoint', ['direct', 'common'])
def test_bash_results_refresh_one_default_partial_txt_without_touching_work(tmp_path, kind, entrypoint):
    work, home = tmp_path / 'work', tmp_path / 'home'
    work.mkdir()
    home.mkdir()
    pair = work / 'runs/selector-pair-v1'
    mbpp = work / 'runs/selection-switch-mbpp-quality-v1'
    rloo = work / 'runs/rloo-selector-v2'
    if kind == 'pair':
        branch_fixture(pair)
    elif kind == 'mbpp':
        mbpp.mkdir(parents=True)
        (mbpp / 'switch.json').write_text(json.dumps({
            'schema': 'offpolicy-selected-prefix-switch/v1', 'dataset': 'mbpp',
            'selector': 'fresh_r', 'accounting': 'matched', 'gate': 'convergence',
            'budget_gpu_seconds': 29040}))
        branch(mbpp, 's0-t25', 'selection_reduced', rewards=[.25, .75], updates=10)
    else:
        _, prepared, _ = rloo_fixture(work / 'inputs')
        out = rloo / 'math500-d0/s0'
        out.parent.mkdir(parents=True)
        prepared.rename(out)
        measured_policy(out, 'fresh_r', .75)
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    gpu_marker = tmp_path / 'gpu-command-called'
    gpu_command = binaries / 'nvidia-smi'
    gpu_command.write_text('#!/bin/sh\nprintf called > ' + shlex.quote(str(gpu_marker)) + '\nexit 99\n')
    gpu_command.chmod(0o755)
    python = binaries / 'cpu-python'
    python.write_text('#!/bin/sh\n[ -z "${CUDA_VISIBLE_DEVICES:-}" ] || exit 98\nexec '
                      + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o755)
    env = {**os.environ, 'HOME': str(home), 'OM_WORK': str(work),
           'PAIR_ROOT': str(pair), 'RLOO_ROOT': str(rloo),
           'PAIR_PYTHON': str(python), 'SWITCH_PYTHON': str(python), 'RLOO_PYTHON': str(python),
           'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'PYTHONDONTWRITEBYTECODE': '1',
           'PATH': str(binaries) + os.pathsep + os.environ['PATH']}
    for variable in ('SWITCH_MBPP_ROOT', 'SWITCH_MBPP_QUALITY_ROOT', 'SWITCH_MBPP_DIFFICULTY_ROOT',
                     'SWITCH_MBPP_LONG_ROOT', 'SWITCH_ROOT', 'OUT_ROOT'):
        env.pop(variable, None)
    script = {'pair': 'run_selector_pair_results.sh', 'mbpp': 'run_mbpp_experiments.sh', 'rloo': 'run_rloo.sh'}[kind]
    command = (['bash', 'scripts/run_paper_results.sh', 'results', kind] if entrypoint == 'common'
               else ['bash', f'scripts/{script}', *([] if kind == 'pair' else ['results'])])
    target = home / {'pair': 'selector-pair-results.txt', 'mbpp': 'mbpp-results.txt', 'rloo': 'rloo-results.txt'}[kind]
    before = source_bytes(work)
    peer = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        for attempt in range(2):
            if attempt:
                target.write_text('previous export must be replaced, not archived')
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
            assert list(home.glob('*.txt')) == [target]
            assert target.stat().st_size < 1_900_000
            assert not list(home.glob('*.tmp.*'))
            text = target.read_text()
            assert 'previous export must be replaced' not in text
            data = json.loads(text.split('DATA_JSON\n', 1)[1])
            if kind == 'pair':
                assert data['complete'] is False and data['rows'] == []
                assert data['branch_measurements'][0]['mean_reward'] == .5
            elif kind == 'mbpp':
                assert [item['status'] for item in data['suites']] == ['exported']
                assert 's0/t25' in text and 'reward= 50.00' in text
            else:
                assert data['complete'] is False and data['points'][0]['status'] == 'incomplete'
                assert data['points'][0]['rows'][0]['mean_reward'] == .75
            assert source_bytes(work) == before
            assert peer.poll() is None
            assert not gpu_marker.exists()
        assert not list(work.rglob('*.txt'))
    finally:
        peer.terminate()
        peer.wait(timeout=5)
