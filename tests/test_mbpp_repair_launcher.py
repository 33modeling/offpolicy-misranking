"""The repair launcher dispatches only its separate root; no GPU execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    scripts = tmp_path / 'repo with spaces/scripts'
    scripts.mkdir(parents=True)
    shutil.copy(ROOT / 'scripts/run_mbpp_repair.sh', scripts)
    (scripts / 'mbpp_controller_identity.py').write_text(
        'import os, sys\n'
        'print(os.environ.get("TEST_OWNER", ""))\n'
        'sys.exit(int(os.environ.get("TEST_OWNER_RC", "0")))\n')
    (scripts / 'mbpp_repair.py').write_text(
        'import json, os, sys\n'
        'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
        'with open(os.environ["PREPARE_LOG"], "a") as f:\n'
        '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'sys.exit(int(os.environ.get("TEST_PREPARE_RC", "0")))\n')
    (scripts / 'mbpp_repair_results.py').write_text(
        'import json, os, sys\n'
        'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
        'with open(os.environ["RESULTS_LOG"], "a") as f:\n'
        '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'sys.exit(int(os.environ.get("TEST_RESULTS_RC", "0")))\n')
    (scripts / 'run_mbpp_experiments.sh').write_text(
        '#!/usr/bin/env bash\nexec "$SWITCH_PYTHON" - "$@" <<\'PY\'\n'
        'import json, os, sys\n'
        'with open(os.environ["DISPATCH_LOG"], "a") as f:\n'
        '    f.write(json.dumps({"args": sys.argv[1:], "env": dict(os.environ)}) + "\\n")\n'
        'sys.exit(int(os.environ.get("TEST_DISPATCH_RC", "0")))\nPY\n')
    env = {**os.environ, 'OM_WORK': str(tmp_path / 'shared work'),
           'SWITCH_PYTHON': sys.executable, 'CUDA_VISIBLE_DEVICES': '0,1,2,3',
           'PREPARE_LOG': str(tmp_path / 'prepare.jsonl'),
           'RESULTS_LOG': str(tmp_path / 'results.jsonl'),
           'DISPATCH_LOG': str(tmp_path / 'dispatch.jsonl')}

    def run(*args, **overrides):
        return subprocess.run(['bash', str(scripts / 'run_mbpp_repair.sh'), *args],
                              cwd=tmp_path, env={**env, **overrides},
                              capture_output=True, text=True, timeout=10)

    return run, env


@pytest.mark.parametrize('mode', [(), ('run',), ('restart',)])
def test_start_prepares_separate_root_then_dispatches_existing_gpu_launcher(launcher, mode):
    run, env = launcher
    result = run(*mode, OUT_ROOT='/wrong/math', SWITCH_ROOT='/wrong/old-root',
                 SWITCH_RUNTIME_REPO='/wrong/runtime')
    assert result.returncode == 0, result.stdout + result.stderr
    source = str(Path(env['OM_WORK']) / 'runs/selection-switch-mbpp-quality-v1')
    target = str(Path(env['OM_WORK']) / 'runs/selection-switch-mbpp-quality-repair-v1')
    assert json.loads(Path(env['PREPARE_LOG']).read_text()) == [
        'prepare', '--source', source, '--root', target]
    dispatched = json.loads(Path(env['DISPATCH_LOG']).read_text())
    assert dispatched['args'] == [mode[0] if mode else 'run', 'quality']
    assert dispatched['env']['SWITCH_MBPP_QUALITY_ROOT'] == target
    assert dispatched['env']['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    assert all(key not in dispatched['env'] for key in ('OUT_ROOT', 'SWITCH_ROOT', 'SWITCH_RUNTIME_REPO'))


@pytest.mark.parametrize('mode', ['status', 'logs', 'stop', 'why'])
def test_readers_and_stop_do_not_prepare_or_train(launcher, mode):
    run, env = launcher
    result = run(mode)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(env['PREPARE_LOG']).exists()
    dispatched = json.loads(Path(env['DISPATCH_LOG']).read_text())
    assert dispatched['args'] == [mode, 'quality']


@pytest.mark.parametrize('rc', ['0', '1'])
def test_results_uses_separate_attempt_cost_export_without_preparing_or_dispatching(launcher, rc):
    run, env = launcher
    result = run('results', TEST_RESULTS_RC=rc)
    assert result.returncode == int(rc)
    assert not Path(env['PREPARE_LOG']).exists()
    assert not Path(env['DISPATCH_LOG']).exists()
    target = str(Path(env['OM_WORK']) / 'runs/selection-switch-mbpp-quality-repair-v1')
    assert json.loads(Path(env['RESULTS_LOG']).read_text()) == ['--root', target]


def test_status_forwards_only_viewer_options(launcher):
    run, env = launcher
    assert run('status', '--watch', '2').returncode == 0
    assert json.loads(Path(env['DISPATCH_LOG']).read_text())['args'] == ['status', 'quality', '--watch', '2']


@pytest.mark.parametrize('rc', ['1', '2', '75', '80'])
def test_failed_preparation_never_dispatches_gpu_work(launcher, rc):
    run, env = launcher
    assert run(TEST_PREPARE_RC=rc).returncode == int(rc)
    assert Path(env['PREPARE_LOG']).exists()
    assert not Path(env['DISPATCH_LOG']).exists()


@pytest.mark.parametrize('mode', ['run', 'restart', 'stop', 'logs'])
def test_other_root_controller_is_never_restarted_or_relabelled(launcher, mode):
    run, env = launcher
    result = run(mode, TEST_OWNER=str(os.getpid()))
    assert result.returncode == 75
    assert 'original controller was not stopped' in result.stderr
    assert not Path(env['PREPARE_LOG']).exists()
    assert not Path(env['DISPATCH_LOG']).exists()


def test_ambiguous_allocation_is_not_overridden(launcher):
    run, env = launcher
    assert run(TEST_OWNER_RC='75').returncode == 75
    assert not Path(env['PREPARE_LOG']).exists()
    assert not Path(env['DISPATCH_LOG']).exists()


def test_existing_repair_controller_can_receive_explicit_restart(launcher):
    run, env = launcher
    target = str(Path(env['OM_WORK']) / 'runs/selection-switch-mbpp-quality-repair-v1')
    owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                             env={**env, 'SWITCH_ROOT': target})
    try:
        result = run('restart', TEST_OWNER=str(owner.pid))
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(Path(env['DISPATCH_LOG']).read_text())['args'] == ['restart', 'quality']
        assert owner.poll() is None
    finally:
        owner.terminate()
        owner.wait(timeout=5)


@pytest.mark.parametrize('relation', ['same', 'child', 'parent'])
def test_source_and_target_must_not_overlap(launcher, relation):
    run, env = launcher
    source = Path(env['OM_WORK']) / 'runs/selection-switch-mbpp-quality-v1'
    target = {'same': source, 'child': source / 'repair', 'parent': source.parent}[relation]
    assert run(MBPP_REPAIR_ROOT=str(target)).returncode == 2
    assert not Path(env['PREPARE_LOG']).exists()
    assert not Path(env['DISPATCH_LOG']).exists()


def test_controller_exit_status_is_preserved(launcher):
    run, env = launcher
    assert run(TEST_DISPATCH_RC='80').returncode == 80


def test_bash_syntax():
    subprocess.run(['bash', '-n', str(ROOT / 'scripts/run_mbpp_repair.sh')], check=True)
