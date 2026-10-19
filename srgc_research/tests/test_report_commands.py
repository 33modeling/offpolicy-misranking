"""Public CPU report commands keep measured values and isolate requested scopes."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import srgc_sr_refresh as extras
from scripts import srgc_switch_validation_report as controls
from srgc_rebuttal.plan import digest, input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, identity
from srgc_rebuttal.tests import test_extra_arm_launch as extra_fixtures
from srgc_rebuttal.tests.test_cluster import write_inputs
from srgc_research.dispatch import olmo_results
from srgc_research.tests.test_submission_time import progress

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cohort(tmp_path):
    plan_path = write_inputs(tmp_path)
    plan = load_plan(plan_path)
    folder = tmp_path / 'runs/seed-5'
    folder.mkdir(parents=True)
    atomic_json(folder / 'run.json', identity(plan_path, plan, 5))
    progress(folder, 'on_policy')
    return plan_path, folder


def test_olmo_costs_are_read_only_and_use_retained_update_times(cohort, capsys):
    plan, folder = cohort
    before = {p: p.read_bytes() for p in plan.parent.rglob('*') if p.is_file()}
    assert olmo_results.main(['costs', '--plan', str(plan), '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    row = next(r for r in report['tasks'] if r['task'] == 'seed-5.on_policy')
    assert row['submission_timing']['gpu_seconds']['primary'] == 1400.
    assert row['submission_timing']['recorded_updates'] == 250
    assert {p: p.read_bytes() for p in plan.parent.rglob('*') if p.is_file()} == before
    assert not (folder.parent / 'results.json').exists()


def test_olmo_public_results_exports_without_loading_model_runtime(cohort, tmp_path):
    plan, folder = cohort
    destination = tmp_path / 'copy/results.txt'
    command = [sys.executable, 'scripts/run_srgc_rebuttal.py', 'results', '--plan', str(plan), '--output', str(destination)]
    result = subprocess.run(command, cwd=REPO, env={**os.environ, 'PAIR_PYTHON': '/missing/model-python'},
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert 'MATH FINAL REWARD (%)' in result.stdout and '0.389' in result.stdout
    assert 'EXCLUDED GPU TIME' in result.stdout
    assert 'RESULT:' not in result.stdout and 'Saved:' not in result.stdout
    assert destination.is_file() and (folder.parent / 'results.json').is_file()
    report = json.loads((folder.parent / 'results.json').read_text())
    assert report['submission_time_basis']['first_update'] == 26
    assert digest(plan) == json.loads((folder / 'run.json').read_text())['plan_sha256']
    assert input_path(plan, load_plan(plan), 5).is_file()


@pytest.fixture
def extra_fixture():
    fixture = extra_fixtures.ExtraArmLaunchTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def test_scoped_results_ignore_other_broken_arms_and_replicates(extra_fixture, capsys):
    fixture = extra_fixture
    plan = fixture.plan()
    path, value = fixture.endpoint(plan, cost=1.)
    (path.parent / 'sr_refresh-pool-endpoint.json').write_text(json.dumps({**value, 'arm': 'sr_refresh-pool'}))
    (path.parent / 'direction_removed-endpoint.json').write_text('unrelated invalid JSON')
    replicate = path.parent / 'replicate-99'
    replicate.mkdir()
    (replicate / 'replicate.json').write_text('unrelated invalid JSON')
    extras.results(SimpleNamespace(plan=plan, json=True, report_scope='pool'))
    report = json.loads(capsys.readouterr().out)
    assert report['errors'] == [] and report['replicates'] == []
    assert 'sr_refresh-pool' in report['rows'][0] and 'direction_removed' not in report['rows'][0]
    assert Path(report['collection']['report_path']).parts[-3:] == ('results', 'pool', 'results.json')
    assert (path.parent / 'direction_removed-endpoint.json').read_text() == 'unrelated invalid JSON'


@pytest.mark.parametrize('scope,excluded', [('timing', 'switch_single'), ('rules', 'switch_fixed50')])
def test_switch_scope_avoids_reading_unrelated_invalid_endpoints(extra_fixture, scope, excluded):
    from scripts.srgc_replicate_worker import Task

    fixture = extra_fixture
    plan = fixture.plan()
    folder, _, _ = fixture.prefix(plan)
    (folder / f'{excluded}-endpoint.json').write_text('unrelated invalid JSON')
    task = Task('math', 5, 0, 'switch_single', plan)
    with patch.object(controls, 'tasks_for', return_value=[task]) as tasks:
        report = controls.collect('math', scope=scope)
    tasks.assert_called_once_with('math', scope)
    assert report['errors'] == [] and excluded not in report['arms']
    assert report['rows'][0]['arms'] == {}


@pytest.mark.parametrize('runner,args', [
    ('run_srgc.sh', ['math', 'costs']),
    ('run_srgc_research.sh', ['math', 'n01', 'results']),
    ('run_srgc_sr_refresh.sh', ['math', 'timing_results']),
])
def test_read_only_shell_modes_fall_back_from_missing_model_python(tmp_path, runner, args):
    binary = tmp_path / 'bin'
    binary.mkdir()
    python = binary / 'python3'
    python.write_text('#!/bin/sh\nprintf "CPU_REPORT %s\\n" "$*"\n')
    python.chmod(0o755)
    env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
           'PAIR_PYTHON': '/missing/model-python', 'SWITCH_PYTHON': '/missing/model-python'}
    result = subprocess.run(['sh', str(REPO / 'scripts' / runner), *args], cwd='/tmp', env=env,
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith('CPU_REPORT ')


@pytest.mark.parametrize('scope', ['timing', 'rules'])
def test_scoped_controls_shell_dispatches_cpu_report_without_training(tmp_path, scope):
    python = tmp_path / 'python'
    python.write_text(f'#!{sys.executable}\nimport json,os,sys\nprint(json.dumps([sys.argv[1:], os.environ.get("CUDA_VISIBLE_DEVICES")]))\n')
    python.chmod(0o755)
    result = subprocess.run(['sh', str(REPO / 'scripts/run_srgc_sr_refresh.sh'), 'all', f'{scope}_json'],
                            env={**os.environ, 'PAIR_PYTHON': str(python)}, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    argv, cuda = json.loads(result.stdout)
    assert argv == ['scripts/srgc_switch_validation_report.py', '--dataset', 'all', '--scope', scope, '--json']
    assert cuda == ''


def test_extra_text_retains_repeated_transitions_and_stream_specific_contrasts():
    from srgc_research.report_text import extra

    output = extra({'dataset': 'math', 'scope': 'all', 'rows': [{
        'seed': 5, 'switch_repeat': {'reward_percent': 80., 'transitions': [
            {'step': 50, 'to': 'sr'}, {'step': 100, 'to': 'on_policy'}]},
        'switch': {'reward_percent': 82.}, 'sr': {'reward_percent': 80.}}], 'replicates': [{
            'seed': 5, 'replicate': 1, 'switch': {'reward_percent': 75.}, 'sr': {'reward_percent': 79.}}]})
    assert 'RECORDED SELECTOR TRANSITIONS' in output
    lines = [line.split() for line in output.splitlines()]
    assert ['5', '0', 'switch_repeat', '50', 'sr'] in lines
    assert ['5', '0', 'switch_repeat', '100', 'on_policy'] in lines
    assert ['5', '0', '2'] in lines and ['5', '1', '-4'] in lines
