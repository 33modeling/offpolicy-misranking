"""Existing-environment startup; no GPU/model/network execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.v8_bootstrap import environment


def file(path,text='fixture'):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return path


def dispatcher_stub(scripts):
    """Process boundary double: echo the source-selection plan instead of claiming GPUs."""
    original=(scripts/'srgc_v8_nodes.py').read_text()
    file(scripts/'v8_dispatch.py',"import json,os,sys\np=json.load(open(sys.argv[1]))\n"
         "os.environ['V8_RUNNER_ROOT']=p['entries'][0]['runner']\nsys.argv=['fixture',*p['argv']]\n"+original)
    file(scripts/'v8_runtime_worker.py','# actor stub is not executed in this startup-only fixture\n')
    file(scripts/'v8_rpc.py','# IPC stub is not executed in this startup-only fixture\n')
    file(scripts/'v8_task_entry.py','# task entry stub\n')
    file(scripts/'v8_generation_probe.py','# decode observer stub\n')
    file(scripts/'v8_progress_text.py','def describe(stage,details):return str(stage)\n')
    file(scripts/'srgc_verifier_fallback.py','# frozen verifier stub\n')


def layout(tmp_path):
    work=tmp_path/'group/minsoo3.kim/offpolicy-misranking'
    python=work/'.venv-cu126/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
    trainer=tmp_path/'user/minsoo3.kim/offpolicy-misranking'
    file(trainer/'srgc_rebuttal/srgc.py')
    bundle=tmp_path/'bundle';tools=bundle/'export-code'
    tools.mkdir(parents=True)
    file(bundle/'v8/experiments/cluster_child.py')
    file(bundle/'v8/experiments/configs/base.json','{}')
    file(bundle/'v8/experiments/run_h100.sh')
    file(bundle/'v8/experiments/switching/cli.py')
    env=dict(GROUP_VOLUME=str(tmp_path/'group'),USER_VOLUME=str(tmp_path/'user'),PATH=os.environ['PATH'])
    return work,trainer,bundle,tools,env


def test_existing_default_environment_and_offline_bundle_need_no_exports(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path)
    before=sorted(str(p) for p in tmp_path.rglob('*'))
    result=environment(tools,env,cwd=tmp_path/'elsewhere')
    assert result['V8_ROOT']==str(work/'v8')
    assert result['V8_CODE_ROOT']==str(trainer)
    assert result['V8_RUNNER_ROOT']==str(bundle/'v8/experiments')
    assert result['V8_PYTHON']==str(work/'.venv-cu126/bin/python')
    assert result['EXPORT_CODE_ROOT']==str(tools)
    assert before==sorted(str(p) for p in tmp_path.rglob('*'))
    assert not (work/'v8').exists()


def test_existing_paths_and_model_environment_overrides_are_preserved(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path)
    env.update(V8_ROOT='/existing/v8',V8_CODE_ROOT='/existing/frozen',V8_RUNNER_ROOT='/existing/runner',
               EXPORT_CODE_ROOT='/existing/export',PAIR_PYTHON=sys.executable,SWITCH_PYTHON=sys.executable)
    result=environment(tools,env)
    assert all(result[k]==v for k,v in env.items())
    # the fallback for both cohorts is the existing venv; PAIR/SWITCH stay per-cohort choices
    assert result['V8_PYTHON']==str(work/'.venv-cu126/bin/python')


def test_source_receipt_finds_original_frozen_checkout_before_tools(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path)
    file(tools/'srgc_rebuttal/srgc.py','new tools must not replace original receipt')
    file(work/'v8/olmo_math/locator.json',json.dumps({'code':{'code_root':str(trainer)}}))
    assert environment(tools,env)['V8_CODE_ROOT']==str(trainer)


def test_bundle_inside_existing_code_tree_reuses_its_ancestor(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path)
    tools=trainer/'offline-v8/export-code';tools.mkdir(parents=True)
    env.pop('USER_VOLUME')
    assert environment(tools,env)['V8_CODE_ROOT']==str(trainer)


def test_explicit_missing_python_is_not_silently_replaced(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path);env['PAIR_PYTHON']='/missing/existing/python'
    with pytest.raises(ValueError,match='Existing Python'):environment(tools,env)


def test_corrupt_source_receipt_does_not_start_a_different_experiment(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path)
    file(work/'v8/olmo_math/locator.json','broken')
    with pytest.raises(ValueError):environment(tools,env)


def test_no_argument_shell_reuses_existing_environment_from_unrelated_directory(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path)
    scripts=tools/'scripts';scripts.mkdir()
    source=Path(__file__).resolve().parents[1]/'scripts'
    for name in ['run_v8_nodes.sh','v8_bootstrap.py','v8_queue_runtime.py','v8_supervisor.py','v8_processes.py','v8_probe.py','v8_control_output.py']:
        shutil.copyfile(source/name,scripts/name)
    file(scripts/'srgc_v8_nodes.py',"import json,os,sys;print(json.dumps({'args':sys.argv[1:],'root':os.environ['V8_ROOT'],'code':os.environ['V8_CODE_ROOT'],'runner':os.environ['V8_RUNNER_ROOT']}))")
    dispatcher_stub(scripts)
    result=subprocess.run(['sh',str(scripts/'run_v8_nodes.sh')],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=5)
    assert result.returncode==0,result.stderr
    actual=json.loads(result.stdout.splitlines()[-1])
    assert {k:v for k,v in actual.items() if k!='runner'}==dict(args=[],root=str(work/'v8'),code=str(trainer))
    assert (Path(actual['runner'])/'cluster_child.py').read_bytes()==(bundle/'v8/experiments/cluster_child.py').read_bytes()
    assert Path(actual['runner'])!=bundle/'v8/experiments'


def test_actual_code_checkout_contains_and_loads_its_runtime_without_a_paper_checkout(tmp_path):
    tools=Path(__file__).resolve().parents[1]
    expected=tools/'v8/experiments'
    for name in ('cluster_child.py','run_h100.sh','switching/cli.py','configs/base.json'):
        assert (expected/name).is_file(), 'the code checkout must ship its runtime, not require a second repository'
    env=environment(tools,{'PATH':os.environ['PATH'],'GROUP_VOLUME':str(tmp_path/'group')},cwd=tmp_path)
    assert env['V8_RUNNER_ROOT']==str(expected)
    result=subprocess.run([env['V8_PYTHON'],'-m','switching.cli','inspect-repo'],cwd=expected,
                          env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['code_root']==str(tools)


# ---- Claude debugging pass 2026-11-08 --------------------------------------------------------
def test_empty_exported_paths_mean_unset_like_the_shell_launchers(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path)
    env.update(OM_WORK='',V8_ROOT='',VENV_DIR='')
    result=environment(tools,env,cwd=tmp_path/'elsewhere')
    assert result['V8_ROOT']==str(work/'v8')        # before: V8_ROOT='' -> relative path refused later
    assert result['V8_PYTHON']==str(work/'.venv-cu126/bin/python')


def test_missing_cohort_interpreter_is_reported_at_startup(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path);env['SWITCH_PYTHON']='/missing/qwen/python'
    with pytest.raises(ValueError,match='SWITCH_PYTHON=/missing/qwen/python'):environment(tools,env)


def test_missing_venv_falls_back_with_a_visible_warning(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path);(work/'.venv-cu126/bin/python').unlink()
    messages=[];result=environment(tools,env,warn=messages.append)
    assert result['V8_PYTHON'] and messages and 'venv not found' in messages[0]


def test_shell_launcher_survives_cdpath_and_a_node_without_python3(tmp_path):
    work,trainer,bundle,tools,env=layout(tmp_path)
    scripts=tools/'scripts';scripts.mkdir()
    source=Path(__file__).resolve().parents[1]/'scripts'
    for name in ['run_v8_nodes.sh','v8_bootstrap.py','v8_queue_runtime.py','v8_supervisor.py','v8_processes.py','v8_probe.py','v8_control_output.py']:
        shutil.copyfile(source/name,scripts/name)
    file(scripts/'srgc_v8_nodes.py',"import json,os,sys;print(json.dumps({'python':sys.executable}))")
    dispatcher_stub(scripts)
    bindir=tmp_path/'bin';bindir.mkdir()
    for tool in ('dirname','sh','env','ps'):
        found=shutil.which(tool)
        if found:(bindir/tool).symlink_to(found)
    env.update(CDPATH=str(tmp_path),PATH=str(bindir))  # no python3 on PATH; CDPATH would print the dir
    result=subprocess.run(['/bin/sh',str(tools/'scripts/run_v8_nodes.sh')],cwd=tools,env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['python']


def test_shell_entries_never_write_bytecode_into_the_code_checkout(tmp_path):
    """Regression: v8_status.sh imported v8_probe and left scripts/__pycache__ in the checkout."""
    source=Path(__file__).resolve().parents[1]/'scripts'
    shells=sorted(p.name for p in source.glob('*v8*.sh'))
    assert {'run_v8_nodes.sh','v8_status.sh','v8_probe.sh','v8_stop.sh'}<=set(shells)
    for name in shells:
        assert 'export PYTHONDONTWRITEBYTECODE=1' in (source/name).read_text(),name
    copy=tmp_path/'checkout'/'scripts';copy.mkdir(parents=True)
    for path in source.glob('*.py'):shutil.copyfile(path,copy/path.name)
    for path in source.glob('*v8*.sh'):shutil.copyfile(path,copy/path.name)
    env={'PATH':os.environ['PATH'],'GROUP_VOLUME':str(tmp_path/'group'),'CDPATH':str(tmp_path)}
    for args in (['v8_status.sh'],['v8_probe.sh'],['run_v8_nodes.sh','--status']):
        subprocess.run(['sh',str(copy/args[0]),*args[1:]],cwd=tmp_path,env=env,capture_output=True,timeout=30)
    assert not list((tmp_path/'checkout').rglob('__pycache__'))
