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


def layout(tmp_path):
    work=tmp_path/'group/minsoo3.kim/offpolicy-misranking'
    python=work/'.venv-cu126/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
    trainer=tmp_path/'user/minsoo3.kim/offpolicy-misranking'
    file(trainer/'srgc_rebuttal/srgc.py')
    bundle=tmp_path/'bundle';tools=bundle/'export-code'
    tools.mkdir(parents=True)
    file(bundle/'v8/experiments/cluster_child.py')
    file(bundle/'v8/experiments/configs/base.json','{}')
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
               EXPORT_CODE_ROOT='/existing/export',PAIR_PYTHON=sys.executable,SWITCH_PYTHON='/existing/qwen')
    result=environment(tools,env)
    assert all(result[k]==v for k,v in env.items())


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
    for name in ['run_v8_nodes.sh','v8_bootstrap.py']:shutil.copyfile(source/name,scripts/name)
    file(scripts/'srgc_v8_nodes.py',"import json,os,sys;print(json.dumps({'args':sys.argv[1:],'root':os.environ['V8_ROOT'],'code':os.environ['V8_CODE_ROOT'],'runner':os.environ['V8_RUNNER_ROOT']}))")
    result=subprocess.run(['sh',str(scripts/'run_v8_nodes.sh')],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==dict(args=[],root=str(work/'v8'),code=str(trainer),runner=str(bundle/'v8/experiments'))
