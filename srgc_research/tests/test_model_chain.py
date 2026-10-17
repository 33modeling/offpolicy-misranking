"""Cross-model dispatch, isolated shell interpreters, and owned process cancellation."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from srgc_research.dispatch import model_chain as chain


def model_of(command):
    return Path(command[1]).stem.removeprefix('run_srgc_')


def test_real_child_processes_continue_llama_to_gemma_and_revisit_yielded_qwen(tmp_path):
    script = tmp_path / 'worker.py'
    script.write_text('''import json,os,sys
from pathlib import Path
root=Path(sys.argv[1]);model=sys.argv[2];path=root/(model+'.json')
assert os.environ['SRGC_MODEL_CHAIN_CHILD']=='1'
attempt=json.loads(path.read_text())['attempt']+1 if path.exists() else 1
path.write_text(json.dumps({'attempt':attempt,'model':model}))
raise SystemExit(76 if model=='qwen35' and attempt==1 else 0)
''')
    jobs, states, sleeps = [], [], []
    def runner(command, environment, heartbeat):
        model = model_of(command)
        jobs.append(model)
        return chain.run_child([sys.executable, str(script), str(tmp_path), model], environment, heartbeat)
    assert chain.drain(list(chain.MODELS), 'all', ['all'], os.environ,
                       lambda *a, **kw: states.append((a, kw)), runner=runner,
                       sleep=lambda seconds: sleeps.append(seconds)) == 0
    assert jobs == ['qwen35', 'llama31', 'gemma4', 'qwen35']
    assert len(sleeps) == 1
    assert states[-1][0] == ('finished',)
    assert all(value=='finished' for value in states[-1][1]['models'].values())


def test_llama_first_rotates_to_gemma_before_qwen():
    order = chain.order_for('llama31')
    jobs = []
    assert order == ['llama31', 'gemma4', 'qwen35']
    assert chain.drain(order, 'mbpp', ['mbpp'], {}, lambda *a, **kw: None,
                       runner=lambda cmd, env, beat: jobs.append(model_of(cmd)) or 0,
                       sleep=lambda n: pytest.fail('completed chain slept')) == 0
    assert jobs == order


@pytest.mark.parametrize('order', ['llama31,llama31', 'qwen35,unknown', '', 'qwen35,gemma4'])
def test_invalid_orders_do_not_launch_jobs(order):
    if order == '':
        assert chain.order_for('llama31', order) == ['llama31','gemma4','qwen35']
    else:
        with pytest.raises(ValueError):
            chain.order_for('llama31', order)


def test_yield_preserves_initial_root_and_flags_and_does_not_spend_failure_budget():
    jobs = []
    initial = ['math','run','--root','/group/custom-llama','--max-attempts','1']
    counts = {key:0 for key in chain.MODELS}
    def runner(command, environment, heartbeat):
        model = model_of(command);counts[model]+=1;jobs.append(command)
        return chain.YIELD if model=='llama31' and counts[model]==1 else 0
    assert chain.drain(chain.order_for('llama31'), 'math', initial, {}, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: None) == 0
    assert jobs[0][2:] == initial
    assert jobs[-1][2:] == initial
    assert all(command[2:]==['math'] for command in jobs[1:3])


def test_failed_model_does_not_hide_gemma_and_retains_failure_exit():
    jobs=[];states=[]
    def runner(command, environment, heartbeat):
        model=model_of(command);jobs.append(model)
        return 2 if model=='llama31' else 0
    assert chain.drain(chain.order_for('llama31'),'all',['all'],{},
                       lambda *a,**kw:states.append((a,kw)),runner=runner)==2
    assert jobs==['llama31','gemma4','qwen35']
    assert states[-1][1]['failures']=={'llama31':2}


@pytest.mark.parametrize('code',[130,137,143])
def test_interruption_stops_chain_without_starting_another_model(code):
    calls=[]
    assert chain.drain(list(chain.MODELS),'all',['all'],{},lambda *a,**kw:None,
                       runner=lambda cmd,*a:calls.append(cmd) or code)==code
    assert len(calls)==1


def test_qwen_resume_clears_only_requested_stop_then_runs_the_same_root():
    jobs=[]
    initial=['math','resume','--root','/group/custom-qwen']
    assert chain.drain(list(chain.MODELS),'math',initial,{},lambda *a,**kw:None,
                       runner=lambda cmd,*a:jobs.append(cmd) or 0)==0
    assert jobs[0][2:]==initial
    assert jobs[1][2:]==['math','run','--root','/group/custom-qwen']
    assert jobs[2][2:]==['math']


@pytest.mark.parametrize('model',chain.MODELS)
@pytest.mark.parametrize('args',[[],['all'],['math','run'],['mbpp','resume']])
def test_public_training_shells_join_chain_and_keep_starting_model(model,args,tmp_path):
    executable=tmp_path/'python'
    executable.write_text('#!/bin/sh\nexec '+sys.executable+' -c \'import json,sys;print(json.dumps(sys.argv[1:]))\' "$@"\n')
    executable.chmod(0o755)
    name={'qwen35':'QWEN','llama31':'LLAMA','gemma4':'GEMMA'}[model]
    env={**os.environ,name+'_PYTHON':str(executable)}
    env.pop('SRGC_MODEL_CHAIN_CHILD',None);env.pop('SRGC_MODEL_CHAIN',None)
    result=subprocess.run(chain.child_command(model,'all',args),env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==['-m','srgc_research.dispatch.model_chain','--start',model,*(args or ['all'])]


def test_chain_signal_forwards_to_owned_child_and_preserves_unrelated_process(tmp_path):
    ready=tmp_path/'ready'
    child_code='import pathlib,sys,time;pathlib.Path(sys.argv[1]).touch();time.sleep(60)'
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    try:
        def heartbeat(pid):
            if ready.exists():
                ready.unlink();os.kill(os.getpid(),signal.SIGTERM)
        assert chain.run_child([sys.executable,'-c',child_code,str(ready)],os.environ,heartbeat)==143
        assert unrelated.poll() is None
    finally:
        unrelated.terminate();unrelated.wait(timeout=5)


def test_auto_llama_handoff_does_not_invoke_manual_gpu_cleanup(monkeypatch):
    from srgc_rebuttal import cluster
    from srgc_research.dispatch import llama_run
    from srgc_research.dispatch.llama31 import cli
    monkeypatch.setenv('SRGC_MODEL_CHAIN_CHILD','1')
    original=cluster.gpu_identity
    with patch.object(llama_run,'clean_start',side_effect=AssertionError('auto handoff must not evict jobs')), \
            patch.object(cluster,'gpu_identity',return_value=('0,1,2,3',('a','b','c','d'))), \
            patch.object(cli,'main',side_effect=lambda args: cluster.gpu_identity() and 0):
        assert llama_run.main(['all','run'])==0
    assert cluster.gpu_identity is original


def test_status_reports_active_gemma_and_marks_old_heartbeat_stale(tmp_path):
    from srgc_rebuttal.runtime import atomic_json
    root=tmp_path/'work/srgc-rebuttal/.model-chain/workers'
    environment={'GROUP_VOLUME':str(tmp_path),'OM_WORK':str(tmp_path/'work')}
    row={'worker_id':'abc','host':'node-a','pid':123,'started':10,'heartbeat':100,
         'status':'running','model':'gemma4','order':['qwen35','llama31','gemma4']}
    atomic_json(root/'abc.json',row)
    before=(root/'abc.json').read_bytes()
    assert chain.snapshot(environment,now=105)[0]['model']=='gemma4'
    assert 'RUNNING model=gemma4' in '\n'.join(chain.render_status(environment,now=105))
    assert chain.snapshot(environment,now=140)[0]['status']=='stale'
    assert (root/'abc.json').read_bytes()==before
    atomic_json(root/'bad.json',{'heartbeat':'bad'})
    assert len(chain.snapshot(environment,now=105))==1


def test_parent_waiting_sigterm_records_stop_and_restores_handler(tmp_path,monkeypatch):
    monkeypatch.setenv('GROUP_VOLUME',str(tmp_path))
    monkeypatch.setenv('OM_WORK',str(tmp_path/'work'))
    def terminate(*args,**kwargs):
        os.kill(os.getpid(),signal.SIGTERM)
    previous=signal.getsignal(signal.SIGTERM)
    with patch.object(chain,'drain',side_effect=terminate):
        assert chain.main(['--start','llama31','all'])==143
    assert signal.getsignal(signal.SIGTERM)==previous
    rows=list((tmp_path/'work/srgc-rebuttal/.model-chain/workers').glob('*.json'))
    row=json.loads(rows[0].read_text());assert row['status']=='stopped';assert row['exit_code']==143
