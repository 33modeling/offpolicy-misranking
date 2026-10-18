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
    jobs, states, sleeps, phases = [], [], [], []
    def runner(command, environment, heartbeat):
        model = model_of(command)
        jobs.append(model)
        phases.append(environment['SRGC_DATASET_PHASE'])
        return chain.run_child([sys.executable, str(script), str(tmp_path), model], environment, heartbeat)
    assert chain.drain(list(chain.MODELS), 'all', ['all'], os.environ,
                       lambda *a, **kw: states.append((a, kw)), runner=runner,
                       sleep=lambda seconds: sleeps.append(seconds)) == 0
    assert jobs == ['qwen35', 'llama31', 'gemma4', 'qwen35', 'qwen35', 'llama31', 'gemma4']
    assert phases == ['math'] * 4 + ['mbpp'] * 3
    assert len(sleeps) == 1
    assert states[-1][0] == ('finished',)
    assert all(value=='finished' for value in states[-1][1]['models'].values())


def test_llama_first_rotates_to_gemma_before_qwen():
    order = chain.order_for('llama31')
    jobs = []
    assert order == ['llama31', 'gemma4', 'qwen35']
    assert chain.drain(order, 'mbpp', ['mbpp'], {}, lambda *a, **kw: None,
                       runner=lambda cmd, env, beat: jobs.append((model_of(cmd), env['SRGC_DATASET_PHASE'])) or 0,
                       sleep=lambda n: pytest.fail('completed chain slept')) == 0
    assert jobs == [(model, phase) for phase in ('math', 'mbpp') for model in order]


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


def test_math_failures_block_mbpp_after_other_models_finish():
    jobs = []
    def runner(command, environment, heartbeat):
        model = model_of(command)
        jobs.append((model, environment['SRGC_DATASET_PHASE']))
        return 2 if model == 'llama31' else 0
    assert chain.drain(list(chain.MODELS), 'all', ['all'], {}, lambda *a, **kw: None, runner=runner) == 2
    assert jobs == [(model, 'math') for model in chain.MODELS]


@pytest.mark.parametrize('start', chain.MODELS)
@pytest.mark.parametrize('failure_code', (1, 2, 124, 134, 139))
def test_failed_saved_model_retries_past_four_errors_before_any_fresh_model(monkeypatch, start, failure_code):
    from srgc_research.dispatch import resume_inventory

    pending = {'llama31': 'unchanged-checkpoint'}
    monkeypatch.setattr(resume_inventory, 'signature', lambda model, *a, **kw: pending.get(model))
    jobs, waits, states = [], [], []

    def runner(command, environment, heartbeat):
        model, recovery = model_of(command), environment['SRGC_RESUME_ONLY'] == '1'
        jobs.append((model, environment['SRGC_DATASET_PHASE'], recovery))
        if recovery and len(jobs) <= 5:
            return failure_code
        if recovery:
            pending.pop(model)
        else:
            assert not pending, 'fresh work started before ERROR recovery finished'
        return 0

    assert chain.drain(chain.order_for(start), 'all', ['all'], {},
                       lambda *a, **kw: states.append((a, kw)), runner=runner,
                       sleep=lambda n: waits.append(n)) == 0
    assert jobs[:6] == [('llama31', 'math', True)] * 6
    assert len(waits) == 5
    assert all(details['model'] == 'llama31' for (status,), details in states if status == 'waiting')
    assert states[-1][1]['failures'] == {}


def test_new_task_error_restarts_same_model_in_recovery_mode_before_fresh_work(monkeypatch):
    from srgc_research.dispatch import resume_inventory

    pending, jobs = {}, []
    monkeypatch.setattr(resume_inventory, 'signature', lambda model, *a, **kw: pending.get(model))

    def runner(command, environment, heartbeat):
        model, recovery = model_of(command), environment['SRGC_RESUME_ONLY'] == '1'
        jobs.append((model, recovery))
        if len(jobs) == 1:
            pending[model] = 'saved-error'
            return 1
        if recovery:
            pending.pop(model)
        else:
            assert not pending
        return 0

    assert chain.drain(list(chain.MODELS), 'math', ['math'], {}, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: None) == 0
    assert jobs[:2] == [('qwen35', False), ('qwen35', True)]


def test_real_error_child_is_restarted_and_keeps_checkpoint_bytes(tmp_path, monkeypatch):
    from srgc_research.dispatch import resume_inventory

    checkpoint = tmp_path / 'sr-latest.pt'
    checkpoint.write_bytes(b'last saved optimizer and selector state')
    receipt = tmp_path / 'failed.json'
    receipt.write_text('{}')
    monkeypatch.setattr(resume_inventory, 'signature', lambda model, *a, **kw:
                        'saved-error' if model == 'gemma4' and receipt.exists() else None)
    jobs = []

    def runner(command, environment, heartbeat):
        model = model_of(command)
        jobs.append((model, environment['SRGC_RESUME_ONLY']))
        assert checkpoint.read_bytes() == b'last saved optimizer and selector state'
        if len(jobs) == 1:
            return chain.run_child([sys.executable, '-c', 'raise SystemExit(2)'], environment, heartbeat)
        receipt.unlink(missing_ok=True)
        return 0

    assert chain.drain(list(chain.MODELS), 'math', ['math'], os.environ, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: None) == 0
    assert jobs[:2] == [('gemma4', '1'), ('gemma4', '1')]
    assert checkpoint.read_bytes() == b'last saved optimizer and selector state'


def test_math_pause_stops_before_any_mbpp_or_other_model_launch():
    jobs, states = [], []
    assert chain.drain(list(chain.MODELS), 'all', ['all'], {},
                       lambda *a, **kw: states.append((a, kw)),
                       runner=lambda cmd, env, beat: jobs.append(env['SRGC_DATASET_PHASE']) or chain.PAUSED) == chain.PAUSED
    assert jobs == ['math']
    assert states[-1][0] == ('stopped',)


def test_repeated_peer_owned_math_yields_cannot_start_mbpp_early():
    jobs, sleeps = [], []
    counts = {model: 0 for model in chain.MODELS}
    def runner(command, environment, heartbeat):
        model, phase = model_of(command), environment['SRGC_DATASET_PHASE']
        jobs.append((model, phase))
        counts[model] += phase == 'math'
        return chain.YIELD if phase == 'math' and model == 'qwen35' and counts[model] < 3 else 0
    assert chain.drain(list(chain.MODELS), 'all', ['all'], {}, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: sleeps.append(n)) == 0
    assert jobs[:5] == [('qwen35', 'math'), ('llama31', 'math'), ('gemma4', 'math'),
                       ('qwen35', 'math'), ('qwen35', 'math')]
    assert jobs[5:] == [(model, 'mbpp') for model in chain.MODELS]
    assert len(sleeps) == 2


def test_all_resume_preserves_root_across_phases_and_clears_each_models_stop_once():
    jobs = []
    initial = ['all', 'resume', '--root', '/group/custom-qwen', '--max-attempts', '1']
    def runner(command, environment, heartbeat):
        jobs.append((model_of(command), environment['SRGC_DATASET_PHASE'], command[2:]))
        return 0
    assert chain.drain(list(chain.MODELS), 'all', initial, {}, lambda *a, **kw: None, runner=runner) == 0
    math_jobs = [(model, args) for model, dataset, args in jobs if dataset == 'math']
    assert math_jobs == [('qwen35', initial), ('qwen35', ['all', 'run', *initial[2:]]),
                         ('llama31', ['all', 'resume']), ('gemma4', ['all', 'resume'])]
    mbpp_jobs = [(model, args) for model, dataset, args in jobs if dataset == 'mbpp']
    assert mbpp_jobs == [('qwen35', ['all', 'run', *initial[2:]]),
                         ('llama31', ['all', 'run']), ('gemma4', ['all', 'run'])]


@pytest.mark.parametrize('model', chain.MODELS)
def test_new_mbpp_pause_is_preserved_after_initial_all_resume(model):
    paused, jobs = False, []

    def runner(command, environment, heartbeat):
        nonlocal paused
        phase, args = environment['SRGC_DATASET_PHASE'], command[2:]
        jobs.append((phase, args))
        if args[1] == 'resume':
            paused = False  # Public resume clears the operator's stop marker.
        if phase == 'math':
            paused = True  # A new MBPP stop arrives while MATH is completing.
            return 0
        return chain.PAUSED if paused else 0

    assert chain.drain([model], 'all', ['all', 'resume'], {}, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: pytest.fail('paused chain slept')) == chain.PAUSED
    assert sum(args[1] == 'resume' for _, args in jobs) == 1
    assert jobs[-1] == ('mbpp', ['all', 'run'])


def test_explicit_mbpp_source_plan_and_interpreter_scope_are_preserved_during_math_phase():
    initial = ['mbpp', 'run', '--source-plan', '/group/code-plan.json', '--root', '/group/custom-llama']
    jobs = []
    def runner(command, environment, heartbeat):
        jobs.append((model_of(command), environment['SRGC_DATASET_PHASE'], command[2:]))
        return 0
    assert chain.drain(chain.order_for('llama31'), 'mbpp', initial, {}, lambda *a, **kw: None, runner=runner) == 0
    assert [(phase, args) for model, phase, args in jobs if model == 'llama31'] == [
        ('math', initial), ('mbpp', initial)]
    assert all(args == ['mbpp'] for model, phase, args in jobs if model != 'llama31')


def test_all_stopped_models_recover_before_any_model_opens_fresh_work(monkeypatch):
    from srgc_research.dispatch import resume_inventory

    pending = {'llama31': 'saved-llama', 'gemma4': 'saved-gemma'}
    monkeypatch.setattr(resume_inventory, 'signature', lambda model, *a, **kw: pending.get(model))
    jobs = []
    def runner(command, environment, heartbeat):
        model, recovery = model_of(command), environment['SRGC_RESUME_ONLY'] == '1'
        jobs.append((model, recovery))
        if recovery:
            pending.pop(model)
        else:
            assert not pending, 'fresh work began while an abandoned model still needed recovery'
        return 0
    assert chain.drain(list(chain.MODELS), 'math', ['math'], {}, lambda *a, **kw: None, runner=runner) == 0
    assert jobs == [('llama31', True), ('gemma4', True),
                    ('qwen35', False), ('llama31', False), ('gemma4', False)]


def test_error_model_is_recovered_before_earlier_interrupted_model(monkeypatch):
    from srgc_research.dispatch import resume_inventory

    pending = {'qwen35': 'interrupted', 'gemma4': 'error-x99'}
    monkeypatch.setattr(resume_inventory, 'signature', lambda model, *a, **kw: pending.get(model))
    monkeypatch.setattr(resume_inventory, 'failed_signature', lambda model, *a, **kw:
                        pending.get(model) if model == 'gemma4' else None)
    jobs = []

    def runner(command, environment, heartbeat):
        model, recovery = model_of(command), environment['SRGC_RESUME_ONLY'] == '1'
        jobs.append((model, recovery))
        if recovery:
            pending.pop(model)
        else:
            assert not pending
        return 0

    assert chain.drain(list(chain.MODELS), 'math', ['math'], {}, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: None) == 0
    assert jobs[:2] == [('gemma4', True), ('qwen35', True)]


def test_failed_llama_recovery_does_not_starve_other_existing_model_errors(monkeypatch):
    from srgc_research.dispatch import resume_inventory

    pending = {'llama31': 'saved-llama-error', 'gemma4': 'saved-gemma-error'}
    monkeypatch.setattr(resume_inventory, 'signature', lambda model, *a, **kw: pending.get(model))
    monkeypatch.setattr(resume_inventory, 'failed_signature', lambda model, *a, **kw: pending.get(model))
    jobs, attempts = [], []

    def runner(command, environment, heartbeat):
        model, recovery = model_of(command), environment['SRGC_RESUME_ONLY'] == '1'
        jobs.append((model, recovery))
        if model == 'llama31' and recovery:
            attempts.append(True)
            if len(attempts) == 1:
                return 2
            assert 'gemma4' not in pending, 'another ERROR model was starved by repeated Llama recovery'
        if recovery:
            pending.pop(model)
        else:
            assert not pending, 'fresh work overtook unfinished ERROR recovery'
        return 0

    assert chain.drain(chain.order_for('llama31'), 'math', ['math'], {}, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: None) == 0
    assert jobs[:3] == [('llama31', True), ('gemma4', True), ('llama31', True)]


def test_success_exit_cannot_hide_an_unchanged_failed_receipt(monkeypatch):
    from srgc_research.dispatch import resume_inventory

    pending = {'llama31': 'failed-task'}
    monkeypatch.setattr(resume_inventory, 'signature', lambda model, *a, **kw: pending.get(model))
    monkeypatch.setattr(resume_inventory, 'failed_signature', lambda model, *a, **kw: pending.get(model))
    jobs = []

    def runner(command, environment, heartbeat):
        model, recovery = model_of(command), environment['SRGC_RESUME_ONLY'] == '1'
        jobs.append((model, recovery))
        if recovery and len(jobs) > 1:
            pending.pop(model)
        if not recovery:
            assert not pending
        return 0

    assert chain.drain(list(chain.MODELS), 'math', ['math'], {}, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: None) == 0
    assert jobs[:2] == [('llama31', True), ('llama31', True)]


def test_cooling_model_recovery_is_revisited_instead_of_starting_other_fresh_models(monkeypatch):
    from srgc_research.dispatch import resume_inventory

    pending = {'llama31': 'stopped-model'}
    monkeypatch.setattr(resume_inventory, 'signature', lambda model, *a, **kw: pending.get(model))
    jobs, waits = [], []
    def runner(command, environment, heartbeat):
        model, recovery = model_of(command), environment['SRGC_RESUME_ONLY'] == '1'
        jobs.append((model, recovery))
        if recovery and len(jobs) == 1:
            return chain.YIELD
        if recovery:
            pending.clear()
        else:
            assert not pending
        return 0
    assert chain.drain(list(chain.MODELS), 'math', ['math'], {}, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: waits.append(n)) == 0
    assert jobs[:2] == [('llama31', True), ('llama31', True)]
    assert len(waits) == 1


def test_stale_completed_receipt_does_not_create_an_endless_recovery_pass(monkeypatch):
    from srgc_research.dispatch import resume_inventory

    monkeypatch.setattr(resume_inventory, 'signature', lambda model, *a, **kw: 'stale-cache' if model == 'qwen35' else None)
    jobs = []
    assert chain.drain(list(chain.MODELS), 'math', ['math'], {}, lambda *a, **kw: None,
                       runner=lambda cmd, env, beat: jobs.append((model_of(cmd), env['SRGC_RESUME_ONLY'])) or 0,
                       sleep=lambda n: pytest.fail('completed recovery slept')) == 0
    assert jobs == [('qwen35', '1'), ('qwen35', '0'), ('llama31', '0'), ('gemma4', '0')]


@pytest.mark.parametrize('code',[130,143])
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
    assert jobs[2][2:]==['math','resume']


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


@pytest.mark.parametrize('bad', [None, [], ['worker'], 7, 'worker',
                               {'started': 'bad'}, {'heartbeat': 'bad'},
                               {'order': 'llama31'}, {'model': {}}, {'devices': []}])
def test_status_isolates_malformed_coordinator_receipts(tmp_path, bad):
    from srgc_rebuttal.runtime import atomic_json

    root = tmp_path / 'work/srgc-rebuttal/.model-chain/workers'
    environment = {'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path / 'work')}
    valid = {'worker_id': 'valid', 'host': 'node-a', 'pid': 123, 'started': 10,
             'heartbeat': 100, 'status': 'running', 'model': 'gemma4',
             'order': list(chain.MODELS), 'devices': '0,1,2,3'}
    payload = {**valid, 'worker_id': 'bad', **bad} if isinstance(bad, dict) else bad
    atomic_json(root / 'bad.json', payload)
    atomic_json(root / 'valid.json', valid)
    before = {p: p.read_bytes() for p in root.iterdir()}
    rows = chain.snapshot(environment, now=105)
    assert len(rows) == 1 and rows[0]['worker_id'] == 'valid'
    assert 'RUNNING model=gemma4' in '\n'.join(chain.render_status(environment, now=105))
    assert {p: p.read_bytes() for p in root.iterdir()} == before


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


@pytest.mark.parametrize('code', [129, 137])
def test_forcibly_killed_worker_is_restarted_before_any_fresh_model(code):
    jobs, waits = [], []
    def runner(command, environment, heartbeat):
        jobs.append(model_of(command))
        return code if len(jobs) <= 5 else 0
    assert chain.drain(list(chain.MODELS), 'math', ['math'], {}, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: waits.append(n)) == 0
    # A successful recovery pass returns before that model's fresh queue.
    assert jobs == ['qwen35'] * 7 + ['llama31', 'gemma4']
    assert len(waits) == 5


def test_surviving_coordinator_retries_a_real_sigkill_child(tmp_path):
    attempts = []
    def runner(command, environment, heartbeat):
        attempts.append(model_of(command))
        if len(attempts) == 1:
            def kill_child(pid):
                os.kill(pid, signal.SIGKILL)
            return chain.run_child([sys.executable, '-c', 'import time;time.sleep(60)'], environment, kill_child)
        return 0
    assert chain.drain(list(chain.MODELS), 'math', ['math'], os.environ, lambda *a, **kw: None,
                       runner=runner, sleep=lambda n: None) == 0
    assert attempts == ['qwen35', 'qwen35', 'qwen35', 'llama31', 'gemma4']
