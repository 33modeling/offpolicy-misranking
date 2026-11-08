"""Actual process disconnect/kill and queue recovery controls; CPU, no H100 allocation."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_retry_debug import prepared

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('close_console',[False,True])
def test_terminal_disconnect_does_not_stop_a_healthy_task(tmp_path,close_console):
    for name in ('v8_supervisor.py','v8_processes.py','v8_probe.py','v8_control_output.py','v8_progress_text.py'):
        shutil.copyfile(ROOT/'scripts'/name,tmp_path/name)
    (tmp_path/'v8_bootstrap.py').write_text('''import time
from pathlib import Path
print('READY',flush=True)
time.sleep(.4)
Path(__file__).with_name('finished').write_text('normal completion after disconnect')
print('FINISHED',flush=True)
''')
    env=dict(os.environ,V8_ROOT=str(tmp_path/'runs'),V8_STOP_WAIT_SECONDS='1')
    process=subprocess.Popen([sys.executable,str(tmp_path/'v8_supervisor.py')],env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    try:
        assert process.stdout.readline().strip()=='READY'
        if close_console:process.stdout.close()
        process.send_signal(signal.SIGHUP)
        if close_console:process.wait(timeout=5)
        else:process.communicate(timeout=5)
        assert (tmp_path/'finished').is_file(),'SIGHUP stopped healthy work instead of surviving the disconnect'
        assert process.returncode==0
    finally:
        if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()


def test_signal_only_termination_is_resumable_without_consuming_training_failure_budget(tmp_path,monkeypatch):
    pool,runtime,task,key,saved=prepared(tmp_path);pool.retry_delay=60
    stop=threading.Event();_,claim=pool.tick(runtime,'terminated-node',stop)
    attempt=claim[-1]
    nodes.atomic(pool.root/'progress'/attempt['id']/'child-failure.json',
        dict(attempt_id=attempt['id'],exit_code=1,failures=[dict(rank=1,signal=9,message='SIGKILL')]))
    monkeypatch.setattr(nodes.time,'time',lambda:100.)
    try:pool.finish(key,1,runtime,stop)
    finally:claim[0].close()
    job=nodes.read(pool.root/'jobs'/(key+'.json'))
    assert job['state']=='interrupted' and nodes.counted_failures(job)==0
    assert not pool.retryable(job,now=159) and pool.retryable(job,now=160)
    assert saved.read_bytes()==b'original step25 optimizer RNG rewards'


@pytest.mark.parametrize('fault',['python-oom','segfault','kill-with-python-error'])
def test_real_code_faults_are_not_refunded_as_node_disconnects(tmp_path,fault):
    pool,runtime,task,key,saved=prepared(tmp_path);stop=threading.Event()
    _,claim=pool.tick(runtime,'node',stop);attempt=claim[-1]
    errors=[dict(rank=1,message='torch.OutOfMemoryError: real CUDA allocation failure',original_error={'message':'OOM'})]
    if fault=='segfault':errors=[dict(rank=1,signal=11,message='SIGSEGV')]
    if fault=='kill-with-python-error':errors.append(dict(rank=2,signal=9,message='SIGKILL during elastic cleanup'))
    nodes.atomic(pool.root/'progress'/attempt['id']/'child-failure.json',
                 dict(attempt_id=attempt['id'],exit_code=1,failures=errors))
    try:pool.finish(key,1,runtime,stop)
    finally:claim[0].close()
    job=nodes.read(pool.root/'jobs'/(key+'.json'))
    assert job['state']=='failed' and nodes.counted_failures(job)==1


def orphan_worker(root,ready):
    pool,runtime,task,key,saved=prepared(Path(root));_,claim=pool.tick(runtime,'lost-node',threading.Event())
    ready.set()
    while True:time.sleep(.1)


def test_sigkill_of_actual_worker_is_taken_over_without_erasing_evidence(tmp_path):
    ctx=mp.get_context('fork');ready=ctx.Event();process=ctx.Process(target=orphan_worker,args=(str(tmp_path),ready))
    process.start()
    try:
        assert ready.wait(5)
        process.kill();process.join(5);assert process.exitcode==-signal.SIGKILL
        # Same persisted queue; do not recreate/initialize it or replace its checkpoint.
        pool=nodes.Pool(tmp_path/'olmo_math/.multi-node',dict(cohort='olmo_math'),retry_delay=0)
        [path]=list((pool.root/'jobs').glob('*.json'));old=nodes.read(path);task=old['task']
        runtime=object.__new__(nodes.Runtime);runtime.root=tmp_path/'olmo_math'
        runtime.plan=lambda phase:([task],False)
        action,claim=pool.tick(runtime,'replacement-node',threading.Event());assert action=='claimed'
        try:
            job=nodes.read(path)
            assert job['attempts'][0]['outcome']=='node_lost'
            assert job['attempts'][1]['resumed_from_attempt']==old['attempts'][0]['id']
            assert nodes.counted_failures(job)==0 and len(job['attempts'])==2
            assert Path(task['parent']['path']).read_bytes()==b'original step25 optimizer RNG rewards'
        finally:claim[0].close()
    finally:
        if process.is_alive():process.kill();process.join()


@pytest.mark.parametrize('rc',[-9,137,-15,143])
@pytest.mark.parametrize('code_fault',[False,True])
def test_direct_signal_exit_keeps_python_fault_priority(tmp_path,rc,code_fault):
    log=tmp_path/'task.log'
    if code_fault:log.write_text('[rank2]: torch.OutOfMemoryError: CUDA out of memory\n')
    assert bool(nodes.forced_termination({},rc,log)) is not code_fault


def test_killed_original_runner_resumes_its_saved_update_and_matches_uninterrupted(tmp_path,monkeypatch):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('V8 runtime source required for original runner checkpoint control')
    monkeypatch.setenv('V8_CODE_ROOT',str(ROOT))
    from test_srgc_v8_nodes import ToyRuntime
    runtime=ToyRuntime(runner,tmp_path/'resumed')
    runtime.barrier('prepare',None,threading.Event(),[])
    task=next(t for t in runtime.plan('train')[0] if t['training_seed']==5 and t['method_id']=='on_monitored')
    runtime.plan=lambda phase:([task],False)
    pool=nodes.Pool(runtime.root/'.multi-node',runtime.binding,retry_delay=0);pool.initialize()
    state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    ctx=mp.get_context('fork');saved=ctx.Event()

    def kill_after_durable_update():
        from switching import snapshots
        from switching.runner import execute_train
        original=snapshots.save
        def save(path,payload,*,immutable):
            result=original(path,payload,immutable=immutable)
            if Path(path).name=='latest.pt' and payload['engine']['step']==26:
                saved.set()
                while True:time.sleep(.1)
            return result
        snapshots.save=save
        _,claim=pool.tick(runtime,'lost-node',threading.Event())
        execute_train(task,runtime.context(5))

    process=ctx.Process(target=kill_after_durable_update);process.start()
    try:
        assert saved.wait(15),'original CPU runner did not save update26'
        assert pool.tick(runtime,'live-peer',threading.Event())[0]=='waiting'
        from switching import snapshots
        from switching.common import read_jsonl
        run=runtime.root/'runs'/task['run_id'];checkpoint=(run/'latest.pt').read_bytes()
        assert snapshots.load(run/'latest.pt')['engine']['step']==26
        process.kill();process.join(5);assert process.exitcode==-signal.SIGKILL
        action,claim=pool.tick(runtime,'replacement-node',threading.Event());assert action=='claimed'
        assert (run/'latest.pt').read_bytes()==checkpoint
        try:
            rc=runtime.execute(task,None,claim[0].fd,threading.Event(),[],claim[-1])
            assert pool.finish(claim[1],rc,runtime,threading.Event())=='done'
        finally:claim[0].close()
        manifest=nodes.read(run/'manifest.json')
        assert [a['start_update'] for a in manifest['attempts']]==[25,26]
        second=manifest['attempts'][1]['attempt_id']
        assert read_jsonl(run/'attempts'/second/'updates.jsonl')[0]['completed_updates']==27
        assert [r['completed_updates'] for r in read_jsonl(run/'history.jsonl')]==list(range(26,76))
        assert [r['completed_update'] for r in read_jsonl(run/'snapshots.jsonl')]==[50,75]
        assert all(r.get('metrics') for r in read_jsonl(run/'history.jsonl'))
        baseline=ToyRuntime(runner,tmp_path/'uninterrupted')
        baseline.barrier('prepare',None,threading.Event(),[])
        control=next(t for t in baseline.plan('train')[0] if t['training_seed']==5 and t['method_id']=='on_monitored')
        from switching.runner import execute_train
        execute_train(control,baseline.context(5))
        reference=baseline.root/'runs'/control['run_id']
        for step in (50,75):
            relative=Path('snapshots')/f'step-{step:06d}.pt'
            assert snapshots.branch_identity(snapshots.load(run/relative))==snapshots.branch_identity(snapshots.load(reference/relative))
        assert [r['metrics'] for r in read_jsonl(run/'history.jsonl')]==[r['metrics'] for r in read_jsonl(reference/'history.jsonl')]
    finally:
        if process.is_alive():process.kill();process.join()
