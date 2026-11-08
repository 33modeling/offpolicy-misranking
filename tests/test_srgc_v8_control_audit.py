"""Adversarial CPU process/control regressions; no model, GPU, service or training execution."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import pytest

from scripts import v8_processes as proc
from scripts import v8_queue_runtime as source

REPO=Path(__file__).resolve().parents[1]


def copy_controls(target):
    for name in ['v8_supervisor.py','v8_processes.py','v8_probe.py','v8_stop.py','v8_control_output.py','v8_progress_text.py']:
        path=REPO/'scripts'/name
        if path.exists():shutil.copyfile(path,target/name)


def wait_for(path,timeout=3):
    end=time.monotonic()+timeout
    while not path.exists() and time.monotonic()<end:time.sleep(.01)
    assert path.exists(),str(path)


def kill_process(pid):
    try:os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:pass


def bootstrap_plan(bootstrap,argv,monkeypatch):
    plans=[]
    class Ready(Exception):pass
    def capture(python,command,env):
        assert Path(command[1]).name=='v8_dispatch.py'
        plans.append(json.loads(Path(command[2]).read_text()));raise Ready()
    monkeypatch.setattr(bootstrap.os,'execvpe',capture)
    with pytest.raises(Ready):bootstrap.main(argv)
    return plans[0]


def test_supervisor_stops_when_console_pipe_is_full(tmp_path):
    copy_controls(tmp_path);pidfile=tmp_path/'bootstrap.pid'
    (tmp_path/'v8_bootstrap.py').write_text('import os,signal,time\nfrom pathlib import Path\n'
        'Path(__file__).with_name("bootstrap.pid").write_text(str(os.getpid()))\n'
        'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
        'for _ in range(32):os.write(1,b"x"*65536)\n'
        'Path(__file__).with_name("work-done").write_text("2MiB emitted without waiting for terminal")\n'
        'while True:time.sleep(.02)\n')
    parent=subprocess.Popen([sys.executable,str(tmp_path/'v8_supervisor.py')],
        env=dict(os.environ,V8_ROOT=str(tmp_path/'v8')),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    child=None
    try:
        wait_for(pidfile);child=int(pidfile.read_text());wait_for(tmp_path/'work-done')
        parent.send_signal(signal.SIGINT);time.sleep(.1);parent.send_signal(signal.SIGINT)
        try:parent.wait(timeout=3)
        except subprocess.TimeoutExpired:pytest.fail('full console pipe prevented bounded Ctrl-C cleanup')
        assert parent.returncode==130
        rows=proc.table();assert child not in rows or rows[child]['state']=='Z'
    finally:
        if parent.poll() is None:parent.kill()
        if child:kill_process(child)
        parent.wait(timeout=3);parent.stdout.close()


def test_stop_command_does_not_wait_for_a_full_console_before_signalling(tmp_path):
    copy_controls(tmp_path)
    rank=tmp_path/'cluster_child.py';rank.write_text('import time\nprint("READY",flush=True)\ntime.sleep(30)\n')
    env=dict(os.environ,V8_ROOT=str(tmp_path/'v8'))
    child=subprocess.Popen([sys.executable,str(rank)],env=env,stdout=subprocess.PIPE,text=True,start_new_session=True)
    readfd,writefd=os.pipe();os.set_blocking(writefd,False)
    try:
        while True:os.write(writefd,b'x'*4096)
    except BlockingIOError:pass
    os.set_blocking(writefd,True)
    parent=None
    try:
        assert child.stdout.readline().strip()=='READY'
        parent=subprocess.Popen([sys.executable,str(tmp_path/'v8_stop.py')],env=env,stdout=writefd,stderr=writefd)
        try:parent.wait(timeout=3)
        except subprocess.TimeoutExpired:pytest.fail('stop command blocked on console before signalling its target')
        assert parent.returncode==0
        child.wait(timeout=2)
    finally:
        if parent is not None:
            if parent.poll() is None:parent.kill()
            parent.wait(timeout=3)
        if child.poll() is None:child.kill()
        child.wait(timeout=3);child.stdout.close();os.close(readfd);os.close(writefd)


def test_stop_catches_rank_spawned_after_first_snapshot(tmp_path):
    rank=tmp_path/'cluster_child.py'
    rank.write_text('import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                    'print("RANK_READY",flush=True)\nwhile True:time.sleep(.02)\n')
    script=tmp_path/'srgc_v8_nodes.py';pidfile=tmp_path/'late.pid'
    script.write_text('import os,subprocess,sys,signal,time\nfrom pathlib import Path\n'
        'def stop(*_):\n'
        ' p=subprocess.Popen([sys.executable,str(Path(__file__).with_name("cluster_child.py"))],stdout=subprocess.PIPE,text=True,start_new_session=True)\n'
        ' assert p.stdout.readline().strip()=="RANK_READY"\n'
        ' Path(__file__).with_name("late.pid").write_text(str(p.pid))\n'
        ' os._exit(0)\n'
        'signal.signal(signal.SIGTERM,stop)\nprint("READY",flush=True)\nwhile True:time.sleep(.02)\n')
    root=str(tmp_path/'v8');parent=subprocess.Popen([sys.executable,str(script)],env=dict(os.environ,V8_ROOT=root),
                                                  stdout=subprocess.PIPE,text=True,start_new_session=True)
    late=None
    try:
        assert parent.stdout.readline().strip()=='READY'
        targets=proc.select(proc.table(),root=root)
        # Stop API used by the standalone tool must refresh this legacy root throughout cleanup.
        if 'root' in __import__('inspect').signature(proc.stop).parameters:
            remaining=proc.stop(targets,root=root,grace=.2)
        else:remaining=proc.stop(targets,grace=.2)
        wait_for(pidfile);late=int(pidfile.read_text());parent.wait(timeout=2)
        rows=proc.table()
        assert not remaining and (late not in rows or rows[late]['state']=='Z'),'rank spawned during stop survived cleanup'
    finally:
        if parent.poll() is None:parent.kill()
        parent.wait(timeout=3);parent.stdout.close()
        if late:kill_process(late)


def make_queue(base,cohort,coordinator,runner):
    queue=base/cohort/'.multi-node';queue.mkdir(parents=True)
    (queue/'binding.json').write_text(json.dumps(dict(schema=1,cohort=cohort,root=str(base/cohort),
        coordinator_sha256=coordinator,runner_sha256=runner)))
    (queue/'state.json').write_text('{"phase":"train","failure":null}')
    return queue


@pytest.mark.parametrize('bad',['unknown_source','malformed_identity','partial_json'])
def test_bad_cohort_source_does_not_block_healthy_cohort(tmp_path,monkeypatch,bad):
    monkeypatch.setattr(tempfile,'tempdir',str(tmp_path))
    base=tmp_path/'experiments';runner=REPO/'v8/experiments'
    broken=make_queue(base,'olmo_math',None if bad=='malformed_identity' else 'f'*64,source.runner_identity(runner))
    if bad=='partial_json':(broken/'binding.json').write_text('{unfinished')
    good=make_queue(base,'qwen_mbpp',source.sha(REPO/'scripts/srgc_v8_nodes.py'),source.runner_identity(runner))
    before={p:p.read_bytes() for p in base.rglob('*') if p.is_file()}
    plan=source.selections(REPO,dict(V8_ROOT=str(base),V8_RUNNER_ROOT=str(runner),V8_PYTHON=sys.executable),[])
    assert len(plan)==1 and plan[0]['cohorts']==['qwen_mbpp']
    assert 'olmo_math' in plan.errors
    assert all(p.read_bytes()==data for p,data in before.items())


def test_different_python_environments_do_not_share_admission_process(tmp_path,monkeypatch):
    from scripts import v8_bootstrap as bootstrap
    left=tmp_path/'left-python';left.symlink_to(sys.executable)
    right=tmp_path/'right-python';right.symlink_to(sys.executable)
    env=dict(V8_ROOT=str(tmp_path/'experiments'),V8_RUNNER_ROOT=str(REPO/'v8/experiments'),
             V8_PYTHON_OLMO_MATH=str(left),V8_PYTHON_QWEN_MBPP=str(right))
    plan=source.selections(REPO,env,[])
    assert len(plan)==2,'separate Python environments would otherwise run admission under the first cohort interpreter'
    assert [g['cohorts'] for g in plan]==[['olmo_math'],['qwen_mbpp']]
    env['V8_CODE_ROOT']=str(REPO);env['V8_PYTHON']=sys.executable
    monkeypatch.setattr(bootstrap,'environment',lambda *a,**k:env)
    plan=bootstrap_plan(bootstrap,[],monkeypatch)
    assert [entry['python'] for entry in plan['entries']]==[str(left),str(right)]


def test_healthy_cohort_runs_but_partial_startup_keeps_failure_exit(tmp_path,monkeypatch):
    from scripts import v8_bootstrap as bootstrap
    monkeypatch.setattr(tempfile,'tempdir',str(tmp_path));base=tmp_path/'experiments';runner=REPO/'v8/experiments'
    make_queue(base,'olmo_math','f'*64,source.runner_identity(runner))
    make_queue(base,'qwen_mbpp',source.sha(REPO/'scripts/srgc_v8_nodes.py'),source.runner_identity(runner))
    env=dict(V8_ROOT=str(base),V8_RUNNER_ROOT=str(runner),V8_PYTHON=sys.executable,V8_CODE_ROOT=str(REPO))
    monkeypatch.setattr(bootstrap,'environment',lambda *a,**k:env)
    plan=bootstrap_plan(bootstrap,[],monkeypatch)
    assert 'olmo_math' in plan['errors']
    assert [entry['cohort'] for entry in plan['entries']]==['qwen_mbpp']


def test_missing_cohort_interpreter_does_not_block_other_environment(tmp_path,monkeypatch):
    from scripts import v8_bootstrap as bootstrap
    monkeypatch.setattr(tempfile,'tempdir',str(tmp_path))
    env=dict(PATH=os.environ['PATH'],TMPDIR=str(tmp_path),V8_ROOT=str(tmp_path/'experiments'),V8_CODE_ROOT=str(REPO),
             V8_RUNNER_ROOT=str(REPO/'v8/experiments'),V8_PYTHON=sys.executable,
             PAIR_PYTHON=str(tmp_path/'missing-python'),SWITCH_PYTHON=sys.executable)
    monkeypatch.setattr(bootstrap.os,'environ',env)
    plan=bootstrap_plan(bootstrap,[],monkeypatch)
    assert 'olmo_math' in plan['errors']
    assert [entry['cohort'] for entry in plan['entries']]==['qwen_mbpp']


def test_git_pull_cannot_change_already_selected_worker_or_child(tmp_path,monkeypatch):
    monkeypatch.setattr(tempfile,'tempdir',str(tmp_path))
    tools=tmp_path/'checkout';(tools/'scripts').mkdir(parents=True)
    shutil.copyfile(REPO/'scripts/srgc_v8_nodes.py',tools/'scripts/srgc_v8_nodes.py')
    shutil.copyfile(REPO/'scripts/srgc_saved_runtime.py',tools/'scripts/srgc_saved_runtime.py')
    helper=tools/'scripts/srgc_verifier_fallback.py'
    shutil.copyfile(REPO/'scripts/srgc_verifier_fallback.py',helper)
    shutil.copytree(REPO/'v8/experiments',tools/'v8/experiments',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    worker=tools/'scripts/srgc_v8_nodes.py';child=tools/'v8/experiments/cluster_child.py'
    expected=(worker.read_bytes(),child.read_bytes(),helper.read_bytes())
    plan=source.selections(tools,dict(V8_ROOT=str(tmp_path/'experiments'),V8_RUNNER_ROOT=str(tools/'v8/experiments'),
                                    V8_PYTHON=sys.executable),[])
    worker.write_text(worker.read_text()+'\n# simulated git pull replacing worker\n')
    child.write_text(child.read_text()+'\n# simulated git pull replacing child\n')
    helper.write_text(helper.read_text()+'\n# simulated git pull replacing verifier\n')
    assert plan[0]['script'].read_bytes()==expected[0],'selected worker changed after checkout update'
    assert (plan[0]['runner']/'cluster_child.py').read_bytes()==expected[1],'selected child changed after checkout update'
    assert (plan[0]['control']/'srgc_verifier_fallback.py').read_bytes()==expected[2],'selected verifier changed after checkout update'
    make_queue(tmp_path/'experiments','olmo_math',plan[0]['coordinator_sha256'],plan[0]['runner_sha256'])
    resumed=source.selections(tools,dict(V8_ROOT=str(tmp_path/'experiments'),V8_RUNNER_ROOT=str(tools/'v8/experiments'),
                                        V8_PYTHON=sys.executable),['olmo_math'])
    assert resumed[0]['script'].read_bytes()==expected[0]
    assert (resumed[0]['runner']/'cluster_child.py').read_bytes()==expected[1]


def test_local_idle_gpu_is_not_evidence_that_a_remote_worker_stalled(tmp_path,monkeypatch):
    from scripts import v8_status
    import socket
    now=time.time();cohort='olmo_math';queue=tmp_path/cohort/'.multi-node';(queue/'jobs').mkdir(parents=True)
    (queue/'state.json').write_text('{"phase":"train"}')
    (queue/'jobs/a.json').write_text(json.dumps(dict(state='running',task=dict(kind='train',run_id='r',horizon=275),
        attempts=[dict(id='a',worker='remote.cluster.example.123',started_at=now-4000)])))
    (queue/'logs').mkdir();log=queue/'logs/a.log';log.write_text('work started\n');os.utime(log,(now-4000,now-4000))
    monkeypatch.setattr(socket,'gethostname',lambda:'local.cluster.example')
    output=v8_status.render([tmp_path],now=now,gpus=[('0',0,0)])
    assert 'WARNING' not in output,'local GPU state was incorrectly attributed to a remote worker'
    assert 'remote.cluster.example' in output,'FQDN was truncated at the first dot'


def test_old_worker_does_not_poison_a_queue_with_a_different_binding(tmp_path):
    from scripts import srgc_v8_nodes as nodes
    import threading
    original=dict(schema=1,cohort='fixture',revision='old')
    updated=dict(original,revision='new')
    class Pool(nodes.Pool):
        calls=0
        def tick(self,*args,**kwargs):
            self.calls+=1
            if self.calls==1:
                nodes.atomic(self.root/'binding.json',updated)
            return super().tick(*args,**kwargs)
    class Idle:
        def verify_node(self):pass
        def barrier(self,*args):pass
    pool=Pool(tmp_path/'queue',original)
    assert nodes.worker_loop(pool,Idle(),'old-worker',threading.Event(),poll=.001)==1
    assert nodes.read(pool.root/'binding.json')==updated
    assert nodes.read(pool.root/'state.json')['failure'] is None,'old worker wrote its mismatch into the new queue'


def test_stale_heartbeat_is_not_proof_that_remote_training_stopped(tmp_path):
    from scripts import v8_status
    import hashlib
    now=time.time();cohort='olmo_math';queue=tmp_path/cohort/'.multi-node';(queue/'jobs').mkdir(parents=True)
    (queue/'state.json').write_text('{"phase":"train"}')
    worker='remote.cluster.example.123'
    (queue/'jobs/a.json').write_text(json.dumps(dict(state='running',task=dict(kind='train',run_id='r',horizon=275),
        attempts=[dict(id='a',worker=worker,started_at=now-4000)])))
    (queue/'workers').mkdir();key=hashlib.sha256(json.dumps(worker).encode()).hexdigest()
    (queue/'workers'/(key+'.json')).write_text(json.dumps(dict(worker=worker,at=now-900)))
    (queue/'logs').mkdir();log=queue/'logs/a.log';log.write_text('PROGRESS stage=policy_update\n');os.utime(log,(now-1,now-1))
    _,rows=v8_status.collect(tmp_path,cohort,now)
    assert rows[0]['status']=='unconfirmed','stale heartbeat alone was labelled as a stopped learner'
    text=v8_status.render([tmp_path],now=now,gpus=[('0',95,60000)],host='local.cluster.example')
    assert 'next worker resumes it' not in text
