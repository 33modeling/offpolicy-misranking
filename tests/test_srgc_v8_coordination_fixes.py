"""Regression tests for the 2026-11-08 coordination audit (COORD-01..10). CPU only: fake coordinators,
real v8_dispatch/runtime actors, GPU probes replaced; no model, GPU or network."""
import importlib
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
import zipfile

import pytest

from scripts import srgc_v8_nodes as nodes

REPO=Path(__file__).resolve().parents[1]
SCRIPTS=REPO/'scripts'

FAKE=r'''
import hashlib,json,os,sys,time,uuid
from pathlib import Path
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(canonical(value)+'\n');os.replace(tmp,path)
class Runtime:
    def __init__(self,runner,root,cohort,python,gpus):
        self.root=Path(root).resolve()/cohort;self.cohort=cohort;self.gpus=gpus
        self.binding=dict(schema=1,cohort=cohort,root=str(self.root),runner_sha256={'fixture':'cpu'},
                          coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.n=int(os.environ.get('FAKE_TASKS','3'));self.sleep=float(os.environ.get('FAKE_SLEEP','0.2'))
    def verify_node(self):pass
    def barrier(self,action,fd,stop,device_fds):self.root.mkdir(parents=True,exist_ok=True)
    def tasks(self):
        return [dict(kind='train',run_id=f'{self.cohort}.s5.m{i}',method_id='on_monitored' if i==0 else f'bank_sr_t{i:03d}',
                     training_seed=5,cohort_id=self.cohort,role='rebuttal_bank',config_hash='c',horizon=275,parent={}) for i in range(self.n)]
    def plan(self,phase):
        if phase=='train':return self.tasks(),True
        if phase=='extension':return [],True
        return [dict(kind='evaluate',run_id=t['run_id'],training_seed=5,completed_update=275,snapshot_path='x',
                     snapshot_sha256=digest(['s',t['run_id']]),evaluation_spec_hash=digest('e'),task_id='eval.'+t['run_id'],
                     split='final',responses=4) for t in self.tasks()],True
    def ready(self,task):return True
    def native_busy(self,task):return False
    def done(self,task):return (self.root/'done'/(task.get('task_id') or task['run_id'])).exists()
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        node=os.environ.get('FAKE_NODE','?');name=task.get('task_id') or task['run_id']
        (self.root/'starts').mkdir(parents=True,exist_ok=True)
        (self.root/'starts'/(name+'.'+node+'.'+attempt['id'][:6])).write_text(attempt['worker'])
        if os.environ.get('FAKE_FAULT_NODE')==node:
            if os.environ.get('FAKE_FAULT')=='actor_exit':os._exit(9)  # e.g. the actor is OOM-killed
            raise RuntimeError('NCCL admission failed on GPUs 0,1,2,3: simulated node-local fabric fault')
        end=time.monotonic()+self.sleep
        while time.monotonic()<end:
            if stop.is_set():return 75
            time.sleep(.01)
        (self.root/'done').mkdir(exist_ok=True);(self.root/'done'/name).write_text(node)
        return 0
'''

NODE=r'''
import os,sys
from pathlib import Path
import v8_dispatch
inventory=[dict(index=str(i),uuid=os.environ['FAKE_NODE']+'-GPU-'+str(i),total_mib=80000,used_mib=0,name='CPU fixture') for i in range(4)]
v8_dispatch.nodes.gpu_inventory=lambda _g:inventory
v8_dispatch.nodes.gpu_lock_root=lambda:Path(os.environ['V8_ROOT'])/'gpu-node-locks'
v8_dispatch.nodes.GpuGate=lambda _g:(lambda:(True,None))
sys.argv=['v8_dispatch.py',sys.argv[1]]
raise SystemExit(v8_dispatch.main())
'''


def fixture_plan(tmp_path,cohorts=('olmo_math','qwen_mbpp')):
    coordinator=tmp_path/'fake_coordinator.py';coordinator.write_text(FAKE)
    root=tmp_path/'v8';plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(schema=1,root=str(root),errors={},argv=['--gpus','0,1,2,3','--poll-seconds','0.02'],
        entries=[dict(cohort=c,script=str(coordinator),runner=str(tmp_path),python=sys.executable) for c in cohorts])))
    return root,plan


def start_node(tmp_path,plan,root,name,**extra):
    env=dict(os.environ,PYTHONPATH=str(SCRIPTS),PYTHONDONTWRITEBYTECODE='1',V8_ROOT=str(root),V8_ALLOW_ANY_ROOT='1',
             FAKE_NODE=name,**extra)
    copy=tmp_path/f'plan-{name}.json';shutil.copyfile(plan,copy)
    return subprocess.Popen([sys.executable,'-c',NODE,str(copy)],env=env,start_new_session=True,
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)


def finish(process,timeout=60):
    try:out,_=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);out,_=process.communicate();pytest.fail('worker did not stop:\n'+out[-3000:])
    return process.returncode,out


def jobs(root,cohort):
    return [json.loads(p.read_text()) for p in (root/cohort/'.multi-node/jobs').glob('*.json')]


@pytest.mark.parametrize('fault',['admission','actor_exit'])
def test_node_local_fault_never_fails_the_shared_queue(tmp_path,fault):
    """COORD-01: one node's NCCL admission failure or dead runtime actor stops only that node."""
    root,plan=fixture_plan(tmp_path)
    bad=start_node(tmp_path,plan,root,'bad',FAKE_FAULT_NODE='bad',FAKE_FAULT=fault,FAKE_SLEEP='0.3')
    deadline=time.monotonic()+20
    while not list(root.glob('*/starts/*.bad.*')) and time.monotonic()<deadline:time.sleep(.02)
    good=start_node(tmp_path,plan,root,'good',FAKE_FAULT_NODE='bad',FAKE_FAULT=fault,FAKE_SLEEP='0.3')
    bad_rc,bad_out=finish(bad);good_rc,good_out=finish(good)
    assert good_rc==0,good_out[-3000:]
    assert bad_rc==1 and 'this node stops serving' in bad_out and 'NOT marked failed' in bad_out,bad_out[-3000:]
    assert 'Traceback' not in bad_out+good_out
    for cohort in ('olmo_math','qwen_mbpp'):
        state=json.loads((root/cohort/'.multi-node/state.json').read_text())
        assert state['phase']=='complete' and state['failure'] is None
    faults=[a for c in ('olmo_math','qwen_mbpp') for j in jobs(root,c) for a in j['attempts'] if a.get('outcome')=='node_failure']
    assert faults and all('node' in a['error'] or 'actor' in a['error'] for a in faults)


def test_ctrl_c_during_a_task_records_interrupted_without_traceback(tmp_path):
    """COORD-04: SIGTERM to every process of the launch (what the supervisor does) during a GPU task."""
    root,plan=fixture_plan(tmp_path,cohorts=('olmo_math',))
    node=start_node(tmp_path,plan,root,'n',FAKE_SLEEP='60',FAKE_TASKS='1')
    deadline=time.monotonic()+20
    while not list(root.glob('olmo_math/starts/*')) and time.monotonic()<deadline:time.sleep(.02)
    os.killpg(node.pid,signal.SIGTERM)
    rc,out=finish(node,timeout=30)
    assert 'Traceback' not in out and 'KeyboardInterrupt' not in out,out[-3000:]
    assert rc==75
    [job]=[j for j in jobs(root,'olmo_math') if j['task']['kind']=='train']
    assert job['state']=='interrupted'
    assert job['attempts'][-1]['exit_code']==75 and job['attempts'][-1]['outcome']=='interrupted' and 'ended_at' in job['attempts'][-1]


def test_node_without_nvidia_smi_is_refused_before_any_queue_write(tmp_path):
    """COORD-02: a login/CPU node (no nvidia-smi) never claims GPU work."""
    bindir=tmp_path/'bin';bindir.mkdir()
    for tool in ('sh',):os.symlink(shutil.which(tool),bindir/tool)
    root=tmp_path/'v8'
    result=subprocess.run([sys.executable,str(SCRIPTS/'srgc_v8_nodes.py'),'olmo_math','--root',str(root),'--gpus','0,1,2,3'],
                          env=dict(PATH=str(bindir),V8_ALLOW_ANY_ROOT='1',HOME=str(tmp_path),PYTHONDONTWRITEBYTECODE='1'),
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==2 and 'nvidia-smi is not available' in result.stderr,result.stderr
    assert not root.exists()


def load_archived(tmp_path,sha):
    with zipfile.ZipFile(REPO/'v8/queue-runtimes.zip') as archive:
        source=archive.read(f'coordinators/{sha}/srgc_v8_nodes.py').decode()
    path=tmp_path/f'archived_{sha[:8]}.py';path.write_text(source)
    spec=importlib.util.spec_from_file_location('_archived_'+sha[:8],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,source


def bare_runtime(module,tmp_path):
    runtime=object.__new__(module.Runtime)
    runtime.environment={'PATH':os.environ['PATH']};runtime.gpus=['0'];runtime.root=tmp_path
    return runtime


def run_with_limit(function,limit):
    result={};thread=threading.Thread(target=lambda:result.setdefault('rc',function()),daemon=True)
    thread.start();thread.join(limit);return result.get('rc'),thread.is_alive()


def test_archived_coordinator_runs_with_current_bounded_process_handling(tmp_path,monkeypatch):
    """COORD-03 (V8-019 recurrence): cbd68a4 joined child stdout without a limit; the actor now uses the
    current process handling for every archived Runtime."""
    monkeypatch.syspath_prepend(str(SCRIPTS));monkeypatch.setenv('V8_STATUS_SECONDS','0')
    worker=importlib.import_module('v8_runtime_worker')
    module,source=load_archived(tmp_path,'9f038a87f7529b368b88e5bf6ac1fade9846db036e26966c09b02512fd8563bf')
    command=['sh','-c','setsid sleep 8 & exit 0']  # a detached rank keeps the child's stdout open
    original=bare_runtime(module,tmp_path)
    rc,alive=run_with_limit(lambda:original.process(command,tmp_path/'old.log',[],threading.Event(),cpu=True),2)
    assert alive,'fixture no longer reproduces the archived unbounded stdout join'
    fixed=bare_runtime(module,tmp_path)
    overrides,corrected=worker.control_overrides(fixed,source)
    assert overrides==['process:current-bounded'] and corrected==[]
    rc,alive=run_with_limit(lambda:fixed.process(command,tmp_path/'new.log',[],threading.Event(),cpu=True),5)
    assert not alive and rc==0


def test_pre_v8_016_coordinator_never_plans_evaluation_with_its_own_planner(tmp_path,monkeypatch):
    """COORD-03 (V8-016 recurrence): c44f6d9 planned evaluation from the matched-horizon run."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    worker=importlib.import_module('v8_runtime_worker');control=importlib.import_module('srgc_v8_nodes')
    module,source=load_archived(tmp_path,'8c81da4b44c17346bb9d36eb59075964f88f8bcce6c9a80776714671527bb6a1')
    runtime=bare_runtime(module,tmp_path);calls=[]
    runtime.plan=lambda phase:calls.append(('original',phase)) or ([],True)
    monkeypatch.setattr(control.Runtime,'plan',lambda self,phase:calls.append(('current',phase)) or ([],True))
    overrides,corrected=worker.control_overrides(runtime,source)
    assert corrected==['evaluation'] and 'plan.evaluation:current-v8-016' in overrides
    runtime.plan('train');runtime.plan('evaluation')
    assert calls==[('original','train'),('current','evaluation')]
    # An evaluation plan cached earlier by that planner is refused, not reused.
    pool=nodes.Pool(tmp_path/'queue',{'cohort':'olmo_math'});pool.initialize()
    nodes.atomic(pool.root/'state.json',dict(phase='evaluation',failure=None,actions=[],failure_history=[]))
    nodes.atomic(pool.root/'plans/evaluation.json',dict(tasks=[],sealed=True))
    class Proxy:corrected_plans=('evaluation',)
    with pytest.raises(ValueError,match='V8-016'):pool.tick(Proxy(),'w',threading.Event())


def test_cohort_python_precedence_matches_the_original_launchers(tmp_path):
    """COORD-05: Qwen uses QWEN_PYTHON before SWITCH/PAIR_PYTHON; OLMo never takes a Qwen/MBPP interpreter."""
    from scripts import v8_queue_runtime as queue
    pythons={}
    for name in ('switch','qwen','pair','fallback'):
        path=tmp_path/name/'python';path.parent.mkdir();os.symlink(sys.executable,path);pythons[name]=str(path)
    env=dict(V8_ROOT=str(tmp_path/'v8'),SWITCH_PYTHON=pythons['switch'],QWEN_PYTHON=pythons['qwen'],PAIR_PYTHON=pythons['pair'])
    assert queue.existing_python({},env,'qwen_mbpp')==pythons['qwen']
    assert queue.existing_python({},env,'qwen_math')==pythons['qwen']
    assert queue.existing_python({},env,'olmo_math')==pythons['pair']
    assert queue.existing_python({},env,'qwen_mbpp',explicit=pythons['fallback'])==pythons['qwen']
    alone=dict(V8_ROOT=env['V8_ROOT'],SWITCH_PYTHON=pythons['switch'])
    assert queue.existing_python({},alone,'olmo_math',explicit=pythons['fallback'])==pythons['fallback']
    for cohort in ('qwen_mbpp','qwen_math','olmo_math'):
        assert queue.existing_python({},env,cohort)==nodes.cohort_python(cohort,'fallback',env)


def test_status_grid_never_hides_extension_or_evaluation_states(tmp_path):
    """COORD-06: an extension shares its branch method_id and evaluations have none."""
    from scripts import v8_status
    queue=tmp_path/'v8/olmo_math/.multi-node'
    def job(task,phase,state):
        nodes.atomic(queue/'jobs'/(nodes.task_key(task)+'.json'),dict(task=task,phase=phase,state=state,attempts=[]))
    nodes.atomic(queue/'state.json',dict(phase='evaluation',failure=None,actions=[],failure_history=[]))
    train=lambda rid,mid:dict(kind='train',run_id=rid,method_id=mid,training_seed=5,horizon=275,parent={'path':'/absent'})
    job(train('s5.on','on_monitored'),'train','done');job(train('s5.sr050','bank_sr_t050'),'train','done')
    job(train('s5.sr050.ext','bank_sr_t050'),'extension','failed')
    for i,state in enumerate(['done','failed','pending']):
        job(dict(kind='evaluate',run_id=f's5.r{i}',training_seed=5,task_id=f'eval.{i}',snapshot_path='/absent',
                 snapshot_sha256=nodes.digest(['s',i]),evaluation_spec_hash=nodes.digest('e')),'evaluation',state)
    out=v8_status.render([tmp_path/'v8'],now=time.time(),gpus=[],host='h',cohorts=('olmo_math',))
    header=next(line for line in out.splitlines() if line.startswith('seed'))
    row=next(line for line in out.splitlines() if line.strip().startswith('5 '))
    assert 'bank_sr_t050 ext' in header and header.rstrip().endswith('eval'),out
    # the branch stays done, its extension shows failed, and the three evaluations are all counted
    assert row.split()[1:4]==['done','done','failed'] and row.rstrip().endswith('1 failed,1 done,1 waiting'),out


def test_queue_written_by_another_account_stops_this_node_without_traceback(tmp_path,capsys):
    """COORD-07: an unwritable lock file is reported with the owning account; nothing is marked failed."""
    if os.geteuid()==0:pytest.skip('root ignores file modes')
    from test_srgc_v8_nodes import FakeRuntime
    runtime=FakeRuntime(tmp_path/'runtime');pool=nodes.Pool(tmp_path/'queue',runtime.binding);pool.initialize()
    (pool.root/'control.lock').touch();os.chmod(pool.root/'control.lock',0o444)
    assert nodes.worker_loop(pool,runtime,'n',threading.Event(),poll=.002)==1
    out=capsys.readouterr().out
    assert 'control.lock' in out and 'Run the worker as the account that created this queue' in out
    assert 'this node stops serving' in out
    assert nodes.read(pool.root/'state.json')['failure'] is None


def test_supervisor_idle_status_reports_the_queue_not_startup(tmp_path):
    """COORD-08: an idle worker (finished its task, waiting for others) is reported as not assigned."""
    from scripts import v8_supervisor
    queue=tmp_path/'v8/olmo_math/.multi-node'
    nodes.atomic(queue/'state.json',dict(phase='train',failure=None,actions=[],failure_history=[]))
    rows={1:dict(ppid=1,start='1',state='S',v8=True,kind=None,env=dict(V8_LAUNCH_ID='L',V8_ROOT=str(tmp_path/'v8')))}
    text=' | '.join(v8_supervisor.before_model_work(rows,'L',tmp_path/'v8',tmp_path/'console.log',host='h'))
    assert 'NOT ASSIGNED olmo_math: queue phase=train' in text and 'startup' not in text.lower()


def test_child_exit_record_turns_torchrun_exit_1_into_an_interruption(tmp_path,monkeypatch):
    """RT exit-75 contract: torchrun exits 1, rank 0 records exit 75 in the per-attempt $V8_EXIT_FILE."""
    record=tmp_path/'a.exit.json'
    assert nodes.child_exit(1,record)==1
    record.write_text(json.dumps(dict(exit_code=75,status='interrupted')));assert nodes.child_exit(1,record)==75
    record.write_text(json.dumps(dict(exit_code=75,status='busy')));assert nodes.child_exit(1,record)==75
    record.write_text(json.dumps(dict(exit_code=1,status='failed')));assert nodes.child_exit(1,record)==1
    assert nodes.child_exit(0,record)==0
    # The actor path (all queue coordinators): V8_EXIT_FILE is set per attempt and read after the child.
    monkeypatch.syspath_prepend(str(SCRIPTS));worker=importlib.import_module('v8_runtime_worker')
    import srgc_rebuttal.cluster as srgc_cluster
    seen={}
    def fake_run_child(command,log,environment,**_):
        seen.update(environment);Path(environment['V8_EXIT_FILE']).write_text(json.dumps(dict(exit_code=75,status='interrupted')))
        return 1
    monkeypatch.setattr(srgc_cluster,'run_child',fake_run_child)
    runtime=type('R',(),{})();runtime.environment={'SRGC_PROGRESS_DIR':str(tmp_path/'progress')};runtime.gpus=['0']
    runtime.cohort='olmo_math';runtime.heartbeat=lambda *_:None
    log=tmp_path/'logs/attempt-1.log';log.parent.mkdir()
    stale=log.with_name('attempt-1.exit.json');stale.write_text(json.dumps(dict(exit_code=75,status='busy')))
    assert worker.run_srgc_child(runtime,['true'],log,[],threading.Event())==75
    assert Path(seen['V8_EXIT_FILE']).parent==tmp_path/'progress'
    assert Path(seen['V8_EXIT_FILE'])!=stale and stale.exists()
    def fails(command,log,environment,**_):return 1   # no record written: stale record must not be reused
    stale.write_text(json.dumps(dict(exit_code=75,status='busy')))
    monkeypatch.setattr(srgc_cluster,'run_child',fails)
    assert worker.run_srgc_child(runtime,['true'],log,[],threading.Event())==1


def test_qwen_binding_records_fla_core_and_old_bindings_still_resume(tmp_path,monkeypatch):
    """The Qwen child enforces fla-core; new Qwen queues record it, old bindings without it keep matching."""
    from scripts import v8_queue_runtime as queue
    assert 'fla-core' in nodes.python_identity(sys.executable,nodes.QWEN_PACKAGES)['packages']
    assert 'fla-core' not in nodes.python_identity(sys.executable)['packages']
    monkeypatch.setenv('V8_CODE_ROOT',str(REPO))
    qwen=nodes.Runtime(REPO/'v8/experiments',tmp_path/'v8','qwen_mbpp',sys.executable,['0','1','2','3'])
    olmo=nodes.Runtime(REPO/'v8/experiments',tmp_path/'v8','olmo_math',sys.executable,['0','1','2','3'])
    assert 'fla-core' in qwen.binding['packages'] and 'fla-core' not in olmo.binding['packages']
    env=dict(V8_ROOT=str(tmp_path/'v8'))
    assert queue.existing_python(qwen.binding,env,'qwen_mbpp')
    old=dict(qwen.binding,packages={k:v for k,v in qwen.binding['packages'].items() if k!='fla-core'})
    assert queue.existing_python(old,env,'qwen_mbpp')
    other=dict(qwen.binding,packages=dict(qwen.binding['packages'],**{'fla-core':'0.0.0-not-installed'}))
    with pytest.raises(ValueError,match='original Python/package versions'):queue.existing_python(other,env,'qwen_mbpp')


def launcher_copy(tmp_path):
    scripts=tmp_path/'scripts';scripts.mkdir()
    shutil.copy(SCRIPTS/'run_v8_nodes.sh',scripts/'run_v8_nodes.sh')
    (scripts/'v8_status.py').write_text('import sys,json;print(json.dumps(sys.argv[1:]))')
    (scripts/'v8_stop.sh').write_text('echo STOP-RAN\n')
    cohorts=tmp_path/'v8/experiments/configs/cohorts';cohorts.mkdir(parents=True);(cohorts/'qwen_math.json').write_text('{}')
    return scripts/'run_v8_nodes.sh'


def test_launcher_words_watch_seconds_other_cohorts_and_stop_note(tmp_path):
    """COORD-09"""
    script=launcher_copy(tmp_path);env={'PATH':'/usr/bin:/bin','HOME':str(tmp_path)}
    run=lambda *args:subprocess.run(['sh',str(script),*args],capture_output=True,text=True,env=env,timeout=30)
    out=run('status','--watch','30');assert out.returncode==0,out.stderr
    assert json.loads(out.stdout)==['--watch','30']
    assert json.loads(run('qwen_math','status').stdout)==['--cohort','qwen_math']
    assert json.loads(run('math','status','--watch').stdout)==['--cohort','olmo_math','--watch']
    out=run('math','stop');assert 'STOP-RAN' in out.stdout and 'all cohorts' in out.stderr
    bindir=tmp_path/'bin';bindir.mkdir()  # no nvidia-smi: status must not query GPUs here
    out=subprocess.run([sys.executable,str(SCRIPTS/'v8_status.py'),'--cohort','qwen_math'],capture_output=True,text=True,
                       env=dict(PATH=str(bindir),V8_ROOT=str(tmp_path/'v8'),HOME=str(tmp_path),PYTHONDONTWRITEBYTECODE='1'),timeout=30)
    assert out.returncode==0 and 'V8 status · ' in out.stdout and 'Qwen MATH not started' in ' '.join(out.stdout.split()),out.stderr


def test_supervisor_waits_for_a_safe_update_boundary_before_sigkill(tmp_path):
    """COORD-10: one Ctrl-C waits up to V8_STOP_WAIT_SECONDS (here a child needing 10.5 s > the old fixed 10 s)."""
    for name in ('v8_supervisor.py','v8_processes.py','v8_probe.py','v8_control_output.py','v8_progress_text.py'):
        shutil.copyfile(SCRIPTS/name,tmp_path/name)
    (tmp_path/'cluster_child.py').write_text(
        'import signal,sys,time\nfrom pathlib import Path\n'
        'def stop(*_):\n time.sleep(10.5);Path(__file__).with_name("saved-at-boundary").write_text("ok");sys.exit(75)\n'
        'signal.signal(signal.SIGTERM,stop)\nprint("RANK_READY",flush=True)\nwhile True:time.sleep(.02)\n')
    (tmp_path/'v8_bootstrap.py').write_text(
        'import signal,subprocess,sys\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
        'p=subprocess.Popen([sys.executable,str(Path(__file__).with_name("cluster_child.py"))])\nsys.exit(p.wait())\n')
    env=dict(os.environ,V8_ROOT=str(tmp_path/'v8'),V8_STOP_WAIT_SECONDS='40',V8_STOP_PROGRESS_SECONDS='1',TMPDIR=str(tmp_path))
    parent=subprocess.Popen([sys.executable,str(tmp_path/'v8_supervisor.py')],env=env,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,text=True)
    try:
        assert parent.stdout.readline().strip()=='RANK_READY'
        start=time.monotonic();parent.send_signal(signal.SIGINT)
        out,_=parent.communicate(timeout=40)
        assert parent.returncode==130,out
        assert (tmp_path/'saved-at-boundary').exists(),'child was killed before its safe boundary:\n'+out
        assert time.monotonic()-start>=10 and 'still waiting' in out and 'V8_STOP_WAIT_SECONDS' in out
    finally:
        if parent.poll() is None:parent.kill();parent.wait()


def test_stop_grace_setting():
    from scripts import v8_processes as proc
    assert proc.stop_grace({})==900 and proc.stop_grace({'V8_STOP_WAIT_SECONDS':'30'})==30
    assert proc.stop_grace({'V8_STOP_WAIT_SECONDS':'nan'})==900 and proc.stop_grace({'V8_STOP_WAIT_SECONDS':'-1'})==900


def test_actor_exits_quietly_when_the_dispatcher_closes_with_an_unread_reply(tmp_path):
    """COORD-04: e.g. Ctrl-C while a finished CPU-stage reply is still unread -> no ConnectionResetError traceback."""
    root,plan=fixture_plan(tmp_path,cohorts=('olmo_math',))
    code='''import json,sys,time
from pathlib import Path
import v8_dispatch,v8_rpc
plan=json.loads(Path(sys.argv[1]).read_text());plan['gpus']=['0','1','2','3'];plan['worker']='fixture.1'
Path(sys.argv[1]).write_text(json.dumps(plan))
actor=v8_dispatch.RuntimeProxy(plan['entries'][0],plan,Path(sys.argv[1]))
assert actor.error is None,actor.error
v8_rpc.send(actor.channel,dict(method='verify_node',args=[]));time.sleep(.5)  # reply arrives, never read
actor.close();print('ACTOR_EXIT',actor.process.returncode,flush=True)
'''
    result=subprocess.run([sys.executable,'-c',code,str(plan)],capture_output=True,text=True,timeout=60,
                          env=dict(os.environ,PYTHONPATH=str(SCRIPTS),PYTHONDONTWRITEBYTECODE='1',V8_ROOT=str(root)))
    assert 'ACTOR_EXIT 0' in result.stdout,result.stdout+result.stderr
    assert 'Traceback' not in result.stderr and 'ConnectionResetError' not in result.stderr,result.stderr


def test_plain_worker_does_not_reblock_a_queue_waiting_for_a_requested_retry(tmp_path):
    """--retry-failed on one node is a queue-level request: plain nodes wait for the retry delay too."""
    from test_srgc_v8_task_failure_isolation import setup,claim
    pool,runtime,stop=setup(tmp_path)
    lease,key,task,attempt=claim(pool,runtime,stop);pool.finish(key,9,runtime,stop);lease.close()
    for _ in range(2):
        lease,other,task,attempt=claim(pool,runtime,stop);runtime.completed.add(task['run_id'])
        pool.finish(other,0,runtime,stop);lease.close()
    retrying=nodes.Pool(pool.root,pool.binding,retry_delay=3600);retrying.initialize(retry=True)
    plain=nodes.Pool(pool.root,pool.binding,retry_delay=3600)
    assert plain.tick(runtime,'plain',stop)[0]=='waiting'
    assert nodes.read(pool.root/'state.json')['failure'] is None
    retrying.retry_delay=0;action,claimed=retrying.tick(runtime,'retry',stop)
    assert action=='claimed' and claimed[1]==key;claimed[0].close()
