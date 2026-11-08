"""CPU multi-process scheduling tests; no model/GPU/scientific measurements.

Run: python -m pytest --noconftest -q tests/test_srgc_v8_nodes.py
The unrelated root historical trainer fixture is not needed by these tests.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes


def task(name,kind='train',parent=None):
    return dict(kind=kind,run_id=name,method_id=name,training_seed=5,cohort_id='fixture',role='smoke',
                config_hash=nodes.digest(name),parent=parent or {},horizon=100,
                snapshot_sha256=nodes.digest(['snapshot',name]),evaluation_spec_hash=nodes.digest(['eval',name]))


class FakeRuntime:
    def __init__(self,path,*,fail=None,no_output=None,overlap_gate=False):
        self.path=Path(path);self.fail=fail;self.no_output=no_output;self.overlap_gate=overlap_gate
        self.binding=dict(schema=nodes.SCHEMA,cohort='fixture',code='fake1')

    def file(self,name):return self.path/name

    def verify_node(self):pass

    def barrier(self,action,fd,stop,device_fds):
        if action=='freeze':
            assert all(self.done(task('branch'+str(i))) for i in range(2))
            assert not self.file('done-eval0').exists()
        if action=='finalize':assert all(self.done(task('eval'+str(i),'evaluate')) for i in range(2))
        self.path.mkdir(parents=True,exist_ok=True)
        nodes.atomic(self.file(action),dict(at=time.time()))

    def plan(self,phase):
        if phase=='train':
            trajectories=[task('on'+str(i)) for i in range(2)]
            for t in trajectories:t['method_id']='on_monitored'
            branches=[task('branch'+str(i),parent={'checkpoint':i}) for i in range(2) if self.file('checkpoint'+str(i)).exists()]
            return trajectories+branches,all(self.done(t) for t in trajectories)
        if phase=='extension':
            assert self.file('freeze').exists()
            return [task('extend')],True
        assert self.done(task('extend'))
        return [task('eval'+str(i),'evaluate') for i in range(2)],True

    def native_busy(self,t):return False

    def ready(self,t):return not t['parent'] or self.file('checkpoint'+str(t['parent']['checkpoint'])).exists()

    def done(self,t):return self.file('done-'+t['run_id']).exists()

    def execute(self,t,matrix,fd,stop,device_fds,attempt):
        name=t['run_id'];self.path.mkdir(parents=True,exist_ok=True)
        nodes.atomic(self.file('start-'+name+'-'+attempt['id']),dict(at=time.time(),worker=attempt['worker']))
        if name.startswith('on'):
            time.sleep(.025);nodes.atomic(self.file('checkpoint'+name[-1]),dict(at=time.time()))
            if self.overlap_gate:
                # Hold the parent open until a worker actually starts a ready branch.
                # A fixed160ms window made fsync/loaded CI decide the scheduling assertion.
                deadline=time.monotonic()+10
                while not list(self.path.glob('start-branch*')) and time.monotonic()<deadline:time.sleep(.005)
            else:time.sleep(.16)
        else:time.sleep(.035)
        if name==self.fail:return 9
        if name!=self.no_output:nodes.atomic(self.file('done-'+name),dict(at=time.time()))
        return 0


def run_fake(root,runtime_path,worker,fail=None,no_output=None,overlap_gate=False):
    runtime=FakeRuntime(runtime_path,fail=fail,no_output=no_output,overlap_gate=overlap_gate)
    try:
        rc=nodes.worker_loop(nodes.Pool(root,runtime.binding),runtime,worker,threading.Event(),poll=.003)
    except Exception:rc=99
    Path(str(root)+'.'+worker+'.rc').write_text(str(rc))


def launch_workers(tmp_path,count=3,**kwargs):
    ctx=mp.get_context('fork')
    root=tmp_path/'queue';runtime=tmp_path/'runtime'
    processes=[ctx.Process(target=run_fake,args=(root,runtime,'node'+str(i)),kwargs=kwargs) for i in range(count)]
    for p in processes:p.start()
    for p in processes:
        p.join(15)
        if p.is_alive():p.terminate();p.join();pytest.fail('worker did not reach a terminal queue state')
    return root,runtime,[int(Path(str(root)+'.node'+str(i)+'.rc').read_text()) for i in range(count)]


def test_three_nodes_share_work_without_duplicates_and_observe_barriers(tmp_path):
    queue,runtime,rc=launch_workers(tmp_path,overlap_gate=True)
    assert rc==[0,0,0]
    assert nodes.read(queue/'state.json')['phase']=='complete'
    starts=list(runtime.glob('start-*'))
    assert len(starts)==7
    owners={nodes.read(path)['worker'] for path in starts}
    assert len(owners)>=2
    assert len(list(queue.glob('jobs/*.json')))==7
    assert all(len(nodes.read(p)['attempts'])==1 for p in queue.glob('jobs/*.json'))
    assert all(nodes.read(p)['state']=='done' for p in queue.glob('jobs/*.json'))
    # An idle node starts a branch while another node still trains its parent trajectory.
    branch=min(nodes.read(p)['at'] for p in runtime.glob('start-branch*'))
    trajectory_end=min(nodes.read(p)['at'] for p in runtime.glob('done-on*'))
    assert branch<trajectory_end
    assert nodes.read(runtime/'freeze')['at']>=max(nodes.read(p)['at'] for p in runtime.glob('done-branch*'))
    assert min(nodes.read(p)['at'] for p in runtime.glob('start-eval*'))>=nodes.read(runtime/'done-extend')['at']


def test_more_workers_than_tasks_wait_instead_of_skipping_phase(tmp_path):
    queue,runtime,rc=launch_workers(tmp_path,count=6)
    assert rc==[0]*6
    assert len(list(runtime.glob('start-*')))==7
    assert nodes.read(queue/'state.json')['phase']=='complete'


@pytest.mark.parametrize('mode',['fail','no_output'])
def test_failed_task_or_success_without_output_never_becomes_complete(tmp_path,mode):
    queue,runtime,rc=launch_workers(tmp_path,**{mode:'branch0'})
    assert all(value!=0 for value in rc)
    state=nodes.read(queue/'state.json')
    assert state['failure'] and state['phase']=='train'
    assert not (runtime/'freeze').exists()
    assert not list(runtime.glob('start-eval*'))


def test_failed_task_retry_preserves_failure_and_attempt_history(tmp_path):
    queue,runtime,rc=launch_workers(tmp_path,fail='branch0')
    pool=nodes.Pool(queue,FakeRuntime(runtime).binding)
    assert nodes.worker_loop(pool,FakeRuntime(runtime),'retry',threading.Event(),poll=.002,retry=True)==0
    state=nodes.read(queue/'state.json')
    assert state['phase']=='complete' and len(state['failure_history'])==1
    failed=next(nodes.read(p) for p in queue.glob('jobs/*.json') if nodes.read(p)['task']['run_id']=='branch0')
    assert [a['exit_code'] for a in failed['attempts']]==[9,0]


def test_conflicting_node_versions_refused_once_training_started(tmp_path):
    pool=nodes.Pool(tmp_path/'queue',{'version':1});pool.initialize()
    nodes.atomic(pool.root/'jobs'/'started.json',dict(task=task('a'),phase='train',state='pending',attempts=[]))
    with pytest.raises(ValueError,match='identity differs'):
        nodes.Pool(pool.root,{'version':2}).initialize()
    assert nodes.read(pool.root/'binding.json')=={'version':1}


def test_corrected_launcher_replaces_a_failed_setup_before_any_training(tmp_path):
    """Regression: a failed/interrupted prepare with an older bundle must not lock the queue forever."""
    old=nodes.Pool(tmp_path/'queue',{'version':1});old.initialize();old.fail('locate failed')
    new=nodes.Pool(old.root,{'version':2});new.initialize()
    state=nodes.read(old.root/'state.json')
    assert nodes.read(old.root/'binding.json')=={'version':2}
    assert state['failure'] is None and state['failure_history'][0]['detail']=='locate failed'
    assert state['actions'][-1]['action']=='retry_failed_setup_with_verified_source'
    assert list((old.root/'setup-history').glob('*.json'))
    # the still-running old worker notices and stops serving this cohort instead of mixing versions
    with old.control() as control:
        assert control is not None
        with pytest.raises(nodes.BindingMismatch,match='replaced'):old.tick(None,'old',threading.Event())


def test_scientific_task_conflict_rejected_but_stage_alias_is_shared(tmp_path):
    pool=nodes.Pool(tmp_path,{'cohort':'fixture'});pool.initialize()
    t=task('same');pool.register(dict(t,stage='bank-branches',deps=['on']), 'train')
    pool.register(dict(t,stage='destination',deps=[]),'train')
    assert len(list(tmp_path.glob('jobs/*.json')))==1
    with pytest.raises(ValueError,match='Conflicting task'):
        pool.register(dict(t,horizon=200),'train')


def test_checkpoint_completion_alone_does_not_skip_held_lease(tmp_path):
    runtime=FakeRuntime(tmp_path/'runtime');runtime.path.mkdir()
    pool=nodes.Pool(tmp_path/'queue',runtime.binding);pool.initialize()
    t=task('on0');t['method_id']='on_monitored';key=pool.register(t,'train')
    state=nodes.read(pool.root/'state.json');state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    nodes.atomic(runtime.file('done-on0'),{})
    held=nodes.Lock(pool.root/'leases'/(key+'.lock'));assert held.acquire()
    try:
        action,claim=pool.tick(runtime,'other',threading.Event())
        if claim:claim[0].close()
        assert nodes.read(pool.root/'state.json')['phase']=='train'
        assert nodes.read(pool.root/'jobs'/(key+'.json'))['state']=='pending'
    finally:held.close()


def test_orphaned_worker_lease_is_reclaimed_without_deleting_old_attempt(tmp_path):
    runtime=FakeRuntime(tmp_path/'runtime');runtime.path.mkdir()
    pool=nodes.Pool(tmp_path/'queue',runtime.binding);pool.initialize()
    state=nodes.read(pool.root/'state.json');state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    only=task('on0');only['method_id']='on_monitored'
    runtime.plan=lambda phase:([only],False)
    action,claim=pool.tick(runtime,'gone',threading.Event());assert action=='claimed'
    lease,key,t,attempt=claim;lease.close()
    action,claim=pool.tick(runtime,'replacement',threading.Event());assert action=='claimed'
    newlease,newkey,_,_=claim
    try:
        assert key==newkey
        job=nodes.read(pool.root/'jobs'/(key+'.json'))
        assert len(job['attempts'])==2
        assert job['attempts'][0]['worker']=='gone'
        assert job['attempts'][0]['lease_released_without_worker_completion']
    finally:newlease.close()


def test_child_inherits_lease_after_parent_reference_closes(tmp_path):
    path=tmp_path/'lease';lease=nodes.Lock(path);assert lease.acquire()
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.25)'],pass_fds=(lease.fd,))
    lease.close()
    other=nodes.Lock(path)
    try:
        assert not other.acquire()
        child.wait(timeout=5)
        assert other.acquire()
    finally:
        other.close()
        if child.poll() is None:child.terminate();child.wait()


def test_empty_or_corrupt_completed_plan_does_not_report_success(tmp_path):
    runtime=FakeRuntime(tmp_path/'runtime');runtime.path.mkdir()
    pool=nodes.Pool(tmp_path/'queue',runtime.binding);pool.initialize()
    state=nodes.read(pool.root/'state.json');state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    runtime.plan=lambda phase:([],True)
    with pytest.raises(ValueError,match='empty experiment'):
        pool.tick(runtime,'n',threading.Event())
    def broken(phase):raise json.JSONDecodeError('bad','bad',0)
    runtime.plan=broken
    with pytest.raises(ValueError,match='no active writer'):
        pool.tick(runtime,'n',threading.Event())


def test_stop_request_exits_interrupted_without_advancing_phase(tmp_path):
    pool=nodes.Pool(tmp_path/'queue',{});stop=threading.Event();stop.set()
    assert nodes.worker_loop(pool,FakeRuntime(tmp_path/'r'),'n',stop,poll=.01)==75
    assert not (pool.root/'state.json').exists()


def test_missing_cpu_command_is_failure_not_infinite_metadata_retry(tmp_path):
    class Bad(FakeRuntime):
        def barrier(self,*args):raise FileNotFoundError('missing python')
    runtime=Bad(tmp_path/'r');pool=nodes.Pool(tmp_path/'q',runtime.binding)
    assert nodes.worker_loop(pool,runtime,'n',threading.Event(),poll=.001)==1
    assert 'missing python' in nodes.read(pool.root/'state.json')['failure']['detail']


def test_mismatched_joining_worker_does_not_poison_existing_queue(tmp_path):
    runtime=FakeRuntime(tmp_path/'runtime');pool=nodes.Pool(tmp_path/'q',runtime.binding);pool.initialize()
    nodes.atomic(pool.root/'workers'/'healthy.json',dict(worker='healthy-node',pid=1,at=time.time(),task=None))
    wrong=nodes.Pool(pool.root,dict(runtime.binding,code='different'))
    assert nodes.worker_loop(wrong,runtime,'bad-node',threading.Event(),poll=.001)==1
    assert nodes.read(pool.root/'state.json')['failure'] is None
    assert nodes.read(pool.root/'binding.json')==runtime.binding


def test_repair_missing_dispatch_matrix_after_coordinator_crash(tmp_path):
    pool=nodes.Pool(tmp_path,{'cohort':'fixture'});pool.initialize()
    t=task('one');key=pool.register(t,'train')
    matrix=tmp_path/'tasks'/(key+'.json');matrix.unlink()
    pool.register(t,'train')
    assert nodes.read(matrix)['tasks']==[t]
    nodes.atomic(matrix,dict(tasks=[dict(t,horizon=999)]))
    with pytest.raises(ValueError,match='task matrix'):
        pool.register(t,'train')


def test_sealed_phase_plan_is_not_recomputed_from_large_ledgers(tmp_path):
    runtime=FakeRuntime(tmp_path/'runtime');runtime.path.mkdir()
    pool=nodes.Pool(tmp_path/'q',runtime.binding);pool.initialize()
    state=nodes.read(pool.root/'state.json');state['phase']='extension';nodes.atomic(pool.root/'state.json',state)
    runtime.plan=lambda phase:([task('extend')],True)
    action,claim=pool.tick(runtime,'first',threading.Event());assert action=='claimed'
    claim[0].close()
    def forbidden(phase):raise AssertionError('sealed plan must be reused')
    runtime.plan=forbidden
    action,claim=pool.tick(runtime,'second',threading.Event());assert action=='claimed'
    claim[0].close()


def test_cpu_barrier_interruption_preserves_resumable_phase(tmp_path):
    stop=threading.Event()
    class Interrupted(FakeRuntime):
        def barrier(self,*args):
            stop.set();raise RuntimeError('CPU phase interrupted')
    runtime=Interrupted(tmp_path/'r');pool=nodes.Pool(tmp_path/'q',runtime.binding)
    assert nodes.worker_loop(pool,runtime,'n',stop,poll=.001)==75
    state=nodes.read(pool.root/'state.json')
    assert state['phase']=='prepare' and state['failure'] is None


def test_stop_between_claim_and_launch_does_not_start_gpu_child(tmp_path):
    stop=threading.Event();runtime=FakeRuntime(tmp_path/'r')
    class StoppingPool(nodes.Pool):
        def tick(self,*args):
            result=super().tick(*args)
            if result[0]=='claimed':stop.set()
            return result
    pool=StoppingPool(tmp_path/'q',runtime.binding)
    assert nodes.worker_loop(pool,runtime,'n',stop,poll=.001)==75
    assert not list(runtime.path.glob('start-*'))
    state=nodes.read(pool.root/'state.json')
    assert state['failure'] is None and state['phase']=='train'


def test_native_training_lock_blocks_dispatch_without_loading_a_model(tmp_path):
    runtime=nodes.Runtime.__new__(nodes.Runtime);runtime.root=tmp_path
    t=task('native');path=tmp_path/'runs'/'native'/'.execution.lock'
    held=nodes.Lock(path);assert held.acquire()
    try:assert runtime.native_busy(t)
    finally:held.close()
    assert not runtime.native_busy(t)


@pytest.mark.parametrize('reward',[float('nan'),float('inf'),True,-.1,1.1])
def test_invalid_existing_evaluation_is_not_completion(tmp_path,reward):
    from types import SimpleNamespace
    runtime=nodes.Runtime.__new__(nodes.Runtime);runtime.root=tmp_path
    runtime.mx=SimpleNamespace(sha256_json=nodes.digest)
    t=dict(task('eval','evaluate'),completed_update=50,split='final',responses=8,question_ids_hash=nodes.digest(['q']))
    path=tmp_path/'evaluations'/f"{t['snapshot_sha256'][:16]}__{t['evaluation_spec_hash'][:12]}.json"
    nodes.atomic(path,dict(checkpoint_hash=t['snapshot_sha256'],evaluation_spec_hash=t['evaluation_spec_hash'],
                          completed_update=50,split='final',responses_per_question=8,n_questions=1,
                          per_question_correct={'q':3},reward_fraction=.375))
    # Deliberately use the permissive JSON writer to create a corrupt existing file.
    rec=nodes.read(path);rec['reward_fraction']=reward;path.write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='finite fraction'):runtime.done(t)


def test_valid_recorded_evaluation_and_training_identity(tmp_path):
    from types import SimpleNamespace
    runtime=nodes.Runtime.__new__(nodes.Runtime);runtime.root=tmp_path
    runtime.mx=SimpleNamespace(sha256_json=nodes.digest)
    t=dict(task('eval','evaluate'),completed_update=50,split='final',responses=8,question_ids_hash=nodes.digest(['q']))
    path=tmp_path/'evaluations'/f"{t['snapshot_sha256'][:16]}__{t['evaluation_spec_hash'][:12]}.json"
    nodes.atomic(path,dict(checkpoint_hash=t['snapshot_sha256'],evaluation_spec_hash=t['evaluation_spec_hash'],
                          completed_update=50,split='final',responses_per_question=8,n_questions=1,
                          per_question_correct={'q':3},reward_fraction=.375))
    assert runtime.done(t)
    t=task('train');manifest=tmp_path/'runs'/'train'/'manifest.json'
    nodes.atomic(manifest,dict(t,status='completed_horizon',completed_updates=100))
    assert runtime.done(t)
    nodes.atomic(manifest,dict(t,training_seed=99,status='completed_horizon',completed_updates=100))
    with pytest.raises(ValueError,match='identity/config'):runtime.done(t)


def test_frozen_input_bundle_change_is_rejected_before_launch(tmp_path):
    from types import SimpleNamespace
    runtime=nodes.Runtime.__new__(nodes.Runtime);runtime.root=tmp_path
    runtime.cfg={'seeds':{'training':[5]}}
    runtime.binding={'trainer_sha256':'expected'}
    runtime.code=SimpleNamespace(implementation_digest=lambda:'expected')
    plan=tmp_path/'plan.json';nodes.atomic(plan,{'model':'fixture'})
    bundle=tmp_path/'bundle.json';nodes.atomic(bundle,{'questions':['q']})
    nodes.atomic(tmp_path/'locator.json',{'code':{'implementation_sha256':'expected'},'seeds':{'5':{
        'plan_path':str(plan),'plan_sha256':nodes.hashlib.sha256(plan.read_bytes()).hexdigest(),
        'input_bundle':str(bundle),'prefix_path':'prefix.pt','prefix_sha256':'a'*64,'prefix_completed_updates':25}}})
    runtime.freeze_inputs();runtime.verify_node()
    nodes.atomic(bundle,{'questions':['different']})
    with pytest.raises(ValueError,match='identity differs'):runtime.verify_node()


def test_active_child_holds_node_and_job_fds_and_handles_stop(tmp_path):
    runtime=nodes.Runtime.__new__(nodes.Runtime);runtime.environment=dict(os.environ);runtime.gpus=['cpu-fixture']
    stop=threading.Event();marker=tmp_path/'safe-stop'
    code="""import signal,sys,time
from pathlib import Path
def stop(*_):
 Path(sys.argv[1]).write_text('checkpoint retained');sys.exit(75)
signal.signal(signal.SIGTERM,stop)
print('ready',flush=True)
while True:time.sleep(.01)
"""
    timer=threading.Timer(.3,stop.set);timer.start()
    lease=nodes.Lock(tmp_path/'lease');assert lease.acquire()
    try:
        rc=runtime.process([sys.executable,'-c',code,str(marker)],tmp_path/'child.log',[lease.fd],stop)
    finally:timer.cancel();lease.close()
    assert rc==75
    assert marker.read_text()=='checkpoint retained'
    assert b'ready' in (tmp_path/'child.log').read_bytes()


@pytest.mark.parametrize('cohort',['olmo_math','qwen_mbpp','all'])
def test_cli_rejects_overlapping_gpu_group_without_starting_a_model(tmp_path,cohort):
    import socket
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('V8 runtime source required for CLI preflight test')
    ids=[tmp_path.name+'-virtual'+str(i) for i in range(4)]
    path=tmp_path/'gpu-node-locks'/(nodes.digest([socket.gethostname(),ids[2]])+'.lock')
    held=nodes.Lock(path);assert held.acquire()
    nosmi=tmp_path/'bin'/'nvidia-smi';nosmi.parent.mkdir();nosmi.write_text('#!/bin/sh\nexit 9\n');nosmi.chmod(0o755)
    env=dict(os.environ,V8_ALLOW_ANY_ROOT='1',V8_CODE_ROOT=str(Path(nodes.__file__).resolve().parents[1]),
             CUDA_VISIBLE_DEVICES='',V8_GPU_LOCK_ROOT=str(tmp_path/'gpu-node-locks'),PATH=f'{nosmi.parent}:/usr/bin:/bin')
    try:
        result=subprocess.run([sys.executable,nodes.__file__,cohort,'--runner-root',runner,
                               '--root',str(tmp_path/'input'),'--python',sys.executable,'--gpus',','.join(ids)],
                              env=env,text=True,capture_output=True,timeout=10)
        assert result.returncode==75,result.stderr+result.stdout
        assert 'overlapping GPU group' in result.stderr
        assert not (tmp_path/'input'/cohort/'.multi-node/state.json').exists()
    finally:held.close()


def test_cli_status_is_read_only_and_needs_no_gpu_or_runner(tmp_path):
    root=tmp_path/'not-created'
    result=subprocess.run([sys.executable,nodes.__file__,'olmo_math','--root',str(root),'--status'],
                          text=True,capture_output=True,timeout=5)
    assert result.returncode==0
    assert json.loads(result.stdout)=={'phase':'not_started'}
    assert not root.exists()


def test_default_command_reports_both_cohorts_without_starting_anything(tmp_path):
    root=tmp_path/'not-created'
    result=subprocess.run([sys.executable,nodes.__file__,'--root',str(root),'--status'],
                          text=True,capture_output=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=={'olmo_math':{'phase':'not_started'},'qwen_mbpp':{'phase':'not_started'}}
    assert not root.exists()


def test_shell_entrypoint_defaults_to_both_cohorts_without_python_overrides(tmp_path):
    env={k:v for k,v in os.environ.items() if k not in ('V8_PYTHON','PAIR_PYTHON','SWITCH_PYTHON')}
    result=subprocess.run(['sh',str(Path(nodes.__file__).with_name('run_v8_nodes.sh')),
                           '--root',str(tmp_path/'not-created'),'--status'],env=env,text=True,capture_output=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert set(json.loads(result.stdout))=={'olmo_math','qwen_mbpp'}
    assert not (tmp_path/'not-created').exists()


def test_automatic_cohorts_keep_their_existing_python_environments(monkeypatch):
    monkeypatch.setenv('PAIR_PYTHON','/fixture/olmo/python')
    monkeypatch.setenv('SWITCH_PYTHON','/fixture/qwen/python')
    assert nodes.cohort_python('olmo_math','fallback')=='/fixture/olmo/python'
    assert nodes.cohort_python('qwen_mbpp','fallback')=='/fixture/qwen/python'
    monkeypatch.setenv('V8_PYTHON_QWEN_MBPP','/fixture/explicit-qwen/python')
    assert nodes.cohort_python('qwen_mbpp','fallback')=='/fixture/explicit-qwen/python'
    monkeypatch.delenv('PAIR_PYTHON')
    assert nodes.cohort_python('olmo_math','fallback')=='fallback'


def run_mixed_fake(root,worker,stop,blocked=False,failed=False):
    pairs=[]
    for cohort in ('a','b'):
        runtime=FakeRuntime(Path(root)/cohort/'runtime',fail='branch0' if failed and cohort=='a' else None)
        runtime.binding['cohort']=cohort
        if blocked and cohort=='a':runtime.ready=lambda _:False
        pairs.append((nodes.Pool(Path(root)/cohort/'queue',runtime.binding),runtime))
    rc=nodes.worker_loop_many(pairs,worker,stop,poll=.003)
    nodes.atomic(Path(root)/(worker+'-exit.json'),{'exit_code':rc})


def test_single_worker_automatically_finishes_both_cohorts(tmp_path):
    run_mixed_fake(tmp_path,'n',threading.Event())
    assert nodes.read(tmp_path/'n-exit.json')['exit_code']==0
    for cohort in ('a','b'):
        assert nodes.read(tmp_path/cohort/'queue/state.json')['phase']=='complete'
        assert len(list((tmp_path/cohort/'runtime').glob('start-*')))==7


def test_blocked_cohort_does_not_leave_worker_idle_while_other_has_work(tmp_path):
    ctx=mp.get_context('fork');stop=ctx.Event()
    process=ctx.Process(target=run_mixed_fake,args=(tmp_path,'n',stop),kwargs={'blocked':True})
    process.start()
    try:
        end=time.monotonic()+10
        while time.monotonic()<end:
            state=tmp_path/'b/queue/state.json'
            if state.exists() and nodes.read(state)['phase']=='complete':break
            time.sleep(.02)
        else:pytest.fail('ready cohort did not finish while first cohort was blocked')
    finally:
        stop.set();process.join(5)
        if process.is_alive():process.terminate();process.join();pytest.fail('mixed worker did not stop')
    assert not list((tmp_path/'a/runtime').glob('start-*'))
    assert nodes.read(tmp_path/'n-exit.json')['exit_code']==75


def test_failed_cohort_retains_failure_while_healthy_cohort_finishes(tmp_path):
    run_mixed_fake(tmp_path,'n',threading.Event(),failed=True)
    assert nodes.read(tmp_path/'n-exit.json')['exit_code']==1
    assert nodes.read(tmp_path/'a/queue/state.json')['failure']
    assert nodes.read(tmp_path/'a/queue/state.json')['phase']!='complete'
    assert nodes.read(tmp_path/'b/queue/state.json')['phase']=='complete'


class ToyRuntime(nodes.Runtime):
    """Actual v8 planner, runner and evaluator with its own CPU ToyBackend."""
    def __init__(self,runner,root,cohort='olmo_math'):
        super().__init__(runner,root,cohort,sys.executable,['0','1','2','3'])
        from switching import backends,coderepo,smoke,spec
        from switching.engine import make_engine_class
        self.backends,self.smoke=backends,smoke
        self.srgc=coderepo.srgc();self.PE=make_engine_class(self.srgc)
        self.feat,self.ans,self.data=backends.toy_problem(0)
        self.cfg=spec.resolve(cohort,role='smoke',extra={'seeds':{'training':[5,6,7]},
             'model':{'repository':'synthetic CPU ToyBackend','revision':'fixture'},
             'training':{'matched_update_endpoint':75,'hard_safety_update_cap':100},
             'bank':{'core_grid':[50],'extended_grid':[25]}})
        self.binding.update(config_sha256=spec.config_hash(self.cfg),software_test_only=True)

    def context(self,seed):
        path=self.root/'inputs'/f'seed-{seed}.json'
        data=self.cluster.load_bundle(path) if path.exists() else self.data
        return self.smoke.toy_ctx(self.root,seed,self.srgc,self.PE,self.feat,self.ans,data,75)

    def barrier(self,action,fd,stop,device_fds):
        from switching.common import atomic_write_json,sha256_file
        if action=='prepare':
            records={}
            for seed in self.cfg['seeds']['training']:
                self.smoke.make_prefix(self.context(seed),seed,25,self.srgc)
                directory=self.root/'prefixes'/f'seed-{seed}'
                bundle=self.root/'inputs'/f'seed-{seed}.json'
                data=dict(self.data,evaluation_ids=self.data['evaluation_ids'][:{5:100,6:70,7:40}[seed]])
                atomic_write_json(bundle,data)
                records[str(seed)]=dict(prefix_path=str(directory/'prefix.pt'),
                         prefix_sha256=sha256_file(directory/'prefix.pt'),prefix_completed_updates=25,
                         input_bundle=str(bundle))
            atomic_write_json(self.root/'locator.json',{'code':{'implementation_sha256':self.binding['trainer_sha256']},'seeds':records})
            self.freeze_inputs()
        elif action=='freeze':
            from switching.bank import SeedBank
            from switching.freeze import freeze_budget,write_frozen_budget
            assert not list((self.root/'evaluations').glob('*.json'))
            rows=[]
            for seed in self.cfg['seeds']['training']:
                rows+=SeedBank(self.root,self.cfg,self.cohort,seed,'rebuttal_bank',evals={},
                               prefix_cost=self.smoke.prefix_cost(self.root,seed)).policy_rows()
            write_frozen_budget(self.root/'frozen/budget.json',freeze_budget(rows))
        else:
            nodes.atomic(self.root/'software-finalized.json',dict(scope='CPU software test, not paper results'))

    def execute(self,t,matrix,fd,stop,device_fds,attempt):
        from switching.runner import execute_train,execute_eval
        ctx=self.context(t['training_seed'])
        result=execute_eval(t,ctx) if t['kind']=='evaluate' else execute_train(t,ctx)
        nodes.atomic(self.root/'executed'/(attempt['id']+'.json'),dict(task=nodes.task_key(t),kind=t['kind'],
                     worker=attempt['worker'],method=t.get('method_id'),result=result))
        return 75 if result.get('status')=='busy' else 0


def toy_worker(runner,root,worker):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['V8_FSYNC']='0'
    os.environ.setdefault('V8_CODE_ROOT',str(Path(__file__).resolve().parents[1]))
    runtime=ToyRuntime(runner,root)
    try:rc=nodes.worker_loop(nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime,worker,threading.Event(),poll=.015)
    except Exception:
        import traceback
        traceback.print_exc();rc=99
    nodes.atomic(Path(root)/(worker+'-exit.json'),{'exit_code':rc})


def test_real_v8_cpu_three_worker_branch_bank(tmp_path,monkeypatch):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('Set V8_NODES_TEST_RUNTIME to v8/experiments for the actual CPU runner integration')
    ctx=mp.get_context('fork')
    processes=[ctx.Process(target=toy_worker,args=(runner,tmp_path,'node'+str(i))) for i in range(3)]
    for process in processes:process.start()
    for process in processes:
        process.join(60)
        if process.is_alive():process.terminate();process.join();pytest.fail('real V8 CPU worker did not finish')
    assert [nodes.read(tmp_path/(f'node{i}-exit.json'))['exit_code'] for i in range(3)]==[0,0,0]
    nodes.atomic(tmp_path/'integration-summary.json',verify_toy_complete(runner,tmp_path,'olmo_math',monkeypatch))


def verify_toy_complete(runner,tmp_path,cohort,monkeypatch):
    root=tmp_path/cohort;queue=root/'.multi-node'
    assert nodes.read(queue/'state.json')['phase']=='complete'
    jobs=[nodes.read(path) for path in queue.glob('jobs/*.json')]
    executions=[nodes.read(path) for path in (root/'executed').glob('*.json')]
    assert len(executions)==len(jobs)==len({row['task'] for row in executions})
    assert len({row['worker'] for row in executions})>=2
    assert all(job['state']=='done' for job in jobs)
    assert all(len(job['attempts'])==1 for job in jobs)
    assert {job['task']['training_seed'] for job in jobs}=={5,6,7}
    assert sum(job['task'].get('method_id')=='bank_sr_t025' for job in jobs if job['phase']=='train')==3
    assert any(job['phase']=='extension' for job in jobs)
    assert any(job['phase']=='evaluation' for job in jobs)
    assert len({job['task']['question_ids_hash'] for job in jobs if job['phase']=='evaluation'})==3
    assert (root/'frozen/budget.json').exists() and (root/'software-finalized.json').exists()
    monkeypatch.setenv('V8_CODE_ROOT',os.environ.get('V8_CODE_ROOT',str(Path(__file__).resolve().parents[1])))
    cfg=ToyRuntime(runner,tmp_path,cohort).cfg
    from switching.bank import SeedBank
    budget=nodes.read(root/'frozen/budget.json')['value']
    evaluated={(job['task']['run_id'],job['task']['completed_update']):job['task']['snapshot_sha256']
               for job in jobs if job['phase']=='evaluation'}
    moved=[];selected=set()
    for seed in [5,6,7]:
        bank=SeedBank(root,cfg,cohort,seed,'rebuttal_bank',evals={})
        for row in bank.policy_rows(budget=budget):
            if row.get('budget_status')!='selected':continue
            tau=row['switch_update']
            branch=bank.branches.get((row['destination'],tau)) if row['status']=='triggered' else None
            choice=bank.budget_endpoint(row['family'],tau,branch,budget,
                       include_check=row['family'] in ('srgc','srgc_uniform_destination'))['endpoint']
            key=(choice['run_id'],choice['completed_update']);selected.add(key)
            assert evaluated.get(key)==choice['snapshot_hash']
            if choice['run_id']!=row['run_id']:moved.append(key)
    assert moved,'CPU fixture must select at least one extension or pre-switch trajectory endpoint'
    return dict(scope='CPU multi-process software integration; not scientific RLVR results',
                 workers=3,training_seeds=[5,6,7],matched_updates=75,safety_cap=100,tasks=len(jobs),
                 no_duplicate_execution=True,all_tasks_done=True,new_GPU_jobs=0,
                 budget_selected_checkpoints=len(selected),budget_endpoint_changed_run_cases=len(moved))


def toy_mixed_worker(runner,root,worker):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['V8_FSYNC']='0'
    os.environ.setdefault('V8_CODE_ROOT',str(Path(__file__).resolve().parents[1]))
    pairs=[]
    for cohort in ('olmo_math','qwen_mbpp'):
        runtime=ToyRuntime(runner,root,cohort)
        pairs.append((nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime))
    try:rc=nodes.worker_loop_many(pairs,worker,threading.Event(),poll=.015)
    except Exception:
        import traceback
        traceback.print_exc();rc=99
    nodes.atomic(Path(root)/(worker+'-exit.json'),{'exit_code':rc})


def test_real_v8_three_workers_automatically_share_both_cohorts(tmp_path,monkeypatch):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('V8 runtime source required for mixed-cohort integration')
    ctx=mp.get_context('fork')
    processes=[ctx.Process(target=toy_mixed_worker,args=(runner,tmp_path,'node'+str(i))) for i in range(3)]
    for process in processes:process.start()
    for process in processes:
        process.join(60)
        if process.is_alive():process.terminate();process.join();pytest.fail('mixed V8 CPU worker did not finish')
    assert [nodes.read(tmp_path/(f'node{i}-exit.json'))['exit_code'] for i in range(3)]==[0,0,0]
    summaries={cohort:verify_toy_complete(runner,tmp_path,cohort,monkeypatch) for cohort in ('olmo_math','qwen_mbpp')}
    owners={cohort:{nodes.read(p)['worker'] for p in (tmp_path/cohort/'executed').glob('*.json')} for cohort in summaries}
    assert owners['olmo_math']&owners['qwen_mbpp'],'workers must move between cohorts without fixed node shares'
    attempts={}
    for cohort in summaries:
        for path in (tmp_path/cohort/'.multi-node/jobs').glob('*.json'):
            for attempt in nodes.read(path)['attempts']:
                attempts.setdefault(attempt['worker'],[]).append((attempt['started_at'],attempt['ended_at']))
    for rows in attempts.values():
        rows.sort()
        assert all(a[1]<=b[0] for a,b in zip(rows,rows[1:])), 'one worker must not overlap GPU jobs across cohorts'
    nodes.atomic(tmp_path/'integration-summary.json',dict(scope='CPU multi-cohort software verification, not scientific RLVR measurements',
                 cohorts=summaries,total_tasks=sum(s['tasks'] for s in summaries.values()),workers=3,
                 no_overlapping_jobs_per_worker=True,no_duplicate_execution=True,new_GPU_jobs=0))


@pytest.mark.parametrize('cohort',['olmo_math','qwen_mbpp'])
def test_current_training_graph_has_55_ready_jobs_after_branch_parents_exist(tmp_path,cohort):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('V8 runtime source required for graph capacity verification')
    sys.path.insert(0,runner)
    from switching import matrix,spec
    cfg=spec.resolve(cohort,role='rebuttal_bank')
    class Locator:
        interval=25
        def prefix(self,seed):
            path=tmp_path/f'prefix-{seed}.pt';path.write_text('synthetic dependency fixture; not a model')
            return dict(kind='prefix',path=str(path),sha256=nodes.digest(seed),completed_update=25,run_id=None)
    loc=Locator();trajectories,_=matrix.build_stage('bank-trajectory',cfg,loc,tmp_path)
    assert len(trajectories)==5
    for task in trajectories:
        root=tmp_path/'runs'/task['run_id'];root.mkdir(parents=True)
        nodes.atomic(root/'manifest.json',dict(status='active'))
        rows=[]
        for update in range(50,275,25):
            path=root/f'step-{update}.pt';path.write_text('synthetic dependency fixture; not a model')
            rows.append(dict(completed_update=update,path=path.name,sha256=nodes.digest([task['run_id'],update]),
                             identity={'state_hash':nodes.digest(update)}))
        (root/'snapshots.jsonl').write_text(''.join(nodes.canonical(row)+'\n' for row in rows))
    tasks=list(trajectories)
    for stage in ('bank-branches','bank-extended','destination'):
        found,_=matrix.build_stage(stage,cfg,loc,tmp_path);tasks+=found
    unique={nodes.task_key(t):t for t in tasks}
    assert len(unique)==55
    assert all(nodes.Runtime.ready(None,t) for t in unique.values())


# ---- Claude debugging pass 2026-11-08: launcher regressions ---------------------------------
def test_cohort_python_follows_the_original_launchers(monkeypatch):
    for name in ('PAIR_PYTHON','SWITCH_PYTHON','QWEN_PYTHON','V8_PYTHON_OLMO_MATH','V8_PYTHON_QWEN_MBPP'):
        monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv('SWITCH_PYTHON','/fixture/mbpp/python')
    # Before: the OLMo MATH cohort fell back to SWITCH_PYTHON (the MBPP/Qwen interpreter).
    assert nodes.cohort_python('olmo_math','venv')=='venv'
    monkeypatch.setenv('QWEN_PYTHON','/fixture/qwen/python')
    assert nodes.cohort_python('qwen_mbpp','venv')=='/fixture/qwen/python'   # run_srgc_qwen35.sh precedence
    assert nodes.cohort_python('olmo_mbpp','venv')=='/fixture/mbpp/python'


def test_gpu_leases_use_the_original_srgc_uuid_names_on_the_group_volume(tmp_path):
    import hashlib
    rows=[dict(index=str(i),uuid=f'GPU-{i}',used_mib=0,total_mib=81559,name='NVIDIA H100 80GB HBM3') for i in range(4)]
    names=[n for _,n in nodes.gpu_lease_names(['0','1','2','3'],rows)]
    assert names==[hashlib.sha256(f'GPU-{i}'.encode()).hexdigest()+'.lock' for i in range(4)]
    env=dict(GROUP_VOLUME=str(tmp_path/'gv'),OM_USER='u')
    assert nodes.gpu_lock_root(env)==tmp_path/'gv/u/offpolicy-misranking/srgc-rebuttal/gpu-node-locks'
    assert nodes.gpu_lock_root(dict(env,OM_WORK=''))==tmp_path/'gv/u/offpolicy-misranking/srgc-rebuttal/gpu-node-locks'
    # an original SRGC worker's lease (srgc_rebuttal.runtime.lease = flock on the same file) excludes V8
    from srgc_rebuttal.runtime import lease
    path=nodes.gpu_lock_root(env)/names[1]
    with lease(path):
        assert not nodes.Lock(path).acquire()


def test_inventory_parses_nvidia_smi_and_busy_gpus_defer_claims(tmp_path):
    class R:
        def __init__(self,out,rc=0):self.stdout,self.returncode=out,rc
    out='0, GPU-a, 3, 81559, NVIDIA H100 80GB HBM3\n1, GPU-b, 70000, 81559, NVIDIA H100 80GB HBM3\n'
    rows=nodes.gpu_inventory(['0','1'],run=lambda *a,**k:R(out))
    assert [r['used_mib'] for r in rows]==[3,70000]
    assert nodes.gpu_inventory(['0','1'],run=lambda *a,**k:R('',9)) is None
    def missing(*a,**k):raise FileNotFoundError('nvidia-smi')
    assert nodes.gpu_inventory(['0'],run=missing) is None
    gate=nodes.GpuGate(['0','1'],inventory=lambda g:rows,processes=lambda:['4242, GPU-b, 69000'])
    ok,why=gate()
    assert not ok and 'GPU 1 70000 MiB' in why and '4242' in why
    runtime=FakeRuntime(tmp_path/'runtime');runtime.path.mkdir()
    pool=nodes.Pool(tmp_path/'q',runtime.binding);pool.initialize()
    state=nodes.read(pool.root/'state.json');state['phase']='extension';nodes.atomic(pool.root/'state.json',state)
    runtime.plan=lambda phase:([task('extend')],True)
    assert pool.tick(runtime,'n',threading.Event(),False)==('gpu_busy',None)
    assert all(not nodes.read(p)['attempts'] for p in pool.root.glob('jobs/*.json'))


def test_one_cohort_setup_failure_does_not_stop_the_other_cohort(tmp_path):
    class Bad(FakeRuntime):
        def barrier(self,action,*args):
            if action=='prepare':raise RuntimeError('CPU stage locate failed exit=1')
            return super().barrier(action,*args)
    bad=Bad(tmp_path/'a'/'runtime');good=FakeRuntime(tmp_path/'b'/'runtime')
    bad.binding=dict(bad.binding,cohort='a');good.binding=dict(good.binding,cohort='b')
    pairs=[(nodes.Pool(tmp_path/'a'/'queue',bad.binding),bad),(nodes.Pool(tmp_path/'b'/'queue',good.binding),good)]
    assert nodes.worker_loop_many(pairs,'n',threading.Event(),poll=.002)==1
    assert nodes.read(tmp_path/'b/queue/state.json')['phase']=='complete'
    assert 'locate failed' in nodes.read(tmp_path/'a/queue/state.json')['failure']['detail']


def test_unsupported_shared_filesystem_lock_explains_the_volume(tmp_path,monkeypatch):
    import errno
    def nolock(fd,op):raise OSError(errno.ENOLCK,'No locks available')
    monkeypatch.setattr(nodes.fcntl,'flock',nolock)
    with pytest.raises(ValueError,match='does not provide flock'):
        nodes.Lock(tmp_path/'x.lock').acquire()


def test_status_shows_running_task_log_and_its_age(tmp_path):
    pool=nodes.Pool(tmp_path/'q',{'cohort':'fixture'});pool.initialize()
    key=pool.register(task('run1'),'train')
    job=nodes.read(pool.root/'jobs'/(key+'.json'));job['attempts']=[dict(id='att1',worker='w1')];job['state']='running'
    nodes.atomic(pool.root/'jobs'/(key+'.json'),job)
    (pool.root/'logs').mkdir();(pool.root/'logs'/'att1.log').write_text('Loading weights: 100%\n')
    held=nodes.Lock(pool.root/'leases'/(key+'.lock'));assert held.acquire()
    try:row=pool.status()['leased_tasks'][0]
    finally:held.close()
    assert row['worker']=='w1' and row['log'].endswith('att1.log') and row['log_written_seconds_ago']>=0


def test_node_without_the_requested_gpus_is_refused_before_any_queue_write(tmp_path):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('V8 runtime source required for CLI preflight test')
    fake=tmp_path/'bin'/'nvidia-smi';fake.parent.mkdir()
    fake.write_text('#!/bin/sh\ncase " $* " in *" -L "*) echo "GPU 0: NVIDIA H100 (UUID: GPU-a)";echo "GPU 1: NVIDIA H100 (UUID: GPU-b)";exit 0;; esac\n'
                    'echo "No devices were found" >&2; exit 6\n')
    fake.chmod(0o755)
    env=dict(os.environ,V8_ALLOW_ANY_ROOT='1',V8_CODE_ROOT=str(Path(nodes.__file__).resolve().parents[1]),
             CUDA_VISIBLE_DEVICES='0,1,2,3',V8_GPU_LOCK_ROOT=str(tmp_path/'locks'),PATH=f"{fake.parent}:/usr/bin:/bin")
    result=subprocess.run([sys.executable,nodes.__file__,'--runner-root',runner,'--root',str(tmp_path/'v8'),
                           '--python',sys.executable],env=env,text=True,capture_output=True,timeout=30)
    assert result.returncode==2 and 'does not expose the GPUs 0,1,2,3' in result.stderr
    assert not (tmp_path/'v8').exists()


# ---- explicit --new-queue (coordinator decision 2026-11-08) -----------------------------------
def tree_digest(root):
    import hashlib
    out={}
    for path in sorted(Path(root).rglob('*')):
        if path.is_file():out[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def started_queue(tmp_path):
    root=tmp_path/'olmo_math'/'.multi-node';pool=nodes.Pool(root,{'cohort':'olmo_math','code':'v1.0.3'});pool.initialize()
    pool.register(task('traj'),'train')
    return root


def test_plain_start_with_another_bundle_still_refuses_with_plain_instructions(tmp_path):
    root=started_queue(tmp_path)
    with pytest.raises(nodes.BindingMismatch) as error:
        nodes.Pool(root,{'cohort':'olmo_math','code':'v1.0.4'}).initialize()
    assert 'run `sh scripts/run_v8_nodes.sh --new-queue` once on one node' in str(error.value)
    assert nodes.read(root/'binding.json')['code']=='v1.0.3'


def test_new_queue_keeps_old_directory_byte_identical_and_relocks(tmp_path,capsys):
    root=started_queue(tmp_path)
    pool=nodes.Pool(root,{'cohort':'olmo_math','code':'v1.0.4'},new_queue=True)
    with pool.control() as control:
        assert control is not None
        before=tree_digest(root)  # includes control.lock, as on a running cluster
        with pytest.raises(nodes.QueueReset) as reset:pool.initialize()
    kept=Path(str(reset.value))
    assert kept.name.startswith('.multi-node.superseded-') and tree_digest(kept)==before
    pool.initialize()  # second call (after re-locking) neither supersedes again nor refuses
    assert nodes.read(root/'binding.json')['code']=='v1.0.4' and not list(root.glob('jobs/*.json'))
    assert len(list(root.parent.glob('.multi-node.superseded-*')))==1
    out=capsys.readouterr().out
    assert str(kept) in out and 'Restart every other node with this same bundle' in out


def test_new_queue_refused_after_a_completed_update(tmp_path):
    root=started_queue(tmp_path)
    log=tmp_path/'olmo_math/runs/r1/attempts/a/updates.jsonl';log.parent.mkdir(parents=True)
    log.write_text('{"completed_updates": 26}\n')
    with pytest.raises(nodes.BindingMismatch,match='updates were already completed'):
        nodes.Pool(root,{'cohort':'olmo_math','code':'v1.0.4'},new_queue=True).initialize()
    assert nodes.read(root/'binding.json')['code']=='v1.0.3' and not list(root.parent.glob('.multi-node.superseded-*'))


def test_new_queue_refused_while_a_job_lease_is_held(tmp_path):
    root=started_queue(tmp_path)
    held=nodes.Lock(root/'leases'/'job.lock');assert held.acquire()
    try:
        with pytest.raises(nodes.BindingMismatch,match='lease'):
            nodes.Pool(root,{'cohort':'olmo_math','code':'v1.0.4'},new_queue=True).initialize()
    finally:held.close()
    assert nodes.read(root/'binding.json')['code']=='v1.0.3'


def test_new_queue_with_the_same_bundle_continues_the_queue(tmp_path):
    root=started_queue(tmp_path)
    nodes.Pool(root,{'cohort':'olmo_math','code':'v1.0.3'},new_queue=True).initialize()
    assert list(root.glob('jobs/*.json')) and not list(root.parent.glob('.multi-node.superseded-*'))


def test_worker_loop_relocks_the_fresh_queue_before_ticking(tmp_path):
    root=started_queue(tmp_path)
    class Rt:
        binding={'cohort':'olmo_math','code':'v1.0.4'}
        def verify_node(self):pass
    pool=nodes.Pool(root,Rt.binding,new_queue=True);seen=[]
    def tick(*args):
        probe=nodes.Lock(root/'control.lock');free=probe.acquire();probe.close()
        seen.append(not free);return 'complete',None
    pool.tick=tick
    assert nodes.worker_loop(pool,Rt(),'w',threading.Event(),poll=.001)==0
    assert seen==[True]  # the worker held the NEW directory's control.lock when it ticked


def test_bundle_mismatch_message_does_not_suggest_retry_failed(tmp_path,capsys):
    runtime=FakeRuntime(tmp_path/'runtime');pool=nodes.Pool(tmp_path/'q',runtime.binding);pool.initialize()
    pool.register(task('started'),'train')
    assert nodes.worker_loop(nodes.Pool(pool.root,dict(runtime.binding,code='newer')),runtime,'n',threading.Event(),poll=.001)==1
    out=capsys.readouterr().out
    assert '--new-queue' in out and '--retry-failed' not in out
