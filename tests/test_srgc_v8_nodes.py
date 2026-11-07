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
    def __init__(self,path,*,fail=None,no_output=None):
        self.path=Path(path);self.fail=fail;self.no_output=no_output
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
            time.sleep(.16)
        else:time.sleep(.035)
        if name==self.fail:return 9
        if name!=self.no_output:nodes.atomic(self.file('done-'+name),dict(at=time.time()))
        return 0


def run_fake(root,runtime_path,worker,fail=None,no_output=None):
    runtime=FakeRuntime(runtime_path,fail=fail,no_output=no_output)
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
    queue,runtime,rc=launch_workers(tmp_path)
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


def test_conflicting_node_versions_refused_before_any_job(tmp_path):
    pool=nodes.Pool(tmp_path/'queue',{'version':1});pool.initialize()
    with pytest.raises(ValueError,match='identity differs'):
        nodes.Pool(pool.root,{'version':2}).initialize()
    assert not list(pool.root.glob('jobs/*.json'))


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
    with pytest.raises(FileNotFoundError):
        nodes.worker_loop(pool,runtime,'n',threading.Event(),poll=.001)
    assert 'missing python' in nodes.read(pool.root/'state.json')['failure']['detail']


def test_mismatched_joining_worker_does_not_poison_existing_queue(tmp_path):
    runtime=FakeRuntime(tmp_path/'runtime');pool=nodes.Pool(tmp_path/'q',runtime.binding);pool.initialize()
    wrong=nodes.Pool(pool.root,dict(runtime.binding,code='different'))
    with pytest.raises(ValueError,match='identity differs'):
        nodes.worker_loop(wrong,runtime,'bad-node',threading.Event(),poll=.001)
    assert nodes.read(pool.root/'state.json')['failure'] is None


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


@pytest.mark.parametrize('cohort',['olmo_math','qwen_mbpp'])
def test_cli_rejects_overlapping_gpu_group_without_starting_a_model(tmp_path,cohort):
    import socket,tempfile
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('V8 runtime source required for CLI preflight test')
    ids=[tmp_path.name+'-virtual'+str(i) for i in range(4)]
    path=Path(tempfile.gettempdir())/'srgc-v8-gpu-locks'/str(os.getuid())/(nodes.digest([socket.gethostname(),ids[2]])+'.lock')
    held=nodes.Lock(path);assert held.acquire()
    env=dict(os.environ,V8_ALLOW_ANY_ROOT='1',V8_CODE_ROOT='/home/kms/dev/offpolicy-misranking',CUDA_VISIBLE_DEVICES='')
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


class ToyRuntime(nodes.Runtime):
    """Actual v8 planner, runner and evaluator with its own CPU ToyBackend."""
    def __init__(self,runner,root):
        super().__init__(runner,root,'olmo_math',sys.executable,['0','1','2','3'])
        from switching import backends,coderepo,smoke,spec
        from switching.engine import make_engine_class
        self.backends,self.smoke=backends,smoke
        self.srgc=coderepo.srgc();self.PE=make_engine_class(self.srgc)
        self.feat,self.ans,self.data=backends.toy_problem(0)
        self.cfg=spec.resolve('olmo_math',role='smoke',extra={'seeds':{'training':[5,6,7]},
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


def test_real_v8_cpu_three_worker_branch_bank(tmp_path):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('Set V8_NODES_TEST_RUNTIME to v8/experiments for the actual CPU runner integration')
    ctx=mp.get_context('fork')
    processes=[ctx.Process(target=toy_worker,args=(runner,tmp_path,'node'+str(i))) for i in range(3)]
    for process in processes:process.start()
    for process in processes:
        process.join(60)
        if process.is_alive():process.terminate();process.join();pytest.fail('real V8 CPU worker did not finish')
    assert [nodes.read(tmp_path/(f'node{i}-exit.json'))['exit_code'] for i in range(3)]==[0,0,0]
    root=tmp_path/'olmo_math';queue=root/'.multi-node'
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
    nodes.atomic(tmp_path/'integration-summary.json',dict(scope='CPU multi-process software integration; not scientific RLVR results',
                 workers=3,training_seeds=[5,6,7],matched_updates=75,safety_cap=100,tasks=len(jobs),
                 no_duplicate_execution=True,all_tasks_done=True,new_GPU_jobs=0))
