"""Sequential cohort scheduling with actual CPU workers/leases, no scientific measurements."""
import multiprocessing as mp
from pathlib import Path
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes


import pytest as _pytest


@_pytest.fixture(autouse=True)
def _cap_without_user_grant(monkeypatch):
    """These tests pin the failure cap itself; the one-time user retry grant (11-13) has its own tests."""
    monkeypatch.setenv('V8_USER_RETRY_GRANT', '0')


class Runtime:
    def __init__(self,base,cohort,*,release=None,fail=False):
        self.root=Path(base)/cohort;self.cohort=cohort;self.release=release;self.fail=fail
        self.binding=dict(schema=1,cohort=cohort,revision='sequential CPU fixture')

    def event(self,stage,task=''):
        nodes.atomic(self.root.parent/'events'/f'{time.time_ns()}-{self.cohort}-{stage}-{task}.json',
                     dict(cohort=self.cohort,stage=stage,task=task))

    def verify_node(self):self.event('verify')

    def barrier(self,stage,*args):self.event(stage)

    def plan(self,phase):
        if phase=='extension':return [],True
        return [dict(kind='train' if phase=='train' else 'evaluate',run_id=f'{self.cohort}-{phase}-{seed}',
                     method_id='on_monitored',training_seed=seed,config_hash=nodes.digest([self.cohort,phase,seed]),
                     parent={},horizon=275,snapshot_sha256=nodes.digest([self.cohort,seed]),
                     evaluation_spec_hash=nodes.digest('CPU evaluation fixture')) for seed in (5,6)],True

    def native_busy(self,task):return False
    def ready(self,task):return True
    def done(self,task):return (self.root/('done-'+task['run_id'])).exists()

    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        self.event('execute',task['run_id'])
        if self.release is not None and self.cohort=='olmo_math' and task['kind']=='train':
            while not self.release.wait(.01):
                if stop.is_set():return 75
        if self.fail:return 1
        (self.root/('done-'+task['run_id'])).write_text('CPU fixture only')
        return 0


def pairs(base,cohorts=nodes.DEFAULT_COHORTS,**kwargs):
    rows=[]
    for name in cohorts:
        runtime=Runtime(base,name,**kwargs)
        rows.append((nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime))
    return rows


def worker(base,name,stop,release):
    code=nodes.worker_loop_many(pairs(base,release=release),name,stop,poll=.005,cohort_order=nodes.DEFAULT_COHORTS)
    nodes.atomic(Path(base)/'exits'/name,dict(code=code))


def events(base):return [nodes.read(p) for p in sorted((base/'events').glob('*.json'))]


def test_all_four_finish_training_evaluation_and_finalize_in_declared_order(tmp_path):
    rows=pairs(tmp_path)
    keep=tmp_path/'olmo_math/step-000025.pt';keep.write_bytes(b'optimizer RNG reward checkpoint sentinel')
    assert nodes.worker_loop_many(rows,'cpu-node',threading.Event(),poll=.001,
                                  cohort_order=nodes.DEFAULT_COHORTS)==0
    recorded=events(tmp_path)
    for first,second in zip(nodes.DEFAULT_COHORTS,nodes.DEFAULT_COHORTS[1:]):
        final=next(i for i,row in enumerate(recorded) if row['cohort']==first and row['stage']=='finalize')
        following=next(i for i,row in enumerate(recorded) if row['cohort']==second)
        assert final<following
    assert all(pool.queue_state()['phase']=='complete' for pool,_ in rows)
    assert keep.read_bytes()==b'optimizer RNG reward checkpoint sentinel'


def test_six_simultaneous_nodes_spill_to_next_cohort_when_two_jobs_are_full(tmp_path):
    # User 11-15: 'mbpp 가 전부 할당 됬으면 다음으로 넘어가야될거 아니냐' - the four nodes that find
    # olmo_math fully allocated take ready work from the next cohorts instead of idling.
    ctx=mp.get_context('fork');stop=ctx.Event();release=ctx.Event();children=[]
    try:
        for i in range(6):
            child=ctx.Process(target=worker,args=(tmp_path,f'node-{i}',stop,release));child.start();children.append(child)
        deadline=time.monotonic()+60   # hang guard under machine load
        while time.monotonic()<deadline:
            claims=[row for row in events(tmp_path) if row['stage']=='execute']
            if any(row['cohort']!='olmo_math' for row in claims):break
            time.sleep(.01)
        assert [row for row in claims if row['cohort']=='olmo_math']==[row for row in claims[:2]]
        assert any(row['cohort']!='olmo_math' for row in claims),'idle nodes did not move to the next cohort'
        release.set()
        for child in children:
            child.join(60)
            assert not child.is_alive() and child.exitcode==0
        assert all(nodes.read(p)['code']==0 for p in (tmp_path/'exits').glob('*'))
        recorded=events(tmp_path)
        claims=[(row['cohort'],row['task']) for row in recorded if row['stage']=='execute']
        assert len(claims)==len(set(claims))==16
    finally:
        stop.set();release.set()
        for child in children:
            child.join(3)
            if child.is_alive():child.kill();child.join()


@pytest.mark.parametrize('phase',['train','primary_evaluation','extension','evaluation','finalize'])
def test_no_ready_jobs_or_late_phase_does_not_mean_complete(tmp_path,phase):
    rows=pairs(tmp_path)
    nodes.atomic(rows[0][0].root/'state.json',dict(phase=phase,failure=None))
    assert nodes.unfinished_cohort(rows,nodes.DEFAULT_COHORTS)=='olmo_math'


def test_complete_label_with_live_lease_still_blocks_next_cohort(tmp_path):
    rows=pairs(tmp_path);pool=rows[0][0]
    nodes.atomic(pool.root/'state.json',dict(phase='complete',failure=None))
    with nodes.Lock(pool.root/'leases/task.lock'):
        assert nodes.unfinished_cohort(rows,nodes.DEFAULT_COHORTS)=='olmo_math'
    assert nodes.unfinished_cohort(rows,nodes.DEFAULT_COHORTS)=='qwen_mbpp'


def test_missing_earlier_runtime_does_not_idle_the_node_but_is_reported(tmp_path,capsys):
    rows=pairs(tmp_path,cohorts=('qwen_mbpp',))
    assert nodes.worker_loop_many(rows,'cpu-node',threading.Event(),poll=.001,
                                  cohort_order=nodes.DEFAULT_COHORTS)==1
    assert rows[0][0].queue_state()['phase']=='complete'
    assert 'BLOCKED: every cohort this node can serve is done, but olmo_math is unavailable' in capsys.readouterr().out
    assert not (tmp_path/'olmo_math').exists()


def test_failed_cohorts_exhaust_finite_budget_and_the_node_exits(tmp_path):
    rows=pairs(tmp_path,fail=True)
    stop=threading.Event();timer=threading.Timer(60,stop.set);timer.start()   # hang guard
    try:assert nodes.worker_loop_many(rows,'cpu-node',stop,poll=.001,cohort_order=nodes.DEFAULT_COHORTS)==1
    finally:timer.cancel();timer.join()
    assert not stop.is_set()
    for path in (rows[0][0].root/'jobs').glob('*.json'):
        assert len(nodes.read(path)['attempts'])==nodes.MAX_ATTEMPTS
    for pool,_ in rows:
        for path in (pool.root/'jobs').glob('*.json'):
            assert len(nodes.read(path)['attempts'])<=nodes.MAX_ATTEMPTS


def test_old_primary_order_scope_no_longer_waits_for_later_cohort(tmp_path):
    rows=pairs(tmp_path);pool,runtime=rows[0]
    pool.initialize()
    runtime.primary_results_first=True;runtime.sequential_cohorts=nodes.DEFAULT_COHORTS
    state=pool.queue_state();state['execution_order']=dict(policy=nodes.PRIMARY_ORDER,cohorts=list(nodes.DEFAULT_COHORTS))
    nodes.atomic(pool.root/'state.json',state)
    assert pool.prioritize_primary_results(runtime)['execution_order']['cohorts']==['olmo_math']
    assert pool.extension_blockers()==['olmo_math']


def test_quiescent_failure_in_active_cohort_is_not_hidden_by_unstarted_future_cohorts(tmp_path):
    from test_srgc_v8_fail_stop import Runtime as DeferredRuntime
    rows=pairs(tmp_path);pool=rows[0][0];pool.initialize()
    runtime=DeferredRuntime(pool.root.parent,pool)
    state=pool.queue_state();state['phase']='extension';nodes.atomic(pool.root/'state.json',state)
    key=pool.register(dict(kind='train',run_id='old-failed',method_id='old-failed',parent={}), 'train')
    path=pool.root/'jobs'/(key+'.json');job=nodes.read(path)
    job.update(state='failed',attempts=[dict(id=f'old-{i}',outcome='failed',exit_code=-11) for i in range(9)])
    nodes.atomic(path,job);before=path.read_bytes()
    stop=threading.Event();timer=threading.Timer(10,stop.set);timer.start()   # hang guard
    try:
        assert nodes.worker_loop_many([(pool,runtime),*rows[1:]],'cpu-node',stop,poll=.001,
                                      cohort_order=nodes.DEFAULT_COHORTS)==1
    finally:timer.cancel();timer.join()
    assert not stop.is_set()
    assert path.read_bytes()==before
    # The later cohorts are served meanwhile; the blocked cohort never executes anything.
    assert not [row for row in events(tmp_path) if row['cohort']=='olmo_math']


@pytest.mark.parametrize('reopened_failure',[False,True])
def test_reopened_completed_cohort_does_not_leave_worker_waiting_on_its_cached_completion(tmp_path,reopened_failure):
    rows=pairs(tmp_path)
    later=rows[1][1];barrier=later.barrier;reopened=[];startup_checks=[]
    def startup_ready():
        startup_checks.append(True)
        return len(startup_checks)>1  # let the worker cache the earlier complete phase first
    later.startup_ready=startup_ready
    def reopen(stage,*args):
        barrier(stage,*args)
        if stage=='prepare' and not reopened:
            # A joining historical worker can restore an earlier phase while
            # checking its missing primary-results receipt. Keep every task and
            # checkpoint; this worker must recheck shared state, not its cache.
            pool=rows[0][0];state=pool.queue_state()
            assert state['phase']=='complete'
            state['phase']='train'
            if reopened_failure:state['failure']=dict(detail='invalid frozen inputs after recheck',task=None)
            nodes.atomic(pool.root/'state.json',state)
            reopened.append(True)
    later.barrier=reopen
    keep=tmp_path/'olmo_math/step-000025.pt';keep.write_bytes(b'optimizer RNG reward checkpoint sentinel')
    stop=threading.Event();timer=threading.Timer(10,stop.set);timer.start()   # hang guard under machine load
    try:
        rc=nodes.worker_loop_many(rows,'cpu-node',stop,poll=.001,cohort_order=nodes.DEFAULT_COHORTS)
    finally:timer.cancel();timer.join()
    assert reopened
    assert rc==(1 if reopened_failure else 0) and not stop.is_set(),'stale completed cache kept a reopened cohort idle forever'
    assert keep.read_bytes()==b'optimizer RNG reward checkpoint sentinel'
    claims=[(row['cohort'],row['task']) for row in events(tmp_path) if row['stage']=='execute']
    if reopened_failure:
        # The reopened, now failed cohort keeps its failure; the node still serves the other cohorts.
        assert rows[0][0].queue_state()['failure']['detail']=='invalid frozen inputs after recheck'
        assert len(claims)==len(set(claims))==16
        assert all(pool.queue_state()['phase']=='complete' for pool,_ in rows[1:])
    else:
        assert all(pool.queue_state()['phase']=='complete' for pool,_ in rows)
        assert len(claims)==len(set(claims))==16,'adopt existing completed outputs instead of rerunning them'


def test_live_lease_after_complete_does_not_idle_the_node_and_accepts_stop(tmp_path):
    rows=pairs(tmp_path);pool=rows[0][0];pool.initialize()
    state=pool.queue_state();state['phase']='complete';nodes.atomic(pool.root/'state.json',state)
    stop=threading.Event();release=threading.Event()
    later=rows[1][1];execute=later.execute
    def first_task_then_stop(task,*args):
        rc=execute(task,*args);stop.set();return rc
    later.execute=first_task_then_stop
    timer=threading.Timer(30,stop.set)   # hang guard
    with nodes.Lock(pool.root/'leases/still-running.lock'):
        timer.start()
        try:rc=nodes.worker_loop_many(rows,'cpu-node',stop,poll=.005,cohort_order=nodes.DEFAULT_COHORTS)
        finally:timer.cancel();timer.join()
    assert rc==75 and stop.is_set()
    recorded=events(tmp_path)
    assert all(row['stage']=='verify' for row in recorded if row['cohort']=='olmo_math')
    assert [row['cohort'] for row in recorded if row['stage']=='execute']==['qwen_mbpp']
    assert pool.queue_state()['phase']=='complete'
