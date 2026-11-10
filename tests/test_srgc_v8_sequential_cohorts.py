"""Sequential cohort scheduling with actual CPU workers/leases, no scientific measurements."""
import multiprocessing as mp
from pathlib import Path
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes


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


def test_six_simultaneous_nodes_never_spill_to_next_cohort_when_two_jobs_are_full(tmp_path):
    ctx=mp.get_context('fork');stop=ctx.Event();release=ctx.Event();children=[]
    try:
        for i in range(6):
            child=ctx.Process(target=worker,args=(tmp_path,f'node-{i}',stop,release));child.start();children.append(child)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            claims=[row for row in events(tmp_path) if row['stage']=='execute']
            if len(claims)>=2:break
            time.sleep(.01)
        assert len(claims)==2
        time.sleep(.2)
        assert {row['cohort'] for row in events(tmp_path)}=={'olmo_math'}
        release.set()
        for child in children:
            child.join(10)
            assert not child.is_alive() and child.exitcode==0
        assert all(nodes.read(p)['code']==0 for p in (tmp_path/'exits').glob('*'))
        recorded=events(tmp_path)
        for first,second in zip(nodes.DEFAULT_COHORTS,nodes.DEFAULT_COHORTS[1:]):
            # A late joining node may verify an already complete queue; it must
            # never execute another task or CPU stage after the next cohort starts.
            end=max(i for i,row in enumerate(recorded) if row['cohort']==first and row['stage']!='verify')
            start=min(i for i,row in enumerate(recorded) if row['cohort']==second)
            assert end<start
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


def test_missing_earlier_runtime_is_not_silently_skipped(tmp_path):
    rows=pairs(tmp_path,cohorts=('qwen_mbpp',))
    assert nodes.worker_loop_many(rows,'cpu-node',threading.Event(),poll=.001,
                                  cohort_order=nodes.DEFAULT_COHORTS)==1
    assert not events(tmp_path)
    assert not (tmp_path/'olmo_math').exists()


def test_failed_active_cohort_exhausts_finite_budget_without_starting_next(tmp_path):
    rows=pairs(tmp_path,fail=True)
    assert nodes.worker_loop_many(rows,'cpu-node',threading.Event(),poll=.001,
                                  cohort_order=nodes.DEFAULT_COHORTS)==1
    assert {row['cohort'] for row in events(tmp_path)}=={'olmo_math'}
    for path in (rows[0][0].root/'jobs').glob('*.json'):
        assert len(nodes.read(path)['attempts'])==nodes.MAX_ATTEMPTS


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
    stop=threading.Event();timer=threading.Timer(2,stop.set);timer.start()
    try:
        assert nodes.worker_loop_many([(pool,runtime),*rows[1:]],'cpu-node',stop,poll=.001,
                                      cohort_order=nodes.DEFAULT_COHORTS)==1
    finally:timer.cancel();timer.join()
    assert not stop.is_set()
    assert path.read_bytes()==before
    assert not events(tmp_path)
