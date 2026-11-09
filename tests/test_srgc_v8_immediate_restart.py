"""Immediate checkpoint restarts and work-conserving dispatch; CPU metadata only."""
from contextlib import contextmanager
import multiprocessing as mp
import json
import threading

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_task_failure_isolation import setup,claim
from test_srgc_v8_fair_dispatch import Runtime,wait_claims,cleanup


def test_default_restart_has_no_failure_cap_or_cooldown(tmp_path):
    pool=nodes.Pool(tmp_path/'queue',dict(cohort='olmo_math'));pool.automatic_retry=True
    job=dict(state='failed',attempts=[dict(id=str(i),outcome='failed',ended_at=10000) for i in range(8)])
    assert pool.max_attempts==0 and pool.retry_delay==0
    assert pool.retryable(job,now=100),'old failure count or another node clock delayed immediate restart'
    job.update(state='interrupted',resume_not_before=10000)
    assert pool.retryable(job,now=100),'legacy cooldown blocked the new immediate policy'


def test_actual_worker_restarts_four_sigsegv_failures_then_completes_without_erasing_checkpoint(tmp_path):
    from test_srgc_v8_retry_debug import prepared
    pool,runtime,task,key,saved=prepared(tmp_path);stop=threading.Event();calls=[]
    runtime.done=lambda _:bool(calls and calls[-1]==0)
    original=saved.read_bytes()
    def execute(task,matrix,fd,stop,device_fds,attempt):
        assert saved.read_bytes()==original
        rc=-11 if len(calls)<4 else 0;calls.append(rc)
        if rc:
            nodes.atomic(pool.root/'progress'/attempt['id']/'child-failure.json',
                         dict(attempt_id=attempt['id'],exit_code=rc,failures=[dict(rank=1,signal=11,message='CPU synthetic SIGSEGV result')]))
        else:stop.set()
        return rc
    runtime.execute=execute
    nodes.worker_loop(pool,runtime,'cpu-restart',stop,poll=.001,retry_delay=0)
    job=nodes.read(pool.root/'jobs'/(key+'.json'))
    assert calls==[-11,-11,-11,-11,0] and job['state']=='done'
    assert [a['outcome'] for a in job['attempts']]==['failed']*4+['done']
    assert nodes.counted_failures(job)==4 and saved.read_bytes()==original
    (tmp_path/'restart-proof.json').write_text(json.dumps(dict(scope='CPU metadata only; no actual GPU SIGSEGV',job=job,checkpoint_preserved=True)))


def test_status_displays_unlimited_immediate_restart_after_many_failures(tmp_path):
    from test_v8_status import status
    from test_srgc_v8_retry_debug import prepared,failed
    pool,runtime,task,key,saved=prepared(tmp_path);path=failed(pool,key,8,nodes.DISPATCH_POLICY)
    pool.enable_automatic_retry('node');job=nodes.read(path)
    row=dict(_log_lines=[],_manifest={},kind='train',total=275,done=25)
    story=status.failure_story(row,job['attempts'][-1],job,task,runtime.root,pool.queue_state(),{nodes.DISPATCH_POLICY},1000)
    assert 'no limit' in story['failure_next'] and 'immediate checkpoint restart' in story['failure_next']
    assert 'no more automatic' not in story['failure_next'] and story['failure_lifetime_count']==8


def test_ready_new_work_is_not_starved_by_immediately_retryable_primary(tmp_path):
    pool,runtime,stop=setup(tmp_path);pool.automatic_retry=True;pool.retry_delay=0;pool.max_attempts=3
    lease,bad,_,_=claim(pool,runtime,stop)
    pool.finish(bad,1,runtime,stop);lease.close()
    lease,key,_,_=claim(pool,runtime,stop)
    try:assert key!=bad,'failed on-policy trajectory starved independent ready branches/seeds'
    finally:lease.close()


def test_busy_gpu_is_rechecked_quickly_after_process_exit(monkeypatch):
    clock=[0.];rows=[dict(index='0',used_mib=70000)];calls=[]
    monkeypatch.setattr(nodes.time,'monotonic',lambda:clock[0])
    def inventory(_):calls.append(clock[0]);return rows
    gate=nodes.GpuGate(['0'],inventory=inventory,processes=lambda:[])
    assert gate()[0] is False
    rows[0]['used_mib']=0;clock[0]=2.
    assert gate()[0] is True,'a finished task left a stale busy cache for30seconds'
    assert len(calls)==2


def test_one_contended_cohort_does_not_idle_ready_other_cohort(tmp_path):
    stop=threading.Event();launched=[];contention=[];errors=[]
    dispatch=tmp_path/'.v8-dispatch.lock'
    class Contended(nodes.Pool):
        @contextmanager
        def control(self):
            probe=nodes.Lock(dispatch);free=probe.acquire();probe.close()
            if not free:contention.append(True);yield None
            else:
                with super().control() as lease:yield lease
    class Ready:
        def __init__(self,name):self.name=name
        def verify_node(self):pass
        def plan(self,phase):return [dict(kind='train',run_id=self.name,method_id='on_monitored',training_seed=5,horizon=275,config_hash=self.name,parent={})],True
        def ready(self,task):return True
        def native_busy(self,task):return False
        def done(self,task):return False
        def execute(self,*_):launched.append(self.name);stop.set();return 75
    pairs=[]
    for cls,name in [(Contended,'a'),(nodes.Pool,'b')]:
        pool=cls(tmp_path/name/'.multi-node',dict(cohort=name));pool.initialize()
        state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
        pairs.append((pool,Ready(name)))
    worker=next('node-'+str(i) for i in range(100) if int(nodes.digest('node-'+str(i))[:8],16)%2==0)
    def run():
        try:nodes.worker_loop_many(pairs,worker,stop,poll=.001)
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=run);thread.start()
    try:thread.join(.5)
    finally:stop.set();thread.join(3)
    assert not thread.is_alive() and not errors,errors
    assert contention and launched==['b'],'metadata contention in a prevented all GPU dispatch'


def four_cohort_worker(root,name,stop):
    pairs=[]
    for cohort in nodes.DEFAULT_COHORTS:
        runtime=Runtime(root,cohort,stop)
        pairs.append((nodes.Pool(root/cohort/'.multi-node',runtime.binding),runtime))
    nodes.worker_loop_many(pairs,name,stop,poll=.003)


def test_24_simultaneous_workers_claim_all_four_cohorts_without_duplicates(tmp_path):
    ctx=mp.get_context('fork');stop=ctx.Event();processes=[]
    for cohort in nodes.DEFAULT_COHORTS:
        runtime=Runtime(tmp_path,cohort,stop);pool=nodes.Pool(tmp_path/cohort/'.multi-node',runtime.binding);pool.initialize()
        state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    nodes.atomic(tmp_path/'olmo_math-prepared',dict(scope='CPU concurrency fixture'))
    try:
        for i in range(24):
            process=ctx.Process(target=four_cohort_worker,args=(tmp_path,'cpu-node-'+str(i),stop))
            process.start();processes.append(process)
        counts=wait_claims(tmp_path,24)
        assert set(counts)==set(nodes.DEFAULT_COHORTS)
    finally:cleanup(processes,stop)
