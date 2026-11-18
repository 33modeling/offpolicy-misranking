"""11-18: SIGSEGVs from the old 30-min periodic faulthandler dump stop counting (retry budget, crash places,
GPU-set quarantine); blocked tasks reopen from their checkpoints; crashes on fixed code still count and a task that
keeps crashing stays bounded. CPU only, real Pool and node loop."""
import os
import socket
import threading

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_retry_debug import prepared

OLD='0'*64   # a controller from before the dump fix


@pytest.fixture(autouse=True)
def _fixture_env(monkeypatch):
    monkeypatch.setenv('V8_USER_RETRY_GRANT','0')   # the refund alone must reopen the task


def crash(i,host,*,elapsed=1835.0,controller=OLD,mode=None,rc=-11,gpus=('0','1','2','3')):
    row=dict(id=f'c{i}',outcome='failed',exit_code=rc,host=host,gpus=list(gpus),worker=host+'.1',
             started_at=1000.0*i,ended_at=1000.0*i+elapsed,controller_sha256=controller)
    if mode is not None:row['periodic_stack_dump']=mode
    return row


def job_with(pool,key,attempts,state='failed'):
    path=pool.root/'jobs'/(key+'.json');job=nodes.read(path);job['state']=state;job['attempts']=attempts
    nodes.atomic(path,job);return path


def test_old_dump_crashes_reopen_a_blocked_task_and_keep_every_record(tmp_path):
    pool,runtime,task,key,saved=prepared(tmp_path)
    attempts=[crash(1,'a'),crash(2,'b'),crash(3,'c'),dict(id='e4',outcome='failed',exit_code=1,ended_at=0)]
    path=job_with(pool,key,attempts)
    before=nodes.read(path)
    assert nodes.native_failure_blocks(before) and nodes.retry_budget_failures(before)==4
    fault=pool.root/'node-native-faults'/'gpuset.json'
    nodes.atomic(fault,dict(schema=1,identity={},failures=[dict(attempt='c1'),dict(attempt='c2')],quarantined=True))
    pool.fail('1 task(s) failed; no independent ready/running work remains.')
    pool.initialize(worker='fresh.1')
    job=nodes.read(path)
    assert job['state']=='pending' and pool.queue_state()['failure'] is None
    assert [a['id'] for a in job['attempts']]==['c1','c2','c3','e4']
    for old,new in zip(before['attempts'],job['attempts']):   # receipts only added, nothing removed
        assert {k:v for k,v in new.items() if k!='periodic_dump_refund'}==old
    assert [a['periodic_dump_refund']['rule'] for a in job['attempts'][:3]]==['launched_by_controller_before_periodic_dump_fix']*3
    assert 'periodic_dump_refund' not in job['attempts'][3]
    assert nodes.retry_budget_failures(job)==1 and nodes.native_crash_places(job)==[]
    assert nodes.fatal_native_failure(job) is None and not nodes.native_failure_blocks(job,('a',('0','1','2','3')))
    assert job['periodic_dump_refunds'][0]['failures_before']==4 and job['periodic_dump_refunds'][0]['failures_after']==1
    row=nodes.read(fault)
    assert row['refunded_attempts']==['c1','c2'] and not row['quarantined'] and len(row['failures'])==2
    assert nodes.node_fault_count(row)==0
    assert any(a['action']=='periodic_dump_crash_refund' for a in pool.queue_state()['actions'])
    pool.host='a';assert pool.retryable(job),'an old crash place may run it again'
    pool.initialize(worker='fresh.2')
    assert nodes.read(path)['attempts']==job['attempts'],'applied once'
    assert saved.read_bytes()==b'original step25 optimizer RNG rewards'


@pytest.mark.parametrize('attempt,refunded',[
    (crash(1,'a',mode='armed',controller=next(iter(nodes.FIXED_DUMP_CONTROLLERS))),True),   # entry without the cancel
    (crash(1,'a',elapsed=600.0),False),                 # before the first 30-min dump: a real crash
    (crash(1,'a',mode='cancelled'),False),              # fixed task entry: counts
    (crash(1,'a',mode='armed-by-env'),False),           # V8_PERIODIC_STACK_DUMP=1 chosen by the user: counts
    (crash(1,'a',rc=-6),False),                         # SIGABRT is not the dump crash
    (crash(1,'a',controller=next(iter(nodes.FIXED_DUMP_CONTROLLERS))),False),   # fixed controller, no stack evidence
])
def test_which_crashes_are_refunded(tmp_path,attempt,refunded):
    pool,runtime,task,key,saved=prepared(tmp_path)
    path=job_with(pool,key,[attempt])
    pool.initialize(worker='w')
    assert nodes.dump_refunded(nodes.read(path)['attempts'][0])==refunded


@pytest.mark.parametrize('tail,refunded',[('Timeout (0:30:00)!\nThread 0x1 (most recent call first):\n  File "x.py", line 1',True),
                                          ('Timeout (0:30:00)!\n...\nFatal Python error: Segmentation fault\n',False)])
def test_stack_file_evidence_on_a_fixed_controller(tmp_path,tail,refunded):
    pool,runtime,task,key,saved=prepared(tmp_path)
    attempt=crash(1,'a',controller=next(iter(nodes.FIXED_DUMP_CONTROLLERS)))
    stack=pool.root/'stacks'/'sr25-rank0.txt';stack.parent.mkdir(parents=True);stack.write_text(tail)
    os.utime(stack,(attempt['ended_at']-5,attempt['ended_at']-5))
    path=job_with(pool,key,[attempt]);pool.initialize(worker='w')
    got=nodes.read(path)['attempts'][0]
    assert nodes.dump_refunded(got)==refunded
    if refunded:assert got['periodic_dump_refund']['stack_file']==str(stack)


def test_refund_runs_periodically_not_only_at_join(tmp_path,monkeypatch):
    pool,runtime,task,key,saved=prepared(tmp_path)
    path=job_with(pool,key,[crash(1,'a'),crash(2,'b'),crash(3,'c')])   # written by another node after this join
    pool.tick(runtime,'node',threading.Event(),claim=False,offer=True)
    assert not nodes.dump_refunded(nodes.read(path)['attempts'][0]),'throttled right after the join'
    pool.refund_checked_at-=nodes.REFUND_CHECK_SECONDS
    action,offer=pool.tick(runtime,'node',threading.Event(),claim=False,offer=True)
    job=nodes.read(path)
    assert all(nodes.dump_refunded(a) for a in job['attempts']) and job['state']=='pending' and action=='ready'


def test_refunds_per_task_are_bounded(tmp_path):
    pool,runtime,task,key,saved=prepared(tmp_path)
    n=nodes.PERIODIC_DUMP_REFUND_LIMIT+3
    path=job_with(pool,key,[crash(i,f'h{i}') for i in range(1,n+1)])
    pool.initialize(worker='w')
    job=nodes.read(path)
    assert sum(map(nodes.dump_refunded,job['attempts']))==nodes.PERIODIC_DUMP_REFUND_LIMIT
    assert nodes.native_failure_blocks(job),'the crashes beyond the limit count again'


def test_new_attempts_record_whether_the_dump_was_cancelled(tmp_path):
    pool,runtime,task,key,saved=prepared(tmp_path)
    assert pool.periodic_stack_dump=='cancelled' and len(pool.task_entry_sha256)==64
    action,claimed=pool.tick(runtime,'node',threading.Event())
    lease,_,_,attempt=claimed;lease.close()
    assert attempt['periodic_stack_dump']=='cancelled' and attempt['task_entry_sha256']==pool.task_entry_sha256


class Resuming:
    """Each launch resumes from the saved step and advances one step, then crashes until step 3."""
    def __init__(self,runtime,saved,crash_rc=-11,forever=False):
        self.runtime=runtime;self.saved=saved;self.crash_rc=crash_rc;self.forever=forever;self.steps=[]
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        ckpt=self.saved.parent/'resume.step'
        step=int(ckpt.read_text()) if ckpt.exists() else 0
        step+=1;ckpt.write_text(str(step));self.steps.append(step)
        if self.forever or step<3:return self.crash_rc
        (self.saved.parent/'done').write_text('ok');return 0


def loop(pool,runtime,seconds=60):
    stop=threading.Event();timer=threading.Timer(seconds,stop.set);timer.start()
    try:return nodes.worker_loop(pool,runtime,'node-x.1',stop,poll=.001,retry_delay=0)
    finally:timer.cancel()


def test_dump_crashes_resume_forward_from_the_checkpoint_on_the_same_node(tmp_path,monkeypatch):
    pool,runtime,task,key,saved=prepared(tmp_path)
    monkeypatch.setattr(nodes,'PERIODIC_DUMP_SECONDS',0)   # CPU stand-in for 'crashed 30 min after start'
    pool.periodic_stack_dump='armed'                        # a task entry that still arms the dump
    fake=Resuming(runtime,saved);runtime.execute=fake.execute
    runtime.done=lambda _:(saved.parent/'done').exists();runtime.ready=lambda _:True;runtime.native_busy=lambda _:False
    loop(pool,runtime)
    job=nodes.read(pool.root/'jobs'/(key+'.json'))
    assert job['state']=='done' and fake.steps==[1,2,3],'each retry continued from the saved step'
    assert [nodes.dump_refunded(a) for a in job['attempts']]==[True,True,False]
    assert not list((pool.root/'node-native-faults').glob('*.json')) or all(
        nodes.node_fault_count(nodes.read(p))==0 for p in (pool.root/'node-native-faults').glob('*.json'))


def test_a_task_that_keeps_crashing_stops_after_the_refund_limit(tmp_path,monkeypatch):
    pool,runtime,task,key,saved=prepared(tmp_path)
    monkeypatch.setattr(nodes,'PERIODIC_DUMP_SECONDS',0);monkeypatch.setattr(nodes,'PERIODIC_DUMP_REFUND_LIMIT',2)
    pool.periodic_stack_dump='armed'
    fake=Resuming(runtime,saved,forever=True);runtime.execute=fake.execute
    runtime.done=lambda _:False;runtime.ready=lambda _:True;runtime.native_busy=lambda _:False
    assert loop(pool,runtime)==1
    job=nodes.read(pool.root/'jobs'/(key+'.json'))
    assert len(job['attempts'])==3 and [nodes.dump_refunded(a) for a in job['attempts']]==[True,True,False]


def test_crashes_on_fixed_code_are_never_refunded(tmp_path):
    pool,runtime,task,key,saved=prepared(tmp_path)
    fake=Resuming(runtime,saved,forever=True);runtime.execute=fake.execute
    runtime.done=lambda _:False;runtime.ready=lambda _:True;runtime.native_busy=lambda _:False
    assert loop(pool,runtime)==1   # one GPU set: a counted crash keeps the task off it, nothing else serves it
    job=nodes.read(pool.root/'jobs'/(key+'.json'))
    assert len(job['attempts'])==1 and not nodes.dump_refunded(job['attempts'][0])
    assert job['attempts'][0]['periodic_stack_dump']=='cancelled'
