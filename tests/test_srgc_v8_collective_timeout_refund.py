"""11-23: Qwen MBPP evals aborted by the frozen 1800 s NCCL wait limit (rank 1, ALLGATHER) stop counting once the
task entry raises the limit; blocked tasks reopen from their checkpoints by themselves, records kept."""
import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_retry_debug import prepared

TIMEOUT_LOG=('[rank1]:[E1117 16:26:22 ProcessGroupNCCL.cpp:632] [Rank 1] Watchdog caught collective operation timeout: '
             'WorkNCCL(SeqNum=15, OpType=ALLGATHER, NumelIn=1, NumelOut=4, Timeout(ms)=1800000) ran for 1800032 '
             'milliseconds before timing out.\nFatal Python error: Aborted\n')


@pytest.fixture(autouse=True)
def _fixture_env(monkeypatch):
    monkeypatch.setenv('V8_USER_RETRY_GRANT','0')


def attempt(i,host,rc,signal=None):
    row=dict(id=f'a{i}',outcome='failed',exit_code=rc,host=host,gpus=['0','1','2','3'],worker=host+'.1',
             started_at=1000.0*i,ended_at=1000.0*i+5400.0,controller_sha256='f'*64,periodic_stack_dump='cancelled')
    if signal:row['child_failure']=dict(failures=[dict(rank=1,signal=signal)])
    return row


def setup(tmp_path,attempts,logs):
    pool,runtime,task,key,saved=prepared(tmp_path)
    path=pool.root/'jobs'/(key+'.json');job=nodes.read(path);job['state']='failed';job['attempts']=attempts
    nodes.atomic(path,job)
    (pool.root/'logs').mkdir(exist_ok=True)
    for aid,text in logs.items():(pool.root/'logs'/(aid+'.log')).write_text('generation lines\n'*50+text)
    pool.fail('15 task(s) failed; no independent ready/running work remains.')
    return pool,path,saved


def test_timeout_aborts_reopen_the_task_and_keep_every_record(tmp_path):
    attempts=[attempt(1,'a',-6),attempt(2,'b',1,signal=6),attempt(3,'c',-6),attempt(4,'d',1,signal=6)]
    pool,path,saved=setup(tmp_path,attempts,{f'a{i}':TIMEOUT_LOG for i in range(1,5)})
    before=nodes.read(path)
    assert nodes.retry_budget_failures(before)>=4
    pool.initialize(worker='fresh.1')
    job=nodes.read(path)
    assert job['state']=='pending' and pool.queue_state()['failure'] is None
    assert [a['periodic_dump_refund']['rule'] for a in job['attempts']]==['nccl_1800s_collective_timeout_fixed_in_task_entry']*4
    assert [a['periodic_dump_refund']['signal'] for a in job['attempts']]==[6]*4
    for old,new in zip(before['attempts'],job['attempts']):
        assert {k:v for k,v in new.items() if k!='periodic_dump_refund'}==old
    assert nodes.retry_budget_failures(job)==0 and nodes.native_crash_places(job)==[]
    pool.host='a';assert pool.retryable(job),'not a node fault: any node may run it'
    pool.initialize(worker='fresh.2');assert nodes.read(path)['attempts']==job['attempts'],'applied once'
    assert saved.read_bytes()==b'original step25 optimizer RNG rewards'


def test_more_than_twenty_timeouts_are_all_refunded(tmp_path):
    attempts=[attempt(i,f'h{i}',1,signal=6) for i in range(1,40)]
    pool,path,_=setup(tmp_path,attempts,{f'a{i}':TIMEOUT_LOG for i in range(1,40)})
    pool.initialize(worker='fresh.1')
    job=nodes.read(path)
    assert sum('periodic_dump_refund' in a for a in job['attempts'])==39 and job['state']=='pending'


@pytest.mark.parametrize('text,refunded',[
    (TIMEOUT_LOG,True),
    (TIMEOUT_LOG.replace('Timeout(ms)=1800000','Timeout(ms)=21600000'),False),   # after the fix: a real hang counts
    ('RuntimeError: CUDA error: an illegal memory access was encountered\n',False),
    ('',False),
])
def test_only_the_1800s_nccl_abort_is_refunded(tmp_path,text,refunded):
    pool,path,_=setup(tmp_path,[attempt(1,'a',-6)],{'a1':text} if text else {})
    job=nodes.read(path)
    assert (nodes.periodic_dump_evidence(pool.root,job,job['attempts'][0]) is not None)==refunded


def test_entry_without_the_fix_refunds_nothing(tmp_path,monkeypatch):
    pool,path,_=setup(tmp_path,[attempt(1,'a',-6)],{'a1':TIMEOUT_LOG})
    real=nodes.Path.read_text
    monkeypatch.setattr(nodes.Path,'read_text',lambda self,*a,**k:'' if self.name=='v8_task_entry.py' else real(self,*a,**k))
    job=nodes.read(path)
    assert nodes.collective_timeout_evidence(pool.root,job['attempts'][0]) is None
