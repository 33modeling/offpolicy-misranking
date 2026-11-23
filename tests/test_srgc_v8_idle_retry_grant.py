"""11-23: a queue whose only unfinished work is blocked tasks no longer stops with idle nodes beside it; each
blocked task restarts from its checkpoint with a fresh failure budget, at most IDLE_RETRY_GRANT_LIMIT times."""
import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_retry_debug import prepared


@pytest.fixture(autouse=True)
def _fixture_env(monkeypatch):
    monkeypatch.delenv("V8_USER_RETRY_GRANT",raising=False);monkeypatch.setenv("V8_IDLE_RETRY_GRANT","1")
    monkeypatch.setattr(nodes.Pool,'grant_user_retry',lambda self,worker:[])   # isolate this policy


def attempt(i):
    return dict(id=f'a{i}',outcome='failed',exit_code=1,host=f'h{i}',gpus=['0','1','2','3'],worker=f'h{i}.1',
                started_at=1000.0*i,ended_at=1000.0*i+60.0,controller_sha256='f'*64,periodic_stack_dump='cancelled')


def blocked(tmp_path,count=4):
    pool,runtime,task,key,saved=prepared(tmp_path)
    path=pool.root/'jobs'/(key+'.json');job=nodes.read(path);job['state']='failed';job['attempts']=[attempt(i) for i in range(1,count+1)]
    nodes.atomic(path,job)
    pool.fail('1 task(s) failed; no independent ready/running work remains.')
    return pool,path,saved


def fail_again(path,start,count=4):
    job=nodes.read(path);job['attempts']+= [attempt(i) for i in range(start,start+count)];job['state']='failed'
    nodes.atomic(path,job)


def test_blocked_task_reopens_instead_of_idle_and_keeps_records(tmp_path):
    pool,path,saved=blocked(tmp_path)
    assert pool.retry_limit_reached(nodes.read(path))
    pool.initialize(worker='fresh.1')
    job=nodes.read(path)
    assert job['state']=='pending' and pool.queue_state()['failure'] is None
    assert job['user_retry_grant']['policy']==nodes.IDLE_RETRY_GRANT and job['user_retry_grant']['after_attempt']=='a4'
    assert len(job['attempts'])==4 and not pool.retry_limit_reached(job)
    assert saved.read_bytes()==b'original step25 optimizer RNG rewards'
    pool.initialize(worker='fresh.2');assert nodes.read(path)['user_retry_grant']==job['user_retry_grant'],'no new attempt: no new grant'


def test_grants_are_bounded_per_task(tmp_path):
    pool,path,_=blocked(tmp_path);start=5
    for n in range(nodes.IDLE_RETRY_GRANT_LIMIT):
        pool.initialize(worker=f'w{n}');job=nodes.read(path)
        assert job['state']=='pending' and job['user_retry_grant']['grant']==n+1
        fail_again(path,start);start+=4;pool.fail('1 task(s) failed; no independent ready/running work remains.')
    pool.initialize(worker='last')
    job=nodes.read(path)
    assert job['state']=='failed' and pool.retry_limit_reached(job)
    assert nodes.grant_policies(job).count(nodes.IDLE_RETRY_GRANT)==nodes.IDLE_RETRY_GRANT_LIMIT
    assert pool.queue_state()['failure'] is not None


def test_quiescent_check_reopens_blocked_tasks(tmp_path):
    pool,path,_=blocked(tmp_path)
    state=pool.queue_state();state['failure']=None;nodes.atomic(pool.root/'state.json',state)
    assert pool.grant_idle_retry('w')==[],'without a stop or a quiescent caller nothing changes'
    assert pool.grant_idle_retry('w',quiescent=True)==[path.stem] and nodes.read(path)['state']=='pending'


def test_setup_failures_are_never_cleared(tmp_path):
    pool,path,_=blocked(tmp_path)
    state=pool.queue_state();state['failure']=dict(detail='input identity changed',task=None,at=1.0)
    nodes.atomic(pool.root/'state.json',state)
    assert pool.grant_idle_retry('w',quiescent=True)==[] and nodes.read(path)['state']=='failed'


def test_switch_off(tmp_path,monkeypatch):
    monkeypatch.setenv('V8_IDLE_RETRY_GRANT','0')
    pool,path,_=blocked(tmp_path)
    assert pool.grant_idle_retry('w',quiescent=True)==[]
