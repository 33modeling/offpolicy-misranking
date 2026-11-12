"""Debug exhausted historical retries and valid outputs hidden behind failed labels; CPU metadata only."""
import threading

import pytest

from scripts import srgc_v8_nodes as nodes


import pytest as _pytest


@_pytest.fixture(autouse=True)
def _cap_without_user_grant(monkeypatch):
    """These tests pin the failure cap itself; the one-time user retry grant (11-13) has its own tests."""
    monkeypatch.setenv('V8_USER_RETRY_GRANT', '0')


def prepared(tmp_path):
    root=tmp_path/'olmo_math'
    runtime=object.__new__(nodes.Runtime);runtime.root=root
    saved=tmp_path/'prefix25.pt';saved.write_bytes(b'original step25 optimizer RNG rewards')
    task=dict(kind='train',run_id='sr25',method_id='bank_sr_t025',training_seed=5,
              cohort_id='olmo_math',role='CPU metadata fixture',config_hash='same-config',
              horizon=275,parent=dict(path=str(saved),completed_update=25))
    runtime.plan=lambda phase:([task],True);runtime.verify_node=lambda:None
    pool=nodes.Pool(root/'.multi-node',dict(cohort='olmo_math'),retry_delay=0)
    pool.initialize();pool.automatic_retry=True
    state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    key=pool.register(task,'train')
    return pool,runtime,task,key,saved


def failed(pool,key,count,policy):
    path=pool.root/'jobs'/(key+'.json');job=nodes.read(path);job['state']='failed'
    job['attempts']=[dict(id='failure-'+str(i),exit_code=1,outcome='failed',ended_at=0,
                          controller_policy=policy,controller_sha256='older-source') for i in range(count)]
    nodes.atomic(path,job)
    return path


def test_old_source_failures_stay_exhausted_across_verified_node_joins(tmp_path):
    pool,runtime,task,key,saved=prepared(tmp_path)
    pool.max_attempts=3
    path=failed(pool,key,5,'older-runtime-policy')
    before=nodes.read(path)
    pool.fail('1 task(s) failed; no independent ready/running work remains.')
    for i in range(4):
        joined=nodes.Pool(pool.root,pool.binding,retry_delay=0,max_attempts=3);joined.automatic_retry=True
        joined.controller_sha256='another-status-hash-'+str(i)
        joined.initialize(worker='restart-'+str(i),recover=joined.controller_sha256)
        assert not joined.retryable(nodes.read(path))
        assert joined.tick(runtime,'node',threading.Event())[0]=='failed'
    assert nodes.read(path)==before
    assert saved.read_bytes()==b'original step25 optimizer RNG rewards'


@pytest.mark.parametrize('aggregate',[False,True])
def test_actual_runtime_verified_completion_is_adopted_even_after_retry_exhaustion(tmp_path,aggregate):
    pool,runtime,task,key,saved=prepared(tmp_path)
    path=failed(pool,key,3,nodes.DISPATCH_POLICY)
    job=nodes.read(path)
    for attempt in job['attempts']:attempt['controller_sha256']=pool.controller_sha256
    nodes.atomic(path,job)
    directory=runtime.root/'runs'/task['run_id'];directory.mkdir(parents=True)
    nodes.atomic(directory/'manifest.json',{k:task[k] for k in ('run_id','method_id','training_seed','cohort_id','role','config_hash')} |
                 dict(status='completed_horizon',completed_updates=275))
    if aggregate:pool.fail('1 task(s) failed; no independent ready/running work remains.')
    stop=threading.Event();launched=[]
    runtime.execute=lambda *args:launched.append(True)
    runtime.barrier=lambda *args:stop.set()
    nodes.worker_loop(pool,runtime,'verified-node',stop,poll=.001,retry_delay=0)
    assert not launched,'complete task was needlessly trained again'
    assert nodes.read(path)['state']=='done','valid completed output was skipped because of its failed label'
    assert nodes.read(path)['attempts']==job['attempts'],'failure evidence was erased while adopting valid outputs'
    assert saved.read_bytes()==b'original step25 optimizer RNG rewards'


def test_native_execution_lock_is_respected_despite_an_exhausted_failed_label(tmp_path):
    pool,runtime,task,key,saved=prepared(tmp_path);failed(pool,key,3,nodes.DISPATCH_POLICY)
    lock=nodes.Lock(runtime.root/'runs'/task['run_id']/'.execution.lock');assert lock.acquire()
    try:
        action,_=pool.tick(runtime,'idle-node',threading.Event())
        assert action=='waiting' and pool.queue_state()['failure'] is None
    finally:lock.close()


def test_invalid_completion_identity_is_not_adopted(tmp_path):
    pool,runtime,task,key,saved=prepared(tmp_path);path=failed(pool,key,3,nodes.DISPATCH_POLICY)
    nodes.atomic(runtime.root/'runs'/task['run_id']/'manifest.json',task |
                 dict(config_hash='different-config',status='completed_horizon',completed_updates=275))
    with pytest.raises(ValueError,match='identity/config differs'):pool.tick(runtime,'node',threading.Event())
    assert nodes.read(path)['state']=='failed' and len(nodes.read(path)['attempts'])==3


def test_unverified_or_broken_retry_boundary_does_not_refund_failures():
    job=dict(attempts=[dict(id='old',exit_code=1)],retry_epoch=dict(policy='fake',after_attempt='old'))
    assert nodes.retry_budget_failures(job)==1
    job['recovery_history']=[dict(controller_policy='fake',previous_attempt='missing')]
    assert nodes.retry_budget_failures(job)==1


def test_status_counts_the_same_lifetime_retry_budget_as_the_dispatcher(tmp_path):
    from test_v8_status import status
    pool,runtime,task,key,saved=prepared(tmp_path);path=failed(pool,key,5,'older-runtime-policy')
    pool.max_attempts=3
    pool.initialize(worker='node',recover=pool.controller_sha256)
    job=nodes.read(path)
    row=dict(_log_lines=[],_manifest={},kind='train',total=275,done=25)
    story=status.failure_story(row,job['attempts'][-1],job,task,runtime.root,pool.queue_state(),{nodes.DISPATCH_POLICY},1000)
    assert story['failure_budget_used']==5 and story['failure_lifetime_count']==5
    assert 'no more automatic' in story['failure_next']


def test_cooldown_uses_the_last_attempt_instead_of_an_older_nodes_future_timestamp(tmp_path):
    pool,runtime,task,key,saved=prepared(tmp_path);pool.retry_delay=60;pool.max_attempts=3
    job=dict(state='failed',attempts=[dict(id='old',outcome='failed',ended_at=10000),
                                    dict(id='new',outcome='failed',ended_at=100)],
             retry_epoch=dict(policy='verified-repair',after_attempt='old'),
             recovery_history=[dict(controller_policy='verified-repair',previous_attempt='old')])
    assert not pool.retryable(job,now=159)
    assert pool.retryable(job,now=160),'older-node clock skew incorrectly extends the current retry cooldown'
