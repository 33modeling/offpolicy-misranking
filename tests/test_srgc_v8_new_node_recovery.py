"""A fresh node needs no recovery flags after the controller fix; failed attempts remain."""
import hashlib
from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_task_failure_isolation import setup


def failed_queue(tmp_path,controller=None):
    pool,runtime,stop=setup(tmp_path)
    runtime.verify_node=lambda:None
    tasks,_=runtime.plan('train')
    for task in tasks:
        key=pool.register(task,'train');path=pool.root/'jobs'/(key+'.json');job=nodes.read(path)
        job['state']='failed';job['attempts']=[dict(id='old-'+key,exit_code=1,controller_sha256=controller)]
        nodes.atomic(path,job)
    pool.fail('3 task(s) failed; no independent ready/running work remains. Failed attempts are preserved; inspect their logs before --retry-failed.')
    return pool,runtime,stop


def test_plain_new_node_reclaims_old_failed_olmo_without_retry_flag_or_checkpoint_reset(tmp_path):
    pool,runtime,stop=failed_queue(tmp_path,'old-controller')
    binding=(pool.root/'binding.json').read_bytes()
    saved=tmp_path/'olmo_math/runs/original/step-000025.pt';saved.parent.mkdir(parents=True)
    saved.write_bytes(b'original checkpoint optimizer RNG intermediate training rewards')
    expected=hashlib.sha256(saved.read_bytes()).hexdigest();started=[]
    def execute(task,matrix,fd,signal,device_fds,attempt):
        started.append(task['run_id']);runtime.completed.add(task['run_id']);stop.set();return 0
    runtime.execute=execute
    assert nodes.worker_loop(pool,runtime,'fresh-node',stop,poll=.001)==75
    assert started,'plain launch left all old failed OLMo jobs blocked'
    assert (pool.root/'binding.json').read_bytes()==binding
    assert hashlib.sha256(saved.read_bytes()).hexdigest()==expected
    for p in (pool.root/'jobs').glob('*.json'):
        job=nodes.read(p)
        assert job['attempts'][0]['exit_code']==1


def test_new_nodes_do_not_repeat_failures_from_the_same_controller(tmp_path):
    pool,runtime,stop=failed_queue(tmp_path)
    for path in (pool.root/'jobs').glob('*.json'):
        job=nodes.read(path);job['attempts'][-1]['controller_sha256']=pool.controller_sha256;nodes.atomic(path,job)
    original={p:p.read_bytes() for p in (pool.root/'jobs').glob('*.json')}
    with pool.control():pool.initialize(worker='another-new-node',recover=pool.controller_sha256)
    assert pool.queue_state()['failure']
    assert all(p.read_bytes()==content for p,content in original.items())


def test_legacy_attempts_without_controller_hash_recover_only_once(tmp_path):
    pool,runtime,stop=failed_queue(tmp_path)
    with pool.control():pool.initialize(worker='node-a',recover=pool.controller_sha256)
    assert pool.queue_state()['failure'] is None
    for path in (pool.root/'jobs').glob('*.json'):
        job=nodes.read(path);assert len(job['recovery_history'])==1
        # Even an older worker writing the legacy failure again cannot cause a retry loop.
        job['state']='failed';nodes.atomic(path,job)
    with pool.control():pool.initialize(worker='node-b',recover=pool.controller_sha256)
    assert all(nodes.read(p)['state']=='failed' and len(nodes.read(p)['recovery_history'])==1
               for p in (pool.root/'jobs').glob('*.json'))


def test_status_only_source_changes_do_not_enable_another_automatic_attempt(tmp_path):
    pool,runtime,stop=failed_queue(tmp_path,'different-source-hash')
    for path in (pool.root/'jobs').glob('*.json'):
        job=nodes.read(path);job['attempts'][-1]['controller_policy']=nodes.DISPATCH_POLICY;nodes.atomic(path,job)
    with pool.control():pool.initialize(worker='new-node',recover=pool.controller_sha256)
    assert pool.queue_state()['failure']
    assert all(nodes.read(p)['state']=='failed' for p in (pool.root/'jobs').glob('*.json'))


def test_active_task_lease_is_not_taken_by_source_recovery(tmp_path):
    pool,runtime,stop=failed_queue(tmp_path)
    path=next((pool.root/'jobs').glob('*.json'));held=nodes.Lock(pool.root/'leases'/(path.stem+'.lock'))
    assert held.acquire();original=path.read_bytes()
    try:
        with pool.control():pool.initialize(worker='new-node',recover=pool.controller_sha256)
        assert path.read_bytes()==original
    finally:held.close()
    with pool.control():pool.initialize(worker='later-node',recover=pool.controller_sha256)
    assert nodes.read(path)['state']=='pending'


def test_input_failure_or_failed_node_verification_cannot_reopen_tasks(tmp_path):
    pool,runtime,stop=failed_queue(tmp_path)
    state=pool.queue_state();state['failure']={'detail':'invalid frozen input identity','task':None}
    nodes.atomic(pool.root/'state.json',state)
    original={p:p.read_bytes() for p in (pool.root/'jobs').glob('*.json')}
    with pool.control():pool.initialize(worker='new-node',recover=pool.controller_sha256)
    assert pool.queue_state()['failure']==state['failure']
    def invalid():raise ValueError('input hash differs')
    runtime.verify_node=invalid
    assert nodes.worker_loop(pool,runtime,'invalid-node',stop,poll=.001)==1
    assert all(p.read_bytes()==content for p,content in original.items())


def test_recovery_resumes_after_job_write_before_aggregate_failure_clear(tmp_path):
    pool,runtime,stop=failed_queue(tmp_path)
    for path in (pool.root/'jobs').glob('*.json'):
        job=nodes.read(path);job['state']='pending'
        job['recovery_history']=[dict(controller_sha256=pool.controller_sha256,previous_attempt=job['attempts'][-1]['id'])]
        nodes.atomic(path,job)
    with pool.control():pool.initialize(worker='recovery-node',recover=pool.controller_sha256)
    assert pool.queue_state()['failure'] is None
    assert len(pool.queue_state()['failure_history'])==1
