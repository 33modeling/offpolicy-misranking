"""Default node launch recovers task faults without a retry flag; CPU metadata only."""
import threading

import pytest

from scripts import srgc_v8_nodes as nodes


@pytest.mark.parametrize('already_failed',[False,True])
def test_default_launch_retries_same_controller_failure_with_checkpoint_and_history(tmp_path,already_failed):
    stop=threading.Event();calls=[]
    pool=nodes.Pool(tmp_path/'olmo_math/.multi-node',{'cohort':'olmo_math'})
    pool.initialize();state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    checkpoint=tmp_path/'latest.pt';checkpoint.write_bytes(b'update25 optimizer RNG training rewards')
    task=dict(kind='train',run_id='sr25',method_id='bank_sr_t025',config_hash='fixture',
              training_seed=5,horizon=275,parent={'path':str(checkpoint)})
    key=pool.register(task,'train');path=pool.root/'jobs'/(key+'.json')
    if already_failed:
        job=nodes.read(path);job['state']='failed'
        job['attempts']=[dict(id='failed-before-join',outcome='failed',exit_code=1,ended_at=0,
                             controller_sha256=pool.controller_sha256,controller_policy=nodes.DISPATCH_POLICY)]
        nodes.atomic(path,job)
        pool.fail('1 task(s) failed; no independent ready/running work remains.')
    class Runtime:
        def verify_node(self):pass
        def plan(self,phase):return [task],True
        def native_busy(self,task):return False
        def ready(self,task):return True
        def done(self,task):return bool(calls and calls[-1]=='success')
        def execute(self,task,*args):
            assert checkpoint.read_bytes()==b'update25 optimizer RNG training rewards'
            if not already_failed and not calls:calls.append('failure');return 1
            calls.append('success');stop.set();return 0
    nodes.worker_loop(pool,Runtime(),'plain-node',stop,poll=.001,retry_delay=0)
    job=nodes.read(path)
    assert job['state']=='done','same-controller failure was left excluded by default'
    assert [a['exit_code'] for a in job['attempts']]==[1,0]
    assert checkpoint.read_bytes()==b'update25 optimizer RNG training rewards'


def test_automatic_recovery_keeps_cooldown_cap_and_structural_failures(tmp_path):
    pool=nodes.Pool(tmp_path/'queue',{'cohort':'olmo_math'});pool.initialize();pool.automatic_retry=True
    pool.retry_delay=60;pool.max_attempts=3
    job=dict(state='failed',attempts=[dict(outcome='failed',ended_at=100)])
    assert not pool.retryable(job,now=159) and pool.retryable(job,now=160)
    job['attempts']*=3
    assert not pool.retryable(job,now=1000) and not pool.retry_later(job)
    pool.fail('frozen input identity differs')
    before=pool.queue_state()['failure'];pool.enable_automatic_retry('node')
    assert pool.queue_state()['failure']==before
