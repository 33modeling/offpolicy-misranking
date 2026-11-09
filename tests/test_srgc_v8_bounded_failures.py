"""Repeated task errors must not consume every node; CPU scheduling only."""
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading

import pytest

from scripts import srgc_v8_nodes as nodes

if os.environ.get('V8_RETRY_TEST_SOURCE'):
    spec=importlib.util.spec_from_file_location('old_retry_nodes',os.environ['V8_RETRY_TEST_SOURCE'])
    nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)


def history(count):
    return [dict(id=f'old-{i}',outcome='failed',exit_code=1,controller_policy='old',ended_at=0)
            for i in range(count)]


@pytest.mark.parametrize('state',['failed','pending','interrupted','running'])
def test_old_failures_cannot_be_refunded_by_node_or_policy_change(tmp_path,state):
    pool=nodes.Pool(tmp_path/'queue',dict(cohort='fixture'));pool.automatic_retry=True
    job=dict(state=state,attempts=history(8),retry_epoch=dict(policy='old',after_attempt='old-7'),
             recovery_history=[dict(controller_policy='old',previous_attempt='old-7')])
    assert not pool.retryable(job),'a historical retry epoch granted an endless new failure budget'
    assert not pool.retry_later(job)
    assert nodes.retry_budget_failures(job)==8


def test_legacy_zero_flag_cannot_enable_infinite_restarts(tmp_path):
    pool=nodes.Pool(tmp_path/'queue',dict(cohort='fixture'),max_attempts=0);pool.automatic_retry=True
    assert not pool.retryable(dict(state='failed',attempts=history(2)))


def test_first_error_can_retry_but_second_error_stops(tmp_path):
    pool=nodes.Pool(tmp_path/'queue',dict(cohort='fixture'));pool.automatic_retry=True
    assert pool.retryable(dict(state='failed',attempts=history(1)))
    assert not pool.retryable(dict(state='failed',attempts=history(2)))
    assert pool.retryable(dict(state='interrupted',attempts=[dict(outcome='node_lost',exit_code=-9)]))


def test_status_distinguishes_new_finite_policy_from_old_live_unlimited_workers(tmp_path):
    from test_v8_status import status
    pool=nodes.Pool(tmp_path/'.multi-node',dict(cohort='fixture'));pool.initialize()
    job=dict(state='failed',attempts=history(2))
    row=dict(_log_lines=[],_manifest={},kind='train',total=275,done=25)
    task=dict(kind='train',horizon=275)
    old=dict(automatic_retry_policy=dict(max_failed_attempts=0,retry_delay=0))
    story=status.failure_story(row,job['attempts'][-1],job,task,tmp_path,old,set(),100)
    assert 'UNBOUNDED legacy retry policy' in story['failure_next']
    pool.enable_automatic_retry('new-node')
    story=status.failure_story(row,job['attempts'][-1],job,task,tmp_path,pool.queue_state(),{nodes.DISPATCH_POLICY},100)
    assert '(2 recorded; limit 2)' in story['failure_next'] and 'no more automatic' in story['failure_next']
    assert 'no limit' not in story['failure_next']


def test_stale_ready_offer_cannot_claim_a_now_exhausted_task(tmp_path):
    runtime=Runtime(tmp_path);pool=nodes.Pool(tmp_path/'.multi-node',runtime.binding);pool.initialize()
    state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    action,offer=pool.tick(runtime,'inspector',threading.Event(),claim=False,offer=True)
    assert action=='ready'
    bad=next(t for t in runtime.plan('train')[0] if t['run_id']=='bad');key=nodes.task_key(bad)
    path=pool.root/'jobs'/(key+'.json');job=nodes.read(path);job.update(state='pending',attempts=history(2))
    nodes.atomic(path,job)
    action,claim=pool.claim_offered(offer,'next-node',threading.Event())
    try:assert action=='claimed' and claim[2]['run_id']=='healthy'
    finally:
        if claim:claim[0].close()
    assert nodes.read(path)['attempts']==job['attempts']


def test_exhausted_primary_still_allows_independent_ready_ext_and_eval(tmp_path):
    from test_srgc_v8_ready_eval_ext import prepared
    pool,runtime,budget=prepared(tmp_path);base=runtime.plan('train')[0][0]
    key=pool.register(base,'train');path=pool.root/'jobs'/(key+'.json');job=nodes.read(path)
    job.update(state='failed',attempts=history(8));nodes.atomic(path,job)
    claims=[]
    try:
        for i in range(2):
            action,claim=pool.tick(runtime,f'free-node-{i}',threading.Event())
            assert action=='claimed';claims.append(claim)
        assert {nodes.read(pool.root/'jobs'/(c[1]+'.json'))['phase'] for c in claims}=={'extension','evaluation'}
        assert nodes.read(path)==job and pool.queue_state()['phase']=='train'
    finally:
        for claim in claims:claim[0].close()


class Runtime:
    binding=dict(cohort='fixture')
    def __init__(self,root):self.root=Path(root)
    def verify_node(self):pass
    def plan(self,phase):
        assert phase=='train'
        return [dict(kind='train',run_id=name,method_id='on_monitored',config_hash=name,
                     training_seed=5,horizon=275,parent={}) for name in ('bad','healthy')],True
    def native_busy(self,task):return False
    def ready(self,task):return True
    def done(self,task):return (self.root/('done-'+task['run_id'])).is_file()
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        assert (self.root/'checkpoint.pt').read_bytes()==b'original optimizer RNG rewards'
        if task['run_id']=='bad':return 1
        (self.root/'done-healthy').write_text('original output');return 0


def run_worker(root,name):
    runtime=Runtime(root);pool=nodes.Pool(Path(root)/'.multi-node',runtime.binding)
    rc=nodes.worker_loop(pool,runtime,name,threading.Event(),poll=.002)
    (Path(root)/(name+'.rc')).write_text(str(rc))


def test_twelve_nodes_share_two_failures_then_stop_claiming_bad_task(tmp_path):
    # The unfixed version never terminates; the test reaps all processes on failure.
    runtime=Runtime(tmp_path);pool=nodes.Pool(tmp_path/'.multi-node',runtime.binding);pool.initialize()
    state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    (tmp_path/'checkpoint.pt').write_bytes(b'original optimizer RNG rewards')
    ctx=mp.get_context('fork');workers=[ctx.Process(target=run_worker,args=(str(tmp_path),f'node-{i}')) for i in range(12)]
    try:
        for worker in workers:worker.start()
        for worker in workers:worker.join(2)
        assert not any(worker.is_alive() for worker in workers),'workers infinitely relaunched the failed task'
        assert all(worker.exitcode==0 for worker in workers)
        jobs=[nodes.read(p) for p in pool.root.glob('jobs/*.json')]
        bad=next(j for j in jobs if j['task']['run_id']=='bad')
        healthy=next(j for j in jobs if j['task']['run_id']=='healthy')
        assert bad['state']=='failed' and len(bad['attempts'])==2
        assert all(a['exit_code']==1 and a['outcome']=='failed' for a in bad['attempts'])
        assert healthy['state']=='done' and len(healthy['attempts'])==1
        assert all((tmp_path/f'node-{i}.rc').read_text()=='1' for i in range(12))
        # Another fresh launch does not reset the cap, even with --retry-failed.
        joined=nodes.Pool(pool.root,pool.binding);joined.automatic_retry=True
        joined.initialize(retry=True,worker='replacement',recover=joined.controller_sha256)
        assert joined.tick(runtime,'replacement',threading.Event())[0]=='failed'
        assert nodes.read(pool.root/'jobs'/(nodes.task_key(bad['task'])+'.json'))['attempts']==bad['attempts']
        assert (tmp_path/'checkpoint.pt').read_bytes()==b'original optimizer RNG rewards'
        (tmp_path/'bounded-proof.json').write_text(json.dumps(dict(scope='CPU synthetic task faults',bad=bad,healthy=healthy)))
    finally:
        for worker in workers:
            if worker.is_alive():worker.terminate()
            worker.join(3)
