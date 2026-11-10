"""Inspected metadata is advisory; current leases and immutable tasks decide claims."""
import threading

import pytest

from scripts import srgc_v8_nodes as nodes


class Runtime:
    def plan(self,phase):
        return [dict(kind='train',run_id=name,method_id='on_monitored',training_seed=seed,
                     config_hash=name,parent={},horizon=275) for seed,name in [(5,'a'),(6,'b')]],True
    def native_busy(self,task):return False
    def done(self,task):return False
    def ready(self,task):return True


def prepared(tmp_path):
    pool=nodes.Pool(tmp_path/'olmo_math/.multi-node',dict(cohort='olmo_math',revision='original'))
    runtime=Runtime();stop=threading.Event()
    with pool.control():
        pool.initialize();state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
        action,offer=pool.tick(runtime,'inspector',stop,claim=False,offer=True)
    assert action=='ready'
    assert all(nodes.read(p)['attempts']==[] for p in (pool.root/'jobs').glob('*.json'))
    return pool,runtime,stop,offer


def test_two_nodes_cannot_claim_the_same_inspected_job_and_claim_never_calls_runtime(tmp_path):
    pool,runtime,stop,offer=prepared(tmp_path)
    def forbidden(*args):raise AssertionError('Original Runtime scan inside the shared dispatch lock')
    runtime.plan=runtime.ready=runtime.done=runtime.native_busy=forbidden
    held=[]
    try:
        for worker in ('node-a','node-b'):
            with pool.control():
                action,claim=pool.claim_offered(offer,worker,stop)
            assert action=='claimed';held.append(claim)
        assert held[0][1]!=held[1][1]
        with pool.control():assert pool.claim_offered(offer,'node-c',stop)==('waiting',None)
    finally:
        for claim in held:claim[0].close()


def test_failure_or_phase_change_after_inspection_cannot_launch_stale_work(tmp_path):
    pool,runtime,stop,offer=prepared(tmp_path)
    with pool.control():
        for item in offer['jobs']:
            path=pool.root/'jobs'/(item['key']+'.json');job=nodes.read(path);job['state']='failed';nodes.atomic(path,job)
        assert pool.claim_offered(offer,'late-node',stop)==('waiting',None)
        state=pool.queue_state();state['phase']='freeze';nodes.atomic(pool.root/'state.json',state)
        assert pool.claim_offered(offer,'late-node',stop)==('stale',None)
        state['phase']='train';state['failure']={'detail':'invalid frozen input'};nodes.atomic(pool.root/'state.json',state)
        assert pool.claim_offered(offer,'late-node',stop)==('stale',None)


def test_changed_source_binding_or_task_is_rejected_after_inspection(tmp_path):
    pool,runtime,stop,offer=prepared(tmp_path)
    with pool.control():
        nodes.atomic(pool.root/'binding.json',dict(pool.binding,revision='foreign'))
        with pytest.raises(nodes.BindingMismatch):pool.claim_offered(offer,'late-node',stop)
        nodes.atomic(pool.root/'binding.json',pool.binding)
        path=pool.root/'jobs'/(offer['jobs'][0]['key']+'.json');job=nodes.read(path)
        job['task']['horizon']=400;nodes.atomic(path,job)
        # V8-158: a stale offer is skipped, never a queue failure (it used to stop every node)
        action,claimed=pool.claim_offered(offer,'late-node',stop)
        if claimed is not None:claimed[0].close()             # the next valid item of the same offer may be taken
        assert claimed is None or claimed[1]!=offer['jobs'][0]['key']
        assert nodes.read(path)['state']=='pending' and not pool.queue_state().get('failure')


def test_recorded_stale_claim_failure_is_cleared_when_a_node_starts(tmp_path):
    """V8-158: queues failed by the old ValueError resume once a node with this code checks them."""
    pool,runtime,stop,offer=prepared(tmp_path)
    pool.fail('ValueError: Inspected task definition changed before claim: '+offer['jobs'][0]['key'])
    assert pool.queue_state()['failure']
    assert pool.recover_budget_api_failure(pool.controller_sha256,'restarted-node')
    state=pool.queue_state()
    assert state['failure'] is None and state['actions'][-1]['action']=='recover_stale_offer_claim_failure'
    assert 'Inspected task definition changed before claim' in state['failure_history'][-1]['detail']
