"""Matched-horizon results for every cohort precede any new extension claim; CPU only."""
import hashlib
from pathlib import Path
import threading
from types import SimpleNamespace

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_nodes import FakeRuntime,task


def publish(root,binding):
    root=Path(root);directory=root/'analysis/matched-update-275';directory.mkdir(parents=True)
    for name in ('analysis.json','policy_outcomes_bank.json','report.md'):(directory/name).write_text('CPU control')
    budget=root/'frozen/budget.json';nodes.atomic(budget,{'value':1,'scope':'CPU software fixture'})
    receipt=root/'.multi-node/primary-results.json'
    nodes.atomic(receipt,dict(policy=nodes.PRIMARY_ORDER,root=str(root.resolve()),analysis_dir=str(directory.relative_to(root)),
        binding_sha256=nodes.digest(binding),budget_sha256=hashlib.sha256(budget.read_bytes()).hexdigest()))
    state=nodes.read(root/'.multi-node/state.json');state['primary_results_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
    nodes.atomic(root/'.multi-node/state.json',state)


def prepared(root,cohort):
    runtime=FakeRuntime(root/cohort/'runtime');runtime.primary_results_first=True
    runtime.primary_cohort_scope=('a','b');runtime.binding['cohort']=cohort
    pool=nodes.Pool(root/cohort/'.multi-node',runtime.binding);pool.initialize()
    runtime.plan=lambda phase:([task('extension')],True)
    runtime.ready=lambda t:True
    state=pool.queue_state();state['phase']='extension';nodes.atomic(pool.root/'state.json',state)
    return pool,runtime


def test_extension_waits_for_other_cohorts_primary_report_and_rechecks_stale_offer(tmp_path):
    left,lr=prepared(tmp_path,'a');right,rr=prepared(tmp_path,'b')
    publish(left.root.parent,left.binding);publish(right.root.parent,right.binding)
    left.prioritize_primary_results(lr)
    action,offer=left.tick(lr,'inspector',threading.Event(),claim=False,offer=True);assert action=='ready'
    saved=(right.root/'primary-results.json').read_bytes();(right.root/'primary-results.json').write_bytes(b'corrupt')
    assert left.claim_offered(offer,'idle-node',threading.Event())==('waiting',None)
    assert left.tick(lr,'idle-node',threading.Event())[0]=='waiting'
    (right.root/'primary-results.json').write_bytes(saved)
    action,claim=left.tick(lr,'idle-node',threading.Event());assert action=='claimed'
    claim[0].close()


def test_reorder_preserves_binding_cached_plans_failures_and_active_job(tmp_path):
    pool,runtime=prepared(tmp_path,'a');binding=(pool.root/'binding.json').read_bytes()
    cached=pool.root/'plans/evaluation.json';nodes.atomic(cached,dict(tasks=[],sealed=True));original=cached.read_bytes()
    t=task('old-extension');key=pool.register(t,'extension')
    held=nodes.Lock(pool.root/'leases'/(key+'.lock'));assert held.acquire()
    try:
        assert pool.prioritize_primary_results(runtime)['phase']=='extension'
        assert pool.tick(runtime,'new-node',threading.Event())[0]=='waiting'
    finally:held.close()
    state=pool.queue_state();failure=dict(detail='1 task(s) failed; no independent ready/running work remains.',task=None)
    state['failure']=failure;nodes.atomic(pool.root/'state.json',state)
    state=pool.prioritize_primary_results(runtime)
    assert state['phase']=='primary_evaluation' and state['failure_history'][-1]==failure
    assert (pool.root/'binding.json').read_bytes()==binding and cached.read_bytes()==original
    assert nodes.read(pool.root/'jobs'/(key+'.json'))['task']==t


def test_primary_evaluation_does_not_wait_for_old_budget_eval_jobs(tmp_path):
    pool,runtime=prepared(tmp_path,'a');runtime.primary_results_first=False
    primary=dict(task('primary','evaluate'),snapshot_sha256='primary',evaluation_spec_hash='spec')
    budget=dict(task('budget','evaluate'),snapshot_sha256='budget',evaluation_spec_hash='spec')
    pool.register(budget,'evaluation');runtime.plan=lambda phase:([primary],True)
    state=pool.queue_state();state['phase']='primary_evaluation';nodes.atomic(pool.root/'state.json',state)
    action,claim=pool.tick(runtime,'node',threading.Event());assert action=='claimed'
    assert claim[2]['snapshot_sha256']=='primary';claim[0].close()
    assert nodes.read(pool.root/'jobs'/(nodes.task_key(budget)+'.json'))['attempts']==[]


def test_primary_override_keeps_old_training_and_budget_fix_without_recursive_closure(monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(Path(nodes.__file__).parent))
    worker=importlib.import_module('v8_runtime_worker');control=importlib.import_module('srgc_v8_nodes')
    runtime=SimpleNamespace(cfg={},mx=object(),cluster=object(),locator=lambda:None,cpu=lambda *a:None,
        plan=lambda phase:([phase],False),barrier=lambda action,*a:action)
    monkeypatch.setattr(control,'primary_evaluation_tasks',lambda r:(['primary'],True))
    monkeypatch.setattr(control.Runtime,'plan',lambda r,phase:(['corrected-budget'],True))
    monkeypatch.setattr(control.Runtime,'barrier',lambda r,action,*a:'primary-report')
    worker.control_overrides(runtime,"row['budget_endpoint_update']")
    assert runtime.plan('train')==(['train'],False)
    assert runtime.plan('primary_evaluation')==(['primary'],True)
    assert runtime.plan('evaluation')==(['corrected-budget'],True)
    assert runtime.barrier('freeze')=='freeze' and runtime.barrier('primary_finalize')=='primary-report'


def test_primary_report_survives_an_unavailable_cohort_without_hanging_worker(tmp_path):
    left,lr=prepared(tmp_path,'a');right,rr=prepared(tmp_path,'b')
    publish(left.root.parent,left.binding)
    def unavailable():raise nodes.NodeFailure('CPU fixture cannot open cohort b')
    rr.verify_node=unavailable
    assert nodes.worker_loop_many([(left,lr),(right,rr)],'node',threading.Event(),poll=.001)==1
    assert nodes.primary_results_ready(left.root.parent)
    assert left.queue_state()['phase']=='extension' and left.queue_state()['failure'] is None
    assert not (lr.path/'start-extension').exists()
