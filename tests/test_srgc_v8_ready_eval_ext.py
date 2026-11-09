"""Ready ext/eval overlap pending training, keeping original budget and evaluation protocols."""
import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from scripts import srgc_v8_nodes as nodes


def prepared(tmp_path,phase='train'):
    root=tmp_path/'olmo_math';pool=nodes.Pool(root/'.multi-node',dict(cohort='olmo_math'));pool.initialize()
    budget=root/'frozen/budget.json';nodes.atomic(budget,dict(value=10,scope='CPU metadata fixture'))
    base=dict(kind='train',run_id='unfinished',method_id='on_monitored',training_seed=5,horizon=275,config_hash='same',parent={})
    extension=dict(base,run_id='extension',method_id='bank_sr_t025')
    evaluation=dict(kind='evaluate',run_id='completed',snapshot_sha256='checkpoint',evaluation_spec_hash='spec',completed_update=275)
    class Runtime:
        primary_results_first=True;streaming_ready_tasks=True
        primary_cohort_scope=('olmo_math','qwen_mbpp')
        def plan(self,phase):return [base],True
        def side_tasks(self):return dict(tasks=[dict(phase='extension',task=extension),dict(phase='evaluation',task=evaluation)],
            budget_ready=True,budget_sha256=hashlib.sha256(budget.read_bytes()).hexdigest())
        def done(self,t):return False
        def native_busy(self,t):return False
        def ready(self,t):return t['run_id']!='unfinished'
    state=pool.queue_state();state['phase']=phase;nodes.atomic(pool.root/'state.json',state)
    return pool,Runtime(),budget


def test_ready_extension_and_evaluation_are_claimed_while_other_training_is_incomplete(tmp_path):
    pool,runtime,budget=prepared(tmp_path);stop=threading.Event();claims=[]
    try:
        for i in range(2):
            action,claim=pool.tick(runtime,'cpu-node-'+str(i),stop)
            assert action=='claimed';claims.append(claim)
        assert {claim[2]['kind'] for claim in claims}=={'train','evaluate'}
        assert {nodes.read(pool.root/'jobs'/(claim[1]+'.json'))['phase'] for claim in claims}=={'extension','evaluation'}
        assert pool.queue_state()['phase']=='train'
        assert len(pool.held_leases())==2
    finally:
        for claim in claims:claim[0].close()


def test_parallel_policy_removes_cross_cohort_primary_result_barrier(tmp_path):
    pool,runtime,budget=prepared(tmp_path,phase='extension')
    action,claim=pool.tick(runtime,'idle-node',threading.Event())
    assert action=='claimed' and pool.extension_blockers()==[]
    try:assert pool.queue_state()['execution_order']['policy']==nodes.CONCURRENT_ORDER
    finally:claim[0].close()
    assert not nodes.primary_results_ready(tmp_path/'qwen_mbpp')


def test_three_ready_kinds_rotate_instead_of_waiting_behind_many_trajectories(tmp_path):
    pool,runtime,budget=prepared(tmp_path);base=runtime.plan('train')[0][0]
    runtime.plan=lambda phase:([dict(base,run_id='trajectory-'+str(i)) for i in range(20)],True)
    runtime.ready=lambda task:True
    claims=[]
    try:
        for i in range(3):
            action,claim=pool.tick(runtime,'cpu-node-'+str(i),threading.Event())
            assert action=='claimed';claims.append(claim)
        phases=[nodes.read(pool.root/'jobs'/(c[1]+'.json'))['phase'] for c in claims]
        assert phases==['train','extension','evaluation'],'ready ext/eval waited behind all training trajectories'
    finally:
        for claim in claims:claim[0].close()


def test_side_offer_rechecks_frozen_budget_before_claim(tmp_path):
    pool,runtime,budget=prepared(tmp_path)
    action,offer=pool.tick(runtime,'inspector',threading.Event(),claim=False,offer=True);assert action=='ready'
    nodes.atomic(budget,dict(value=999,scope='tampered CPU fixture'))
    assert pool.claim_offered(offer,'idle-node',threading.Event())==('stale',None)
    assert all(not nodes.read(path)['attempts'] for path in pool.root.glob('jobs/*.json'))


def test_original_runtime_actor_transports_ready_side_work_without_new_helper_modules(tmp_path,monkeypatch):
    import importlib
    from test_srgc_v8_coordination_fixes import fixture_plan
    monkeypatch.syspath_prepend(str(Path(nodes.__file__).parent))
    monkeypatch.setenv('V8_CODE_ROOT',str(Path(__file__).resolve().parents[1]))
    dispatch=importlib.import_module('v8_dispatch');root,planfile=fixture_plan(tmp_path,cohorts=('olmo_math',))
    plan=json.loads(planfile.read_text());plan.update(worker='cpu-rpc',gpus=['0','1','2','3'])
    script=Path(plan['entries'][0]['script'])
    script.write_text(script.read_text()+"\nRuntime.streaming_ready_tasks=True\nRuntime.side_tasks=lambda self:dict(tasks=[],budget_ready=False,reason='CPU transport fixture')\n")
    planfile.write_text(json.dumps(plan))
    actor=dispatch.RuntimeProxy(plan['entries'][0],plan,planfile)
    try:
        assert actor.streaming_ready_tasks is True and actor.error is None
        assert actor.side_tasks()==dict(tasks=[],budget_ready=False,reason='CPU transport fixture')
    finally:actor.close()


def test_original_cpu_runner_starts_ext_and_eval_before_unrelated_uniform_training(tmp_path,monkeypatch):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('original V8 runtime required')
    monkeypatch.setenv('V8_CODE_ROOT',str(Path(__file__).resolve().parents[1]))
    from test_srgc_v8_nodes import ToyRuntime,verify_toy_complete
    class ReadyToy(ToyRuntime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.streaming_ready_tasks=True
        def ready(self,task):
            if task.get('method_id','').startswith('bank_uniform_'):
                if not all((self.root/('side-started-'+name+'.json')).exists() for name in ('extension','evaluate')):return False
            return super().ready(task)
        def execute(self,task,*args):
            side='evaluate' if task['kind']=='evaluate' else 'extension' if task.get('method',{}).get('extension_of') else None
            if side:
                budget=self.root/'frozen/budget.json';assert budget.exists()
                nodes.atomic(self.root/('side-started-'+side+'.json'),dict(kind=side,run_id=task['run_id'],
                    budget_sha256=hashlib.sha256(budget.read_bytes()).hexdigest(),source='CPU original runner'))
            return super().execute(task,*args)
        def barrier(self,action,*args):
            if action=='freeze':
                assert (self.root/'frozen/budget.json').exists();return
            return super().barrier(action,*args)
    runtime=ReadyToy(runner,tmp_path);pool=nodes.Pool(runtime.root/'.multi-node',runtime.binding)
    stop=threading.Event();timer=threading.Timer(25,stop.set);timer.start()
    try:rc=nodes.worker_loop(pool,runtime,'cpu-parallel',stop,poll=.001)
    finally:timer.cancel()
    assert rc==0,'ready ext/eval still waited for unrelated unfinished uniform training'
    budget=(runtime.root/'frozen/budget.json').read_bytes();sha=hashlib.sha256(budget).hexdigest()
    for name in ('extension','evaluate'):
        assert nodes.read(runtime.root/('side-started-'+name+'.json'))['budget_sha256']==sha
    assert nodes.primary_results_ready(runtime.root)
    verify_toy_complete(runner,tmp_path,'olmo_math',monkeypatch,min_workers=1)


def test_budget_waits_for_selected_on_policy_endpoint_even_if_another_branch_completed(tmp_path,monkeypatch):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('original V8 runtime required')
    monkeypatch.syspath_prepend(runner)
    from switching import bank
    cfg=dict(seeds=dict(training=[5]),training=dict(matched_update_endpoint=275),runtime=dict(gpu_type='CPU fixture',world_size=4))
    runtime=SimpleNamespace(root=tmp_path,cohort='olmo_math',cfg=cfg,
        cluster=SimpleNamespace(prefix_cost_from_receipts=lambda *a:None),locator=lambda:None,
        mx=SimpleNamespace(build_stage=lambda *a,**k:([],[])))
    monkeypatch.setattr(nodes.Runtime,'plan',lambda *a:([],True))
    for name,status,update in [('other-branch','completed_horizon',275),('selected-on','active',250)]:
        nodes.atomic(tmp_path/'runs'/name/'manifest.json',dict(cohort_id='olmo_math',role='rebuttal_bank',
            training_seed=5,status=status,completed_updates=update))
    class Bank:
        def __init__(self,*args,**kwargs):pass
        def policy_rows(self):return [dict(policy_id='srgc_current',seed=5,run_id='selected-on',status='not_triggered',
            continuation_gpu_seconds=10,cold_start_gpu_seconds=None)]
    monkeypatch.setattr(bank,'SeedBank',Bank)
    result=nodes.ready_side_tasks(runtime)
    assert result['budget_ready'] is False and not result['tasks']
    assert not (tmp_path/'frozen/budget.json').exists()
