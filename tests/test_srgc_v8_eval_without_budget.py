"""A failed reference seed must not stop another immutable checkpoint's final evaluation."""
import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from scripts import srgc_v8_nodes as nodes


def partial_reference(tmp_path,monkeypatch):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('original V8 runtime required')
    monkeypatch.syspath_prepend(runner)
    from switching import matrix
    root=tmp_path/'olmo_math';run=root/'runs/healthy';run.mkdir(parents=True)
    snapshot=run/'step-275.pt';snapshot.write_bytes(b'CPU immutable checkpoint fixture, not real weights')
    sha=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    nodes.atomic(run/'manifest.json',dict(run_id='healthy',cohort_id='olmo_math',role='rebuttal_bank',
        training_seed=5,status='completed_horizon',completed_updates=275,task=dict(method={})))
    (run/'snapshots.jsonl').write_text(json.dumps(dict(completed_update=275,path=snapshot.name,sha256=sha))+'\n')
    cfg=dict(seeds=dict(training=[5,6]),training=dict(matched_update_endpoint=275),verifier='unchanged-fixture',
             evaluation=dict(final=dict(responses=2,seed_purpose='final')),runtime=dict(gpu_type='CPU fixture',world_size=4))
    binding=dict(cohort='olmo_math',scope='CPU metadata fixture')
    class Runtime:
        primary_results_first=True;streaming_ready_tasks=True
        def locator(self):return SimpleNamespace(loc=dict(seeds={'5':dict(input_bundle='seed5'),'6':dict(input_bundle='seed6')}))
        def plan(self,phase):return [dict(kind='train',run_id='failed-seed6',training_seed=6,method_id='on_monitored',horizon=275,config_hash='same',parent={})],False
        def side_tasks(self):return nodes.ready_side_tasks(self)
        def ready(self,task):return task['kind']=='evaluate'
        def done(self,task):return False
        def native_busy(self,task):return False
    runtime=Runtime();runtime.root=root;runtime.cohort='olmo_math';runtime.cfg=cfg;runtime.binding=binding
    runtime.mx=matrix;runtime.cluster=SimpleNamespace(load_bundle=lambda p:dict(evaluation_ids=['fixed-q1','fixed-q2']))
    pool=nodes.Pool(root/'.multi-node',binding);pool.initialize()
    state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    return pool,runtime


def test_missing_reference_seed_does_not_block_ready_checkpoint_evaluation(tmp_path,monkeypatch):
    pool,runtime=partial_reference(tmp_path,monkeypatch)
    action,claim=pool.tick(runtime,'spare-node',threading.Event())
    assert action=='claimed','healthy checkpoint evaluation waited for a different reference seed'
    try:
        assert claim[2]['kind']=='evaluate' and claim[2]['completed_update']==275
        assert not (runtime.root/'frozen/budget.json').exists()
        assert (pool.root/'budget-rule.json').exists()
    finally:claim[0].close()


def test_early_evaluation_rechecks_its_precommitted_rule_before_claim(tmp_path,monkeypatch):
    pool,runtime=partial_reference(tmp_path,monkeypatch)
    action,offer=pool.tick(runtime,'inspector',threading.Event(),claim=False,offer=True);assert action=='ready'
    path=pool.root/'budget-rule.json';rule=nodes.read(path);rule['seeds']=[5];nodes.atomic(path,rule)
    assert pool.claim_offered(offer,'spare-node',threading.Event())==('stale',None)
    assert all(not nodes.read(p)['attempts'] for p in pool.root.glob('jobs/*.json'))


def test_evaluation_planning_error_does_not_discard_ready_extensions(tmp_path,monkeypatch):
    pool,runtime=partial_reference(tmp_path,monkeypatch)
    nodes.atomic(runtime.root/'frozen/budget.json',dict(value=10))
    extension=dict(kind='train',run_id='ready-extension',method_id='bank_sr_t025',training_seed=5,horizon=275,parent={})
    runtime.mx=SimpleNamespace(build_stage=lambda *a,**k:([extension],[]),eval_tasks=lambda *a,**k:[])
    def failed_plan(*args):raise ValueError('Selected budget checkpoint is missing or changed: another-run@300')
    monkeypatch.setattr(nodes.Runtime,'plan',failed_plan)
    result=nodes.ready_side_tasks(runtime)
    assert result['tasks']==[dict(phase='extension',task=extension)]
    assert 'Selected budget checkpoint' in result['planning_errors']['evaluation']


def test_cannot_claim_a_precommit_after_untracked_evaluation_results_already_exist(tmp_path,monkeypatch):
    pool,runtime=partial_reference(tmp_path,monkeypatch)
    nodes.atomic(runtime.root/'evaluations/old.json',dict(scope='untracked fixture'))
    with pytest.raises(ValueError,match='before budget|precommit'):nodes.ready_side_tasks(runtime)


def test_original_runner_evaluates_before_a_blocked_reference_seed_can_train(tmp_path,monkeypatch):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('original V8 runtime required')
    monkeypatch.setenv('V8_CODE_ROOT',str(Path(__file__).resolve().parents[1]))
    from test_srgc_v8_nodes import ToyRuntime,verify_toy_complete
    class PartialToy(ToyRuntime):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.streaming_ready_tasks=True
        def ready(self,task):
            if task['kind']=='train' and task['training_seed']==7 and not (self.root/'early-evaluation.json').exists():return False
            return super().ready(task)
        def execute(self,task,*args):
            early=task['kind']=='evaluate' and not (self.root/'frozen/budget.json').exists()
            result=super().execute(task,*args)
            if early:
                assert (self.root/'.multi-node/budget-rule.json').exists()
                assert not (self.root/'frozen/budget.json').exists()
                nodes.atomic(self.root/'early-evaluation.json',dict(task=task,scope='CPU original evaluator; no H100 result',
                    blocked_reference_seed=7,budget_not_frozen=True))
            return result
        def barrier(self,action,*args):
            if action=='freeze':
                if not (self.root/'frozen/budget.json').exists():nodes.freeze_precommitted_budget(self)
                return
            return super().barrier(action,*args)
    runtime=PartialToy(runner,tmp_path);pool=nodes.Pool(runtime.root/'.multi-node',runtime.binding)
    stop=threading.Event();timer=threading.Timer(25,stop.set);timer.start()
    try:rc=nodes.worker_loop(pool,runtime,'cpu-early-eval',stop,poll=.001)
    finally:timer.cancel()
    assert rc==0,'completed checkpoints still waited for the blocked reference seed'
    early=nodes.read(runtime.root/'early-evaluation.json');assert early['budget_not_frozen']
    budget=nodes.read(runtime.root/'frozen/budget.json')
    assert budget['selection_protocol']==nodes.BUDGET_RULE and budget['evaluation_records_present_at_numeric_freeze']
    assert budget['budget_rule_receipt_sha256']==hashlib.sha256((pool.root/'budget-rule.json').read_bytes()).hexdigest()
    verify_toy_complete(runner,tmp_path,'olmo_math',monkeypatch,min_workers=1)
