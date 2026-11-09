"""Exact archived budget functions and narrowly scoped recovery; no model/GPU work."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from scripts import srgc_v8_nodes as nodes

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('index',range(7))
def test_ready_budget_accepts_each_exact_archived_freeze_function(tmp_path,monkeypatch,index):
    runner=os.environ.get('V8_NODES_TEST_RUNTIME')
    if not runner:pytest.skip('original V8 runtime required')
    monkeypatch.syspath_prepend(runner)
    from switching import bank,freeze
    with zipfile.ZipFile(ROOT/'v8/queue-runtimes.zip') as z:
        catalog=json.loads(z.read('catalog.json'));saved=catalog['runners'][index]
        source=z.read(saved['root']+'/switching/freeze.py').decode()
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='freeze_budget')
    namespace=dict(vars(freeze));exec(compile(ast.Module(body=[fn],type_ignores=[]),'<archived:'+saved['source_commit']+'>','exec'),namespace)
    monkeypatch.setattr(freeze,'freeze_budget',namespace['freeze_budget'])
    cfg=dict(seeds=dict(training=[5]),training=dict(matched_update_endpoint=275),runtime=dict(gpu_type='CPU fixture',world_size=4))
    runtime=SimpleNamespace(root=tmp_path,cohort='olmo_math',cfg=cfg,locator=lambda:None,
        cluster=SimpleNamespace(prefix_cost_from_receipts=lambda *a:None),mx=SimpleNamespace(build_stage=lambda *a,**k:([],[])))
    nodes.atomic(tmp_path/'runs/selected-on/manifest.json',dict(cohort_id='olmo_math',role='rebuttal_bank',
        training_seed=5,status='completed_horizon',completed_updates=275))
    class Bank:
        def __init__(self,*args,**kwargs):pass
        def policy_rows(self):return [dict(policy_id='srgc_current',seed=5,run_id='selected-on',status='not_triggered',
            continuation_gpu_seconds=10,cold_start_gpu_seconds=None)]
    monkeypatch.setattr(bank,'SeedBank',Bank);monkeypatch.setattr(nodes.Runtime,'plan',lambda *a:([],True))
    result=nodes.ready_side_tasks(runtime)
    assert result['budget_ready'] is True
    budget=nodes.read(tmp_path/'frozen/budget.json');assert budget['value']==10 and budget['n_seeds']==1


@pytest.mark.parametrize('verified,detail,cleared',[
    (True,"ValueError: olmo_math: TypeError: freeze_budget() got an unexpected keyword argument 'expected_seeds'",True),
    (False,"ValueError: olmo_math: TypeError: freeze_budget() got an unexpected keyword argument 'expected_seeds'",False),
    (True,'invalid frozen input identity',False),
    (True,'ValueError: Selected budget checkpoint is missing or changed: another-run@300',True),
])
def test_only_verified_known_controller_api_failure_is_reopened(tmp_path,verified,detail,cleared):
    pool=nodes.Pool(tmp_path/'olmo_math/.multi-node',dict(cohort='olmo_math'));pool.initialize()
    checkpoint=tmp_path/'latest.pt';checkpoint.write_bytes(b'original model optimizer RNG rewards')
    state=pool.queue_state();state['phase']='train';state['failure']=dict(detail=detail,task=None,at=1)
    nodes.atomic(pool.root/'state.json',state);original=state['failure'];binding=(pool.root/'binding.json').read_bytes()
    pool.automatic_retry=True;pool.initialize(worker='verified-node',recover=pool.controller_sha256 if verified else None)
    state=pool.queue_state()
    assert (state['failure'] is None)==cleared
    if cleared:assert original in state['failure_history']
    if 'Selected budget checkpoint' in detail:assert state['unresolved_evaluation_planner_failure']==original
    assert (pool.root/'binding.json').read_bytes()==binding
    assert checkpoint.read_bytes()==b'original model optimizer RNG rewards'
