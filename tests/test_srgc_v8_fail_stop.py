"""Actual worker-loop regressions for nine old failures, native -11, and ready eval."""
import hashlib
import importlib.util
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from scripts import srgc_v8_nodes as nodes


import pytest as _pytest


@_pytest.fixture(autouse=True)
def _cap_without_user_grant(monkeypatch):
    """These tests pin the failure cap itself; the one-time user retry grant (11-13) has its own tests."""
    monkeypatch.setenv('V8_USER_RETRY_GRANT', '0')

if os.environ.get('V8_FAIL_STOP_SOURCE'):
    spec=importlib.util.spec_from_file_location('old_fail_stop',os.environ['V8_FAIL_STOP_SOURCE'])
    nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)


class Runtime:
    primary_results_first=True;streaming_ready_tasks=True
    def __init__(self,root,pool,evaluate=False):
        self.root=root;self.pool=pool;self.calls=[];self.evaluate=evaluate
        self.snapshot=root/'step-275.pt';self.snapshot.write_bytes(b'immutable CPU checkpoint sentinel')
        rule=dict(policy=nodes.BUDGET_RULE,binding_sha256=nodes.digest(pool.binding),horizon=275,seeds=[5])
        nodes.atomic(pool.root/'budget-rule.json',rule)
    def verify_node(self):pass
    def plan(self,phase):raise ValueError('unavailable failed parent')
    def side_tasks(self):
        task=dict(kind='evaluate',run_id='healthy',training_seed=5,split='final',completed_update=275,
                  snapshot_path=str(self.snapshot),snapshot_sha256='saved',evaluation_spec_hash='original')
        return dict(tasks=[dict(phase='evaluation',task=task)] if self.evaluate else [],budget_ready=False,
                    budget_rule_sha256=hashlib.sha256((self.pool.root/'budget-rule.json').read_bytes()).hexdigest(),
                    reason='reference_policy_costs_incomplete')
    def native_busy(self,task):return False
    def ready(self,task):return task['kind']=='evaluate' and self.snapshot.is_file()
    def done(self,task):return task['kind']=='evaluate' and (self.root/'eval-done').exists()
    def execute(self,task,*args):
        self.calls.append(task['kind']);assert task['kind']=='evaluate'
        (self.root/'eval-done').write_text('CPU metadata only');return 0


def failed_root(tmp_path,phase='extension',evaluate=False):
    root=tmp_path/'olmo_math';pool=nodes.Pool(root/'.multi-node',dict(cohort='olmo_math'))
    pool.initialize();state=pool.queue_state();state['phase']=phase;nodes.atomic(pool.root/'state.json',state)
    task=dict(kind='train',run_id='old-failed',method_id='on_monitored',training_seed=6,
              horizon=275,config_hash='same',parent={})
    key=pool.register(task,'train');path=pool.root/'jobs'/(key+'.json');job=nodes.read(path)
    job.update(state='failed',attempts=[dict(id=f'old-{i}',outcome='failed',exit_code=-11) for i in range(9)])
    nodes.atomic(path,job)
    return pool,Runtime(root,pool,evaluate),path


@pytest.mark.parametrize('phase',['extension','evaluation'])
@pytest.mark.parametrize('evaluate',[False,True])
def test_nine_failed_attempts_outside_current_plan_cannot_idle_worker_forever(tmp_path,phase,evaluate):
    pool,runtime,path=failed_root(tmp_path,phase,evaluate);before=path.read_bytes();stop=threading.Event()
    timer=threading.Timer(.5,stop.set);timer.start()
    try:rc=nodes.worker_loop(pool,runtime,'real-loop',stop,poll=.001)
    finally:timer.cancel();timer.join()
    assert rc==1 and not stop.is_set(),'9/2 failed job was hidden by the deferred phase and kept the worker looping'
    assert runtime.calls==(['evaluate'] if evaluate else [])
    assert path.read_bytes()==before
    assert runtime.snapshot.read_bytes()==b'immutable CPU checkpoint sentinel'
    assert pool.queue_state()['phase']==phase,'incomplete experiment was falsely advanced/completed'


@pytest.mark.parametrize('launcher_rc',[-11,1])
def test_first_sigsegv_is_retried_only_on_another_host_even_when_torchrun_wraps_it_as_exit1(tmp_path,launcher_rc):
    """v25: a first SIGSEGV is resumed from its checkpoint on another host, never on the host where it crashed."""
    pool=nodes.Pool(tmp_path/'queue',dict(cohort='fixture'));pool.automatic_retry=True;pool.host='crash-host'
    job=dict(state='failed',attempts=[dict(id='first',outcome='failed',exit_code=launcher_rc,host='crash-host',
        child_failure=dict(failures=[dict(signal=11,message='torchrun launcher exit -11; no Python traceback')]))])
    assert not pool.retryable(job),'SIGSEGV was sent back to the host where it crashed'
    assert pool.retry_later(job),'another host may still resume it: not a queue-wide failure'
    assert not nodes.native_failure_blocks(job)
    pool.host='other-host';assert pool.retryable(job)
    second=dict(job,attempts=job['attempts']+[dict(job['attempts'][0],id='second',host='other-host')])
    assert nodes.native_failure_blocks(second) and not pool.retryable(second) and not pool.retry_later(second)
    pool.initialize();key=pool.register(dict(kind='train',run_id='fatal',parent={}), 'train')
    path=pool.root/'jobs'/(key+'.json');saved=nodes.read(path);saved.update(job);nodes.atomic(path,saved)
    pool.initialize(recover=pool.controller_sha256,worker='fresh-node')
    assert nodes.read(path)==saved,'new controller/node reset the native-fault exclusion'


def test_held_native_execution_lock_prevents_false_quiescence_exit(tmp_path):
    pool,runtime,path=failed_root(tmp_path)
    lock=nodes.Lock(pool.root.parent/'runs/old-failed/.execution.lock');assert lock.acquire()
    try:assert nodes.quiescent_failure([(pool,runtime)]) is None
    finally:lock.close()
    assert nodes.quiescent_failure([(pool,runtime)])['faults']


def test_other_cohort_waiting_on_unavailable_parent_does_not_keep_failed_worker_alive(tmp_path):
    pool,runtime,path=failed_root(tmp_path);other=nodes.Pool(tmp_path/'qwen_mbpp/.multi-node',dict(cohort='qwen_mbpp'))
    other.initialize();state=other.queue_state();state['phase']='train';nodes.atomic(other.root/'state.json',state)
    class Waiting:
        def verify_node(self):pass
        def plan(self,phase):return [dict(kind='train',run_id='missing-parent',method_id='on_monitored',parent={})],False
        def native_busy(self,task):return False
        def ready(self,task):return False
        def done(self,task):return False
        def execute(self,*args):raise AssertionError('unready task launched')
    stop=threading.Event();timer=threading.Timer(.5,stop.set);timer.start()
    try:rc=nodes.worker_loop_many([(pool,runtime),(other,Waiting())],'multi-loop',stop,poll=.001)
    finally:timer.cancel();timer.join()
    assert rc==1 and not stop.is_set(),'another blocked cohort kept all nodes in an idle loop'
    assert other.queue_state()['failure'] is None,'worker-local exit poisoned a healthy shared queue'


def test_already_planned_ready_budget_eval_runs_even_if_global_eval_planner_omits_it(tmp_path):
    pool,runtime,path=failed_root(tmp_path,'extension')
    budget=runtime.root/'frozen/budget.json';nodes.atomic(budget,dict(value=10))
    task=dict(kind='evaluate',run_id='healthy',training_seed=5,split='final',completed_update=300,
              snapshot_path=str(runtime.snapshot),snapshot_sha256='already-pinned',evaluation_spec_hash='already-pinned-spec')
    key=pool.register(task,'evaluation')
    runtime.side_tasks=lambda:dict(tasks=[],budget_ready=True,budget_sha256=hashlib.sha256(budget.read_bytes()).hexdigest(),
        planning_errors=dict(evaluation='another selected checkpoint missing'))
    stop=threading.Event();timer=threading.Timer(.5,stop.set);timer.start()
    try:rc=nodes.worker_loop(pool,runtime,'ready-eval',stop,poll=.001)
    finally:timer.cancel();timer.join()
    assert runtime.calls==['evaluate'],'an already registered immutable eval stayed ready forever behind another plan error'
    assert nodes.read(pool.root/'jobs'/(key+'.json'))['state']=='done'
    assert rc==1 and not stop.is_set()


def test_extension_planner_error_does_not_discard_independent_evaluations(tmp_path,monkeypatch):
    nodes.atomic(tmp_path/'frozen/budget.json',dict(value=10))
    task=dict(kind='evaluate',snapshot_sha256='original',evaluation_spec_hash='original')
    def blocked(*args,**kwargs):raise ValueError('extension parent unavailable')
    runtime=SimpleNamespace(root=tmp_path,cfg=dict(seeds=dict(training=[5]),training=dict(matched_update_endpoint=275)),
        locator=lambda:None,mx=SimpleNamespace(build_stage=blocked))
    monkeypatch.setattr(nodes.Runtime,'plan',lambda *args:([task],True))
    result=nodes.ready_side_tasks(runtime)
    assert result['tasks']==[dict(phase='evaluation',task=task)]
    assert 'extension parent unavailable' in result['planning_errors']['extension']
