"""Actual worker-loop regressions for nine old failures, native -11, and ready eval."""
import hashlib
import importlib.util
import os
from pathlib import Path
import threading

import pytest

from scripts import srgc_v8_nodes as nodes

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
def test_first_sigsegv_is_terminal_even_when_torchrun_wraps_it_as_exit1(tmp_path,launcher_rc):
    pool=nodes.Pool(tmp_path/'queue',dict(cohort='fixture'));pool.automatic_retry=True
    job=dict(state='failed',attempts=[dict(id='first',outcome='failed',exit_code=launcher_rc,
        child_failure=dict(failures=[dict(signal=11,message='torchrun launcher exit -11; no Python traceback')]))])
    assert not pool.retryable(job),'SIGSEGV was scheduled again despite no repair'
    assert not pool.retry_later(job)
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
