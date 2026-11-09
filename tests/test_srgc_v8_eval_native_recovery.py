"""Bounded automatic recovery of the same immutable evaluation after exit -11."""
import importlib.util
import os
from pathlib import Path
import threading

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_child_errors import child_runtime

if os.environ.get('V8_EVAL_RECOVERY_SOURCE'):
    spec=importlib.util.spec_from_file_location('old_eval_recovery',os.environ['V8_EVAL_RECOVERY_SOURCE'])
    nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)


def prepared(tmp_path,old_failures=0,failures_remaining=0,complete=False):
    root=tmp_path/'olmo_math';pool=nodes.Pool(root/'.multi-node',dict(cohort='olmo_math'));pool.initialize()
    checkpoint=root/'step-275.pt';checkpoint.write_bytes(b'original immutable checkpoint sentinel')
    task=dict(kind='evaluate',run_id='same-run',training_seed=5,completed_update=275,
        snapshot_path=str(checkpoint),snapshot_sha256='same-checkpoint',evaluation_spec_hash='same-spec',
        split='final',responses=4,question_ids_hash='same-questions')
    key=pool.register(task,'evaluation');path=pool.root/'jobs'/(key+'.json')
    job=nodes.read(path)
    if old_failures:
        job.update(state='failed',attempts=[dict(id=f'old-{i}',outcome='failed',exit_code=-11,
            controller_policy='fatal-fault-quiescence-v21',worker='old-node') for i in range(old_failures)])
        nodes.atomic(path,job)
    state=pool.queue_state();state['phase']='evaluation'
    if old_failures:state['failure']=dict(detail=f'{old_failures} task(s) failed; no independent ready/running work remains.',task=None)
    nodes.atomic(pool.root/'state.json',state)
    result=root/'saved-result.json'
    if complete:result.write_bytes(b'original completed evaluation result')
    class Runtime:
        def __init__(self):self.calls=[]
        def verify_node(self):pass
        def plan(self,phase):return [task],True
        def ready(self,t):return checkpoint.is_file()
        def native_busy(self,t):return False
        def done(self,t):return result.is_file()
        def barrier(self,*args):pass
        def execute(self,t,*args):
            assert t==task and checkpoint.read_bytes()==b'original immutable checkpoint sentinel'
            self.calls.append(dict(t))
            if len(self.calls)<=failures_remaining:return -11
            result.write_bytes(b'original completed evaluation result');return 0
    return pool,Runtime(),path,checkpoint,result


@pytest.mark.parametrize('old_failures',[0,9])
def test_plain_launch_recovers_native_eval_without_replacing_its_checkpoint_or_spec(tmp_path,old_failures):
    pool,runtime,path,checkpoint,result=prepared(tmp_path,old_failures,failures_remaining=0 if old_failures else 1)
    assert nodes.worker_loop(pool,runtime,'new-node',threading.Event(),poll=.001)==0
    job=nodes.read(path)
    assert len(runtime.calls)==(1 if old_failures else 2),'native eval never received its bounded automatic replay'
    assert job['state']=='done' and len(job['attempts'])==(old_failures+1 if old_failures else 2)
    assert job['evaluation_native_recovery']['max_failed_replays']==1
    assert checkpoint.read_bytes()==b'original immutable checkpoint sentinel'


@pytest.mark.parametrize('old_failures',[0,9])
def test_repeated_native_eval_is_finite_and_new_nodes_cannot_renew_recovery(tmp_path,old_failures):
    pool,runtime,path,checkpoint,result=prepared(tmp_path,old_failures,failures_remaining=100)
    assert nodes.worker_loop(pool,runtime,'first-node',threading.Event(),poll=.001)==1
    assert len(runtime.calls)==(1 if old_failures else 2)
    before=nodes.read(path)
    for i in range(3):
        joined=nodes.Pool(pool.root,pool.binding);joined.controller_sha256=f'later-code-{i}'
        assert nodes.worker_loop(joined,runtime,f'new-node-{i}',threading.Event(),poll=.001)==1
        assert nodes.read(path)==before,'restarting a node renewed the recovery budget'
    assert not result.exists()


def test_finished_native_eval_is_adopted_without_running_it_again(tmp_path):
    pool,runtime,path,checkpoint,result=prepared(tmp_path,9,complete=True)
    original=nodes.read(path)['attempts'];content=result.read_bytes()
    assert nodes.worker_loop(pool,runtime,'new-node',threading.Event(),poll=.001)==0
    assert runtime.calls==[] and nodes.read(path)['attempts']==original
    assert nodes.read(path)['state']=='done' and result.read_bytes()==content


def test_training_sigsegv_is_not_reopened_by_the_evaluation_repair(tmp_path):
    pool,runtime,path,checkpoint,result=prepared(tmp_path,9)
    job=nodes.read(path);job['task']=dict(kind='train',run_id='native-failed-training',parent={})
    assert not pool.arm_evaluation_recovery(job,'new-node')
    pool.automatic_retry=True;assert not pool.retryable(job)


def test_changed_task_cannot_reuse_an_evaluation_recovery_receipt(tmp_path):
    pool,runtime,path,checkpoint,result=prepared(tmp_path,9);job=nodes.read(path)
    assert pool.arm_evaluation_recovery(job,'new-node')
    job['task']['snapshot_sha256']='different-checkpoint'
    assert nodes.evaluation_recovery_attempts(job) is None
    pool.automatic_retry=True;assert not pool.retryable(job)


def test_changed_binding_cannot_reuse_an_evaluation_recovery_receipt(tmp_path):
    pool,runtime,path,checkpoint,result=prepared(tmp_path,9);job=nodes.read(path)
    assert pool.arm_evaluation_recovery(job,'new-node')
    job['evaluation_native_recovery']['binding_sha256']='different-input-binding'
    pool.automatic_retry=True;assert not pool.retryable(job)


def test_status_shows_bounded_eval_recovery_instead_of_permanent_native_block(tmp_path):
    from test_v8_status import status
    pool,runtime,path,checkpoint,result=prepared(tmp_path,9)
    pool.automatic_retry=True;pool.initialize(worker='new-node',recover=pool.controller_sha256)
    job=nodes.read(path);row=dict(_log_lines=[],_manifest={},kind='evaluate',done=None,total=None)
    story=status.failure_story(row,job['attempts'][-1],job,job['task'],checkpoint.parent,pool.queue_state(),{nodes.DISPATCH_POLICY},100)
    assert 'evaluation recovery queued' in story['failure_next']
    assert '0 of 1' in story['failure_next'] and 'lifetime failures 9' in story['failure_next']


def test_actual_cpu_torchrun_eval_recovers_after_one_native_sigsegv(tmp_path,child_runtime):
    run,progress,log=child_runtime
    pool,runtime,path,checkpoint,result=prepared(tmp_path);calls=[]
    def execute(task,matrix,fd,stop,device_fds,attempt):
        mode='segfault' if not calls else 'success';calls.append(mode)
        rc=run(mode,pool.root/'progress'/attempt['id'])
        if rc==0:result.write_bytes(b'original completed evaluation result')
        return rc
    runtime.execute=execute
    assert nodes.worker_loop(pool,runtime,'real-cpu-node',threading.Event(),poll=.001)==0
    job=nodes.read(path)
    assert calls==['segfault','success'] and job['state']=='done'
    assert [a['outcome'] for a in job['attempts']]==['failed','done']
    assert any(f.get('signal')==11 for f in job['attempts'][0]['child_failure']['failures'])
    assert checkpoint.read_bytes()==b'original immutable checkpoint sentinel'
