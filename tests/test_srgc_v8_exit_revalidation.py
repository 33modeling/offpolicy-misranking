"""Healthy work appearing after an idle scan must prevent a stale worker exit."""
import hashlib
import threading

import pytest

from scripts import srgc_v8_nodes as nodes


@pytest.mark.parametrize('hidden_failure',[False,True])
def test_newly_available_checkpoint_is_rechecked_before_worker_exits(tmp_path,hidden_failure):
    root=tmp_path/'olmo_math';pool=nodes.Pool(root/'.multi-node',dict(cohort='olmo_math'));pool.initialize()
    bad=dict(kind='train',run_id='bad',method_id='on_monitored',training_seed=6,horizon=275,config_hash='bad',parent={})
    healthy=dict(kind='train',run_id='healthy',method_id='bank_sr_t025',training_seed=5,horizon=275,config_hash='healthy',parent={})
    if hidden_failure:
        healthy=dict(kind='evaluate',run_id='healthy',training_seed=5,split='final',completed_update=275,
                     snapshot_sha256='fixed',evaluation_spec_hash='fixed')
        nodes.atomic(pool.root/'budget-rule.json',dict(policy=nodes.BUDGET_RULE,binding_sha256=nodes.digest(pool.binding),horizon=275,seeds=[5]))
    key=pool.register(bad,'train');path=pool.root/'jobs'/(key+'.json');job=nodes.read(path)
    job.update(state='failed',attempts=[dict(id=f'old-{i}',outcome='failed',exit_code=1) for i in range(9)])
    nodes.atomic(path,job);state=pool.queue_state();state['phase']='extension' if hidden_failure else 'train';nodes.atomic(pool.root/'state.json',state)
    stop=threading.Event();calls=[];observations=[];checkpoint=root/'checkpoint.pt'
    class Runtime:
        primary_results_first=hidden_failure;streaming_ready_tasks=hidden_failure
        def verify_node(self):pass
        def plan(self,phase):return [bad,healthy],True
        def side_tasks(self):
            return dict(tasks=[dict(phase='evaluation',task=healthy)],budget_ready=False,
                        budget_rule_sha256=hashlib.sha256((pool.root/'budget-rule.json').read_bytes()).hexdigest())
        def native_busy(self,task):return False
        def done(self,task):return False
        def ready(self,task):
            if task['run_id']=='bad':return False
            observations.append(True)
            if not checkpoint.exists():
                # Another node's publication happens just after the first observation.
                checkpoint.write_bytes(b'immutable CPU checkpoint sentinel');return False
            return True
        def execute(self,task,*args):calls.append(task['run_id']);stop.set();return 75
    timer=threading.Timer(.5,stop.set);timer.start()
    try:nodes.worker_loop(pool,Runtime(),'race-node',stop,poll=.001)
    finally:timer.cancel();timer.join()
    assert calls==['healthy'],'stale no-work observation terminated a node with a newly ready task'
    assert len(observations)>=2
    assert nodes.read(path)['attempts']==job['attempts']


def test_worker_retirement_does_not_count_old_failures_as_new_model_crashes(tmp_path):
    pool=nodes.Pool(tmp_path/'queue',dict(cohort='fixture'));pool.initialize()
    key=pool.register(dict(kind='train',run_id='same',parent={}), 'train');path=pool.root/'jobs'/(key+'.json')
    job=nodes.read(path);job['state']='failed'
    job['attempts']=[dict(id=f'old-{i}',worker='old-worker',outcome='failed',exit_code=1) for i in range(9)]
    job['attempts'] += [dict(id='new-done',worker='current',outcome='done',exit_code=0),
        dict(id='new-native',worker='current',outcome='failed',exit_code=-11)]
    nodes.atomic(path,job)
    report=nodes.record_worker_retirement([(pool,None)],'current','no ready work')
    assert report['exit_kind']=='blocked_work' and report['exit_code']==1
    assert report['current_worker_counts']==dict(done=1,native_crash=1)
    assert report['lifetime_task_failure_count']==10 and report['previous_worker_failure_count']==9
    assert nodes.read(path)==job


def test_unreadable_exit_diagnostics_do_not_mask_the_original_failure(tmp_path):
    pool=nodes.Pool(tmp_path/'queue',dict(cohort='fixture'));pool.initialize()
    path=pool.root/'jobs/broken.json';path.parent.mkdir(exist_ok=True);path.write_text('{incomplete')
    report=nodes.record_worker_retirement([(pool,None)],'current','original error')
    assert report['exit_code']==1 and report['reason']=='original error'
    assert report['unreadable_records'][0]['path']==str(path)
