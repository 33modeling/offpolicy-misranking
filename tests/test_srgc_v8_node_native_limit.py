"""Native failures across different tasks must not burn the same GPU node forever."""
import threading

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_eval_native_recovery import prepared


def many_evaluations(tmp_path):
    pool,runtime,path,checkpoint,result=prepared(tmp_path,failures_remaining=100)
    original=nodes.read(path)['task']
    tasks=[original,*[dict(original,run_id=f'eval-{i}',evaluation_spec_hash=f'fixed-spec-{i}') for i in range(19)]]
    runtime.plan=lambda phase:(tasks,True)
    runtime.gpus=['fixture-GPU-0']
    def fail(task,*_):
        assert checkpoint.read_bytes()==b'original immutable checkpoint sentinel'
        runtime.calls.append(dict(task))
        return -11
    runtime.execute=fail
    for task in tasks:pool.register(task,'evaluation')
    return pool,runtime,checkpoint


def test_different_eval_tasks_cannot_accumulate_unbounded_native_faults_on_one_node(tmp_path):
    pool,runtime,checkpoint=many_evaluations(tmp_path)
    assert nodes.worker_loop(pool,runtime,'node-first-process',threading.Event(),poll=.001)==1
    assert len(runtime.calls)==2,'node kept launching different evals after repeated native faults'
    assert checkpoint.read_bytes()==b'original immutable checkpoint sentinel'
    assert not pool.queue_state().get('failure'),'node quarantine poisoned the shared queue'
    before=sum(len(nodes.read(p)['attempts']) for p in (pool.root/'jobs').glob('*.json'))
    joined=nodes.Pool(pool.root,pool.binding)
    assert nodes.worker_loop(joined,runtime,'node-new-process',threading.Event(),poll=.001)==1
    assert len(runtime.calls)==2,'restarting the same node renewed its native-fault budget'
    assert sum(len(nodes.read(p)['attempts']) for p in (pool.root/'jobs').glob('*.json'))==before


def test_native_failed_node_does_not_block_other_gpu_or_other_cohort(tmp_path):
    pool,runtime,checkpoint=many_evaluations(tmp_path)
    assert nodes.worker_loop(pool,runtime,'bad-node',threading.Event(),poll=.001)==1
    healthy_calls=[]
    runtime.gpus=['fixture-GPU-1']
    def healthy(task,*_):
        healthy_calls.append(task['run_id'])
        runtime.done=lambda t:t['run_id'] in healthy_calls
        return 0
    runtime.execute=healthy
    assert nodes.worker_loop(pool,runtime,'healthy-node',threading.Event(),poll=.001)==0
    assert len(healthy_calls)==20 and len(set(healthy_calls))==20


def test_other_cohort_still_runs_after_node_quarantine(tmp_path):
    bad,runtime,checkpoint=many_evaluations(tmp_path/'bad')
    good,healthy,path,_,result=prepared(tmp_path/'good')
    good.binding['cohort']='qwen_mbpp'
    nodes.atomic(good.root/'binding.json',good.binding)
    # A terminal phase avoids unrelated cross-cohort budget dependencies in this fixture.
    healthy.gpus=runtime.gpus
    assert nodes.worker_loop_many([(bad,runtime),(good,healthy)],'both',threading.Event(),poll=.001)==1
    assert len(runtime.calls)==2 and len(healthy.calls)==1 and result.is_file()
