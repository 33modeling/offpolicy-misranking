"""Reuse SRGC's task-level failure semantics; preserve failed attempts and independent ready work."""
import threading

from scripts import srgc_v8_nodes as nodes


class Runtime:
    def __init__(self):self.completed=set()
    def plan(self,phase):
        return [dict(kind='train',run_id=name,method_id='on_monitored' if name=='first' else name,
                     training_seed=seed,cohort_id='olmo_math',config_hash=name,horizon=275,parent={})
                for name,seed in [('first',5),('other-seed',6),('ready-sr25',5)]],True
    def done(self,task):return task['run_id'] in self.completed
    def ready(self,task):return True
    def native_busy(self,task):return False


def setup(tmp_path):
    pool=nodes.Pool(tmp_path/'olmo_math/.multi-node',dict(schema=1,cohort='olmo_math',fixture=True))
    pool.initialize();state=nodes.read(pool.root/'state.json');state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    return pool,Runtime(),threading.Event()


def claim(pool,runtime,stop):
    action,claimed=pool.tick(runtime,'node',stop)
    assert action=='claimed'
    return claimed


def test_failed_task_does_not_block_other_seed_or_ready_sr25(tmp_path):
    pool,runtime,stop=setup(tmp_path)
    lease,key,task,attempt=claim(pool,runtime,stop)
    pool.finish(key,9,runtime,stop);lease.close()
    assert nodes.read(pool.root/'state.json')['failure'] is None
    assert nodes.read(pool.root/'jobs'/(key+'.json'))['state']=='failed'
    done=[]
    for _ in range(2):
        lease,next_key,task,attempt=claim(pool,runtime,stop)
        assert next_key!=key,'failed work was silently retried'
        runtime.completed.add(task['run_id']);done.append(task['run_id'])
        pool.finish(next_key,0,runtime,stop);lease.close()
    assert set(done)=={'other-seed','ready-sr25'}
    action,_=pool.tick(runtime,'node',stop)
    assert action=='failed','unresolved task failure must not become successful completion'
    assert nodes.read(pool.root/'state.json')['phase']=='train'
    assert len(nodes.read(pool.root/'jobs'/(key+'.json'))['attempts'])==1


def test_old_cohort_wide_task_failure_is_demoted_without_retry_reset_or_checkpoint_change(tmp_path):
    pool,runtime,stop=setup(tmp_path)
    lease,key,task,attempt=claim(pool,runtime,stop)
    job=nodes.read(pool.root/'jobs'/(key+'.json'));job['state']='failed'
    job['attempts'][-1]['exit_code']=9;nodes.atomic(pool.root/'jobs'/(key+'.json'),job);lease.close()
    pool.fail('task exit=9; output verified=False',key)
    before=(pool.root/'jobs'/(key+'.json')).read_bytes()
    saved=tmp_path/'olmo_math/runs/checkpoint/step-000025.pt';saved.parent.mkdir(parents=True)
    saved.write_bytes(b'original optimizer RNG intermediate rewards')
    inode=(pool.root/'binding.json').stat().st_ino;binding=(pool.root/'binding.json').read_bytes()
    pool.initialize()
    state=nodes.read(pool.root/'state.json')
    assert state['failure'] is None
    assert state['failure_history'][-1]['task']==key
    assert (pool.root/'jobs'/(key+'.json')).read_bytes()==before
    assert (pool.root/'binding.json').read_bytes()==binding and (pool.root/'binding.json').stat().st_ino==inode
    assert saved.read_bytes()==b'original optimizer RNG intermediate rewards'
    lease,next_key,task,attempt=claim(pool,runtime,stop)
    assert next_key!=key;lease.close()


def test_structural_or_unknown_failure_is_never_silently_unblocked(tmp_path):
    pool,runtime,stop=setup(tmp_path)
    for detail,key in [('invalid frozen input identity',None),('task exit=1; output verified=False','unknown-job')]:
        state=nodes.read(pool.root/'state.json');state['failure']=None;nodes.atomic(pool.root/'state.json',state)
        pool.fail(detail,key);original=nodes.read(pool.root/'state.json')['failure']
        pool.initialize()
        assert nodes.read(pool.root/'state.json')['failure']==original


def test_explicit_retry_reopens_only_failed_tasks_even_without_cohort_failure(tmp_path):
    pool,runtime,stop=setup(tmp_path)
    lease,key,task,attempt=claim(pool,runtime,stop)
    pool.finish(key,9,runtime,stop);lease.close()
    assert nodes.read(pool.root/'state.json')['failure'] is None
    pool.initialize(retry=True)
    lease,retry_key,task,attempt=claim(pool,runtime,stop)
    assert retry_key==key
    assert len(nodes.read(pool.root/'jobs'/(key+'.json'))['attempts'])==2
    lease.close()


def test_legacy_worker_failure_published_after_join_does_not_reblock_ready_jobs(tmp_path):
    pool,runtime,stop=setup(tmp_path)
    lease,key,task,attempt=claim(pool,runtime,stop)
    pool.finish(key,9,runtime,stop);lease.close()
    # An old coordinator can still be running one of the five MBPP jobs.
    pool.fail('task exit=9; output verified=False',key)
    action,next_claim=pool.tick(runtime,'already-joined-node',stop)
    assert action=='claimed' and next_claim[1]!=key
    assert nodes.read(pool.root/'state.json')['failure'] is None
    assert len(nodes.read(pool.root/'jobs'/(key+'.json'))['attempts'])==1
    next_claim[0].close()
