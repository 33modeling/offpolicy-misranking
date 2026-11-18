"""11-16 '노드가 낭비되잖아': spare nodes of a full first cohort must reach ready work in later cohorts (CPU fixtures).

Reproduced with 16 CPU nodes over the real v8_dispatch -> v8_runtime_worker -> worker_loop_many path (slow
inspections, slow node verification, a runtime that fails on some nodes): every idle node re-inspected every
cohort's whole task DAG under that cohort's control.lock, so spare nodes waited minutes on the lock while ready
work sat in the next cohort, and their status said WAIT_QUEUE_LOCK of the full first cohort."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_sequential_cohorts import Runtime, events

ROOT=Path(__file__).resolve().parents[1]
FULL,NEXT='qwen_mbpp','olmo_math'


@pytest.fixture(autouse=True)
def _cap_without_user_grant(monkeypatch):
    monkeypatch.setenv('V8_USER_RETRY_GRANT','0')


class Actor(Runtime):
    """Inspected/verified off the loop like the dispatcher's runtime actors; tasks run until stopped."""
    independent_startup=True
    def __init__(self,base,cohort,tasks=2,verify=None):
        super().__init__(base,cohort);self.tasks=tasks;self.verify_gate=verify
    def verify_node(self):
        self.event('verify')
        if self.verify_gate is not None:self.verify_gate.wait(30)
    def plan(self,phase):
        if phase!='train':return [],True
        return [dict(kind='train',run_id=f'{self.cohort}-train-{i}',method_id='on_monitored',training_seed=i,
                     config_hash=nodes.digest([self.cohort,i]),parent={},horizon=275) for i in range(self.tasks)],True
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        self.event('execute',task['run_id'])
        while not stop.wait(.02):pass
        return 75


def queue(base,runtime):
    pool=nodes.Pool(runtime.root/'.multi-node',runtime.binding)
    pool.initialize();nodes.atomic(pool.root/'state.json',dict(phase='train',failure=None,actions=[],failure_history=[]))
    for task in runtime.plan('train')[0]:pool.register(task,'train')
    return pool


def hold_all_leases(pool):
    """Other nodes run every task of the full cohort: their task leases are held."""
    held=[]
    for path in pool.root.glob('jobs/*.json'):
        lease=nodes.Lock(pool.root/'leases'/(path.stem+'.lock'));assert lease.acquire();held.append(lease)
    return held


def node_rows(base,next_tasks,verify=None):
    rows=[]
    for name,tasks in ((FULL,2),(NEXT,next_tasks)):
        runtime=Actor(base,name,tasks,verify if name==NEXT else None)
        rows.append((nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime))
    return rows


def start(rows,name,stop,poll):
    thread=threading.Thread(target=nodes.worker_loop_many,args=(rows,name,stop),
                            kwargs=dict(poll=poll,cohort_order=(FULL,NEXT)),daemon=True)
    thread.start();return thread


def until(condition,seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if condition():return True
        time.sleep(.02)
    return False


def test_idle_nodes_do_not_reinspect_a_full_cohort_on_every_poll(tmp_path,monkeypatch):
    """Four nodes, the first cohort full on other nodes, three ready tasks next: three claim them, one has
    nothing. Before: every node re-inspected both whole queues every poll under their control locks."""
    monkeypatch.setattr(nodes,'ACTIVE_RESERVE',0)   # held leases stand for busy nodes; spill mechanics only
    monkeypatch.setattr(nodes,'SPILL_STALL_SECONDS',.3)
    full=queue(tmp_path,Actor(tmp_path,FULL));queue(tmp_path,Actor(tmp_path,NEXT,3))
    held=hold_all_leases(full)
    inspections={FULL:0,NEXT:0};original=nodes.Pool.tick
    def counted(self,*args,**kwargs):
        inspections[self.binding['cohort']]+=1;return original(self,*args,**kwargs)
    monkeypatch.setattr(nodes.Pool,'tick',counted)
    stop=threading.Event()
    threads=[start(node_rows(tmp_path,3),f'node{n}',stop,.05) for n in range(4)]
    try:
        assert until(lambda:len([r for r in events(tmp_path) if r['stage']=='execute'])==3,20)
        before=dict(inspections);time.sleep(2)   # 2 s with one idle node and nothing new anywhere
        after=dict(inspections)
    finally:
        stop.set()
        for t in threads:t.join(10)
        for lease in held:lease.close()
    runs=[r['task'] for r in events(tmp_path) if r['stage']=='execute']
    assert sorted(runs)==[f'{NEXT}-train-{i}' for i in range(3)]          # each ready task claimed once
    # The idle node reuses the last inspection while jobs/ and state.json are unchanged (one per minute).
    assert after[FULL]-before[FULL]<=1 and after[NEXT]-before[NEXT]<=1,(before,after)


def test_spare_nodes_claim_from_a_recent_inspection_of_another_node_once_each(tmp_path,monkeypatch):
    """The next cohort's ready offer, found by another node, is claimed through the same claim_offered()
    revalidation under its control lock, without a whole-DAG inspection per node; each task exactly once."""
    monkeypatch.setattr(nodes,'ACTIVE_RESERVE',0)   # held leases stand for busy nodes; spill mechanics only
    monkeypatch.setattr(nodes,'SPILL_STALL_SECONDS',.3)
    full=queue(tmp_path,Actor(tmp_path,FULL));nxt=queue(tmp_path,Actor(tmp_path,NEXT,2))
    held=hold_all_leases(full)
    jobs=sorted(p.stem for p in nxt.root.glob('jobs/*.json'))
    offer=dict(phase='train',side_budget_sha256=None,side_rule_sha256=None,
               jobs=[dict(key=k,phase='train',retry=False,task=nodes.semantic_task(nodes.read(nxt.root/'jobs'/(k+'.json'))['task'])) for k in jobs])
    nodes.atomic(tmp_path/'.v8-scan-share'/(NEXT+'.json'),dict(schema=1,cohort=NEXT,binding_sha256=nodes.digest(nxt.binding),
                 at=time.time(),worker='other-node',action='ready',offer=offer,observation=dict(ready=2),marks=None))
    inspections={FULL:0,NEXT:0};original=nodes.Pool.tick
    def counted(self,*args,**kwargs):
        inspections[self.binding['cohort']]+=1;return original(self,*args,**kwargs)
    monkeypatch.setattr(nodes.Pool,'tick',counted)
    stop=threading.Event()
    threads=[start(node_rows(tmp_path,2),f'spare{n}',stop,.05) for n in range(3)]
    try:
        ok=until(lambda:len([r for r in events(tmp_path) if r['stage']=='execute'])==2,15)
        time.sleep(.5)
    finally:
        stop.set()
        for t in threads:t.join(10)
        for lease in held:lease.close()
    runs=[r['task'] for r in events(tmp_path) if r['stage']=='execute']
    assert ok and sorted(runs)==[f'{NEXT}-train-0',f'{NEXT}-train-1'],runs   # no double claim, third node idle
    jobs=[nodes.read(p) for p in nxt.root.glob('jobs/*.json')]
    assert all(len(j['attempts'])==1 for j in jobs)   # one claim per task (then stopped)
    assert inspections[NEXT]<=1,inspections   # the third node may inspect once after its share was used


def test_status_of_a_spilled_node_names_the_cohort_it_works_on_and_what_it_serves(tmp_path,monkeypatch):
    """Before: 'WAIT_QUEUE_LOCK qwen_mbpp' for minutes while the node was verifying/starting the next cohort."""
    monkeypatch.setattr(nodes,'ACTIVE_RESERVE',0)   # held leases stand for busy nodes; spill mechanics only
    monkeypatch.setattr(nodes,'SPILL_STALL_SECONDS',.3)
    full=queue(tmp_path,Actor(tmp_path,FULL));queue(tmp_path,Actor(tmp_path,NEXT,1))
    held=hold_all_leases(full)
    gate=threading.Event();stop=threading.Event()
    rows=node_rows(tmp_path,1,verify=gate)
    # Other nodes hold the full cohort's queue lock as soon as this node has joined it.
    lock=open(full.root/'control.lock','a+')
    joined=lambda:any(r['cohort']==FULL and r['stage']=='verify' for r in events(tmp_path))
    thread=threading.Thread(target=nodes.worker_loop_many,args=(rows,'spill-node',stop),
        kwargs=dict(poll=.05,cohort_order=(FULL,NEXT),not_served={'qwen_math':'qwen_math: configured Python is unavailable'}),daemon=True)
    try:
        thread.start()
        assert until(joined,10) and until(lambda:nodes.Pool(full.root,full.binding).root.joinpath('binding.json').exists(),5)
        time.sleep(.3);fcntl.flock(lock,fcntl.LOCK_EX)
        path=tmp_path/'.v8-worker-status'/(nodes.digest('spill-node')+'.json')
        def checking_next():
            try:row=nodes.read(path)
            except (OSError,ValueError):return False
            return row.get('cohort')==NEXT and row.get('state')=='CHECKING'
        assert until(checking_next,10),nodes.read(path)
        row=nodes.read(path)
        assert row['nothing_claimable_in']==FULL and row['serves']==[FULL,NEXT]
        assert row['not_serving']=={'qwen_math':'qwen_math: configured Python is unavailable'}
        assert row['cohorts'][NEXT]=='node verification'
    finally:
        gate.set();stop.set();thread.join(10);lock.close()
        for lease in held:lease.close()


def test_sequential_claims_do_not_probe_every_task_lease_under_the_dispatch_lock(tmp_path,monkeypatch):
    """The claim order is the cohort rank; the per-claim lease count (one flock per task lease on the shared
    volume, under the one cross-cohort dispatch lock) was computed and never used."""
    from test_srgc_v8_sequential_cohorts import pairs
    import inspect as stack
    probes=[];original=nodes.Pool.held_leases
    def held(self):
        if stack.stack()[1].function=='_worker_loop_many':probes.append(self.binding['cohort'])
        return original(self)
    monkeypatch.setattr(nodes.Pool,'held_leases',held)
    rows=pairs(tmp_path,cohorts=(FULL,NEXT))
    assert nodes.worker_loop_many(rows,'cpu-node',threading.Event(),poll=.001,cohort_order=(FULL,NEXT))==0
    assert len([r for r in events(tmp_path) if r['stage']=='execute'])==8 and probes==[]


def gate_node(tmp_path,poll):
    queue(tmp_path,Actor(tmp_path,FULL,1))
    rows=[(nodes.Pool(r.root/'.multi-node',r.binding),r) for r in [Actor(tmp_path,FULL,1)]]
    stop=threading.Event()
    thread=threading.Thread(target=nodes.worker_loop_many,args=(rows,'gate-node',stop),
                            kwargs=dict(poll=poll,cohort_order=(FULL,)),daemon=True)
    return thread,stop


def executed(tmp_path):return any(r['stage']=='execute' for r in events(tmp_path))


def test_node_with_a_ready_offer_retries_a_busy_dispatch_lock_without_a_full_poll(tmp_path):
    gate=open(tmp_path/'.v8-dispatch.lock','a+');fcntl.flock(gate,fcntl.LOCK_EX)   # another node is claiming
    thread,stop=gate_node(tmp_path,3)
    status=tmp_path/'.v8-worker-status'/(nodes.digest('gate-node')+'.json')
    def offered():
        try:return nodes.read(status).get('ready')==1
        except (OSError,ValueError):return False
    try:
        thread.start()
        assert until(offered,20);time.sleep(.3)   # the node holds a ready offer and found the gate busy
        released=time.monotonic();gate.close()
        assert until(lambda:executed(tmp_path),10)
        delay=time.monotonic()-released
    finally:
        stop.set();thread.join(10)
    assert delay<1.0,delay   # before: the next try came one whole poll (3 s) later


def test_finished_verification_and_inspection_are_collected_without_waiting_a_poll(tmp_path):
    thread,stop=gate_node(tmp_path,3)
    started=time.monotonic()
    try:
        thread.start();assert until(lambda:executed(tmp_path),20)
        took=time.monotonic()-started
    finally:
        stop.set();thread.join(10)
    assert took<2.0,took   # before: verification, inspection and claim each waited a 3 s poll (about 6-9 s)


def test_failed_runtime_start_reports_its_own_error(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('idle_dispatch',ROOT/'scripts/v8_dispatch.py')
    dispatch=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatch)
    plan=dict(root=str(tmp_path),gpus=['0','1','2','3'])
    actor=dispatch.RuntimeProxy(dict(cohort='qwen_math',python=sys.executable,runner=str(tmp_path)),plan,tmp_path/'plan.json',
                                asynchronous=True,lazy=True)
    actor.error='ValueError: qwen_math: original Python/package versions are not available'
    actor.on_ready=lambda proxy:proxy.streaming_ready_tasks   # the policy receipt reads attributes of a started actor
    with pytest.raises(Exception,match='runtime actor unavailable on this node: ValueError: qwen_math: original Python') as caught:
        actor.verify_node()
    assert type(caught.value).__name__=='NodeFailure'


def test_startup_lists_every_served_and_not_served_cohort(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('idle_dispatch2',ROOT/'scripts/v8_dispatch.py')
    dispatch=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatch)
    lines=dispatch.serving_lines('h1.7',['qwen_mbpp','olmo_math'],{'qwen_math':'no queue'})
    assert 'NODE h1.7 SERVES 2: qwen_mbpp -> olmo_math' in lines[1] and lines[2]=='[v8] NODE h1.7 NOT SERVING qwen_math: no queue'
    import v8_queue_runtime as q
    env=dict(os.environ,V8_ROOT=str(tmp_path/'v8'),V8_RUNNER_ROOT=str(ROOT/'v8/experiments'),V8_PYTHON=sys.executable)
    env.pop('V8_COHORTS',None)
    plan=q.selections(ROOT,env,['qwen_mbpp'],serve_remaining=True)
    assert plan.order==['qwen_mbpp'] and {'olmo_math','olmo_mbpp','qwen_math'}<=set(plan.unqueued)
    assert all('no queue under' in reason for reason in plan.unqueued.values())
    from v8_progress_text import scheduler_text
    text=scheduler_text(dict(at=time.time(),state='CHECKING',cohort='olmo_math',reason='node verification',
                             serves=['qwen_mbpp','olmo_math'],not_serving={'qwen_math':'no queue'},nothing_claimable_in='qwen_mbpp'),time.time())
    assert 'serves qwen_mbpp -> olmo_math' in text and 'NOT serving qwen_math: no queue' in text and 'nothing claimable in Qwen / MBPP' in text
