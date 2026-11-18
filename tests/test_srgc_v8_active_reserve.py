"""11-18 'qwen_mbpp 먼저 끝내': the first unfinished cohort keeps reserve nodes, does not lose a node while it is
still being inspected, and is retried soon after a node-local give-up. Real node loop, forked CPU workers, no GPUs."""
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_sequential_cohorts import Runtime

FIRST,LATER='qwen_mbpp','olmo_mbpp'
ORDER=(FIRST,LATER)


@pytest.fixture(autouse=True)
def _fixture_env(monkeypatch):
    monkeypatch.setenv('V8_USER_RETRY_GRANT','0')
    monkeypatch.delenv('V8_ACTIVE_RESERVE',raising=False)


def wait_file(path,timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline and not Path(path).exists():time.sleep(.01)
    return Path(path).exists()


class Flow(Runtime):
    """FIRST: train task P, then C1/C2 that wait for P's checkpoint. LATER: six independent train tasks.
    Train tasks block until a release file exists (shared across forked workers)."""
    pending_barrier=None
    independent_startup=True

    def __init__(self,base,cohort,gpus=()):
        super().__init__(base,cohort);self.gpus=tuple(gpus)

    def event(self,stage,task=''):
        nodes.atomic(self.root.parent/'events'/f'{time.time_ns()}-{self.cohort}-{stage}-{task}.json',
                     dict(cohort=self.cohort,stage=stage,task=task,pid=os.getpid(),gpus=list(self.gpus)))

    def plan(self,phase):
        if phase!='train':return super().plan(phase)
        names=['P','C1','C2'] if self.cohort==FIRST else [f'L{i}' for i in range(6)]
        return [dict(kind='train',run_id=f'{self.cohort}-{name}',method_id='on_monitored',training_seed=i,
                     config_hash=nodes.digest([self.cohort,name]),parent={},horizon=275,
                     snapshot_sha256=nodes.digest([self.cohort,name]),evaluation_spec_hash=nodes.digest('CPU'))
                for i,name in enumerate(names)],True

    def ready(self,task):
        if self.cohort==FIRST and task['run_id'] in (f'{FIRST}-C1',f'{FIRST}-C2'):
            return (self.root/f'done-{FIRST}-P').exists()
        return True

    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        self.event('execute',task['run_id'])
        if task['kind']=='train':
            gate=self.root.parent/('release-'+('P' if task['run_id']==f'{FIRST}-P' else self.cohort))
            if task['run_id'] in (f'{FIRST}-C1',f'{FIRST}-C2'):gate=None
            while gate is not None and not gate.exists():
                if stop.wait(.01):return 75
        (self.root/('done-'+task['run_id'])).write_text('CPU fixture only')
        return 0

    def begin_barrier(self,action,fd,stop,device_fds):self.barrier(action);self.pending_barrier=(action,os.dup(fd))

    def poll_barrier(self,stop):
        if self.pending_barrier is None:return None
        action,fd=self.pending_barrier;self.pending_barrier=None;return action,fd,None


def flow_pairs(base,gpus=()):
    rows=[]
    for name in ORDER:
        runtime=Flow(base,name,gpus)
        rows.append((nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime))
    return rows


def worker(base,name,stop,gpus=()):
    code=nodes.worker_loop_many(flow_pairs(base,gpus),name,stop,poll=.005,cohort_order=ORDER)
    nodes.atomic(Path(base)/'exits'/name,dict(code=code))


def executes(base,cohort=None,task=None):
    rows=[nodes.read(p) for p in sorted((Path(base)/'events').glob('*.json'))] if (Path(base)/'events').is_dir() else []
    return [r for r in rows if r['stage']=='execute' and (cohort is None or r['cohort']==cohort)
            and (task is None or r['task']==task)]


def statuses(base):
    folder=Path(base)/'.v8-worker-status'
    return [nodes.read(p) for p in folder.glob('*.json')] if folder.is_dir() else []


def until(check,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if check():return True
        time.sleep(.02)
    return check()


class Stops(list):
    """One stop event per worker process: a SIGKILLed worker may die holding its own event's internal lock."""
    def set(self):
        for event in self:event.set()


def start(ctx,base,names,stop,gpus=None):
    children=[]
    for i,name in enumerate(names):
        event=ctx.Event();stop.append(event)
        child=ctx.Process(target=worker,args=(base,name,event,(gpus or {}).get(name,())));child.start()
        child.stop_event=event;children.append(child)
    return children


def finish(children,stop,timeout=90):
    for child in children:
        child.join(timeout)
    alive=[c for c in children if c.is_alive()]
    for child in children:child.stop_event.set()
    for child in children:
        child.join(3)
        if child.is_alive():child.kill();child.join()
    return alive


def release_all(base):
    for name in ('release-P','release-'+FIRST,'release-'+LATER):(Path(base)/name).write_text('go')


def test_two_idle_nodes_stay_with_the_running_first_cohort_and_take_its_children_at_once(tmp_path):
    ctx=mp.get_context('fork');stop=Stops()
    children=start(ctx,tmp_path,[f'node-{i}' for i in range(5)],stop)
    try:
        assert until(lambda:executes(tmp_path,FIRST,f'{FIRST}-P'),30)
        # 5 nodes: one runs P, two stay in reserve for FIRST, the other two take LATER work.
        assert until(lambda:len(executes(tmp_path,LATER))>=2,40)
        time.sleep(8)   # past the reserve warm-up: no third node moves on
        assert len(executes(tmp_path,LATER))==2
        assert until(lambda:sum(s.get('state')=='WAIT_RESERVE' for s in statuses(tmp_path))==2,20)
        reserve=[s for s in statuses(tmp_path) if s.get('state')=='WAIT_RESERVE']
        assert all(s['cohort']==FIRST and 'reserve node' in s['reason'] for s in reserve)
        before=time.monotonic();(tmp_path/'release-P').write_text('go')
        assert until(lambda:len(executes(tmp_path,FIRST))==3,15),'children of P not claimed by the reserve'
        took=time.monotonic()-before
        assert took<15 and len(executes(tmp_path,LATER))==2,'LATER tasks were still blocked; the reserve ran C1/C2'
        release_all(tmp_path)
        assert not finish(children,stop)
        assert all(nodes.read(p)['code']==0 for p in (tmp_path/'exits').glob('*'))
        claims=[(r['cohort'],r['task']) for r in executes(tmp_path)]
        assert len(claims)==len(set(claims)),'a task ran twice'
        assert all(pool.queue_state()['phase']=='complete' for pool,_ in flow_pairs(tmp_path))
    finally:
        release_all(tmp_path);finish(children,stop,timeout=5)


def test_killed_node_task_is_resumed_by_a_reserve_node_without_a_second_concurrent_start(tmp_path):
    ctx=mp.get_context('fork');stop=Stops()
    children=start(ctx,tmp_path,[f'node-{i}' for i in range(4)],stop)
    try:
        assert until(lambda:executes(tmp_path,FIRST,f'{FIRST}-P'),30)
        assert until(lambda:len(executes(tmp_path,LATER))>=1,40)
        time.sleep(8)
        assert len(executes(tmp_path,LATER))==1   # 4 nodes: P + 2 reserve + 1 moved on
        victim=executes(tmp_path,FIRST,f'{FIRST}-P')[0]['pid']
        os.kill(victim,signal.SIGKILL)   # whole node lost mid-task; its lease is released by the kernel
        assert until(lambda:len(executes(tmp_path,FIRST,f'{FIRST}-P'))==2,15),'orphaned P not resumed by a reserve node'
        assert len(executes(tmp_path,LATER))==1
        job=nodes.read([p for p in (tmp_path/FIRST/'.multi-node/jobs').glob('*.json')
                        if nodes.read(p)['task']['run_id']==f'{FIRST}-P'][0])
        first,second=job['attempts']
        assert first['outcome']=='node_lost' and second['resumed_from_attempt']==first['id']
        assert executes(tmp_path,FIRST,f'{FIRST}-P')[1]['pid']!=victim
        release_all(tmp_path)
        rest=[c for c in children if c.pid!=victim]
        assert not finish(rest,stop)
        children=rest
        assert all(nodes.read(p)['code']==0 for p in (tmp_path/'exits').glob('*'))
        assert all(pool.queue_state()['phase']=='complete' for pool,_ in flow_pairs(tmp_path))
    finally:
        release_all(tmp_path);finish(children,stop,timeout=5)


def crashed_place_job(base,gpus):
    """P failed with a native crash (not the periodic dump: 5 s after start) on this host's GPU set `gpus`."""
    pool,runtime=flow_pairs(base)[0]
    pool.initialize();state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    for task in runtime.plan('train')[0]:pool.register(task,'train')
    path=[p for p in (pool.root/'jobs').glob('*.json') if nodes.read(p)['task']['run_id']==f'{FIRST}-P'][0]
    job=nodes.read(path);job['state']='failed'
    job['attempts']=[dict(id='crash-0',outcome='failed',exit_code=-11,host=socket.gethostname(),gpus=list(gpus),
                          worker='node-a.1',started_at=100.0,ended_at=105.0,controller_sha256='old')]
    nodes.atomic(path,job)
    return path


def test_crash_place_node_moves_on_and_another_node_runs_the_failed_task(tmp_path):
    A,B=('0','1','2','3'),('4','5','6','7')
    path=crashed_place_job(tmp_path,A)
    ctx=mp.get_context('fork');stop=Stops();shared=tmp_path/'.v8-scan-share'/f'{FIRST}.json';rows=[]
    children=start(ctx,tmp_path,['node-a'],stop,gpus={'node-a':A})
    try:
        # The crash place cannot run P; it does not hold itself in reserve for a task it may not run.
        deadline=time.monotonic()+40
        while time.monotonic()<deadline and not executes(tmp_path,LATER):
            if shared.exists():
                try:rows.append(nodes.read(shared))
                except (OSError,ValueError):pass
            time.sleep(.005)
        assert executes(tmp_path,LATER) and not executes(tmp_path,FIRST)
        assert not any(r.get('action')=='waiting' and (r['observation'].get('recovery_wait') or r['observation'].get('failed'))
                       for r in rows),'a place-dependent waiting scan was shared'
        children+=start(ctx,tmp_path,['node-b'],stop,gpus={'node-b':B})
        assert until(lambda:executes(tmp_path,FIRST,f'{FIRST}-P'),30)
        assert executes(tmp_path,FIRST,f'{FIRST}-P')[0]['gpus']==list(B)
        (tmp_path/'release-P').write_text('go')
        assert until(lambda:len(executes(tmp_path,FIRST))==3,30)
        release_all(tmp_path)
        assert not finish(children,stop)
        job=nodes.read(path)
        assert job['state']=='done' and [a['id'] for a in job['attempts']][0]=='crash-0'
        assert len(executes(tmp_path,FIRST,f'{FIRST}-P'))==1
    finally:
        release_all(tmp_path);finish(children,stop,timeout=5)


def test_slow_first_cohort_inspection_keeps_the_node_for_it(tmp_path,monkeypatch):
    """F8: the node has moved on (the first cohort had nothing ready) and holds a ready later offer while a new
    inspection of the first cohort runs for 4 s and then finds a ready task. Before: the later task was claimed
    once that inspection was older than the claim grace. Now the node waits for the first cohort's inspection."""
    monkeypatch.setattr(nodes,'SPILL_STALL_SECONDS',.3);monkeypatch.setattr(nodes,'SPILL_LOCK_GRACE',.5)
    monkeypatch.setattr(nodes,'SCAN_SHARE_SECONDS',0)   # every loop inspects the first cohort itself
    rows=flow_pairs(tmp_path);pool,first=rows[0];gate=threading.Event();ticks={'n':0}
    first.ready=lambda task:gate.is_set() and task['run_id']==f'{FIRST}-P'
    tick=pool.tick
    def slow_tick(*args,**kwargs):
        ticks['n']+=1
        if ticks['n']>=2 and not gate.is_set():time.sleep(4);gate.set()   # slow, then P is ready
        return tick(*args,**kwargs)
    pool.tick=slow_tick
    stop=threading.Event();claimed=[]
    for _,runtime in rows:
        def execute(task,matrix,fd,stop_,device_fds,attempt,runtime=runtime):
            claimed.append(runtime.cohort);stop.set();return 75
        runtime.execute=execute
    timer=threading.Timer(60,stop.set);timer.start()
    try:nodes.worker_loop_many(rows,'solo',stop,poll=.005,cohort_order=ORDER)
    finally:timer.cancel()
    assert ticks['n']>=2 and (rows[1][0].root/'state.json').exists(),'the node did move on to the later cohort'
    assert claimed==[FIRST]


def test_first_cohort_give_up_is_retried_within_120_seconds(tmp_path,monkeypatch,capsys):
    monkeypatch.delenv('V8_GIVEUP_RETRY_SECONDS',raising=False)
    rows=flow_pairs(tmp_path)
    def broken():raise nodes.NodeFailure('fixture: runtime actor unavailable on this node')
    rows[0][1].verify_node=broken
    stop=threading.Event();timer=threading.Timer(60,stop.set);timer.start()
    def watch():
        while not stop.is_set():
            if (tmp_path/LATER/'.multi-node/state.json').exists():break
            time.sleep(.01)
        time.sleep(.5);stop.set()
    threading.Thread(target=watch,daemon=True).start()
    try:nodes.worker_loop_many(rows,'solo',stop,poll=.005,cohort_order=ORDER)
    finally:timer.cancel()
    out=capsys.readouterr().out
    assert f'retries {FIRST} in 120s' in out
    assert nodes.giveup_retry_delay(1)==600   # other cohorts keep the 600 s .. 3600 s backoff


def test_reserve_off_restores_plain_spill(tmp_path,monkeypatch):
    monkeypatch.setattr(nodes,'ACTIVE_RESERVE',0)
    ctx=mp.get_context('fork');stop=Stops()
    children=start(ctx,tmp_path,[f'node-{i}' for i in range(3)],stop)
    try:
        assert until(lambda:executes(tmp_path,FIRST,f'{FIRST}-P') and len(executes(tmp_path,LATER))>=2,40)
        release_all(tmp_path)
        assert not finish(children,stop)
    finally:
        release_all(tmp_path);finish(children,stop,timeout=5)


def test_reserve_nodes_share_an_inspection_while_a_first_cohort_task_is_blocked_for_good(tmp_path,monkeypatch):
    """A task blocked for good (failure cap) gives the same answer on every node, so the reserve nodes reuse one
    inspection instead of each inspecting the whole queue under its control lock on every poll (11-16 lock storm)."""
    from test_srgc_v8_idle_1116 import Actor,queue,node_rows,start as start_thread
    from test_srgc_v8_sequential_cohorts import events
    full_name,next_name='qwen_mbpp','olmo_math'
    monkeypatch.setattr(nodes,'SPILL_STALL_SECONDS',.3)
    full=queue(tmp_path,Actor(tmp_path,full_name,3));queue(tmp_path,Actor(tmp_path,next_name,2))
    paths=sorted(full.root.glob('jobs/*.json'));held=[]
    for path in paths[:2]:   # two tasks run on other nodes
        lease=nodes.Lock(full.root/'leases'/(path.stem+'.lock'));assert lease.acquire();held.append(lease)
    job=nodes.read(paths[2]);job['state']='failed'
    job['attempts']=[dict(id=f'f{i}',outcome='failed',exit_code=1,error='task exit=1; output verified=False',
                          host='elsewhere',gpus=['0'],worker='elsewhere.1',started_at=1.0+i,ended_at=2.0+i)
                     for i in range(nodes.MAX_ATTEMPTS+2)]
    nodes.atomic(paths[2],job)
    inspections={full_name:0,next_name:0};original=nodes.Pool.tick
    def counted(self,*args,**kwargs):
        inspections[self.binding['cohort']]+=1;return original(self,*args,**kwargs)
    monkeypatch.setattr(nodes.Pool,'tick',counted)
    stop=threading.Event()
    threads=[start_thread(node_rows(tmp_path,2),f'node{n}',stop,.05) for n in range(4)]
    try:
        assert until(lambda:len([r for r in events(tmp_path) if r['stage']=='execute'])==2,30)
        time.sleep(12)   # past the reserve warm-up
        before=dict(inspections);time.sleep(3);after=dict(inspections)
        assert sum(s.get('state')=='WAIT_RESERVE' for s in statuses(tmp_path))==2
    finally:
        stop.set()
        for t in threads:t.join(10)
        for lease in held:lease.close()
    assert after[full_name]-before[full_name]<=4,(before,after)
    assert len(nodes.read(paths[2])['attempts'])==nodes.MAX_ATTEMPTS+2   # the blocked task was not started
