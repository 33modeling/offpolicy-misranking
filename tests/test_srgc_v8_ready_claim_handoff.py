"""Exercise the inspection-to-claim race with real flock contenders, no GPUs."""
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes

if os.environ.get('V8_HANDOFF_SOURCE'):
    spec=importlib.util.spec_from_file_location('previous_v8_handoff',os.environ['V8_HANDOFF_SOURCE'])
    nodes=importlib.util.module_from_spec(spec);spec.loader.exec_module(nodes)


def asynchronous_pairs(base):
    from test_srgc_v8_sequential_cohorts import pairs
    rows=pairs(base)
    for _,runtime in rows:
        runtime.independent_startup=True;runtime.pending_barrier=None
        def begin(action,fd,stop,device_fds,runtime=runtime):
            runtime.barrier(action,fd,stop,device_fds)
            runtime.pending_barrier=(action,os.dup(fd))
        def poll(stop,runtime=runtime):
            if runtime.pending_barrier is None:return None
            action,held=runtime.pending_barrier;runtime.pending_barrier=None
            return action,held,None
        runtime.begin_barrier=begin;runtime.poll_barrier=poll
    return rows


def concurrent_worker(base,name,stop,release):
    rows=asynchronous_pairs(base)
    rows[0][1].release=release
    code=nodes.worker_loop_many(rows,name,stop,poll=.002,cohort_order=nodes.DEFAULT_COHORTS)
    nodes.atomic(Path(base)/'exits'/name,dict(code=code))


def test_six_new_async_workers_fill_active_jobs_and_finish_all_four_in_order(tmp_path):
    from test_srgc_v8_sequential_cohorts import events
    ctx=mp.get_context('fork');stop=ctx.Event();release=ctx.Event();children=[]
    try:
        for i in range(6):
            child=ctx.Process(target=concurrent_worker,args=(tmp_path,f'new-worker-{i}',stop,release))
            child.start();children.append(child)
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            claims=[row for row in events(tmp_path) if row['stage']=='execute']
            if len(claims)>=2:break
            time.sleep(.005)
        assert len(claims)==2 and {row['cohort'] for row in claims}=={'olmo_math'}
        release.set()
        for child in children:
            child.join(8)
            assert not child.is_alive() and child.exitcode==0
        assert len(list((tmp_path/'exits').glob('*')))==6
        assert all(nodes.read(path)['code']==0 for path in (tmp_path/'exits').glob('*'))
        recorded=events(tmp_path)
        actual=[(row['cohort'],row['task']) for row in recorded if row['stage']=='execute']
        assert len(actual)==len(set(actual))==16
        for first,second in zip(nodes.DEFAULT_COHORTS,nodes.DEFAULT_COHORTS[1:]):
            end=max(i for i,row in enumerate(recorded) if row['cohort']==first and row['stage']!='verify')
            start=min(i for i,row in enumerate(recorded) if row['cohort']==second)
            assert end<start
    finally:
        stop.set();release.set()
        for child in children:
            child.join(1)
            if child.is_alive():child.kill();child.join()


@pytest.mark.parametrize('stop_before_claim',[False,True])
def test_another_inspector_cannot_overtake_a_ready_claim_and_stop_releases_lease(tmp_path,monkeypatch,stop_before_claim):
    rows=asynchronous_pairs(tmp_path);stop=threading.Event();original=nodes.TaskScan
    overtaken=[];observed=[]
    def inspect(*args,**kwargs):
        scan=original(*args,**kwargs);event=scan.finished;pool=args[0]
        class Finished:
            def set(self):event.set()
            def is_set(self):
                ready=event.is_set()
                if ready and scan.result and scan.result[0]=='ready':
                    contender=nodes.Lock(pool.root/'control.lock')
                    try:
                        overtaken.append(contender.acquire())
                        observed.append(pool.binding['cohort'])
                    finally:contender.close()
                    if stop_before_claim:stop.set()
                return ready
        scan.finished=Finished()
        return scan
    monkeypatch.setattr(nodes,'TaskScan',inspect)
    watchdog=threading.Timer(4,stop.set);watchdog.start()
    try:
        code=nodes.worker_loop_many(rows,'handoff-node',stop,poll=.001,cohort_order=nodes.DEFAULT_COHORTS)
    finally:watchdog.cancel();watchdog.join()
    assert observed and not any(overtaken),'another inspector acquired control.lock before the ready claim'
    assert code==(75 if stop_before_claim else 0)
    for pool,_ in rows:
        contender=nodes.Lock(pool.root/'control.lock')
        try:assert contender.acquire(),'finished/stopped worker leaked its inspection lock'
        finally:contender.close()
    if stop_before_claim:
        assert all(not nodes.read(path)['attempts'] for path in rows[0][0].root.glob('jobs/*.json'))
        assert not (rows[1][0].root/'state.json').exists()
    else:
        assert all(pool.queue_state()['phase']=='complete' for pool,_ in rows)
