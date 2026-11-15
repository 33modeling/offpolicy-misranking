"""11-15: a node whose cohort has nothing claimable now takes ready work from the next cohort (CPU fixture)."""
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_sequential_cohorts import Runtime, events, pairs


@pytest.fixture(autouse=True)
def _cap_without_user_grant(monkeypatch):
    monkeypatch.setenv('V8_USER_RETRY_GRANT', '0')


class Gated(Runtime):
    """Tasks exist but wait on something outside this node (parents/budget) until `open` is set."""
    def __init__(self, base, cohort, gate):
        super().__init__(base, cohort);self.gate=gate;self.ready_calls=0

    def ready(self, task):
        self.ready_calls+=1
        return self.gate.is_set()


ORDER=('olmo_mbpp','olmo_math')


def run(rows, stop, *, seconds=60, poll=.005):
    result={};timer=threading.Timer(seconds, stop.set);timer.start()   # hang guard
    try:result['rc']=nodes.worker_loop_many(rows, 'spill-node', stop, poll=poll, cohort_order=ORDER)
    finally:timer.cancel();timer.join()
    return result['rc']


def test_first_cohort_without_claimable_task_hands_the_node_to_the_next_cohort(tmp_path):
    gate=threading.Event();stop=threading.Event()
    first=Gated(tmp_path, 'olmo_mbpp', gate);rows=pairs(tmp_path, cohorts=('olmo_math',))
    rows.insert(0, (nodes.Pool(first.root/'.multi-node', first.binding), first))
    def watch():
        deadline=time.monotonic()+60
        while time.monotonic()<deadline and not stop.is_set():
            if rows[1][0].root.joinpath('state.json').exists() and rows[1][0].queue_state()['phase']=='complete':
                gate.set();return
            time.sleep(.01)
    threading.Thread(target=watch, daemon=True).start()
    assert run(rows, stop)==0 and not stop.is_set()
    claims=[(row['cohort'], row['task']) for row in events(tmp_path) if row['stage']=='execute']
    # The blocked cohort did not idle the node: the next cohort ran while it waited, then it resumed.
    assert [c for c, _ in claims[:4]]==['olmo_math']*4
    assert {c for c, _ in claims[4:]}=={'olmo_mbpp'} and len(claims)==len(set(claims))==8
    assert all(pool.queue_state()['phase']=='complete' for pool, _ in rows)


def test_first_cohort_lock_held_by_other_nodes_does_not_keep_the_node_idle(tmp_path, monkeypatch):
    """11-16 'mbpp는 꽉찼는데 그걸 기다리고 있음', '다른걸로 안넘어가고': other nodes keep the first cohort's
    queue control lock busy almost all the time; this node must still serve the next cohort."""
    import fcntl
    monkeypatch.setattr(nodes, 'SPILL_STALL_SECONDS', .3)
    gate=threading.Event();stop=threading.Event();done=threading.Event()
    first=Gated(tmp_path, 'olmo_mbpp', gate);rows=pairs(tmp_path, cohorts=('olmo_math',))
    rows.insert(0, (nodes.Pool(first.root/'.multi-node', first.binding), first))
    lock_path=rows[0][0].root/'control.lock'
    def other_nodes():   # hold the lock >99.5% of the time, like many nodes polling a full queue
        while not done.is_set():
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, 'a+') as fh:
                fcntl.flock(fh, fcntl.LOCK_EX);time.sleep(.2);fcntl.flock(fh, fcntl.LOCK_UN)
            time.sleep(.0005)
    holder=threading.Thread(target=other_nodes, daemon=True);holder.start()
    def watch():
        deadline=time.monotonic()+60
        while time.monotonic()<deadline and not stop.is_set():
            if rows[1][0].root.joinpath('state.json').exists() and rows[1][0].queue_state()['phase']=='complete':
                done.set();gate.set();return
            time.sleep(.01)
    threading.Thread(target=watch, daemon=True).start()
    try:assert run(rows, stop, seconds=20)==0 and not stop.is_set()
    finally:done.set();holder.join(5)
    claims=[c for c, _ in ((row['cohort'], row['task']) for row in events(tmp_path) if row['stage']=='execute')]
    assert claims[:4]==['olmo_math']*4 and claims.count('olmo_mbpp')==4

def test_nothing_claimable_anywhere_waits_with_bounded_polls_and_does_not_exit(tmp_path):
    gate=threading.Event();stop=threading.Event()
    rows=[]
    for name in ORDER:
        runtime=Gated(tmp_path, name, gate);rows.append((nodes.Pool(runtime.root/'.multi-node', runtime.binding), runtime))
    started=time.monotonic()
    assert run(rows, stop, seconds=1.0, poll=.1)==75   # still waiting when stopped: no exit while work is pending
    elapsed=time.monotonic()-started
    assert elapsed>=.9
    calls=sum(r.ready_calls for _, r in rows)
    # 2 cohorts x 2 tasks per scan at most every poll (0.1 s): about 40 in 1 s; a busy loop makes thousands.
    assert 0<calls<=80, calls
    assert not [row for row in events(tmp_path) if row['stage']=='execute']


def test_node_exits_when_every_listed_cohort_is_complete(tmp_path):
    rows=pairs(tmp_path, cohorts=ORDER)
    assert run(rows, threading.Event())==0
    assert all(pool.queue_state()['phase']=='complete' for pool, _ in rows)
    claims=[(row['cohort'], row['task']) for row in events(tmp_path) if row['stage']=='execute']
    assert len(claims)==len(set(claims))==8
    # Without waiting, the earlier cohort still comes first.
    assert [c for c, _ in claims[:4]]==['olmo_mbpp']*4



# Launch path: which cohorts one node serves, in what claim order.
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def _runner(tmp_path, names):
    folder=tmp_path/'runner/configs/cohorts';folder.mkdir(parents=True)
    for name in names:(folder/(name+'.json')).write_text('{}')
    return tmp_path/'runner'


def _queue(root, name):
    path=root/name/'.multi-node';path.mkdir(parents=True);(path/'binding.json').write_text('{}')


def test_named_cohort_first_then_other_cohorts_that_already_have_queues(tmp_path):
    import v8_queue_runtime as q
    runner=_runner(tmp_path, ['olmo_math','qwen_mbpp','olmo_mbpp','qwen_math','llama_mbpp','gemma_math'])
    root=tmp_path/'v8';_queue(root,'olmo_math');_queue(root,'llama_mbpp');_queue(root,'qwen_mbpp')
    # Default (no flag): the named cohort first, then only cohorts that already have a queue, MBPP first.
    # olmo_mbpp, qwen_math and gemma_math have no queue, so they are never added (no new queue is started).
    assert q.cohort_list('qwen_mbpp', None, root, runner)==(['qwen_mbpp','llama_mbpp','olmo_math'],{'llama_mbpp','olmo_math'})
    order,auto=q.cohort_list('all', None, root, runner)
    assert order==['qwen_mbpp','olmo_mbpp','olmo_math','qwen_math','llama_mbpp'] and auto=={'llama_mbpp'}
    # An exact list is used as given (no automatic additions).
    assert q.cohort_list('all', 'olmo_mbpp, qwen_mbpp', root, runner)==(['olmo_mbpp','qwen_mbpp'],set())
    assert q.cohort_list('qwen_math', 'olmo_mbpp,qwen_math', root, runner)==(['qwen_math','olmo_mbpp'],set())
    with pytest.raises(ValueError):q.cohort_list('all', 'a;rm -rf', root, runner)


def test_dispatch_uses_the_plan_order_and_does_not_pass_cohorts_flag_to_the_worker(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('spill_dispatch', ROOT/'scripts/v8_dispatch.py')
    dispatch=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatch)
    path=tmp_path/'plan.json'
    path.write_text(json.dumps(dict(root=str(tmp_path/'queue'), argv=['--cohorts','qwen_mbpp,olmo_mbpp','--gpus','0,1,2,3','--cohorts=x'],
        errors={}, order=['qwen_mbpp','olmo_mbpp'],
        entries=[dict(cohort=c, script=str(ROOT/'scripts/srgc_v8_nodes.py'), runner=str(ROOT/'v8/experiments'), python=sys.executable)
                 for c in ('olmo_mbpp','qwen_mbpp')])))
    seen=[]
    def main(args, *, selected_cohorts, cohort_order):
        seen.append((args, cohort_order));return 0
    monkeypatch.setattr(dispatch.nodes, 'main', main)
    monkeypatch.setattr(sys, 'argv', ['v8_dispatch.py', str(path)])
    assert dispatch.main()==0
    assert seen==[(['--gpus','0,1,2,3'], ('qwen_mbpp','olmo_mbpp'))]


def test_launcher_default_serves_the_remaining_cohorts_after_the_named_one(tmp_path):
    import os
    import v8_queue_runtime as q
    env=dict(os.environ, V8_ROOT=str(tmp_path/'v8'), V8_RUNNER_ROOT=str(ROOT/'v8/experiments'), V8_PYTHON=sys.executable)
    env.pop('V8_COHORTS', None)
    plan=q.selections(ROOT, env, ['qwen_mbpp'], serve_remaining=True)
    assert plan.order==['qwen_mbpp'] and not plan.errors          # no other queue exists: nothing else is started
    _queue(tmp_path/'v8','olmo_math');_queue(tmp_path/'v8','olmo_mbpp')
    plan=q.selections(ROOT, env, ['qwen_mbpp'], serve_remaining=True)
    # existing queues only (these fake ones cannot be opened, so they are skipped, not a launch error)
    assert plan.order==['qwen_mbpp'] and list(plan.skipped)==['olmo_mbpp','olmo_math'] and not plan.errors
    assert q.selections(ROOT, env, ['qwen_mbpp','--cohorts','qwen_mbpp'], serve_remaining=True).order==['qwen_mbpp']
    assert q.selections(ROOT, env, ['qwen_mbpp']).order==['qwen_mbpp']   # direct callers keep one cohort
