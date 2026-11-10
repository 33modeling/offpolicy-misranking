"""V8-159: node verification runs without the queue control.lock, and a node that cannot serve the active
sequential cohort exits instead of idling. User 11-11: 'v8 코드 지금 lock 걸려서 아무것도 시작못해',
'모든 시작노드가 idle임'. CPU workers and real flock only; no GPU or model."""
import multiprocessing as mp
from pathlib import Path
import threading
import time

from scripts import srgc_v8_nodes as nodes


def slow_pairs(base, verify_seconds, limit):
    from test_srgc_v8_ready_claim_handoff import asynchronous_pairs
    rows = asynchronous_pairs(base)
    for pool, runtime in rows:
        original = runtime.verify_node
        def verify_node(original=original, runtime=runtime):
            original(); time.sleep(verify_seconds)
        runtime.verify_node = verify_node; runtime.verify_timeout = limit
    return rows


def worker(base, name, stop, verify_seconds, limit):
    code = nodes.worker_loop_many(slow_pairs(base, verify_seconds, limit), name, stop, poll=.005,
                                  cohort_order=nodes.DEFAULT_COHORTS)
    nodes.atomic(Path(base) / 'exits' / name, dict(code=code))


def test_starting_nodes_verify_in_parallel_not_one_by_one_behind_the_queue_lock(tmp_path):
    # Each verification takes 0.4 s against a 0.6 s limit: one at a time behind control.lock the third node
    # would wait 0.8 s and time out; without the lock all three verify together and every cohort finishes.
    ctx = mp.get_context('fork'); stop = ctx.Event(); children = []
    try:
        for i in range(3):
            child = ctx.Process(target=worker, args=(tmp_path, f'start-{i}', stop, .4, .6)); child.start(); children.append(child)
        for child in children:
            child.join(30)
            assert not child.is_alive()
        codes = {p.name: nodes.read(p)['code'] for p in (tmp_path / 'exits').glob('*')}
        assert codes == {f'start-{i}': 0 for i in range(3)}, codes
        for name in nodes.DEFAULT_COHORTS:
            assert nodes.read(tmp_path / name / '.multi-node' / 'state.json')['phase'] == 'complete'
    finally:
        stop.set()
        for child in children:
            if child.is_alive():child.terminate()


def test_queue_lock_is_free_while_a_node_verifies(tmp_path):
    rows = slow_pairs(tmp_path, .5, 5)
    stop = threading.Event(); seen = []
    def probe():
        time.sleep(.2)
        with rows[0][0].control() as control:seen.append(control is not None)
        stop.set()
    thread = threading.Thread(target=probe); thread.start()
    nodes.worker_loop_many(rows, 'verifying-node', stop, poll=.005, cohort_order=nodes.DEFAULT_COHORTS)
    thread.join()
    assert seen == [True], 'another node could not take control.lock while this node was verifying'


def test_node_that_cannot_serve_the_active_cohort_exits_instead_of_idling(tmp_path, capsys):
    rows = slow_pairs(tmp_path, 1.0, .1)       # verification longer than its limit: this node gives up olmo_math
    start = time.monotonic()
    code = nodes.worker_loop_many(rows, 'slow-node', threading.Event(), poll=.005, cohort_order=nodes.DEFAULT_COHORTS)
    assert code == 1 and time.monotonic() - start < 3
    out = capsys.readouterr().out
    assert 'timed out' in out and ('EXIT: this node cannot serve the active experiment olmo_math' in out
                                    or 'STOPPED: olmo_math is not complete' in out)   # exits either way, never idles
