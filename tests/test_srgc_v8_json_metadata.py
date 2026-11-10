"""Corrupt fixed JSON must not wait for unrelated training; JSONL waits are bounded."""
import json
import threading

import pytest

from scripts import srgc_v8_nodes as nodes


def fixture(tmp_path):
    pool=nodes.Pool(tmp_path/'olmo_math/.multi-node',dict(cohort='olmo_math'));pool.initialize()
    state=pool.queue_state();state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    key=pool.register(dict(kind='train',run_id='unrelated',parent={}), 'train')
    held=nodes.Lock(pool.root/'leases'/(key+'.lock'));assert held.acquire()
    return pool,held


def test_corrupt_cached_json_fails_with_path_even_while_another_task_holds_a_lease(tmp_path):
    pool,held=fixture(tmp_path)
    path=pool.root/'plans/train.json';path.parent.mkdir();path.write_text('{broken')
    original=path.read_bytes()
    try:
        with pytest.raises(ValueError,match='Invalid fixed JSON') as caught:
            pool.tick(object(),'node',threading.Event(),claim=False)
        assert str(path) in str(caught.value) and 'line 1' in str(caught.value)
        assert path.read_bytes()==original
    finally:held.close()


def test_jsonl_same_bad_line_cannot_retry_forever_as_other_lines_are_appended(tmp_path,monkeypatch):
    pool,held=fixture(tmp_path);clock={'now':10.}
    monkeypatch.setattr(nodes.time,'monotonic',lambda:clock['now'])
    path=tmp_path/'snapshots.jsonl';path.write_text('{broken\n')
    def error():
        value=json.JSONDecodeError('bad row','bad',0);value.filename=str(path);return value
    try:
        assert pool.metadata_wait(error())==('waiting',None)
        path.write_text('{broken\n'+json.dumps({'new':'valid row'})+'\n')
        clock['now']=41.
        with pytest.raises(ValueError,match='same JSONL line') as caught:pool.metadata_wait(error())
        assert str(path) in str(caught.value)
    finally:held.close()


def test_different_partial_jsonl_line_gets_its_own_bounded_wait(tmp_path,monkeypatch):
    pool,held=fixture(tmp_path);clock={'now':10.}
    monkeypatch.setattr(nodes.time,'monotonic',lambda:clock['now'])
    path=tmp_path/'updates.jsonl';path.write_text('{unfinished')
    first=json.JSONDecodeError('unfinished','x',0);first.filename=str(path)
    try:
        assert pool.metadata_wait(first)==('waiting',None)
        clock['now']=41.
        second=json.JSONDecodeError('unfinished','x',0);second.filename=str(path);second.lineno=2
        assert pool.metadata_wait(second)==('waiting',None)
    finally:held.close()


def test_side_metadata_error_is_not_reset_by_an_unrelated_successful_base_plan(tmp_path,monkeypatch):
    pool,held=fixture(tmp_path);clock={'now':10.}
    monkeypatch.setattr(nodes.time,'monotonic',lambda:clock['now'])
    path=tmp_path/'side.jsonl';path.write_text('{unfinished')
    task=nodes.read(next((pool.root/'jobs').glob('*.json')))['task']
    class Runtime:
        streaming_ready_tasks=True
        def plan(self,phase):return [task],True
        def side_tasks(self):
            error=json.JSONDecodeError('unfinished','x',0);error.filename=str(path);raise error
    try:
        assert pool.tick(Runtime(),'node',threading.Event(),claim=False)==('waiting',None)
        clock['now']=41.
        with pytest.raises(ValueError,match='same JSONL line'):pool.tick(Runtime(),'node',threading.Event(),claim=False)
    finally:held.close()


def test_console_does_not_dump_full_queue_json_on_an_idle_worker(tmp_path,capsys):
    from test_srgc_v8_sequential_cohorts import pairs
    rows=pairs(tmp_path);rows[0][1].ready=lambda task:False
    stop=threading.Event();timer=threading.Timer(.08,stop.set);timer.start()
    try:nodes.worker_loop_many(rows,'node',stop,poll=.005,cohort_order=nodes.DEFAULT_COHORTS)
    finally:timer.cancel();timer.join()
    output=capsys.readouterr().out
    assert 'WAIT_CHECKPOINT' in output
    assert 'known_task_counts' not in output and '"dispatch_workers"' not in output
