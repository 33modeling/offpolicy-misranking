"""Native signal evidence and exact MBPP launch controls; CPU fixtures only."""
import importlib
import json
from pathlib import Path

import pytest

from test_srgc_v8_child_errors import child_runtime

ROOT=Path(__file__).resolve().parents[1]


def test_actual_torchrun_sigsegv_keeps_rank_stack_and_stage(child_runtime):
    run,directory,log=child_runtime
    assert run('segfault') not in (0,75)
    report=json.loads((directory/'child-failure.json').read_text())
    [fault]=[f for f in report['failures'] if f.get('rank')==1 and f.get('signal')==11]
    stack=Path(fault['native_fault_log']).read_text()
    assert 'Segmentation fault' in stack and 'string_at' in stack
    assert fault['native_invocation'] and 'SIGSEGV' in log.read_text()
    assert fault['last_progress']['stage']=='gradient_scoring'
    assert 'NATIVE STACK (original fault-handler dump' in log.read_text() and 'string_at' in log.read_text()


def test_signal_and_original_python_fault_are_both_preserved(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    (tmp_path/'rank-0-error.json').write_text(json.dumps(dict(rank=0,invocation='current',error={'message':'ValueError: original input fault'})))
    log=tmp_path/'task.log';log.write_text('rank 1 exitcode : -11\n')
    env=dict(V8_FAILURE_TOKEN='current',V8_EXIT_FILE=str(tmp_path/'exit.json'))
    assert worker.child_result(1,tmp_path,env,log,'task')==1
    failures=json.loads((tmp_path/'child-failure.json').read_text())['failures']
    assert any(f.get('signal')==11 for f in failures)
    assert any('original input fault' in f['message'] for f in failures)


def test_direct_launcher_sigsegv_is_not_lost_without_rank_summary(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    log=tmp_path/'task.log';log.write_text('Fatal Python error: Segmentation fault\n')
    env=dict(V8_FAILURE_TOKEN='current',V8_EXIT_FILE=str(tmp_path/'exit.json'))
    assert worker.child_result(-11,tmp_path,env,log,'task')==-11
    failures=json.loads((tmp_path/'child-failure.json').read_text())['failures']
    assert any(f.get('signal')==11 and f['rank'] is None for f in failures)


@pytest.mark.parametrize('reason',['old-invocation','outside-progress'])
def test_stale_or_outside_native_dump_is_not_attributed_to_current_rank(tmp_path,monkeypatch,reason):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    dump=tmp_path/'native.log' if reason=='old-invocation' else tmp_path.parent/(tmp_path.name+'-outside.log')
    dump.write_text('Fatal Python error: Segmentation fault\n')
    meta=dict(rank=1,pid=123,invocation='old' if reason=='old-invocation' else 'current',log=str(dump))
    (tmp_path/'native-fault-rank-1.json').write_text(json.dumps(meta))
    log=tmp_path/'task.log';log.write_text('rank 1 exitcode : -11\n')
    env=dict(V8_FAILURE_TOKEN='current',V8_EXIT_FILE=str(tmp_path/'exit.json'))
    assert worker.child_result(1,tmp_path,env,log,'task')==1
    failures=json.loads((tmp_path/'child-failure.json').read_text())['failures']
    assert len(failures)==1 and failures[0]['rank'] is None
    assert failures[0]['signal']==11 and 'native_fault_log' not in failures[0]


def test_long_native_dump_keeps_fatal_header_without_reusing_old_pid_stage(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    dump=tmp_path/'native.log';dump.write_text('Fatal Python error: Segmentation fault\n'+'x'*200000)
    meta=dict(rank=1,pid=123,invocation='current',log=dump.name)
    (tmp_path/'native-fault-rank-1.json').write_text(json.dumps(meta))
    (tmp_path/'rank-1.json').write_text(json.dumps(dict(pid=122,stage='stale-stage')))
    log=tmp_path/'task.log';log.write_text('rank 1 exitcode : -11\n')
    env=dict(V8_FAILURE_TOKEN='current',V8_EXIT_FILE=str(tmp_path/'exit.json'))
    assert worker.child_result(1,tmp_path,env,log,'task')==1
    [fault]=json.loads((tmp_path/'child-failure.json').read_text())['failures']
    assert fault['rank']==1 and fault['signal']==11 and fault['last_progress'] is None
    assert fault['native_python_stack'].startswith('Fatal Python error: Segmentation fault')


@pytest.mark.parametrize('malformed',['metadata','stage'])
def test_valid_json_with_wrong_shape_does_not_replace_signal_exit(tmp_path,monkeypatch,malformed):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    dump=tmp_path/'native.log';dump.write_text('Fatal Python error: Segmentation fault\n')
    meta=[] if malformed=='metadata' else dict(rank=1,pid=123,invocation='current',log=dump.name)
    (tmp_path/'native-fault-rank-1.json').write_text(json.dumps(meta))
    (tmp_path/'rank-1.json').write_text('[]')
    log=tmp_path/'task.log';log.write_text('rank 1 exitcode : -11\n')
    env=dict(V8_FAILURE_TOKEN='current',V8_EXIT_FILE=str(tmp_path/'exit.json'))
    assert worker.child_result(1,tmp_path,env,log,'task')==1
    [fault]=json.loads((tmp_path/'child-failure.json').read_text())['failures']
    assert fault['signal']==11
    if malformed=='stage':assert fault['rank']==1 and fault['last_progress'] is None
