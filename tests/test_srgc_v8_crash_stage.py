"""A native crash names the rank's last stage from the real progress record (which has no pid)."""
import importlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def setup(tmp_path,monkeypatch,event,started):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    dump=tmp_path/'native.log';dump.write_text('Fatal Python error: Segmentation fault\n\n'
        'Current thread 0x01 (most recent call first):\n  File "/x/torch/nn/modules/module.py", line 1, in _call_impl\n')
    meta=dict(rank=1,pid=123,invocation='current',log=dump.name)
    if started is not None:meta['started_at']=started
    (tmp_path/'native-fault-rank-1.json').write_text(json.dumps(meta))
    (tmp_path/'rank-1.json').write_text(json.dumps(event))
    log=tmp_path/'task.log';log.write_text('rank 1 exitcode : -11\n')
    env=dict(V8_FAILURE_TOKEN='current',V8_EXIT_FILE=str(tmp_path/'exit.json'))
    assert worker.child_result(1,tmp_path,env,log,'task')==1
    [fault]=json.loads((tmp_path/'child-failure.json').read_text())['failures']
    return fault,log


def test_progress_record_without_pid_names_the_stage(tmp_path,monkeypatch):
    # Exactly what srgc_rebuttal.progress.record writes: stage, updated, details; no pid.
    now=time.time()
    fault,log=setup(tmp_path,monkeypatch,dict(stage='rollout',updated=now),now-10)
    assert fault['last_progress']['stage']=='rollout'
    assert 'last stage=rollout' in log.read_text() and 'unrecorded' not in log.read_text()


def test_record_older_than_this_invocation_is_not_used(tmp_path,monkeypatch):
    now=time.time()
    fault,_=setup(tmp_path,monkeypatch,dict(stage='old',updated=now-100),now)
    assert fault['last_progress'] is None


def test_legacy_meta_without_start_time_still_uses_attempt_record(tmp_path,monkeypatch):
    fault,_=setup(tmp_path,monkeypatch,dict(stage='policy_update',updated=1.0),None)
    assert fault['last_progress']['stage']=='policy_update'


def test_fault_handler_meta_and_probe_record_carry_identity(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    entry=importlib.import_module('v8_task_entry')
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path));monkeypatch.setenv('RANK','0')
    monkeypatch.setattr(entry,'_NATIVE_FAULT_STATE',None)
    import faulthandler
    try:entry.install_native_fault_handler()
    finally:faulthandler.disable()
    meta=json.loads((tmp_path/'native-fault-rank-0.json').read_text())
    assert meta['started_at']>0 and meta['executable'] and 'host' in meta
    source=(ROOT/'scripts/v8_generation_probe.py').read_text()
    assert "details.setdefault('pid',os.getpid())" in source
