"""Real original SRGC child lifecycle on CPU subprocesses; no model or GPU access."""
import importlib
import os
from pathlib import Path
import sys
import subprocess
import threading
import time
import types

import pytest

REPO=Path(__file__).resolve().parents[1]


def test_v8_reuses_real_srgc_no_progress_timeout_and_preserves_checkpoints(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    worker=importlib.import_module('v8_runtime_worker')
    beats=[];checkpoint=tmp_path/'step-000025.pt';checkpoint.write_bytes(b'original optimizer RNG rewards')
    runtime=types.SimpleNamespace(cohort='olmo_math',gpus=[],heartbeat=beats.append,
            environment=dict(os.environ,V8_TASK_STALL_SECONDS='.1',SRGC_PROGRESS_DIR=str(tmp_path/'progress')))
    started=time.monotonic()
    result=worker.run_srgc_child(runtime,[sys.executable,'-c','import time;print("reading model",flush=True);time.sleep(30)'],
                                 tmp_path/'task.log',[],threading.Event(),interval=.02)
    assert result==124 and time.monotonic()-started<3
    assert 'no task progress' in (tmp_path/'task.log').read_text()
    assert checkpoint.read_bytes()==b'original optimizer RNG rewards'
    assert beats and all(beat.startswith('child:') for beat in beats)


def test_srgc_real_work_records_keep_child_alive_and_held_task_fd_survives(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    worker=importlib.import_module('v8_runtime_worker')
    from scripts import srgc_v8_nodes as nodes
    runtime=types.SimpleNamespace(cohort='olmo_math',gpus=[],heartbeat=lambda _:None,
        environment=dict(os.environ,V8_TASK_STALL_SECONDS='.15',SRGC_PROGRESS_DIR=str(tmp_path/'progress')))
    held=nodes.Lock(tmp_path/'task.lock');assert held.acquire()
    code='''import os,sys,time
from pathlib import Path
fd=int(sys.argv[1]);assert os.fstat(fd).st_ino==Path(sys.argv[2]).stat().st_ino
p=Path(os.environ['SRGC_PROGRESS_DIR'])/'rank-0.json'
for i in range(6):p.write_text(str(i));time.sleep(.06)
'''
    try:
        assert worker.run_srgc_child(runtime,[sys.executable,'-c',code,str(held.fd),str(tmp_path/'task.lock')],
                   tmp_path/'task.log',[held.fd],threading.Event(),interval=.02)==0
    finally:held.close()


def test_frozen_actor_outside_checkout_can_import_original_srgc_without_pythonpath(tmp_path):
    controls=tmp_path/'private-cache/control';controls.mkdir(parents=True)
    for name in ('v8_runtime_worker.py','v8_rpc.py'):
        (controls/name).write_bytes((REPO/'scripts'/name).read_bytes())
    script=controls/'probe.py'
    script.write_text('''import os,sys,threading,types
from pathlib import Path
from v8_runtime_worker import run_srgc_child
r=types.SimpleNamespace(cohort='olmo_math',gpus=[],heartbeat=lambda _:None,environment=dict(os.environ))
assert run_srgc_child(r,[sys.executable,'-c','from srgc_rebuttal.progress import signature;print("original SRGC child reached")'],
 Path(sys.argv[1]),[],threading.Event(),interval=.01)==0
''')
    env=dict(os.environ,V8_CODE_ROOT=str(REPO),SRGC_PROGRESS_DIR=str(tmp_path/'progress'))
    env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(script),str(tmp_path/'task.log')],cwd=tmp_path,
                          env=env,capture_output=True,text=True,timeout=4)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'original SRGC child reached' in (tmp_path/'task.log').read_text()
