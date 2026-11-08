"""Actual CPU torchrun + SRGC lifecycle; no GPU/model/training measurements."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import types

import pytest

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def child_runtime(tmp_path,monkeypatch):
    if importlib.util.find_spec('torch') is None:pytest.skip('existing CPU torch environment required')
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    worker=importlib.import_module('v8_runtime_worker')
    script=tmp_path/'cluster_child.py'
    script.write_text('''import os,sys,time
from pathlib import Path
sys.path.insert(0,REPO_RUNNER)
def main():
 from switching.runner import write_exit_marker
 mode=os.environ['FAULT_MODE'];rank=int(os.environ['RANK'])
 if mode=='busy' or (mode=='mixed' and rank==0):
  if rank==0:write_exit_marker(75,'busy',task='fixture-task')
  if mode=='mixed':
   while not (Path(os.environ['SRGC_PROGRESS_DIR'])/'rank-1-error.json').exists():time.sleep(.01)
  raise SystemExit(75)
 if rank==1:
  if mode=='mixed':
   while not Path(os.environ['V8_EXIT_FILE']).exists():time.sleep(.01)
  if mode=='systemexit':raise SystemExit('PINNED INPUT CHECK FAILED ON RANK 1')
  raise ValueError('ORIGINAL RANK 1 FAILURE; DO NOT REPLACE WITH ChildFailedError')
 time.sleep(15)
'''.replace('REPO_RUNNER',repr(str(REPO/'v8/experiments'))))
    import hashlib
    runtime=types.SimpleNamespace(cohort='olmo_math',gpus=[],heartbeat=lambda _:None,
        environment=dict(os.environ,V8_CODE_ROOT=str(REPO),V8_TASK_STALL_SECONDS='60',
          SRGC_PROGRESS_DIR=str(tmp_path/'progress'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
          V8_ORIGINAL_CHILD_SHA256=hashlib.sha256(script.read_bytes()).hexdigest()))
    def run(mode):
        runtime.environment['FAULT_MODE']=mode
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--max_restarts=0',
                 str(REPO/'scripts/v8_task_entry.py'),str(script)]
        return worker.run_srgc_child(runtime,command,tmp_path/'task.log',[],threading.Event(),
                                     label='fixture-task',interval=.05)
    return run,tmp_path/'progress',tmp_path/'task.log'


def test_real_torchrun_busy_exit75_is_not_a_failed_training_task(child_runtime):
    run,progress,log=child_runtime
    assert run('busy')==75,'torchrun exit1 hid the original resumable busy exit75'


@pytest.mark.parametrize('mode',['fault','mixed','systemexit'])
def test_actual_rank_error_is_retained_and_busy_marker_cannot_hide_it(child_runtime,mode):
    run,progress,log=child_runtime
    assert run(mode)!=0
    failure=json.loads((progress/'child-failure.json').read_text())
    assert failure['failures'] and failure['failures'][0]['rank']==1
    expected='PINNED INPUT CHECK FAILED' if mode=='systemexit' else 'ORIGINAL RANK 1 FAILURE'
    assert expected in failure['failures'][0]['message']
    assert expected in log.read_text()
    assert failure['exit_code']!=75,'a real rank fault was hidden by rank0 busy marker'
