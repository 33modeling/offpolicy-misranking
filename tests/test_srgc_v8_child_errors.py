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
def install_diagnostics(*args):
 import faulthandler
 handle=(Path(os.environ['SRGC_PROGRESS_DIR'])/'legacy-diagnostics.log').open('a')
 faulthandler.enable(file=handle,all_threads=True)
 return handle
def main():
 from switching.runner import write_exit_marker
 mode=os.environ['FAULT_MODE'];rank=int(os.environ['RANK'])
 if mode=='success':return
 if mode=='busy' or (mode in ('mixed','mixed_signal') and rank==0):
  if rank==0:write_exit_marker(75,'busy',task='fixture-task')
  if mode in ('mixed','mixed_signal'):
   while not (Path(os.environ['SRGC_PROGRESS_DIR'])/'rank-1-error.json').exists():time.sleep(.01)
  raise SystemExit(75)
 if rank==1:
  if mode in ('segfault','overwritten-segfault'):
   import ctypes,resource,json
   if mode=='overwritten-segfault':legacy_handle=install_diagnostics()
   resource.setrlimit(resource.RLIMIT_CORE,(0,0))
   (Path(os.environ['SRGC_PROGRESS_DIR'])/'rank-1.json').write_text(json.dumps(dict(pid=os.getpid(),stage='gradient_scoring',scope='CPU synthetic crash fixture')))
   ctypes.string_at(0)
  if mode in ('mixed','mixed_signal'):
   while not Path(os.environ['V8_EXIT_FILE']).exists():time.sleep(.01)
  if mode=='mixed_signal':
   import signal
   os.kill(os.getpid(),signal.SIGKILL)
  if mode=='systemexit':raise SystemExit('PINNED INPUT CHECK FAILED ON RANK 1')
  raise ValueError('ORIGINAL RANK 1 FAILURE; DO NOT REPLACE WITH ChildFailedError')
 time.sleep(15)
'''.replace('REPO_RUNNER',repr(str(REPO/'v8/experiments'))))
    import hashlib
    runtime=types.SimpleNamespace(cohort='olmo_math',gpus=[],heartbeat=lambda _:None,
        environment=dict(os.environ,V8_CODE_ROOT=str(REPO),V8_TASK_STALL_SECONDS='60',
          SRGC_PROGRESS_DIR=str(tmp_path/'progress'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
          V8_ORIGINAL_CHILD_SHA256=hashlib.sha256(script.read_bytes()).hexdigest()))
    def run(mode,directory=None):
        runtime.environment['FAULT_MODE']=mode
        if directory is not None:runtime.environment['SRGC_PROGRESS_DIR']=str(directory)
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


def test_busy_marker_does_not_hide_a_real_signal_killed_rank(child_runtime):
    run,progress,log=child_runtime
    assert run('mixed_signal') not in (0,75)
    failure=json.loads((progress/'child-failure.json').read_text())
    assert any(f.get('signal')==9 and 'SIGKILL' in f['message'] for f in failure['failures'])


def test_rank_failure_is_kept_in_the_task_attempt_and_status(tmp_path,monkeypatch):
    from test_srgc_v8_task_failure_isolation import setup,claim
    from scripts import srgc_v8_nodes as nodes
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    status=importlib.import_module('v8_status')
    pool,runtime,stop=setup(tmp_path)
    lease,key,task,attempt=claim(pool,runtime,stop)
    report=dict(attempt_id=attempt['id'],exit_code=1,
                failures=[dict(rank=2,message='ValueError: original failing model/input check')])
    nodes.atomic(pool.root/'progress'/attempt['id']/'child-failure.json',report)
    try:pool.finish(key,1,runtime,stop)
    finally:lease.close()
    job=nodes.read(pool.root/'jobs'/(key+'.json'))
    assert job['state']=='failed' and job['attempts'][-1]['child_failure']==report
    state,rows=status.collect(tmp_path,'olmo_math',0)
    assert any(row['error'] and 'rank 2: ValueError: original failing' in row['error'] for row in rows)


@pytest.mark.parametrize('marker',[
    dict(exit_code=75,status='busy',task='different-task'),
    dict(exit_code=75.0,status='busy',task='fixture-task'),
    dict(exit_code=75,status='failed',task='fixture-task'),
    dict(exit_code=1,status='busy',task='fixture-task')])
def test_invalid_or_foreign_exit_marker_does_not_turn_failure_into_interruption(tmp_path,monkeypatch,marker):
    monkeypatch.syspath_prepend(str(REPO/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    path=tmp_path/'exit.json';path.write_text(json.dumps(marker));log=tmp_path/'task.log';log.write_text('failure\n')
    env=dict(V8_EXIT_FILE=str(path),V8_FAILURE_TOKEN='current')
    assert worker.child_result(1,tmp_path,env,log,'fixture-task')==1


def test_previous_invocation_rank_record_cannot_override_the_current_exit(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    (tmp_path/'rank-1-error.json').write_text(json.dumps(dict(invocation='old',rank=1,error={'message':'old failure'})))
    path=tmp_path/'exit.json';path.write_text(json.dumps(dict(exit_code=75,status='busy',task='fixture-task')))
    log=tmp_path/'task.log';log.write_text('current busy\n')
    assert worker.child_result(1,tmp_path,dict(V8_EXIT_FILE=str(path),V8_FAILURE_TOKEN='current'),log,'fixture-task')==75


def test_recorded_fault_cannot_be_reported_as_exit75(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'));worker=importlib.import_module('v8_runtime_worker')
    (tmp_path/'rank-1-error.json').write_text(json.dumps(dict(invocation='current',rank=1,
        error={'message':{'message':'ValueError: actual failure'}})))
    log=tmp_path/'task.log';log.write_text('actual failure\n')
    assert worker.child_result(75,tmp_path,dict(V8_EXIT_FILE=str(tmp_path/'exit.json'),V8_FAILURE_TOKEN='current'),
                               log,'fixture-task')==1


def test_existing_logs_show_original_rank_exception_instead_of_elastic_wrapper(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'));status=importlib.import_module('v8_status')
    lines=['[rank1]: ValueError: pinned input identity mismatch','Traceback (most recent call last):',
           '    raise ChildFailedError(', 'torch.distributed.elastic.multiprocessing.errors.ChildFailedError:',
           'error_file: <N/A>', 'traceback: To enable traceback see the error docs']
    assert status.last_error_line(lines)=='[rank1]: ValueError: pinned input identity mismatch'
