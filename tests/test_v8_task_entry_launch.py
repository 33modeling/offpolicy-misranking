"""CPU: the real SRGC child launch (run_srgc_child -> v8_task_entry -> child); no GPU, model or measurements.

1. The task entry exports V8_STACK_DUMP_SECONDS=0 to the rank, so the frozen cluster_child.install_diagnostics
   arms no periodic faulthandler dump even when it is called directly (outside the entry's wrapper) and the node
   environment carries a positive value. V8_PERIODIC_STACK_DUMP=1 keeps the old dump.
2. A slow MBPP checker bootstrap start inside a launched rank: the reward is computed and the rank exits 0.
   A start past the startup window is still a rank failure (infrastructure error, never a 0 reward).
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import types

import pytest

REPO=Path(__file__).resolve().parents[1]

CHILD='''import json,os,sys,time,types
from pathlib import Path
sys.path.insert(0,REPO_RUNNER);sys.path.insert(0,REPO_ROOT)
def install_diagnostics(*args):return None
def main():
 out=Path(os.environ['SRGC_PROGRESS_DIR'])
 seen=dict(stack_dump_seconds=os.environ.get('V8_STACK_DUMP_SECONDS'))
 if os.environ['FIXTURE_MODE']=='dump':
  # The frozen child's own function, not the entry's wrapped module attribute.
  import cluster_child
  handle=cluster_child.install_diagnostics(os.environ['FIXTURE_ROOT'],'fixture-task',0)
  time.sleep(2.5);handle.flush()
 else:
  entry=sys.modules['__main__']
  import importlib.util
  spec=importlib.util.spec_from_file_location('srgc_rebuttal.launch_fixture',Path(REPO_ROOT)/'srgc_rebuttal/verifiers.py')
  verifiers=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifiers)
  verifiers.CODE_TIMEOUT_SECONDS=1.0
  entry._MBPP_EXEC_BOOTSTRAP='import time;time.sleep(1.5)\\n'+entry._MBPP_EXEC_BOOTSTRAP
  ctx=types.SimpleNamespace(backend=types.SimpleNamespace(verifier=verifiers.code_reward),
                            verifier_identity=dict(code_verifier_version=verifiers.CODE_REWARD_VERSION))
  entry.repair_context(ctx)
  record=dict(answer='assert add(2,3)==5')
  seen['rewards']=[ctx.backend.verifier(record,code) for code in
      ('def add(a,b): return a+b','def add(a,b): return a-b','while True: pass',
       'import time\\ntime.sleep(0.2)\\ndef add(a,b): return a+b')]
 (out/'fixture-result.json').write_text(json.dumps(seen))
'''


@pytest.fixture
def launch(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    worker=importlib.import_module('v8_runtime_worker')
    script=tmp_path/'cluster_child.py'
    script.write_text(CHILD.replace('REPO_RUNNER',repr(str(REPO/'v8/experiments'))).replace('REPO_ROOT',repr(str(REPO))))
    def run(mode,**environment):
        progress=tmp_path/f'progress-{mode}-{len(list(tmp_path.iterdir()))}'
        base={k:v for k,v in os.environ.items() if k not in ('V8_PERIODIC_STACK_DUMP','V8_MBPP_STARTUP_SECONDS')}
        runtime=types.SimpleNamespace(cohort='qwen_mbpp',gpus=[],heartbeat=lambda _:None,
            environment=dict(base,V8_CODE_ROOT=str(REPO),V8_TASK_STALL_SECONDS='60',RANK='0',
                             SRGC_PROGRESS_DIR=str(progress),FIXTURE_ROOT=str(tmp_path),FIXTURE_MODE=mode,
                             V8_STACK_DUMP_SECONDS='1',
                             V8_ORIGINAL_CHILD_SHA256=hashlib.sha256(script.read_bytes()).hexdigest(),**environment))
        log=progress.with_suffix('.log')
        rc=worker.run_srgc_child(runtime,[sys.executable,str(REPO/'scripts/v8_task_entry.py'),str(script)],
                                 log,[],threading.Event(),label='fixture-task',interval=.05)
        result=progress/'fixture-result.json'
        return rc,(json.loads(result.read_text()) if result.exists() else None),log.read_text()
    return run,tmp_path


def test_launched_rank_gets_no_periodic_stack_dump_by_default(launch):
    run,root=launch
    rc,seen,log=run('dump')
    assert rc==0,log[-2000:]
    assert seen['stack_dump_seconds']=='0'
    stacks=(root/'.multi-node/stacks/fixture-task-rank0.txt').read_text()
    assert stacks.startswith('== ') and 'Timeout (' not in stacks


def test_periodic_stack_dump_opt_in_keeps_the_node_value(launch):
    run,root=launch
    rc,seen,log=run('dump',V8_PERIODIC_STACK_DUMP='1')
    assert rc==0,log[-2000:]
    assert seen['stack_dump_seconds']=='1'
    assert 'Timeout (' in (root/'.multi-node/stacks/fixture-task-rank0.txt').read_text()


def test_slow_mbpp_checker_start_in_a_launched_rank_computes_the_reward(launch):
    run,_=launch
    rc,seen,log=run('mbpp')
    assert rc==0,log[-2000:]
    assert seen['rewards']==[1.0,0.0,0.0,1.0]


def test_mbpp_checker_start_past_its_window_still_fails_the_rank(launch):
    run,_=launch
    rc,seen,log=run('mbpp',V8_MBPP_STARTUP_SECONDS='0.3')
    assert rc!=0 and seen is None
    assert 'MBPP checker limits/setup failed after exec: startup timeout' in log
