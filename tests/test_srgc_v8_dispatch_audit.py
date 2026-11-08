"""Adversarial dispatch control tests; CPU fixtures, never load a model."""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes
from scripts import v8_queue_runtime as source

REPO=Path(__file__).resolve().parents[1]


def test_claim_requires_successful_node_verification_even_after_control_contention(tmp_path):
    stop=threading.Event();launched=[]
    class Runtime:
        def verify_node(self):raise ValueError('this node has invalid frozen inputs')
        def plan(self,phase):
            return [dict(kind='train',run_id='r',method_id='on_monitored',training_seed=5,
                         horizon=275,config_hash='fixture',parent={})],False
        def ready(self,task):return True
        def native_busy(self,task):return False
        def done(self,task):return False
        def execute(self,*args):launched.append(True);stop.set();return 75
    class ContendedOnce(nodes.Pool):
        calls=0
        @contextmanager
        def control(self):
            self.calls+=1
            if self.calls==1:yield None
            else:
                with super().control() as lease:yield lease
    pool=ContendedOnce(tmp_path/'olmo_math/.multi-node',dict(cohort='olmo_math'))
    pool.initialize();state=nodes.read(pool.root/'state.json');state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    assert nodes.worker_loop_many([(pool,Runtime())],'node',stop,poll=.001)==1
    assert not launched,'an unverified joining node executed a GPU task'
    assert nodes.read(pool.root/'state.json')['failure'] is None


def test_qwen_interpreter_precedence_matches_original_launcher(tmp_path):
    qwen=tmp_path/'qwen-python';qwen.symlink_to(sys.executable)
    switch=tmp_path/'switch-python';switch.symlink_to(sys.executable)
    env=dict(V8_ROOT=str(tmp_path/'v8'),QWEN_PYTHON=str(qwen),SWITCH_PYTHON=str(switch),V8_PYTHON=sys.executable)
    assert source.existing_python({},env,'qwen_mbpp')==str(qwen)


def test_wedged_actor_initialization_is_bounded_and_child_is_reaped(tmp_path):
    frozen=tmp_path/'original.py'
    frozen.write_text('import os,time\nfrom pathlib import Path\n'
                     'class Runtime:\n def __init__(self,*args):\n'
                     '  Path(__file__).with_suffix(".pid").write_text(str(os.getpid()))\n  time.sleep(30)\n')
    entry=dict(cohort='olmo_math',python=sys.executable,script=str(frozen),runner=str(tmp_path))
    plan=dict(root=str(tmp_path/'v8'),gpus=['0','1','2','3'],worker='fixture.1',entries=[entry])
    planfile=tmp_path/'plan.json';planfile.write_text(json.dumps(plan))
    code='''import json,sys
from pathlib import Path
from v8_dispatch import RuntimeProxy
p=Path(sys.argv[1]);plan=json.loads(p.read_text())
actor=RuntimeProxy(plan['entries'][0],plan,p)
assert actor.error and 'timed out' in actor.error,actor.error
assert actor.process.poll() is not None,'failed initialization leaked its child'
actor.close()
'''
    child=subprocess.Popen([sys.executable,'-c',code,str(planfile)],start_new_session=True,
                           env=dict(os.environ,PYTHONPATH=str(REPO/'scripts'),V8_RUNTIME_INIT_TIMEOUT='0.2'),
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        try:out,err=child.communicate(timeout=4)
        except subprocess.TimeoutExpired:pytest.fail('one wedged runtime blocks both cohorts without a startup timeout')
        assert child.returncode==0,out+err
    finally:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=3)


def test_idle_workers_do_not_rescan_every_task_before_entering_dispatch(tmp_path):
    stop=threading.Event();calls=[]
    class Runtime:
        def verify_node(self):pass
        def plan(self,phase):
            calls.append(phase)
            return [dict(kind='train',run_id='r',method_id='on_monitored',training_seed=5,
                         horizon=275,config_hash='fixture',parent={})],False
        def ready(self,task):return True
        def native_busy(self,task):return False
        def done(self,task):return False
        def execute(self,*args):stop.set();return 75
    pool=nodes.Pool(tmp_path/'olmo_math/.multi-node',dict(cohort='olmo_math'));pool.initialize()
    state=nodes.read(pool.root/'state.json');state['phase']='train';nodes.atomic(pool.root/'state.json',state)
    assert nodes.worker_loop_many([(pool,Runtime())],'node',stop,poll=.001)==75
    assert calls==['train'],'duplicate full DAG scans retain cohort locks ahead of the dispatch gate'
