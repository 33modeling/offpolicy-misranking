"""Recovered-node eligibility with synthetic admission receipts, never an H100 claim."""
import socket
import threading
import json
from pathlib import Path
import sys

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_eval_native_recovery import prepared


def run(pool,runtime,worker):
    stop=threading.Event();timer=threading.Timer(1,stop.set);timer.start()
    try:return nodes.worker_loop(pool,runtime,worker,stop,poll=.001)
    finally:timer.cancel();timer.join()


def recovered(tmp_path,*,passes=True):
    pool,runtime,path,checkpoint,result=prepared(tmp_path)
    runtime.gpus=['0','1','2','3'];pool.gpus=tuple(runtime.gpus);pool.host=socket.gethostname()
    job=nodes.read(path);job.update(state='failed',attempts=[dict(id='old-crash',outcome='failed',exit_code=-11,
        host=pool.host,gpus=runtime.gpus,worker=pool.host+'.1')]);nodes.atomic(path,job)
    checks=[]
    def check(fds):
        checks.append(True)
        if not passes:raise RuntimeError('synthetic failed NCCL admission')
        receipt=tmp_path/'synthetic-admission.json';nodes.atomic(receipt,dict(state='passed',scope='CPU fixture'))
        return dict(receipt=str(receipt))
    runtime.fresh_admission=check
    return pool,runtime,path,checkpoint,result,checks


def test_restart_recovers_previously_crashed_gpu_set_without_resetting_task_history(tmp_path):
    pool,runtime,path,checkpoint,result,checks=recovered(tmp_path);before=checkpoint.read_bytes()
    assert not pool.retryable(nodes.read(path))
    assert run(pool,runtime,'restarted-node')==0
    assert len(checks)==1 and len(runtime.calls)==1 and result.is_file()
    job=nodes.read(path);assert job['state']=='done' and job['attempts'][0]['id']=='old-crash'
    assert len(job['attempts'])==2 and checkpoint.read_bytes()==before


def test_failed_probe_claims_no_training_and_plain_restart_has_finite_probe_budget(tmp_path):
    pool,runtime,path,checkpoint,result,checks=recovered(tmp_path,passes=False)
    before=nodes.read(path);saved=checkpoint.read_bytes()
    for i in range(3):
        joined=nodes.Pool(pool.root,pool.binding);joined.host=pool.host
        assert run(joined,runtime,'restart-'+str(i))==1
    after=nodes.read(path)
    assert len(checks)==2 and not runtime.calls
    assert after['attempts']==before['attempts'] and after['task']==before['task']
    assert checkpoint.read_bytes()==saved and not pool.queue_state().get('failure')


def test_readmission_cannot_renew_exhausted_task_failure_budget(tmp_path):
    pool,runtime,path,checkpoint,result,checks=recovered(tmp_path)
    job=nodes.read(path);job['attempts'].append(dict(job['attempts'][0],id='second-crash'))
    nodes.atomic(path,job);pool.readmitted=True;pool.readmit_fault_baseline=0
    assert not pool.retryable(job) and nodes.retry_budget_failures(job)==2


def test_post_readmission_native_crash_does_not_loop_on_the_same_node(tmp_path):
    pool,runtime,path,checkpoint,result,checks=recovered(tmp_path)
    runtime.execute=lambda *args:-11
    assert run(pool,runtime,'restarted-node')==1
    assert len(checks)==1 and len(nodes.read(path)['attempts'])==2
    joined=nodes.Pool(pool.root,pool.binding);joined.host=pool.host
    assert run(joined,runtime,'another-pid')==1
    assert len(checks)==1 and len(nodes.read(path)['attempts'])==2


def test_original_runtime_actor_rechecks_admission_with_inherited_gpu_lease(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from v8_dispatch import RuntimeProxy
    from test_srgc_v8_dispatch_runtime import FAKE
    original=tmp_path/'original.py'
    original.write_text(FAKE.replace('__TAG__',repr('admission CPU fixture'))+'''
 def admit(self,device_fds):
  assert self.admitted is None
  assert len(device_fds)==1
  os.fstat(device_fds[0])
  self.root.mkdir(parents=True,exist_ok=True)
  receipt=self.root/'admission.json'
  receipt.write_text(json.dumps(dict(state='passed',scope='CPU FD transfer fixture')))
  self.admitted=dict(receipt=str(receipt));return self.admitted
''')
    entry=dict(cohort='olmo_math',script=str(original),runner=str(tmp_path),python=sys.executable)
    plan=dict(root=str(tmp_path/'runs'),gpus=['0','1','2','3'],worker='host.42',entries=[entry])
    planfile=tmp_path/'plan.json';planfile.write_text(json.dumps(plan))
    actor=RuntimeProxy(entry,plan,planfile)
    try:
        assert actor.error is None
        with nodes.Lock(tmp_path/'gpu.lock') as lease:
            report=actor.fresh_admission((lease.fd,))
            assert nodes.read(report['receipt'])['state']=='passed'
            probe=nodes.Lock(tmp_path/'gpu.lock');assert not probe.acquire()
    finally:actor.close()
