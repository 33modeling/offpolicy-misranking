"""11-16 '모르겠으면 주석이라도 잘 나오게 하던가': an idle node says every 30 s, in plain words and per cohort it
serves, why it has no task; the same text is in its worker status (idle_reason) and in v8_status's node line."""
import importlib
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_idle_1116 import Actor, queue, hold_all_leases, until

FULL,DEPS,DOWN='qwen_mbpp','olmo_mbpp','qwen_math'


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv('V8_USER_RETRY_GRANT','0')
    monkeypatch.setenv('V8_GIVEUP_RETRY_SECONDS','600')


@pytest.mark.parametrize('obs,text',[
    (None,'tasks not inspected yet'),
    (dict(phase='train',ready=0,leased=6,dependencies=0),'train: none ready, 6 running on other nodes'),
    (dict(phase='train',ready=0,leased=0,dependencies=12),'train: none ready, 12 waiting for parent checkpoints'),
    (dict(phase='evaluation',ready=0,deferred_budget=True),'evaluation: none ready, waiting for the frozen evaluation budget'),
    (dict(phase='train',ready=0,recovery_wait=True,failed=1),
     'train: none ready, failed tasks wait for their retry time or another GPU host, 1 failed for good'),
    (dict(phase='train',ready=0,unsealed_plan=True),
     'train: none ready, all listed tasks are done; the planner has not added the next phase yet'),
    (dict(phase='train',ready=0,leased=0),'train: none ready, nothing else to do in this phase now'),
    (dict(phase='train',ready=2,leased=1),'train: 2 ready, 1 running on other nodes'),
    (dict(phase='train',native_busy=1),'train: none ready, 1 wait for a process still using the same run folder'),
    (dict(phase='extension',extension_waiting_for=['olmo_math']),'extension waits for the matched-step results of olmo_math'),
    (dict(phase='train',metadata_error='x.jsonl line 3'),'a task record is being written by another node; reading it again shortly'),
])
def test_observation_in_plain_words(obs,text):
    assert nodes.observation_text(obs)==text
    assert 'WAIT_' not in text


class Waiting(Actor):
    def ready(self,task):return False   # parent checkpoints not there yet


class Broken(Actor):
    def verify_node(self):raise nodes.NodeFailure(f'{self.cohort}: runtime actor unavailable on this node: fixture start failure')


def test_idle_node_explains_each_cohort(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(nodes,'ACTIVE_RESERVE',0)   # held leases stand for busy nodes; spill mechanics only
    monkeypatch.setattr(nodes,'SPILL_STALL_SECONDS',.3);monkeypatch.setattr(nodes,'IDLE_REPORT_SECONDS',.3)
    full=queue(tmp_path,Actor(tmp_path,FULL,2));queue(tmp_path,Waiting(tmp_path,DEPS,3))
    held=hold_all_leases(full)
    rows=[]
    for runtime in (Actor(tmp_path,FULL,2),Waiting(tmp_path,DEPS,3),Broken(tmp_path,DOWN)):
        rows.append((nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime))
    stop=threading.Event();worker='h12.7'
    thread=threading.Thread(target=nodes.worker_loop_many,args=(rows,worker,stop),
                            kwargs=dict(poll=.05,cohort_order=(FULL,DEPS,DOWN)),daemon=True)
    thread.start()
    status_path=tmp_path/'.v8-worker-status'/(nodes.digest(worker)+'.json')
    def explained():
        try:reason=nodes.read(status_path).get('idle_reason') or ''
        except (OSError,ValueError):return False
        return 'running on other nodes' in reason and 'parent checkpoints' in reason and 'NOT SERVING until' in reason
    try:
        assert until(explained,20)
        time.sleep(.8)
        record=nodes.read(status_path)
        status=importlib.import_module('v8_status')
        table=status.node_table([tmp_path],[FULL,DEPS,DOWN],time.time(),this_host='other-host')
    finally:
        stop.set();thread.join(10)
        for lease in held:lease.close()
    out=capsys.readouterr().out
    lines=[l for l in out.splitlines() if l.startswith(f'[v8] NODE {worker} IDLE ')]
    assert len(lines)>=2,out[-2000:]   # repeated while idle
    line=lines[-1]
    assert f'{FULL} — train: none ready, 2 running on other nodes' in line
    assert f'{DEPS} — train: none ready, 3 waiting for parent checkpoints' in line
    assert f'{DOWN} — NOT SERVING until ' in line and '(retry 1): node verification failed: NodeFailure' in line
    assert line.index(FULL)<line.index(DEPS)<line.index(DOWN)   # claim order
    assert 'WAIT_' not in line
    reason=record['idle_reason']
    assert reason.startswith('IDLE ') and f'{FULL} — train: none ready, 2 running on other nodes' in reason
    doing=[r['doing'] for r in table if r.get('state')=='idle']
    assert doing and doing[0].startswith('IDLE ') and 'waiting for parent checkpoints' in doing[0],table


def test_starting_cohort_is_named_with_its_verification_time(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(nodes,'IDLE_REPORT_SECONDS',.3)
    gate=threading.Event()
    runtime=Actor(tmp_path,DEPS,1,verify=gate)
    rows=[(nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime)]
    stop=threading.Event()
    thread=threading.Thread(target=nodes.worker_loop_many,args=(rows,'h3.1',stop),
                            kwargs=dict(poll=.05,cohort_order=(DEPS,)),daemon=True)
    thread.start()
    try:
        time.sleep(1)
    finally:
        gate.set();stop.set();thread.join(10)
    out=capsys.readouterr().out
    assert f'[v8] NODE h3.1 IDLE ' in out and f'{DEPS} — starting (node verification ' in out,out[-1500:]
