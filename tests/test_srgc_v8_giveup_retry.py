"""11-16 'idle 고쳐': a node-local give-up is retried on this node after a bounded backoff (CPU fixtures).

Before: a runtime that failed to start, a node verification timeout or a launch failure marked that cohort
terminal on this node for the rest of the process, so a node that had given up every remaining cohort idled
(or exited) beside ready work. Now such a cohort comes back after V8_GIVEUP_RETRY_SECONDS (600, doubling up to
3600); native-crash quarantine, a changed queue identity and queue-level failures keep the old behaviour."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from scripts import srgc_v8_nodes as nodes
from test_srgc_v8_sequential_cohorts import Runtime, events

ROOT=Path(__file__).resolve().parents[1]
FIRST,SECOND='olmo_math','qwen_mbpp'


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setenv('V8_USER_RETRY_GRANT','0')
    monkeypatch.setenv('V8_GIVEUP_RETRY_SECONDS','0.3')


class Flaky(Runtime):
    """The second cohort's runtime fails on this node the first `failures` times, then works."""
    def __init__(self,base,cohort,where,failures=1,independent=False):
        super().__init__(base,cohort);self.where=where;self.failures=failures;self.independent_startup=independent
        self.restarts=0
    def _fault(self,stage):
        if self.where==stage and self.failures>0:
            self.failures-=1;self.event('fault-'+stage)
            raise nodes.NodeFailure(f'{self.cohort}: runtime actor unavailable on this node: fixture {stage} fault')
    def verify_node(self):
        self._fault('verify');super().verify_node()
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        self._fault('execute');return super().execute(task,matrix,fd,stop,device_fds,attempt)
    def restart(self):self.restarts+=1;self.event('restart')
    # Independent (actor-like) startup: CPU stages are begun and collected off the loop.
    pending_barrier=None
    def begin_barrier(self,action,fd,stop,device_fds):self.barrier(action);self.pending_barrier=(action,os.dup(fd))
    def poll_barrier(self,stop):
        action,fd=self.pending_barrier;self.pending_barrier=None;return action,fd,None


def rows(base,where,**kwargs):
    out=[]
    for name in (FIRST,SECOND):
        runtime=Flaky(base,name,where if name==SECOND else None,**kwargs)
        out.append((nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime))
    return out


def run(pairs,name='cpu-node',seconds=60):
    stop=threading.Event();result={}
    thread=threading.Thread(target=lambda:result.setdefault('code',nodes.worker_loop_many(
        pairs,name,stop,poll=.01,cohort_order=(FIRST,SECOND))),daemon=True)
    thread.start();thread.join(seconds);stop.set();thread.join(10)
    return result.get('code')


def test_retry_kinds():
    fault=nodes.NodeFailure('x')
    for kind in ('node verification','queue initialization','node check','node launch','node runtime',
                 'queue access','task inspection','final readiness check','queue recovery check'):
        assert nodes.giveup_retryable(kind,fault),kind
    assert nodes.giveup_retryable('queue access',PermissionError('denied'))
    # A verification that answers 'frozen inputs differ' is deterministic: not retried (the node exits as before).
    assert not nodes.giveup_retryable('node verification',ValueError('this node has invalid frozen inputs'))
    # Never retried: this GPU set's native-crash quarantine/re-admission, queue identity, queue-level failures.
    assert not nodes.giveup_retryable('native crash quarantine',fault)
    assert not nodes.giveup_retryable('GPU re-admission',fault)
    assert not nodes.giveup_retryable('extension deferred',fault)
    assert not nodes.giveup_retryable('queue identity',nodes.BindingMismatch('moved'))
    assert not nodes.giveup_retryable('node verification',nodes.BindingMismatch('queue identity changed'))
    assert not nodes.giveup_retryable('stage',ValueError('recorded in the queue'))
    assert not nodes.giveup_retryable('CPU stage',RuntimeError('recorded in the queue'))


def test_backoff_doubles_to_one_hour(monkeypatch):
    monkeypatch.delenv('V8_GIVEUP_RETRY_SECONDS')
    assert [nodes.giveup_retry_delay(n) for n in range(1,7)]==[600,1200,2400,3600,3600,3600]
    assert nodes.giveup_retry_delay(10**6)==3600
    monkeypatch.setenv('V8_GIVEUP_RETRY_SECONDS','0')       # never a busy loop
    assert nodes.giveup_retry_delay(1)==600
    monkeypatch.setenv('V8_GIVEUP_RETRY_SECONDS','5')
    assert [nodes.giveup_retry_delay(n) for n in (1,2,3)]==[5,10,20]


@pytest.mark.parametrize('where,independent',[('verify',False),('verify',True),('execute',False)])
def test_given_up_cohort_is_served_again_after_backoff(tmp_path,capsys,where,independent):
    """Runtime start / node verification / launch fails once for the second cohort: before, the node gave it
    up for good and exited 1 once the first cohort was done; now it retries and finishes both."""
    pairs=rows(tmp_path,where,independent=independent)
    started=time.monotonic()
    assert run(pairs)==0
    out=capsys.readouterr().out
    assert f'NOT SERVING {SECOND} until ' in out and '(retry 1): ' in out,out[-3000:]
    assert f'SERVING {SECOND} again (retry 1)' in out
    assert all(pool.queue_state()['phase']=='complete' for pool,_ in pairs)
    recorded=[(r['cohort'],r['stage']) for r in events(tmp_path)]
    fault=recorded.index((SECOND,'fault-'+where));restart=recorded.index((SECOND,'restart'))
    assert fault<restart and pairs[1][1].restarts==1   # the runtime is recreated before serving it again
    assert time.monotonic()-started>=.3


def test_node_waits_for_retry_instead_of_exiting_and_says_until_when(tmp_path,capsys,monkeypatch):
    """While the only remaining cohort waits for its retry the node neither exits nor busy-loops, and its
    status names the retry time."""
    monkeypatch.setenv('V8_GIVEUP_RETRY_SECONDS','30')
    pairs=rows(tmp_path,'verify',failures=10**6)
    stop=threading.Event();result={}
    thread=threading.Thread(target=lambda:result.setdefault('code',nodes.worker_loop_many(
        pairs,'cpu-node',stop,poll=.01,cohort_order=(FIRST,SECOND))),daemon=True)
    thread.start()
    try:
        status=tmp_path/'.v8-worker-status'/(nodes.digest('cpu-node')+'.json')
        def waiting():
            try:return nodes.read(status).get('state')=='WAIT_RETRY'
            except (OSError,ValueError):return False
        deadline=time.monotonic()+30
        while time.monotonic()<deadline and not waiting():time.sleep(.05)
        time.sleep(1)
        assert thread.is_alive() and 'code' not in result     # did not exit beside a pending retry
        row=nodes.read(status)
        assert row['state']=='WAIT_RETRY' and row['retrying'][SECOND]['retry']==1
        assert row['reason'].startswith(f'NOT SERVING {SECOND} until ') and '(retry 1): node check failed' in row['reason']
        assert row['not_serving'][SECOND].startswith('until ')
        assert len([r for r in events(tmp_path) if r['stage']=='fault-verify'])==1   # no retry before the backoff
    finally:
        stop.set();thread.join(10)
    from v8_progress_text import scheduler_text
    text=scheduler_text(row,time.time())
    assert f'NOT SERVING {SECOND} until ' in text and '(retry 1)' in text


def test_non_retryable_give_up_still_exits(tmp_path,capsys,monkeypatch):
    """A native-crash quarantine of this GPU set is not retried: the node exits as before once nothing else
    remains, and its status lists no pending retry."""
    pairs=rows(tmp_path,None)
    original=nodes.node_native_hold
    monkeypatch.setattr(nodes,'node_native_hold',lambda pool,runtime:'fixture quarantine'
                        if pool.binding['cohort']==SECOND else original(pool,runtime))
    assert run(pairs,seconds=30)==1
    out=capsys.readouterr().out
    assert f'NOT SERVING {SECOND} from now: native crash quarantine' in out and 'retry 1' not in out
    row=nodes.read(tmp_path/'.v8-worker-status'/(nodes.digest('cpu-node')+'.json'))
    assert row['retrying']=={} and SECOND in row['not_serving']


def test_runtime_proxy_restart_recreates_a_dead_actor(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('retry_dispatch',ROOT/'scripts/v8_dispatch.py')
    dispatch=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatch)
    entry=dict(cohort=SECOND,python=sys.executable,runner=str(tmp_path))
    proxy=dispatch.RuntimeProxy(entry,dict(root=str(tmp_path),gpus=[]),tmp_path/'plan.json',asynchronous=True,lazy=True)
    proxy.error='RuntimeError: runtime actor exited during initialization'
    proxy.process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    started=[]
    monkeypatch.setattr(proxy,'_start',lambda:(started.append(1),setattr(proxy,'error','fixture: not started')))
    process=proxy.process;proxy.restart()
    assert process.poll() is not None   # the old actor is stopped
    assert proxy.error is None and proxy.channel is None and proxy.process is None and not proxy.initialized
    proxy.startup_ready()
    assert started==[1]   # the same lazy start path as at startup
