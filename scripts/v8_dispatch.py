"""Versioned sequential-cohort dispatcher over exact numerical runtimes; local socketpairs only."""
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
import uuid

import srgc_v8_nodes as nodes
from v8_rpc import send,receive


class RuntimeProxy:
    def __init__(self,entry,plan,planfile,*,asynchronous=False):
        self.cohort=entry['cohort'];self.python=entry['python'];self.runner=Path(entry['runner'])
        self.root=Path(plan['root'])/self.cohort;self.error=None;self.process=None
        self.gpus=list(plan['gpus'])
        self.independent_startup=asynchronous;self.initialized=False;self.pending_barrier=None;self.on_ready=None
        self.stop_event=None;self.overrides=[];self.corrected_plans=()
        self.binding=dict(schema=1,cohort=self.cohort,root=str(self.root))
        self.channel,child=socket.socketpair()
        try:
            timeout=float(os.environ.get('V8_RUNTIME_INIT_TIMEOUT','120'));self.started=time.monotonic();self.timeout=timeout
            if not math.isfinite(timeout) or timeout<=0:raise ValueError('V8_RUNTIME_INIT_TIMEOUT must be finite and positive')
            self.channel.settimeout(timeout)
            print(f'[v8] {self.cohort}: opening original runtime with {self.python}; startup timeout={timeout:g}s',flush=True)
            self.process=subprocess.Popen([self.python,str(Path(__file__).with_name('v8_runtime_worker.py')),
                str(child.fileno()),str(planfile),self.cohort],env=os.environ,pass_fds=(child.fileno(),),stdin=subprocess.DEVNULL)
            child.close()
            if not asynchronous:self._initialize()
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
            self.close(grace=.2)  # initialization has not launched any training task
        finally:
            child.close()
            if self.channel.fileno()!=-1:self.channel.settimeout(None)

    def _initialize(self):
        reply,fds=receive(self.channel)
        for fd in fds:os.close(fd)
        if reply is None:raise RuntimeError('runtime actor exited during initialization')
        if not reply.get('ok'):raise ValueError(reply.get('error','actor initialization failed'))
        # Pool keeps this same dictionary, created before the asynchronous reply.
        self.binding.clear();self.binding.update(reply['binding']);self.initialized=True
        self.primary_results_first=reply.get('primary_results_first',False)
        self.streaming_ready_tasks=reply.get('streaming_ready_tasks',False)
        self.overrides=list(reply.get('overrides',[]));self.corrected_plans=tuple(reply.get('corrected_plans',()))
        if self.on_ready:self.on_ready(self)

    def startup_ready(self):
        if self.error or self.initialized:return True
        try:
            if select.select([self.channel],[],[],0)[0]:
                self.channel.settimeout(min(1,self.timeout));self._initialize();return True
            if time.monotonic()-self.started>=self.timeout:raise TimeoutError('runtime initialization timed out')
            return False
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}';self.close(grace=.2);return True
        finally:
            if self.channel.fileno()!=-1:self.channel.settimeout(None)

    def call(self,method,*args,fds=(),stop=None):
        if self.error:raise nodes.NodeFailure(f'{self.cohort}: runtime actor unavailable on this node: {self.error}')
        try:
            send(self.channel,dict(method=method,args=args),fds)
            reply,received=receive(self.channel)
        except (OSError,EOFError,ValueError) as exc:
            # The local actor process died or its socket broke: a fault of this node, not of the queue.
            raise nodes.NodeFailure(f'{self.cohort}: runtime actor on this node failed during {method}: '
                                    f'{type(exc).__name__}: {exc}') from None
        for fd in received:os.close(fd)
        if reply is None:raise nodes.NodeFailure(f'{self.cohort}: runtime actor on this node exited during {method}')
        if reply.get('interrupted'):
            # A stop signal reached the actor. Report it through the shared stop event; the reply itself is
            # complete, so the caller can still record the attempt (no KeyboardInterrupt traceback).
            for event in (stop,self.stop_event):
                if event is not None:event.set()
        if not reply.get('ok'):
            if reply.get('type')=='JSONDecodeError':
                error=json.JSONDecodeError(reply['error'],'',0)
                error.filename=reply.get('filename')
                error.lineno=reply.get('line',1);error.colno=reply.get('column',1)
                error.args=(reply['error'],)
                raise error
            if reply.get('type')=='PermissionError':
                raise nodes.NodeFailure(f'{self.cohort}: this account cannot access queue/run files: {reply.get("error")}')
            raise ValueError(f'{self.cohort}: {reply.get("type")}: {reply.get("error")}')
        return reply.get('result')

    def verify_node(self):return self.call('verify_node')
    def fresh_admission(self,device_fds):return self.call('fresh_admission',fds=device_fds)
    def plan(self,phase):return self.call('plan',phase)
    def side_tasks(self):return self.call('side_tasks')
    def ready(self,task):return self.call('ready',task)
    def native_busy(self,task):return self.call('native_busy',task)
    def done(self,task):return self.call('done',task)
    def barrier(self,action,fd,stop,device_fds):return self.call('barrier',action,fds=(fd,*device_fds),stop=stop)

    def begin_barrier(self,action,fd,stop,device_fds):
        # Keep the exact control lease alive in BOTH manager and original actor. Other
        # cohorts may initialize/prepare/claim while this CPU-only barrier is running.
        held=os.dup(fd)
        try:send(self.channel,dict(method='barrier',args=[action]),(fd,*device_fds))
        except BaseException:os.close(held);raise
        self.pending_barrier=(action,held)

    def poll_barrier(self,stop):
        if self.pending_barrier is None or not select.select([self.channel],[],[],0)[0]:return None
        action,fd=self.pending_barrier;self.pending_barrier=None
        error=None
        try:
            try:reply,received=receive(self.channel)
            except (OSError,EOFError,ValueError) as exc:
                raise nodes.NodeFailure(f'runtime actor on this node failed during CPU stage {action}: {exc}') from None
            for received_fd in received:os.close(received_fd)
            if reply is None:raise nodes.NodeFailure('runtime actor on this node exited during CPU stage '+action)
            if reply.get('interrupted'):stop.set()
            if not reply.get('ok'):raise RuntimeError(reply.get('error','CPU stage failed'))
        except Exception as exc:error=exc
        return action,fd,error
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        return self.call('execute',task,str(matrix),attempt,fds=(fd,*device_fds),stop=stop)

    def close(self,grace=2):
        self.channel.close()
        if self.process is not None:
            try:self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=grace)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
        if self.pending_barrier is not None:
            os.close(self.pending_barrier[1]);self.pending_barrier=None


def main():
    planfile=Path(sys.argv[1]);plan=json.loads(planfile.read_text());actors=[]
    plan['worker']=socket.gethostname()+'.'+str(os.getpid())
    # The actor receives this private per-launch plan, with paths already frozen by bootstrap.
    runfile=planfile.with_name(planfile.stem+'-'+str(os.getpid())+'-'+uuid.uuid4().hex+'.json')
    nodes.atomic(runfile,plan)
    entries={entry['cohort']:entry for entry in plan['entries']}
    requested=set(entries)|set(plan['errors'])
    cohort_order=tuple(c for c in nodes.DEFAULT_COHORTS if c in requested)+tuple(sorted(requested-set(nodes.DEFAULT_COHORTS)))
    controls={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ('v8_dispatch.py','v8_runtime_worker.py','v8_rpc.py','srgc_v8_nodes.py',
                           'v8_task_entry.py','v8_generation_probe.py','v8_progress_text.py','srgc_verifier_fallback.py')}
    plan['dispatch_policy']=nodes.DISPATCH_POLICY
    plan['controller_sha256']=controls['srgc_v8_nodes.py']
    print(f'[v8] dispatch policy={nodes.DISPATCH_POLICY} controller={controls["srgc_v8_nodes.py"][:12]}',flush=True)
    def factory(runner,root,cohort,python,gpus):
        plan['root']=str(Path(root).resolve());plan['gpus']=gpus
        nodes.atomic(runfile,plan)
        def record_policy(actor):
            record=dict(schema=1,policy=nodes.DISPATCH_POLICY,controller_source=controls,
                        original_runtime_binding=actor.binding,numerical_source_changed=False,
                        cohort_order=dict(policy=nodes.COHORT_ORDER,cohorts=list(cohort_order),spill_to_next=False),
                        execution_order=nodes.CONCURRENT_ORDER if actor.streaming_ready_tasks else
                                        nodes.PRIMARY_ORDER if actor.primary_results_first else 'original-runtime-order',
                        conditional_runtime_repairs=dict(strict_math_verifier='recorded-seed-fallback',
                            mbpp_subprocess='hash-pinned-limits-after-exec',native_fault='per-rank-invocation-stack',
                            checkpoint_evaluation='matched-horizon-before-numeric-budget-with-precommitted-cost-rule',
                            rollout_memory='cache-release-and-one-identical-retry',
                            legacy_dense_memory='hash-pinned-per-prompt-release-and-host-optimizer-snapshot',
                            olmo_memory='zero-dropout-activation-recomputation-and-generation-last-logit',
                            default_task_recovery=dict(max_counted_failures=nodes.MAX_ATTEMPTS,retry_delay=nodes.RETRY_DELAY,
                                                       unlimited_failures=False,retry_count_scope='task-lifetime',
                                                       native_crash_relaunch=False,quiescent_errors='exit-worker',
                                                       final_exit_check='refresh-original-ready-tasks',
                                                       evaluation_sigsegv_replay=dict(policy=nodes.EVAL_NATIVE_RECOVERY,max_failed_replays=1)),
                            node_native_faults=dict(max_failures=nodes.NODE_NATIVE_FAILURE_LIMIT,
                                scope='GPU-set/cohort/controller',new_pid_resets=False,
                                shared_queue_failed=False),
                            per_rank_receipt='progress/<attempt>/runtime-repair-rank-N.json'))
            if actor.overrides:record['control_overrides']=actor.overrides
            receipt=Path(root)/'.v8-dispatch-policies'/(cohort+'-'+nodes.digest(record)+'.json')
            if receipt.exists():
                if nodes.read(receipt)!=record:raise ValueError('dispatch policy receipt changed')
            else:nodes.atomic(receipt,record)
        actor=RuntimeProxy(entries[cohort],plan,runfile,asynchronous=True);actors.append(actor)
        actor.primary_cohort_scope=tuple(dict.fromkeys([*entries,*plan['errors']]))
        actor.primary_unavailable_cohorts=tuple(plan['errors'])
        actor.on_ready=record_policy
        return actor
    original=nodes.Runtime;nodes.Runtime=factory
    args=[];i=0
    while i<len(plan['argv']):
        token=plan['argv'][i]
        if token in ('--root','--runner-root','--python','--gpus','--poll-seconds','--max-attempts','--retry-delay'):
            if token!='--runner-root':args.extend(plan['argv'][i:i+2])
            i+=2;continue
        if token.startswith('-') and not token.startswith('--runner-root='):args.append(token)
        i+=1
    if len(entries)==1:args.insert(0,next(iter(entries)))
    for cohort,error in plan['errors'].items():print(f'V8 startup: {cohort} blocked: {error}',file=sys.stderr,flush=True)
    try:
        result=nodes.main(args,selected_cohorts=tuple(entries),cohort_order=cohort_order)
        return result or (1 if plan['errors'] else 0)
    except (OSError,ValueError,RuntimeError) as error:
        print(f'V8 queue error: {type(error).__name__}: {error}',file=sys.stderr,flush=True);return 1
    finally:
        for actor in actors:actor.close()
        nodes.Runtime=original


if __name__=='__main__':raise SystemExit(main())
