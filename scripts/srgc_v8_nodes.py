#!/usr/bin/env python3
"""Shared-filesystem V8 workers: one four-GPU task per node, dynamic DAG dispatch.

Run the same command on each allocated node (no cross-node DDP rendezvous):
  python scripts/srgc_v8_nodes.py --runner-root /path/to/v8/experiments

V8_ROOT must be the same shared path on every node. V8_CODE_ROOT remains the
frozen trainer checkout. No web service, paid model API or cluster allocator is
started. The launcher uses torchrun only when the user runs a worker command.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import errno
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import shutil
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

SCHEMA = 1
DISPATCH_POLICY = 'retain-ready-offer-control-until-sequential-claim-v34'
COHORT_ORDER = 'earlier-cohort-first-next-cohort-when-nothing-claimable-v2'
SPILL_LOCK_GRACE = float(os.environ.get('V8_SPILL_LOCK_GRACE', '10'))  # seconds
# An earlier cohort that gives this node no task for this long, for any reason (including other nodes
# holding its queue lock all the time), no longer holds the node back (user 11-16: 'mbpp는 꽉찼는데 그걸
# 기다리고 있음', '다른걸로 안넘어가고'). Earlier cohorts keep claim priority on every later loop.
SPILL_STALL_SECONDS = float(os.environ.get('V8_SPILL_STALL_SECONDS', '30'))
# Shared inspection result (scheduling cache under V8_ROOT/.v8-scan-share, never a queue file). Every idle node
# re-inspected every cohort's whole task DAG under that cohort's control.lock, so with many nodes the lock was
# always held by an inspection that found nothing new and a spare node waited on it (WAIT_QUEUE_LOCK) for
# minutes before it could inspect, initialize or claim a later cohort. A node now reuses another node's
# inspection when it is younger than this: a ready offer is claimed through the same claim_offered()
# revalidation under the control lock (no claim rule changes), and 'nothing ready' with unchanged
# jobs/ and state.json is not inspected again until it expires.
SCAN_SHARE_SECONDS = float(os.environ.get('V8_SCAN_SHARE_SECONDS', '60'))
# 11-18 (user: 'qwen_mbpp 먼저 끝내'): while the first unfinished cohort has a task running on another node (or one
# this node may retry after its delay), this many of its live idle nodes stay with it instead of spilling, so an
# orphaned or newly ready task is claimed within seconds. Nodes beyond the reserve spill as before. 0 turns it off.
try:ACTIVE_RESERVE = max(0, int(os.environ.get('V8_ACTIVE_RESERVE', '2')))
except ValueError:ACTIVE_RESERVE = 2
ACTIVE_GIVEUP_RETRY_CAP = 120.0   # seconds: a node-local give-up of the first unfinished cohort is retried this soon
EVAL_NATIVE_RECOVERY = 'evaluation-sigsegv-recovery-v1'
# User 11-13 '자꾸 에러 난거 먼저 처리해서 다시 시작하게 해야지': every task that was BLOCKED (failure cap or native
# crashes on two places) when this policy first met it gets ONE fresh failure budget, counted from its last
# attempt. Never renewed by a restart/new node/hash; attempts, failures and checkpoints are kept; a native
# crash place stays excluded for that task.
USER_RETRY_GRANT = 'user-retry-blocked-v1-20261113'
# 11-17: every V8 SIGSEGV since 11-08 came from the pinned child's 30-min faulthandler watchdog
# (dump_traceback_later), which v8_task_entry.py now cancels. Tasks blocked by those crashes get one fresh
# budget, once per task, from a node whose checkout carries that fix (user: '새노드 잡히면 알아서 복구되면서 시작').
WATCHDOG_FIX_GRANT = 'periodic-stack-dump-sigsegv-fixed-v1-20261117'
GRANT_POLICIES = (USER_RETRY_GRANT, WATCHDOG_FIX_GRANT)
# 11-18: a SIGSEGV from that periodic dump is not a task or GPU fault. Such an attempt keeps its record and gets a
# periodic_dump_refund receipt; it then counts toward no failure cap, crash place or GPU-set quarantine, so blocked
# tasks resume from their checkpoints (user: 'qwen_mbpp 먼저 끝내'). Evidence, SIGSEGV at least 30 min after start:
# the attempt's task entry armed the dump, or it was launched before the fix by a controller that predates it, or its
# rank stack file ends in a 'Timeout (0:' dump. Crashes on fixed code count as before; at most
# PERIODIC_DUMP_REFUND_LIMIT refunds per task, so no task restarts forever.
PERIODIC_DUMP_REFUND = 'periodic-stack-dump-crash-refund-v1-20261118'
PERIODIC_DUMP_REFUND_LIMIT = 20
COLLECTIVE_TIMEOUT_REFUND_LIMIT = 100   # per task; see collective_timeout_evidence
PERIODIC_DUMP_SECONDS = 1800   # dump_traceback_later(1800, repeat=True) in the pinned child
# srgc_v8_nodes.py at fc352ed..e34e750: its checkout's task entry cancels the dump (9b31912)
FIXED_DUMP_CONTROLLERS = frozenset({'b6342f58767e056c08958b1028475e384df217e179f029c4295327d9a5eb6444'})
REFUND_CHECK_SECONDS = float(os.environ.get('V8_REFUND_CHECK_SECONDS', '300'))
NODE_NATIVE_FAILURE_LIMIT = 2
# Node verification hashes this node's frozen trainer and inputs; on a shared volume with many nodes starting
# together it takes minutes. It runs without the queue control.lock and has its own limit (user 11-11:
# 'v8 코드 지금 lock 걸려서 아무것도 시작못해', '모든 시작노드가 idle임'); the 120 s runtime-start timeout is not it.
NODE_VERIFY_TIMEOUT = float(os.environ.get('V8_NODE_VERIFY_TIMEOUT', '1800'))
# A task that died of a native signal (SIGSEGV etc.) is retried from its last checkpoint on a host where it has
# not crashed (user 11-10: 'failed로 된거 복구 되는거지? ... 디버깅 계속 해보고 수정해'). Repeated crashes on one
# GPU node (V8-144) say nothing about the task; crashes on this many different hosts point at the task itself.
TASK_NATIVE_HOST_LIMIT = 3   # 11-16: one more host before a native crash is blamed on the task
TERMINAL = {'completed_horizon', 'completed_budget'}
PHASES = ('train', 'primary_evaluation', 'extension', 'evaluation')
CPU_PHASES = ('prepare', 'freeze', 'primary_finalize', 'finalize')
PRIMARY_ORDER = 'matched-results-before-extension-v1'
CONCURRENT_ORDER = 'ready-checkpoint-evaluation-extension-v1'
BUDGET_RULE = 'original-cost-rule-precommitted-v1'
DEFAULT_COHORTS = ('olmo_math', 'qwen_mbpp', 'olmo_mbpp', 'qwen_math')
CORRECTED_PLANNER = 'current-v8-016'  # evaluation planner replaced for pre-V8-016 coordinators


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    try:return json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        error=json.JSONDecodeError(f'{path}: {exc.msg}',exc.doc,exc.pos)
        error.filename=str(path)
        raise error from exc


def metadata_error_path(error):
    if getattr(error,'filename',None):return str(error.filename)
    trace=error.__traceback__;found=None
    while trace is not None:
        value=trace.tb_frame.f_locals.get('path')
        if isinstance(value,(str,Path)) and Path(value).suffix in ('.json','.jsonl'):found=str(value)
        trace=trace.tb_next
    return found


def atomic(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as handle:
            handle.write(canonical(value)+'\n');handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


class WorkerStatus:
    """Diagnostic receipts only: scheduler activity is never training progress."""
    def __init__(self,base,worker,controller):
        self.path=Path(base)/'.v8-worker-status'/(digest(worker)+'.json')
        self.worker=worker;self.controller=controller;self.last=None;self.written=-float('inf');self.since=time.time()
        self.context=None  # callable -> extra fields on every record (cohorts served / not served on this node)

    def publish(self,state,cohort,reason,**details):
        now=time.time();signature=(state,cohort,reason)
        if signature!=self.last:self.since=now
        elif time.monotonic()-self.written<15:return
        self.last=signature;self.written=time.monotonic()
        extra={}
        if self.context is not None:
            try:extra=self.context()
            except Exception:extra={}  # diagnostics never stop the scheduler
        record=dict(schema=1,worker=self.worker,pid=os.getpid(),launch_id=os.environ.get('V8_LAUNCH_ID'),
                    controller_sha256=self.controller,at=now,since=self.since,state=state,cohort=cohort,
                    reason=reason,**{**extra,**details})
        try:atomic(self.path,record)
        except OSError as exc:print(f'[V8 STATUS ERROR] cannot save worker diagnosis: {exc}',flush=True)
        counts=' '.join(f'{k}={details[k]}' for k in ('phase','kind','method','seed','ready','leased','native_busy','dependencies','failed') if k in details)
        stamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now))
        print(f'[V8 STATUS] {stamp} node={self.worker} state={state} experiment={cohort or "-"} '
              f'{reason}; {counts} same_state={now-self.since:.0f}s',flush=True)


class Lock:
    def __init__(self,path):
        self.path=Path(path);self.fd=None

    def acquire(self):
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.fd=os.open(self.path,os.O_CREAT|os.O_RDWR,0o660)
        except PermissionError as exc:
            raise NodeFailure(access_message(self.path,exc)) from None
        try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd);self.fd=None;return False
        except OSError as exc:
            os.close(self.fd);self.fd=None
            if exc.errno in (errno.ENOLCK,errno.ENOSYS,errno.EOPNOTSUPP,errno.EINVAL):
                raise ValueError(f'{self.path.parent}: this filesystem does not provide flock() locks ({exc.strerror}). '
                                 'V8_ROOT/OM_WORK must be the same shared group volume the SRGC queue already uses.') from None
            raise
        return True

    def close(self):
        if self.fd is not None:
            # Close this reference, not LOCK_UN: a torchrun child can inherit
            # the same open-file description and must keep it after parent loss.
            os.close(self.fd);self.fd=None

    def __enter__(self):
        if not self.acquire():raise BlockingIOError(str(self.path))
        return self

    def __exit__(self,*_):self.close()


class BindingMismatch(ValueError):
    pass


class NodeFailure(RuntimeError):
    """A fault of this node or of its local runtime process: NCCL admission, a dead runtime actor,
    missing GPUs, local file access. It is never written into the shared queue; this node stops
    serving the cohort and the other nodes continue."""


def access_message(path,exc):
    import getpass,pwd
    target=Path(path)
    owner='?'
    for candidate in (target,*target.parents):
        try:owner=pwd.getpwuid(candidate.stat().st_uid).pw_name;break
        except (OSError,KeyError):continue
    return (f'{target}: {exc.strerror or exc} for account {getpass.getuser()} (queue files owned by {owner}). '
            'Run the worker as the account that created this queue; nothing was changed.')


MAX_ATTEMPTS=4     # initial task fault plus three immediate retries, across all workers (user 11-16:
                   # '빨리 실패한거에 노드 다시 붙여야지'); still bounded, so a broken task cannot relaunch forever
RETRY_DELAY=0.0    # restart from the saved checkpoint without a cooldown


def failure_limit(value):
    """Legacy zero must not re-enable unlimited torchrun relaunches."""
    return value if type(value) is int and value>0 else MAX_ATTEMPTS


def dump_refunded(attempt):
    """The attempt carries a periodic-dump crash refund receipt (kept as history, counted nowhere)."""
    return isinstance(attempt,dict) and (attempt.get('periodic_dump_refund') or {}).get('policy')==PERIODIC_DUMP_REFUND


def counted_failures(job):
    """Original SRGC accounting: intentional interruptions and node-local launch faults are refunded."""
    count=0
    for attempt in job.get('attempts',[]):
        if dump_refunded(attempt):continue
        outcome=attempt.get('outcome')
        if outcome is None and attempt.get('exit_code') not in (None,0,75):outcome='failed'  # pre-outcome records
        count+=outcome=='failed'
    return count


def misclassified_stops(job):
    """Recognize the old whole-history SIGSEGV check; retain every original field."""
    prior_native=False;corrected=[]
    for attempt in job.get('attempts',[]):
        native=native_signal(attempt)
        if (prior_native and not native and attempt.get('id') and attempt.get('outcome')=='failed'
                and attempt.get('exit_code')==75 and not attempt.get('output_validation_error')
                and not (attempt.get('child_failure') or {}).get('failures')
                and str(attempt.get('error','')).startswith('task exit=75; output verified=')):
            corrected.append(attempt['id'])
        prior_native=prior_native or bool(native)
    return corrected


def budget_attempts(job,attempts):
    ignored=set(misclassified_stops(job))
    return [a for a in attempts if a.get('id') not in ignored]


def granted_view(job):
    """The job as seen by its one user retry grant: only attempts after the grant, no older replay receipt."""
    grant=job.get('user_retry_grant') if isinstance(job,dict) else None
    if (not isinstance(grant,dict) or grant.get('policy') not in GRANT_POLICIES or not isinstance(job.get('task'),dict)
            or grant.get('task_sha256')!=digest(semantic_task(job['task']))):return None
    attempts=[a for a in job.get('attempts',[]) if isinstance(a,dict)]
    ids=[a.get('id') for a in attempts]
    if grant.get('after_attempt') not in ids:return None
    view=dict(job,attempts=attempts[ids.index(grant['after_attempt'])+1:])
    view.pop('evaluation_native_recovery',None);view.pop('user_retry_grant',None)
    return view


def retry_budget_failures(job):
    """Lifetime task faults, except the single user-requested native evaluation replay.

    Historical retry_epoch records stay intact as evidence, but no longer grant
    another failure budget. Genuine node loss/intentional stops are still refunded.
    Native crashes count once per host (v25): relaunches on one crashing node are one node fault.
    """
    view=granted_view(job)
    if view is not None:return retry_budget_failures(view)
    recovery=evaluation_recovery_attempts(job)
    if recovery is not None:return counted_failures(dict(attempts=budget_attempts(job,recovery)))
    attempts=budget_attempts(job,[a for a in job.get('attempts',[]) if isinstance(a,dict)])
    other=[a for a in attempts if not fatal_native_failure(dict(attempts=[a]))]
    return counted_failures(dict(attempts=other))+native_crash_count(job)


def attempt_host(attempt):
    """The attempt's host (recorded since v25), else from the worker id '<host>.<pid>'."""
    if attempt.get('host'):return str(attempt['host'])
    worker=str(attempt.get('worker') or '')
    return worker.rsplit('.',1)[0] if '.' in worker else worker


def native_crash_count(job):
    """Native crashes that use the retry budget: each attempt with a recorded host (v25 never sends a retry back
    to a crash host), and once per host for older records (repeated relaunches on one node, V8-136/V8-144)."""
    recorded=0;legacy=set()
    for index,attempt in enumerate(job.get('attempts',[])):
        if not (isinstance(attempt,dict) and fatal_native_failure(dict(attempts=[attempt]))):continue
        if attempt.get('host'):recorded+=1
        else:legacy.add(attempt_host(attempt) or f'unknown-node-{attempt.get("id") or index}')
    return recorded+len(legacy)


def native_crash_places(job):
    """(host, GPU set or None) of each native crash of this task, each once, in order. The GPU set is recorded
    since v25 (the quarantine unit of node_native_identity); None means unknown, i.e. any GPU set of that host.
    A crash without a recorded worker counts on its own (its node is unknown)."""
    places=[]
    for index,attempt in enumerate(job.get('attempts',[])):
        if isinstance(attempt,dict) and fatal_native_failure(dict(attempts=[attempt])):
            host=attempt_host(attempt) or f'unknown-node-{attempt.get("id") or index}'
            gpus=tuple(str(g) for g in attempt.get('gpus') or ()) if attempt.get('host') else ()
            place=(host,gpus or None)
            if place not in places:places.append(place)
    return places


def native_crash_hosts(job):
    """Readable names of the places a task crashed natively ('host' or 'host GPU 0,1,2,3')."""
    return [host+(' GPU '+','.join(gpus) if gpus else '') for host,gpus in native_crash_places(job)]


def crashed_here(job,place):
    host,gpus=place
    return any(h==host and (not g or not gpus or tuple(g)==tuple(gpus)) for h,g in native_crash_places(job))


def evaluation_recovery_attempts(job):
    """One explicit evaluation recovery allowance; never renewed by a new node/hash."""
    receipt=job.get('evaluation_native_recovery')
    if (job.get('task',{}).get('kind')!='evaluate' or not isinstance(receipt,dict)
            or receipt.get('policy')!=EVAL_NATIVE_RECOVERY or receipt.get('max_failed_replays')!=1
            or receipt.get('task_sha256')!=digest(semantic_task(job['task']))):return None
    attempts=job.get('attempts',[]);matches=[i for i,a in enumerate(attempts) if a.get('id')==receipt.get('after_attempt')]
    if len(matches)!=1:return None
    index=matches[0];fault=fatal_native_failure(dict(attempts=[attempts[index]]))
    if not fault or fault['signal']!=11:return None
    return attempts[index+1:]


def native_failure_blocks(job,place=None):
    """Where a native crash stops a task: on the GPU set (host, GPUs) where it already crashed natively
    (place given), and everywhere once it crashed TASK_NATIVE_HOST_LIMIT times on different places. Without
    `place` this is the queue-wide answer: False means another node may still resume it from its checkpoint."""
    view=granted_view(job)
    if view is not None:   # after the user grant: never back to a crash place; the cap counts new crashes only
        return bool(place is not None and crashed_here(job,place)) or native_failure_blocks(view)
    fault=fatal_native_failure(job)
    if not fault:return False
    if place is not None and crashed_here(job,place):return True
    replay=evaluation_recovery_attempts(job)
    if replay is None:return native_crash_count(job)>=TASK_NATIVE_HOST_LIMIT
    # A replay crash, including one mislabelled as an interruption, never buys more work.
    return bool(fatal_native_failure(dict(attempts=replay))) or counted_failures(dict(attempts=budget_attempts(job,replay)))>=1


def native_signal(attempt):
    """The attempt's native crash signal as recorded, refunded or not."""
    rc=attempt.get('exit_code')
    if rc in (-4,-6,-7,-8,-11,132,134,135,136,139):
        return dict(signal=-rc if rc<0 else rc-128,attempt=attempt.get('id'))
    for failure in (attempt.get('child_failure') or {}).get('failures',[]):
        if isinstance(failure,dict) and failure.get('signal') in (4,6,7,8,11):
            return dict(signal=failure['signal'],attempt=attempt.get('id'))
    return None


def fatal_native_failure(job):
    """A native crash is not repaired by another identical torchrun invocation.
    Crashes refunded as periodic-dump SIGSEGVs (PERIODIC_DUMP_REFUND) are not counted."""
    for attempt in reversed(job.get('attempts',[])):
        if not isinstance(attempt,dict) or dump_refunded(attempt):continue
        fault=native_signal(attempt)
        if fault:return fault
    return None


def collective_timeout_evidence(root,attempt):
    """A SIGABRT from the frozen child's 1800 s NCCL wait limit (srgc_rebuttal/distributed.py), now raised by
    v8_task_entry.extend_collective_timeout. 11-17: all 15 stopped Qwen MBPP evals were a rank that finished its
    eval share early and aborted after exactly 1800000 ms in ALLGATHER while another rank still generated.
    Not a node or GPU fault: refunded like the periodic-dump crash once the task entry carries the fix."""
    try:fixed='extend_collective_timeout()' in Path(__file__).with_name('v8_task_entry.py').read_text()
    except OSError:fixed=False
    if not fixed or not attempt.get('id'):return None
    log=Path(root)/'logs'/(str(attempt['id'])+'.log');text=_tail(log,1048576)
    if 'Watchdog caught collective operation timeout' not in text or 'Timeout(ms)=1800000)' not in text:return None
    return dict(rule='nccl_1800s_collective_timeout_fixed_in_task_entry',log=str(log))


def periodic_dump_evidence(root,job,attempt):
    """Why this attempt's SIGSEGV came from the old 30-min periodic faulthandler dump, or None."""
    if not isinstance(attempt,dict) or dump_refunded(attempt):return None
    fault=native_signal(attempt)
    # Recorded as SIGABRT (-6) or, by torchrun, as exit 1 with the rank's SIGABRT in the log: read the log either way.
    if (fault and fault['signal']==6) or attempt.get('outcome')=='failed' or attempt.get('exit_code') not in (None,0,75):
        evidence=collective_timeout_evidence(root,attempt)
        if evidence is not None:return evidence
    if not fault or fault['signal']!=11:return None
    started,ended=attempt.get('started_at'),attempt.get('ended_at')
    if not all(isinstance(v,(int,float)) for v in (started,ended)) or ended-started<PERIODIC_DUMP_SECONDS-10:
        return None   # the dump first fires 30 min after the child starts
    mode=attempt.get('periodic_stack_dump')
    if mode=='armed':return dict(rule='task_entry_armed_periodic_dump',task_entry_sha256=attempt.get('task_entry_sha256'))
    if mode is None and attempt.get('controller_sha256') not in FIXED_DUMP_CONTROLLERS:
        return dict(rule='launched_by_controller_before_periodic_dump_fix',controller_sha256=attempt.get('controller_sha256'))
    if mode not in (None,'unknown'):return None   # 'cancelled' (no dump ran) or 'armed-by-env' (V8_PERIODIC_STACK_DUMP=1)
    task=job.get('task') if isinstance(job.get('task'),dict) else {}
    label=task.get('task_id') or task.get('run_id');stacks=Path(root)/'stacks'
    if not label or not stacks.is_dir():return None
    for path in sorted(stacks.glob(f'{label}-rank*.txt')):
        try:mtime=path.stat().st_mtime
        except OSError:continue
        if not started<=mtime<=ended+120:continue
        text=_tail(path,65536);at=text.rfind('Timeout (0:')
        if at>=0 and 'Fatal Python error' not in text[at:]:
            return dict(rule='rank_stack_file_ends_in_periodic_dump',stack_file=str(path))
    return None


def refund_periodic_dump(root,job,worker):
    """Mark this job's periodic-dump SIGSEGV attempts (receipt on the attempt, nothing removed). Returns their ids."""
    attempts=[a for a in job.get('attempts',[]) if isinstance(a,dict)]
    used=sum(dump_refunded(a) and not a['periodic_dump_refund'].get('rule','').startswith('nccl_') for a in attempts);marked=[]
    for attempt in attempts:
        evidence=periodic_dump_evidence(root,job,attempt)
        if evidence is None:continue
        # The 1800 s NCCL abort cannot recur once the entry raises the limit (the rule needs Timeout(ms)=1800000),
        # so those refunds have their own bound and never use up the periodic-dump budget.
        if evidence['rule'].startswith('nccl_'):
            if sum(1 for a in attempts if (a.get('periodic_dump_refund') or {}).get('rule','').startswith('nccl_'))>=COLLECTIVE_TIMEOUT_REFUND_LIMIT:continue
        elif used>=PERIODIC_DUMP_REFUND_LIMIT:continue
        timeout=evidence['rule'].startswith('nccl_')
        started,ended=attempt.get('started_at'),attempt.get('ended_at')
        elapsed=round(ended-started) if all(isinstance(v,(int,float)) for v in (started,ended)) else None
        attempt['periodic_dump_refund']=dict(policy=PERIODIC_DUMP_REFUND,worker=worker,at=time.time(),
            signal=6 if timeout else 11,elapsed_seconds=elapsed,**evidence,
            reason=('sigabrt_from_frozen_1800s_nccl_collective_timeout_now_raised' if timeout else
                    'sigsegv_from_30min_periodic_faulthandler_dump')+'; not counted as a task failure, crash place or node fault')
        used+=not timeout;marked.append(attempt.get('id'))
    return marked


def node_fault_count(row):
    """Native crashes recorded for a GPU set, without refunded periodic-dump crashes."""
    refunded=set(row.get('refunded_attempts') or ())
    return sum(1 for f in row.get('failures',[]) if f.get('attempt') not in refunded)


def node_native_identity(pool,runtime):
    """A new PID/worker name must not give the same GPU set another crash budget."""
    return dict(host=socket.gethostname(),gpus=list(getattr(runtime,'gpus',())),
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                binding_sha256=digest(pool.binding),controller_sha256=pool.controller_sha256)


def node_native_path(pool,runtime):
    return pool.root/'node-native-faults'/(digest(node_native_identity(pool,runtime))+'.json')


def record_node_native_fault(pool,runtime,attempt):
    """Append current native failures under the existing cohort control lease."""
    fault=fatal_native_failure(dict(attempts=[attempt]))
    if not fault:return
    path=node_native_path(pool,runtime)
    row=read(path) if path.exists() else dict(schema=1,identity=node_native_identity(pool,runtime),
        max_native_failures=NODE_NATIVE_FAILURE_LIMIT,failures=[])
    if not any(f.get('attempt')==attempt['id'] for f in row['failures']):
        row['failures'].append(dict(attempt=attempt['id'],worker=attempt.get('worker'),
            signal=fault['signal'],at=time.time(),log=str(pool.root/'logs'/(attempt['id']+'.log'))))
        row['quarantined']=node_fault_count(row)>=NODE_NATIVE_FAILURE_LIMIT
        atomic(path,row)


def node_native_hold(pool,runtime):
    path=node_native_path(pool,runtime)
    if not path.exists():return None
    row=read(path)
    if row.get('identity')!=node_native_identity(pool,runtime):raise NodeFailure('Node native-fault identity differs')
    if getattr(pool,'readmitted',False):
        if node_fault_count(row)<=pool.readmit_fault_baseline:return None
        return 'A new native crash followed successful GPU re-admission; stop this node instead of repeating recovery'
    if node_fault_count(row)>=NODE_NATIVE_FAILURE_LIMIT:
        return (f'This GPU set recorded {node_fault_count(row)} native crashes for this cohort/controller; '
                f'no further tasks on this GPU set. Other nodes/cohorts and saved checkpoints are retained. '
                f'Original evidence: {path}; underlying native cause remains unconfirmed')
    return None


def recheck_crashed_node(pool,runtime,device_fds,worker,diagnostic):
    """One fresh verified admission after a native fault; never renew task failure budgets."""
    if not hasattr(runtime,'fresh_admission'):return
    identity=node_native_identity(pool,runtime)
    path=pool.root/'node-native-reentry'/(digest(identity)+'.json')
    faults=node_native_path(pool,runtime)
    count=node_fault_count(read(faults)) if faults.exists() else 0
    previous=read(path) if path.exists() else None
    if previous and previous.get('state')=='passed' and previous.get('fault_count')==count:
        receipt=Path(previous['receipt'])
        if receipt.is_file() and hashlib.sha256(receipt.read_bytes()).hexdigest()==previous['receipt_sha256']:
            pool.readmitted=True;pool.readmit_fault_baseline=count;return
    if previous and previous.get('state')=='passed' and previous.get('fault_count')!=count:
        raise NodeFailure('A new native crash followed verified re-admission; stop repeated recovery; evidence: '+str(path))
    needed=count>=NODE_NATIVE_FAILURE_LIMIT
    for source in pool.root.glob('jobs/*.json'):
        job=read(source)
        if job.get('state')!='done' and crashed_here(job,pool.place()) and not native_failure_blocks(job) and not pool.retry_limit_reached(job):
            needed=True;break
    if not needed:return
    if previous and previous.get('state')=='passed':
        raise NodeFailure('A new native crash followed verified re-admission; stop repeated recovery; evidence: '+str(path))
    history=list((previous or {}).get('attempts',[]))
    if previous and not history:history=[dict(state=previous.get('state'),at=previous.get('at'))]
    if len(history)>=2:raise NodeFailure('GPU/NCCL re-admission reached its two-probe limit; evidence: '+str(path))
    lease=Lock(path.with_suffix('.lock'))
    if not lease.acquire():raise NodeFailure('GPU re-admission is already in progress on this GPU set')
    try:
        if (read(path) if path.exists() else None)!=previous:raise NodeFailure('GPU re-admission record changed on another worker')
        if read(pool.root/'binding.json')!=pool.binding:raise BindingMismatch('Queue identity changed before GPU re-admission')
        history.append(dict(state='checking',worker=worker,at=time.time()))
        record=dict(schema=1,identity=identity,state='checking',worker=worker,at=time.time(),fault_count=count,attempts=history)
        atomic(path,record)
        diagnostic.publish('CHECKING',pool.binding['cohort'],'rechecking four-GPU NCCL after a native crash; no task claimed')
        try:
            result=runtime.fresh_admission(device_fds)
            receipt=Path(result['receipt']);checked=read(receipt)
            if checked.get('state')!='passed':raise NodeFailure('Fresh GPU/NCCL admission did not pass')
            if read(pool.root/'binding.json')!=pool.binding:raise BindingMismatch('Queue identity changed during GPU re-admission')
            record.update(state='passed',receipt=str(receipt),receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),completed_at=time.time())
            record['attempts'][-1].update(state='passed',completed_at=record['completed_at'])
            atomic(path,record);pool.readmitted=True;pool.readmit_fault_baseline=count
            diagnostic.publish('RECOVERED',pool.binding['cohort'],'fresh four-GPU NCCL admission passed; existing task failure caps retained')
        except Exception as exc:
            record.update(state='failed',error=f'{type(exc).__name__}: {exc}',completed_at=time.time());atomic(path,record)
            record['attempts'][-1].update(state='failed',error=record['error']);atomic(path,record)
            raise NodeFailure('GPU re-admission failed; no training task was claimed: '+str(exc)) from exc
    finally:lease.close()


def aggregate_task_failure(failure):
    return (isinstance(failure,dict) and failure.get('task') is None and
            bool(re.match(r'^\d+ task\(s\) failed; no independent ready/running work remains\.',str(failure.get('detail','')))))


def forced_termination(attempt,rc,log):
    """Positive signal evidence only; original Python/CUDA faults retain failure priority."""
    allowed={1,2,9,15}
    failures=(attempt.get('child_failure') or {}).get('failures') or []
    if failures:
        if not all(isinstance(f,dict) and f.get('signal') in allowed and not f.get('original_error')
                   and not f.get('traceback') for f in failures):return None
        signals=sorted({f['signal'] for f in failures})
    elif type(rc) is int and rc in (-1,-2,-9,-15,129,130,137,143):
        signals=[-rc if rc<0 else rc-128]
    else:return None
    try:
        with Path(log).open('rb') as handle:
            handle.seek(max(0,Path(log).stat().st_size-131072))
            lines=handle.read().decode(errors='replace').splitlines()
        for line in lines:
            plain=re.sub(r'^\[rank\d+\]:\s*','',line.strip())
            if re.match(r'^(?:[\w.]+(?:Error|Exception)|SystemExit):',plain) and 'ChildFailedError' not in plain.split(':',1)[0]:
                return None
    except OSError:pass
    return dict(signals=signals,cause='forced_signal_cause_unconfirmed',evidence='process exit or signal-only rank report')


CRASH_RECORD='crash-fields-v1'
_FRAME=re.compile(r'^\s+File "(?P<file>[^"]+)", line (?P<line>\d+) in (?P<func>.+?)\s*$')
_C_FRAME=re.compile(r'^\s+(?:Binary file "(?P<binary>[^"]+)"|(?P<cfile>\S+\.so[.\d]*))(?:, at (?P<at>.+?))?\s*$')
CRASH_CLASSES=(  # (class, signals or None for any, log text); first match wins
    ('nccl_watchdog_abort',(6,),('Watchdog caught collective operation timeout','ProcessGroupNCCL\'s watchdog',
                                  'NCCL watchdog','Some NCCL operations have failed or timed out')),
    ('cuda_error',None,('CUDA error:','cudaErrorIllegalAddress','an illegal memory access','CUBLAS_STATUS_','CUDNN_STATUS_')),
    ('gpu_out_of_memory',None,('CUDA out of memory','OutOfMemoryError')),
    ('nccl_error',None,('NCCL error','ncclUnhandledCudaError','ncclSystemError','ncclInternalError')),
    ('shared_memory_bus_error',(7,),('/dev/shm','shm_open','Bus error')),
    ('cpp_exception_abort',(6,),('terminate called after throwing','std::bad_alloc','c10::Error')),
    ('host_killed',(9,),('',)),
)


def crash_frames(text,limit=12):
    """Top frames of the crashing thread from a faulthandler dump: 'func (file:line)' (Python) and
    'binary at symbol' (C stack, Python 3.14+), most recent call first."""
    lines=str(text or '').splitlines()
    start=next((i for i,l in enumerate(lines) if l.startswith('Current thread')),None)
    if start is None:start=next((i for i,l in enumerate(lines) if l.startswith('Thread ')),None)
    if start is None:start=next((i for i,l in enumerate(lines) if 'Fatal Python error' in l),0)
    frames=[]
    for line in lines[start+1:]:
        if frames and (line.startswith('Thread ') or line.startswith('Current thread') or line.startswith('Extension modules')):break
        match=_FRAME.match(line)
        if match:frames.append(f"{match['func']} ({match['file']}:{match['line']})")
        elif 'C stack' in line or 'Binary file' in line:
            c=_C_FRAME.match(line)
            if c and (c['binary'] or c['cfile']):frames.append(f"[C] {Path(c['binary'] or c['cfile']).name}"+(f" at {c['at']}" if c['at'] else ''))
        if len(frames)>=limit:break
    return frames


def _tail(path,size=131072):
    try:
        with Path(path).open('rb') as handle:
            handle.seek(max(0,Path(path).stat().st_size-size))
            return handle.read().decode(errors='replace')
    except OSError:return ''


def crash_fields(root,attempt,rc,task=None):
    """exit code/signal, stage, top native frames and a cause class for one failed attempt.
    Reads only this attempt's files: child-failure summary, rank progress, fault dumps, log tail, stacks."""
    root=Path(root);aid=attempt.get('id') or '';progress=root/'progress'/aid
    failures=[f for f in (attempt.get('child_failure') or {}).get('failures',[]) if isinstance(f,dict)]
    fault=fatal_native_failure(dict(attempts=[attempt]))
    signal_number=fault['signal'] if fault else (-rc if type(rc) is int and rc<0 else rc-128 if type(rc) is int and 128<rc<160 else None)
    if signal_number is None:signal_number=next((f['signal'] for f in failures if f.get('signal')),None)
    crashed=next((f for f in failures if f.get('signal')==signal_number and f.get('rank') is not None),None) or \
            next((f for f in failures if f.get('rank') is not None),None)
    stage=dict(stage=None,rank=None,source=None)
    if crashed and isinstance(crashed.get('last_progress'),dict):
        stage=dict(stage=crashed['last_progress'].get('stage'),rank=crashed.get('rank'),source='crashed rank')
    else:
        newest=None
        for path in sorted(progress.glob('rank-*.json')) if progress.is_dir() else ():
            if not re.fullmatch(r'rank-\d+\.json',path.name):continue
            try:event=read(path)
            except (OSError,ValueError):continue
            if isinstance(event,dict) and (newest is None or (event.get('updated') or 0)>(newest[1].get('updated') or 0)):
                newest=(path,event)
        if newest:stage=dict(stage=newest[1].get('stage'),rank=int(newest[0].stem.split('-')[1]),source='latest rank record')
    stacks=[];sources=[]
    if crashed and crashed.get('native_python_stack'):
        stacks.append(crashed['native_python_stack']);sources.append(crashed.get('native_fault_log'))
    for path in sorted(progress.glob('native-fault-rank-*.log')) if progress.is_dir() else ():
        if str(path) not in sources:
            text=_tail(path,65536)
            if 'Fatal Python error' in text:stacks.append(text);sources.append(str(path))
    log=root/'logs'/(aid+'.log');log_text=_tail(log)
    if 'Fatal Python error' in log_text:
        stacks.append(log_text[log_text.rfind('Fatal Python error'):]);sources.append(str(log))
    label=(task or {}).get('task_id') or (task or {}).get('run_id')
    started=attempt.get('started_at') or 0
    for path in sorted((root/'stacks').glob(f'{label}-rank*.txt')) if label and (root/'stacks').is_dir() else ():
        try:
            if path.stat().st_mtime<started:continue
        except OSError:continue
        text=_tail(path,65536)
        if 'Fatal Python error' in text:
            stacks.append(text[text.rfind('Fatal Python error'):]);sources.append(str(path))
    frames=next((f for f in (crash_frames(t) for t in stacks) if f),[])
    crash_class=None
    for name,signals,needles in CRASH_CLASSES:
        if signals is not None and signal_number not in signals:continue
        if any(n and n in log_text for n in needles) or (signals and needles==('',)):crash_class=name;break
    if crash_class is None and signal_number:crash_class='native_crash_unclassified'
    notable=[l.strip()[:300] for l in log_text.splitlines()
             if re.search(r'Fatal Python error|Segmentation fault|Aborted|terminate called|CUDA error|NCCL (?:WARN|error)|Xid|'
                          r'watchdog|illegal memory|out of memory|exitcode\s*:\s*-\d+|ChildFailedError|Bus error',l,re.I)][-12:]
    try:signal_name=signal.Signals(signal_number).name if signal_number else None
    except ValueError:signal_name=f'signal {signal_number}'
    return dict(record=CRASH_RECORD,signal=signal_number,signal_name=signal_name,stage=stage['stage'],stage_rank=stage['rank'],
                stage_source=stage['source'],crashed_rank=crashed.get('rank') if crashed else None,
                crash_frames=frames,crash_class=crash_class,crash_log_lines=notable,
                crash_evidence=[s for s in dict.fromkeys(sources) if s])


def crash_line(attempt):
    env=attempt.get('env') or {};torch=(env.get('torch') or {})
    return (f"CRASH exit={attempt.get('exit_code')} {attempt.get('signal_name') or ''} host={attempt.get('host')} "
            f"gpus={','.join(attempt.get('gpus') or [])} stage={attempt.get('stage') or 'unrecorded'} "
            f"class={attempt.get('crash_class')} python={env.get('executable') or '?'} torch={torch.get('version') or '?'}"
            f"/cuda{torch.get('cuda') or '?'} transformers={(env.get('packages') or {}).get('transformers') or '?'} "
            f"frames: {' <- '.join((attempt.get('crash_frames') or [])[:4]) or 'none recorded'}")


def primary_results_ready(cohort_root):
    """Small immutable receipts only; never scan model/checkpoint bodies in the claim gate."""
    root=Path(cohort_root);queue=root/'.multi-node'
    try:
        state=read(queue/'state.json');path=queue/'primary-results.json'
        payload=path.read_bytes();receipt=json.loads(payload)
        return (state.get('primary_results_receipt_sha256')==hashlib.sha256(payload).hexdigest()
                and receipt.get('policy')==PRIMARY_ORDER and receipt.get('root')==str(root.resolve())
                and receipt.get('binding_sha256')==digest(read(queue/'binding.json'))
                and receipt.get('budget_sha256')==hashlib.sha256((root/'frozen/budget.json').read_bytes()).hexdigest()
                and all((root/receipt['analysis_dir']/name).is_file()
                        for name in ('analysis.json','policy_outcomes_bank.json','report.md')))
    except (OSError,ValueError,KeyError,TypeError):return False


def primary_evaluation_tasks(runtime):
    """Exactly the original matched-horizon checkpoints; extensions are evaluated later."""
    training,sealed=runtime.plan('train')
    if not sealed or not training or not all(runtime.done(t) for t in training):
        raise ValueError('Matched-horizon training is incomplete; primary results are not ready')
    tasks=[];loc=runtime.locator();H=runtime.cfg['training']['matched_update_endpoint']
    for seed in runtime.cfg['seeds']['training']:
        data=runtime.cluster.load_bundle(loc.loc['seeds'][str(seed)]['input_bundle'])
        runs=[]
        for path in sorted((runtime.root/'runs').glob('*/manifest.json')):
            record=read(path)
            if (record.get('training_seed'),record.get('cohort_id'),record.get('role'))!=(seed,runtime.cohort,'rebuttal_bank'):continue
            if record.get('task',{}).get('method',{}).get('extension_of'):continue
            runs.append(path.parent.name)
        for rid in runs:
            found=runtime.mx.eval_tasks(runtime.cfg,data,runtime.root,[rid],split='final',updates={H})
            if not found:raise ValueError('Matched-horizon checkpoint missing: '+rid+'@'+str(H))
            tasks+=found
    if not tasks:raise ValueError('No matched-horizon evaluations; refusing to publish empty primary results')
    return tasks,True


def matched_ready_evaluations(runtime):
    """Only published matched-horizon checkpoints; no other seed or budget is needed to score them."""
    if not list((runtime.root/'runs').glob('*/snapshots.jsonl')):return []
    loc=runtime.locator();tasks=[];H=runtime.cfg['training']['matched_update_endpoint']
    for seed in runtime.cfg['seeds']['training']:
        runs=[]
        for path in sorted((runtime.root/'runs').glob('*/manifest.json')):
            row=read(path)
            if (row.get('cohort_id'),row.get('role'),row.get('training_seed'))!=(runtime.cohort,'rebuttal_bank',seed):continue
            if row.get('task',{}).get('method',{}).get('extension_of'):continue
            index=path.parent/'snapshots.jsonl'
            if not index.exists():continue
            published=[json.loads(line) for line in index.read_text().splitlines() if line.strip()]
            if not any(s.get('completed_update')==H and (path.parent/s['path']).is_file() for s in published):continue
            runs.append(path.parent.name)
        if not runs:continue
        data=runtime.cluster.load_bundle(loc.loc['seeds'][str(seed)]['input_bundle'])
        tasks.extend(t for t in runtime.mx.eval_tasks(runtime.cfg,data,runtime.root,runs,split='final',updates={H})
                     if Path(t['snapshot_path']).is_file())
    return tasks


def precommit_budget_rule(runtime):
    """Fix the exact cost-only formula and all seeds before permitting any early evaluation."""
    from switching import freeze
    path=runtime.root/'.multi-node/budget-rule.json'
    source=hashlib.sha256(Path(freeze.__file__).read_bytes()).hexdigest()
    expected=runtime.binding.get('runner_sha256',{}).get('switching/freeze.py')
    if expected and expected!=source:raise ValueError('Frozen budget implementation differs from this binding')
    identity=runtime.root/'.multi-node/input-identity.json'
    rule=dict(schema=1,policy=BUDGET_RULE,root=str(runtime.root.resolve()),binding_sha256=digest(runtime.binding),
              config_sha256=digest(runtime.cfg),seeds=list(runtime.cfg['seeds']['training']),
              horizon=runtime.cfg['training']['matched_update_endpoint'],reference_policy='srgc_current',
              freeze_implementation_sha256=source,input_identity_sha256=hashlib.sha256(identity.read_bytes()).hexdigest() if identity.exists() else None,
              reads_evaluation_results=False,requires_all_reference_endpoints=True)
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='committed_at'}!=rule:
            raise ValueError('Budget precommit identity/config/rule changed; original receipt retained')
    else:
        if list((runtime.root/'evaluations').glob('*.json')):
            raise ValueError('Cannot precommit budget rule after untracked evaluation results already exist')
        atomic(path,dict(rule,committed_at=time.time()))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ready_side_tasks(runtime):
    """Original cost-only budget and exact planners; never wait for unrelated arms/cohorts."""
    budget_path=runtime.root/'frozen/budget.json';seeds=runtime.cfg['seeds']['training']
    H=runtime.cfg['training']['matched_update_endpoint']
    rule_path=runtime.root/'.multi-node/budget-rule.json';rule_sha=None
    def waiting(reason,error=None):
        evaluations=matched_ready_evaluations(runtime)
        rule=precommit_budget_rule(runtime) if evaluations or rule_path.exists() else None
        result=dict(tasks=[dict(phase='evaluation',task=t) for t in evaluations],budget_ready=False,reason=reason,
                    evaluation_independent_of_budget=True,ready_matched_evaluations=len(evaluations),budget_rule_sha256=rule)
        if error:result['budget_error']=error
        return result
    if not budget_path.exists():
        if rule_path.exists():rule_sha=precommit_budget_rule(runtime)
        elif list((runtime.root/'evaluations').glob('*.json')):
            raise ValueError('Evaluation records exist before budget freeze; cannot certify cost-only selection')
        complete=set()
        for path in (runtime.root/'runs').glob('*/manifest.json'):
            row=read(path)
            if (row.get('cohort_id')==runtime.cohort and row.get('role')=='rebuttal_bank'
                    and row.get('status')=='completed_horizon' and row.get('completed_updates',-1)>=H):
                complete.add(row.get('training_seed'))
        if not set(seeds)<=complete:return waiting('reference_policy_costs_incomplete')
        from switching.bank import SeedBank
        from switching.freeze import freeze_budget,write_frozen_budget
        rows=[]
        for seed in seeds:
            rows+=SeedBank(runtime.root,runtime.cfg,runtime.cohort,seed,'rebuttal_bank',evals={},
                           prefix_cost=runtime.cluster.prefix_cost_from_receipts(runtime.root,seed)).policy_rows()
        reference=[r for r in rows if r['policy_id']=='srgc_current']
        if (set(seeds)!={r['seed'] for r in reference}
                or any(r.get('status') not in ('triggered','not_triggered')
                       or r.get('continuation_gpu_seconds') is None for r in reference)):
            return waiting('reference_policy_costs_incomplete')
        for row in reference:
            path=runtime.root/'runs'/row['run_id']/'manifest.json'
            if not path.is_file():return waiting('reference_endpoint_unfinished')
            endpoint=read(path)
            if (endpoint.get('cohort_id'),endpoint.get('role'),endpoint.get('training_seed'))!=(runtime.cohort,'rebuttal_bank',row['seed']):
                raise ValueError('Reference policy endpoint identity differs from this cohort/seed')
            if endpoint.get('status') not in TERMINAL or endpoint.get('completed_updates',-1)<H:
                return waiting('reference_endpoint_unfinished')
        arguments=dict(files_read=[str(runtime.root/'runs')])
        # Earlier pinned runners lack this optional keyword. The controller above
        # already requires every configured seed and its complete selected endpoint.
        if 'expected_seeds' in inspect.signature(freeze_budget).parameters:arguments['expected_seeds']=seeds
        try:budget=freeze_budget(rows,**arguments)
        except ValueError as exc:return waiting('budget_calculation_error',str(exc))
        budget.update(gpu_type=runtime.cfg['runtime']['gpu_type'],gpu_count=runtime.cfg['runtime']['world_size'])
        if rule_sha:
            budget.update(selection_protocol=BUDGET_RULE,budget_rule_receipt_sha256=rule_sha,
                          evaluation_records_present_at_numeric_freeze=bool(list((runtime.root/'evaluations').glob('*.json'))),
                          formula_precommitted_before_evaluation=True,numeric_budget_timing='after_all_reference_costs',
                          legacy_rule_timing_text_preserved=True,
                          rule_timing_note='Saved rule text describes legacy order; actual timing follows selection_protocol and the immutable precommit receipt.')
        write_frozen_budget(budget_path,budget)
        print(f'[v8] {runtime.cohort}: all reference costs complete; original cost-only budget frozen; precommit={rule_sha or "legacy-order"}',flush=True)
    loc=runtime.locator();budget=read(budget_path)['value']
    errors={}
    try:extension,notes=runtime.mx.build_stage('budget-extension',runtime.cfg,loc,runtime.root,seeds=seeds,budget=budget)
    except (ValueError,json.JSONDecodeError) as exc:
        extension,notes=[],[];errors['extension']=f'{type(exc).__name__}: {exc}'
    try:evaluations,_=Runtime.plan(runtime,'evaluation')
    except (ValueError,json.JSONDecodeError) as exc:
        errors['evaluation']=f'{type(exc).__name__}: {exc}'
        evaluations=matched_ready_evaluations(runtime)
    result=dict(tasks=[dict(phase='extension',task=t) for t in extension]+
                      [dict(phase='evaluation',task=t) for t in evaluations],
                budget_ready=True,budget_sha256=hashlib.sha256(budget_path.read_bytes()).hexdigest(),waiting_parents=notes,
                planning_errors=errors,evaluation_independent_of_budget=True)
    return result


def freeze_precommitted_budget(runtime):
    """Keep the saved calculator; its CLI's legacy timing guard cannot describe early-eval scheduling."""
    precommit_budget_rule(runtime)
    state=ready_side_tasks(runtime)
    if not state.get('budget_ready'):
        raise ValueError('Reference-policy cost budget is not ready; early checkpoint evaluations are retained: '+
                         str(state.get('budget_error') or state.get('reason')))


def capture_primary_results(runtime):
    """Preserve the first complete matched-update report independently of later extension reports."""
    tasks,_=primary_evaluation_tasks(runtime)
    if not all(runtime.done(t) for t in tasks):raise ValueError('Matched-horizon evaluations are incomplete')
    H=runtime.cfg['training']['matched_update_endpoint'];analysis=runtime.root/'analysis'
    target=analysis/f'matched-update-{H}';receipt_path=target/'receipt.json'
    if target.exists():
        receipt=read(receipt_path)
        if receipt.get('binding_sha256')!=digest(runtime.binding):raise ValueError('Primary result identity changed')
        if receipt.get('budget_sha256')!=hashlib.sha256((runtime.root/'frozen/budget.json').read_bytes()).hexdigest():
            raise ValueError('Frozen budget differs from saved primary results')
        for name,sha in receipt['files'].items():
            if hashlib.sha256((target/name).read_bytes()).hexdigest()!=sha:raise ValueError('Saved primary result changed: '+name)
    else:
        for name in ('analysis.json','policy_outcomes_bank.json','report.md'):
            if not (analysis/name).is_file():raise ValueError('Primary analysis output missing: '+name)
        stage=analysis/('.primary-'+uuid.uuid4().hex);stage.mkdir()
        try:
            files={}
            for path in sorted(analysis.iterdir()):
                if not path.is_file():continue
                content=path.read_bytes();(stage/path.name).write_bytes(content)
                files[path.name]=hashlib.sha256(content).hexdigest()
            receipt=dict(schema=1,policy=PRIMARY_ORDER,root=str(runtime.root.resolve()),horizon=H,
                         binding_sha256=digest(runtime.binding),analysis_dir=str(target.relative_to(runtime.root)),
                         budget_sha256=hashlib.sha256((runtime.root/'frozen/budget.json').read_bytes()).hexdigest(),
                         evaluation_tasks=[task_key(t) for t in tasks],files=files,recorded_at=time.time())
            atomic(stage/'receipt.json',receipt);os.replace(stage,target)
        finally:
            if stage.exists():shutil.rmtree(stage)
    atomic(runtime.root/'.multi-node/primary-results.json',receipt)


class QueueReset(Exception):
    """--new-queue replaced the queue directory; the caller must re-acquire control.lock in the new one."""


def mismatch_help(cohort='<cohort>'):
    return ('This queue was started with a different bundle. Either keep using that bundle on every node, or stop '
            f'the workers on ALL nodes and run `sh scripts/run_v8_nodes.sh {cohort} --new-queue` once on one node (allowed '
            'only while no update has completed; without a cohort word --new-queue applies to every cohort).')


MISMATCH_HELP=mismatch_help()


def completed_training(cohort_root):
    """True once any v8 run of this cohort has durably recorded a completed update."""
    runs=Path(cohort_root)/'runs'
    if not runs.is_dir():return False
    for log in runs.rglob('updates.jsonl'):
        try:
            if log.stat().st_size>0:return True
        except OSError:return True
    return False


def binding_changes(old,new):
    changed=[]
    for key in sorted(set(old)|set(new)):
        a,b=old.get(key),new.get(key)
        if a==b:continue
        if isinstance(a,dict) and isinstance(b,dict):
            changed+=[f'{key}:{k}' for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
        else:changed.append(key)
    return changed


def semantic_task(task):
    return {k:v for k,v in task.items() if k not in ('stage','deps','aliases')}


def task_key(task):
    if task['kind']=='train':return digest(['train',task['run_id']])
    return digest(['evaluate',task['snapshot_sha256'],task['evaluation_spec_hash']])


class Pool:
    """All queue writes occur under control.lock; jobs have independent leases."""
    def __init__(self,root,binding,*,new_queue=False,retry_failed=False,max_attempts=MAX_ATTEMPTS,retry_delay=RETRY_DELAY):
        self.root=Path(root);self.binding=binding;self.new_queue=new_queue
        self.controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.retry_failed=retry_failed;self.max_attempts=max_attempts;self.retry_delay=retry_delay
        self.automatic_retry=False  # enabled by the node loop only after normal node admission
        self.host=socket.gethostname()  # a native-crash retry never returns to the GPU set where the task crashed
        self.gpus=()                    # set by the worker loop from its runtime (node_native_identity's GPU set)
        # Recorded on every attempt: whether this checkout's task entry still lets the pinned child's periodic dump run.
        try:
            entry=Path(__file__).with_name('v8_task_entry.py').read_bytes()
            self.task_entry_sha256=hashlib.sha256(entry).hexdigest()
            self.periodic_stack_dump=('armed' if b'cancel_dump_traceback_later' not in entry else
                                      'armed-by-env' if os.environ.get('V8_PERIODIC_STACK_DUMP')=='1' else 'cancelled')
        except OSError:self.task_entry_sha256=None;self.periodic_stack_dump='unknown'
        self.refund_checked_at=-float('inf');self.refund_cleared=set()
        self.root.mkdir(parents=True,exist_ok=True)

    def retry_requested(self,state=None):
        """Task restarts admitted on this node; keep the existing shared retry marker."""
        return self.automatic_retry or self.retry_failed or bool((state or {}).get('retry_failed_requested_at'))

    def metadata_wait(self,error):
        """Only a known append-only file may be transient, for at most 30 unchanged seconds."""
        path=metadata_error_path(error)
        active=bool(self.held_leases())
        detail=f'file={path or "unidentified"}; {error}'
        if not active:raise ValueError('Invalid run metadata with no active writer; '+detail) from error
        if not path or Path(path).suffix!='.jsonl':
            raise ValueError('Invalid fixed JSON; an unrelated running task cannot repair it; '+detail) from error
        try:Path(path).stat();signature=(path,error.lineno)
        except OSError:raise ValueError('Unreadable append-only metadata; '+detail) from error
        now=time.monotonic();previous=getattr(self,'metadata_error_wait',None)
        if previous is None or previous[0]!=signature:self.metadata_error_wait=(signature,now)
        elif now-previous[1]>=30:
            raise ValueError('The same JSONL line has remained invalid for 30 seconds despite a live lease; '+detail) from error
        self.observation=dict(checked_at=time.time(),phase=self.queue_state()['phase'],metadata_error=detail,
                              metadata_wait_seconds=now-self.metadata_error_wait[1])
        return 'waiting',None

    def place(self):
        return (getattr(self,'host',None) or socket.gethostname(),tuple(str(g) for g in getattr(self,'gpus',()) or ()))

    def retry_limit_reached(self,job):
        if granted_view(job) is not None:return retry_budget_failures(job)>=failure_limit(self.max_attempts)
        if evaluation_recovery_attempts(job) is not None:return retry_budget_failures(job)>=1
        return retry_budget_failures(job)>=failure_limit(self.max_attempts)

    def retryable(self,job,now=None,state=None):
        """One immediate retry by default; repeated faults stay excluded across joins."""
        receipt=job.get('evaluation_native_recovery')
        if receipt and receipt.get('binding_sha256')!=digest(self.binding):return False
        place=None if getattr(self,'readmitted',False) else self.place()
        if native_failure_blocks(job,place) or self.retry_limit_reached(job):return False
        if job.get('state')=='interrupted' and job.get('resume_not_before'):
            return self.retry_delay<=0 or (time.time() if now is None else now)>=job['resume_not_before']
        if job.get('state')!='failed':return True
        if not self.retry_requested(state) or self.retry_limit_reached(job):return False
        if self.retry_delay<=0:return True
        attempts=job.get('attempts',[]);last=attempts[-1] if attempts else {}
        ended=last.get('ended_at') or last.get('started_at') or 0
        return (time.time() if now is None else now)>=ended+self.retry_delay

    def retry_later(self,job,state=None):
        if native_failure_blocks(job) or self.retry_limit_reached(job):return False
        if job.get('state')=='interrupted' and job.get('resume_not_before'):
            return self.retry_delay>0 and time.time()<job['resume_not_before']
        return job.get('state')=='failed' and self.retry_requested(state) and not self.retry_limit_reached(job)

    def held_leases(self):
        """Jobs whose lease a live worker or its training child still holds."""
        held=[]
        for path in sorted((self.root/'leases').glob('*.lock')):
            lease=Lock(path)
            try:free=lease.acquire()
            except NodeFailure:free=False  # unreadable here: never assume a free slot
            if free:lease.close()
            else:held.append(path.stem)
        return held

    def supersede(self):
        """Explicit --new-queue: keep the old queue untouched under a new name and start a fresh one.
        Called with this queue's control.lock held; raises QueueReset so the caller re-locks the new directory."""
        self.new_queue=False
        cohort=self.binding.get('cohort','?')
        if completed_training(self.root.parent):
            raise BindingMismatch(f'{cohort}: --new-queue refused: updates were already completed in {self.root.parent/"runs"}; '
                                  'this queue must be continued with the bundle that started it.')
        held=self.held_leases()
        if held:
            raise BindingMismatch(f'{cohort}: --new-queue refused: {len(held)} job lease(s) are still held by a running '
                                  'worker or training child. Stop the workers on ALL nodes first.')
        kept=self.root.with_name(self.root.name+'.superseded-'+time.strftime('%Y%m%dT%H%M%S'))
        while kept.exists():kept=kept.with_name(kept.name+'x')
        os.replace(self.root,kept)
        # A still-running worker of another node may recreate the folder (heartbeat, control.lock) right after
        # the rename. Take the fresh folder's own control.lock before writing its identity record.
        self.root.mkdir(parents=True,exist_ok=True)
        fresh=Lock(self.root/'control.lock');deadline=time.monotonic()+30
        while not fresh.acquire():
            if time.monotonic()>deadline:
                raise BindingMismatch(f'{cohort}: --new-queue moved the old queue to {kept}, but another running worker keeps '
                                      f'the fresh queue locked. Stop the workers on ALL nodes, then run the plain command.')
            time.sleep(.05)
        try:
            if list(self.root.glob('jobs/*.json')):
                raise BindingMismatch(f'{cohort}: another running worker started jobs in the fresh queue {self.root} during '
                                      f'--new-queue (old queue kept as {kept}). Stop the workers on ALL nodes first.')
            atomic(self.root/'binding.json',self.binding)
        finally:fresh.close()
        print(f'[v8] {cohort}: --new-queue kept the previous queue unchanged as {kept} and started a fresh queue in '
              f'{self.root}. Restart every other node with this same bundle (plain command, without --new-queue).',flush=True)
        raise QueueReset(str(kept))

    @contextmanager
    def control(self):
        lock=Lock(self.root/'control.lock')
        try:
            yield lock if lock.acquire() else None
        finally:lock.close()

    def initialize(self,retry=False,worker=None,*,recover=None):
        path=self.root/'binding.json'
        if self.new_queue and path.exists():
            if read(path)==self.binding:
                print(f"[v8] {self.binding.get('cohort')}: the queue already uses this bundle; --new-queue not needed, continuing it.",flush=True)
            else:self.supersede()
        self.new_queue=False
        if path.exists():
            if read(path)!=self.binding:
                old=read(path)
                state_path=self.root/'state.json'
                state=read(state_path) if state_path.exists() else {}
                stable=('schema','cohort','root','config_sha256','trainer_sha256','world_size','python_version','packages')
                if (state.get('phase')=='prepare' and state.get('failure')
                        and not list(self.root.glob('jobs/*.json'))
                        and not (self.root/'input-identity.json').exists()
                        and all(old.get(k)==self.binding.get(k) for k in stable)):
                    atomic(self.root/'setup-history'/(digest(old)+'.json'),dict(binding=old,state=state))
                    state['failure_history'].append(state['failure']);state['failure']=None
                    state['actions'].append(dict(action='retry_failed_setup_with_verified_source',at=time.time()))
                    atomic(path,self.binding);atomic(state_path,state)
                else:
                    # An idle old worker may still use this queue even with no held job lease or
                    # completed update. Moving control.lock would also split its locking domain.
                    # Keep the original binding and inode; supervisor/stop/probe remain usable.
                    raise BindingMismatch(
                        f"{self.binding.get('cohort')}: {mismatch_help(self.binding.get('cohort','<cohort>'))} (Worker/runner/config/trainer identity differs "
                        f"from this queue: {', '.join(binding_changes(old,self.binding)[:12])}; record: {path})")
        else:atomic(path,self.binding)
        state_path=self.root/'state.json'
        if not state_path.exists():atomic(state_path,dict(phase='prepare',failure=None,actions=[],failure_history=[]))
        state=read(state_path)
        for job_path in self.root.glob('jobs/*.json'):
            job=read(job_path);corrected=misclassified_stops(job)
            if not corrected or (job.get('stop_accounting_correction') or {}).get('attempts')==corrected:continue
            lease=Lock(self.root/'leases'/(job_path.stem+'.lock'))
            if not lease.acquire():continue
            try:
                job=read(job_path);corrected=misclassified_stops(job)
                if not corrected:continue
                job['stop_accounting_correction']=dict(policy=DISPATCH_POLICY,attempts=corrected,worker=worker,at=time.time(),
                    reason='exit75 after an older native crash was incorrectly counted as a new failure; raw attempts retained')
                atomic(job_path,job)
                print(f'[v8] REFUNDED_EXIT75 {job_path.stem}: {len(corrected)} old stop(s) were miscounted; '
                      'original failures, attempts and checkpoints retained',flush=True)
            finally:lease.close()
        failure=state.get('failure')
        # Original SRGC keeps a failed task separate from independent ready work.
        # Older V8 finish() promoted any single task exit into a cohort-wide stop.
        # Preserve that incident and the failed job; do not retry or reset either.
        if (isinstance(failure,dict) and re.fullmatch(r'[0-9a-f]{64}',str(failure.get('task','')))
                and str(failure.get('detail','')).startswith('task exit=')):
            failed_path=self.root/'jobs'/(str(failure['task'])+'.json')
            if failed_path.exists() and read(failed_path).get('state')=='failed':
                state.setdefault('failure_history',[]).append(failure)
                state.setdefault('task_failures',[]).append(failure)
                state.setdefault('actions',[]).append(dict(action='isolate_legacy_task_failure',task=failure['task'],at=time.time()))
                state['failure']=None;atomic(state_path,state)
                print(f'[v8] {self.binding.get("cohort")}: kept failed task {failure["task"]}; '
                      'independent ready jobs may continue. The failed task was not retried.',flush=True)
        self.refund_periodic_dump_crashes(worker,force=True)
        if recover:
            self.recover_budget_api_failure(recover,worker)
            self.recover_failed(recover,worker)
            self.grant_user_retry(worker)
            self.grant_watchdog_fix_retry(worker)
        if self.automatic_retry:self.enable_automatic_retry(worker)
        state=read(state_path)
        if retry:
            # Queue-level failures (CPU stage, failed tasks blocking all work) are cleared here. Failed tasks keep
            # their attempt history; tick() retries them under the original SRGC --max-attempts/--retry-delay rules.
            self.retry_failed=True;state['retry_failed_requested_at']=time.time()
            if state['failure']:
                state['failure_history'].append(state['failure']);state['failure']=None
            atomic(state_path,state)

    def refund_periodic_dump_crashes(self,worker,force=False):
        """PERIODIC_DUMP_REFUND, called under control.lock at join and every REFUND_CHECK_SECONDS: periodic-dump
        SIGSEGVs recorded by any node stop counting; a task they blocked reopens from its checkpoint."""
        now=time.monotonic()
        if not force and now-getattr(self,'refund_checked_at',-float('inf'))<REFUND_CHECK_SECONDS:return []
        self.refund_checked_at=now
        if not isinstance(getattr(self,'refund_cleared',None),set):self.refund_cleared=set()
        reopened=[];refunded=0;ids=set()
        for path in sorted(self.root.glob('jobs/*.json')):
            try:job=read(path)
            except (OSError,ValueError):continue
            if not isinstance(job,dict):continue
            attempts=[a for a in job.get('attempts',[]) if isinstance(a,dict)]
            ids.update(a.get('id') for a in attempts if dump_refunded(a))
            fresh=[a for a in attempts if not dump_refunded(a) and a.get('id') not in self.refund_cleared]
            if not any(periodic_dump_evidence(self.root,job,a) for a in fresh):
                # A finished attempt's evidence never changes: do not read its stack files again.
                self.refund_cleared.update(a.get('id') for a in fresh if a.get('ended_at') is not None and a.get('id'))
                continue
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():continue   # running: refunded at a later check
            try:
                job=read(path)
                blocked=native_failure_blocks(job) or self.retry_limit_reached(job)
                failures=retry_budget_failures(job);places=native_crash_hosts(job)
                marked=refund_periodic_dump(self.root,job,worker)
                if not marked:continue
                ids.update(marked);refunded+=len(marked)
                job.setdefault('periodic_dump_refunds',[]).append(dict(policy=PERIODIC_DUMP_REFUND,attempts=marked,
                    failures_before=failures,failures_after=retry_budget_failures(job),crash_places_before=places,
                    crash_places_after=native_crash_hosts(job),worker=worker,at=time.time()))
                if blocked and not (native_failure_blocks(job) or self.retry_limit_reached(job)) and job.get('state')=='failed':
                    job['state']='pending';reopened.append(path.stem)
                atomic(path,job)
            finally:lease.close()
        for path in sorted(self.root.glob('node-native-faults/*.json')):
            try:row=read(path)
            except (OSError,ValueError):continue
            mine=sorted(ids.intersection(f.get('attempt') for f in row.get('failures',[]) if isinstance(f,dict)))
            if not mine or mine==sorted(row.get('refunded_attempts') or ()):continue
            row['refunded_attempts']=mine;row['refund_policy']=PERIODIC_DUMP_REFUND
            row['quarantined']=node_fault_count(row)>=NODE_NATIVE_FAILURE_LIMIT;atomic(path,row)
        if refunded:
            state=self.queue_state()
            if reopened and state.get('failure') and aggregate_task_failure(state['failure']):
                state.setdefault('failure_history',[]).append(state['failure']);state['failure']=None
            state.setdefault('actions',[]).append(dict(action='periodic_dump_crash_refund',policy=PERIODIC_DUMP_REFUND,
                attempts=refunded,reopened=reopened,worker=worker,at=time.time()))
            atomic(self.root/'state.json',state)
            print(f'[v8] {self.binding.get("cohort")}: {refunded} SIGSEGV(s) from the old 30-min stack dump no longer '
                  f'count as failures; {len(reopened)} blocked task(s) resume from their checkpoints (records kept)',flush=True)
        return reopened

    def grant_user_retry(self,worker):
        """USER_RETRY_GRANT: blocked tasks get one fresh failure budget, once per task, history kept."""
        if os.environ.get('V8_USER_RETRY_GRANT','1')=='0':return []   # tests that pin the cap itself
        state=self.queue_state();failure=state.get('failure')
        if failure and not aggregate_task_failure(failure):return []   # input/setup/identity failures stay
        granted=[]
        for path in sorted(self.root.glob('jobs/*.json')):
            job=read(path)
            if job.get('state') not in ('failed','pending','interrupted') or not job.get('attempts'):continue
            if isinstance(job.get('user_retry_grant'),dict) and job['user_retry_grant'].get('policy')==USER_RETRY_GRANT:continue
            if not (native_failure_blocks(job) or self.retry_limit_reached(job)):continue
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():continue
            try:
                job=read(path)
                if job.get('user_retry_grant') or not job.get('attempts'):continue
                last=job['attempts'][-1]
                job['user_retry_grant']=dict(policy=USER_RETRY_GRANT,after_attempt=last.get('id'),
                    task_sha256=digest(semantic_task(job['task'])),failures_before=retry_budget_failures(job),
                    crash_places=native_crash_hosts(job),worker=worker,at=time.time(),
                    reason='user_requested_restart_of_blocked_tasks_20261113')
                if last.get('id') is None:continue
                job['state']='pending';atomic(path,job);granted.append(path.stem)
            finally:lease.close()
        if granted:
            state=self.queue_state()
            if state.get('failure') and aggregate_task_failure(state['failure']):
                state.setdefault('failure_history',[]).append(state['failure']);state['failure']=None
            state.setdefault('actions',[]).append(dict(action='user_retry_grant',policy=USER_RETRY_GRANT,
                                                       tasks=granted,worker=worker,at=time.time()))
            atomic(self.root/'state.json',state)
            print(f'[v8] {self.binding.get("cohort")}: {len(granted)} blocked task(s) restart once more from their '
                  'checkpoints (user retry grant; failures and logs kept; never on the node where they crashed)',flush=True)
        return granted

    def grant_watchdog_fix_retry(self,worker):
        """WATCHDOG_FIX_GRANT: a task blocked by native crashes restarts from its checkpoint once more, once per task,
        only when this checkout's task entry cancels the periodic dump. History and the earlier grant are kept."""
        if '0' in (os.environ.get('V8_WATCHDOG_FIX_GRANT','1'),os.environ.get('V8_USER_RETRY_GRANT','1')):return []   # tests that pin the cap
        try:fixed='cancel_dump_traceback_later' in Path(__file__).with_name('v8_task_entry.py').read_text()
        except OSError:fixed=False
        if not fixed:return []
        state=self.queue_state();failure=state.get('failure')
        if failure and not aggregate_task_failure(failure):return []   # input/setup/identity failures stay
        granted=[]
        for path in sorted(self.root.glob('jobs/*.json')):
            job=read(path)
            if job.get('state') not in ('failed','pending','interrupted') or not job.get('attempts'):continue
            if (job.get('user_retry_grant') or {}).get('policy')==WATCHDOG_FIX_GRANT:continue
            if not fatal_native_failure(job) or not (native_failure_blocks(job) or self.retry_limit_reached(job)):continue
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():continue
            try:
                job=read(path);last=job['attempts'][-1] if job.get('attempts') else {}
                if (job.get('user_retry_grant') or {}).get('policy')==WATCHDOG_FIX_GRANT or last.get('id') is None:continue
                if job.get('user_retry_grant'):job.setdefault('retry_grant_history',[]).append(job['user_retry_grant'])
                job['user_retry_grant']=dict(policy=WATCHDOG_FIX_GRANT,after_attempt=last['id'],
                    task_sha256=digest(semantic_task(job['task'])),failures_before=retry_budget_failures(job),
                    crash_places=native_crash_hosts(job),worker=worker,at=time.time(),
                    reason='sigsegv_from_periodic_faulthandler_dump_fixed_in_task_entry_20261117')
                job['state']='pending';atomic(path,job);granted.append(path.stem)
            finally:lease.close()
        if granted:
            state=self.queue_state()
            if state.get('failure') and aggregate_task_failure(state['failure']):
                state.setdefault('failure_history',[]).append(state['failure']);state['failure']=None
            state.setdefault('actions',[]).append(dict(action='watchdog_fix_retry_grant',policy=WATCHDOG_FIX_GRANT,
                                                       tasks=granted,worker=worker,at=time.time()))
            atomic(self.root/'state.json',state)
            print(f'[v8] {self.binding.get("cohort")}: {len(granted)} task(s) stopped by the 30-min stack-dump SIGSEGV '
                  'restart from their checkpoints (dump now off; failures and logs kept)',flush=True)
        return granted

    def recover_budget_api_failure(self,controller,worker):
        """Verified recovery: fix the known API regression, or allow independent work beside a blocked eval plan."""
        if controller!=self.controller_sha256:return False
        state=self.queue_state();failure=state.get('failure')
        if state.get('phase') not in PHASES or not isinstance(failure,dict) or failure.get('task') is not None:
            return False
        detail=str(failure.get('detail',''))
        api=bool(re.search(r'''freeze_budget\(\) got an unexpected keyword argument ['"]expected_seeds['"]''',detail))
        evaluation='Selected budget checkpoint is missing or changed:' in detail
        stale_claim='Inspected task definition changed before claim:' in detail   # controller defect, v35
        if not api and not evaluation and not stale_claim:return False
        state.setdefault('failure_history',[]).append(failure);state['failure']=None
        if evaluation:state['unresolved_evaluation_planner_failure']=failure
        state.setdefault('actions',[]).append(dict(action=('recover_stale_offer_claim_failure' if stale_claim else
            'recover_verified_legacy_budget_keyword_failure' if api else 'resume_independent_work_with_blocked_evaluation_plan'),
            controller_sha256=controller,controller_policy=DISPATCH_POLICY,worker=worker,at=time.time()))
        atomic(self.root/'state.json',state)
        print(f'[v8] {self.binding.get("cohort")}: resumed independent work after verified controller recovery; '
              f'evaluation planner error remains recorded={evaluation}; original failure/checkpoints/bindings retained',flush=True)
        return True

    def enable_automatic_retry(self,worker):
        """Default task recovery after admission; never clear input/setup/identity failures."""
        state=self.queue_state();failure=state.get('failure')
        policy=dict(policy=DISPATCH_POLICY,max_failed_attempts=failure_limit(self.max_attempts),retry_delay=self.retry_delay,
                    unlimited_failures=False,retry_count_scope='task-lifetime')
        changed=state.get('automatic_retry_policy')!=policy or not state.get('retry_failed_requested_at')
        state['automatic_retry_policy']=policy
        # Older dispatchers already understand this shared request and cooldown.
        state.setdefault('retry_failed_requested_at',time.time())
        aggregate=aggregate_task_failure(failure)
        if aggregate and any(job.get('phase')==state['phase'] and self.retry_later(job,state)
                             for job in (read(p) for p in self.root.glob('jobs/*.json'))):
            state.setdefault('failure_history',[]).append(failure);state['failure']=None;changed=True
        if changed:
            state.setdefault('actions',[]).append(dict(action='enable_default_task_retry',worker=worker,at=time.time(),**policy))
            atomic(self.root/'state.json',state)

    def recover_failed(self,controller,worker):
        """After node verification, retry an older controller's failed tasks once per fix.

        This is control recovery, not a new experiment: binding, inputs, attempts and checkpoints stay.
        An input/setup failure or a task that already failed on this controller is never cleared here.
        """
        state=self.queue_state();failure=state.get('failure')
        aggregate=aggregate_task_failure(failure)
        if failure and not aggregate:return
        recovered=[];resumable=False
        for path in sorted(self.root.glob('jobs/*.json')):
            job=read(path);history=job.get('recovery_history',[])
            if job['state'] in ('failed','interrupted','pending') and not job.get('evaluation_native_recovery') and job.get('task',{}).get('kind')=='evaluate':
                lease=Lock(self.root/'leases'/(path.stem+'.lock'))
                if not lease.acquire():continue
                try:
                    if self.arm_evaluation_recovery(job,worker):
                        job['state']='pending';atomic(path,job);recovered.append(path.stem)
                finally:lease.close()
                if path.stem in recovered:continue
            already=any(row.get('controller_sha256')==controller or row.get('controller_policy')==DISPATCH_POLICY
                        for row in history)
            if already and job['state'] in ('pending','interrupted'):resumable=True
            if (job['state']!='failed' or already or fatal_native_failure(job) or native_failure_blocks(job)
                    or self.retry_limit_reached(job)):continue
            attempts=job.get('attempts',[])
            if attempts and (attempts[-1].get('controller_sha256')==controller
                             or attempts[-1].get('controller_policy')==DISPATCH_POLICY):continue
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():continue
            try:
                job.setdefault('recovery_history',[]).append(dict(controller_sha256=controller,
                    controller_policy=DISPATCH_POLICY,worker=worker,
                    previous_attempt=attempts[-1].get('id') if attempts else None,at=time.time(),
                    reason='verified_node_join_after_controller_update'))
                job['state']='pending';atomic(path,job);recovered.append(path.stem)
            finally:lease.close()
        if aggregate and (recovered or resumable):
            state.setdefault('failure_history',[]).append(failure);state['failure']=None
        if recovered or (aggregate and resumable):
            state.setdefault('actions',[]).append(dict(action='recover_failed_after_controller_update',
                controller_sha256=controller,worker=worker,tasks=recovered,at=time.time()))
            atomic(self.root/'state.json',state)
            print(f'[v8] {self.binding.get("cohort")}: resumed {len(recovered)} older-controller failed task(s) '
                  'after node verification; original attempts/checkpoints retained; no retry flag needed',flush=True)

    def arm_evaluation_recovery(self,job,worker):
        """Latest user request: recover an eval SIGSEGV once without resetting its history."""
        if job.get('task',{}).get('kind')!='evaluate' or job.get('evaluation_native_recovery'):return False
        attempts=job.get('attempts',[])
        if not attempts:return False
        fault=fatal_native_failure(dict(attempts=[attempts[-1]]))
        if not fault or fault['signal']!=11:return False
        job['evaluation_native_recovery']=dict(policy=EVAL_NATIVE_RECOVERY,max_failed_replays=1,
            after_attempt=attempts[-1]['id'],task_sha256=digest(semantic_task(job['task'])),
            binding_sha256=digest(self.binding),worker=worker,at=time.time(),
            reason='user_requested_bounded_recovery_of_native_evaluation_failure')
        print(f'[v8] evaluation recovery armed once: {task_key(job["task"])}; '
              f'{counted_failures(job)} original failures retained; identical checkpoint/spec; not unlimited',flush=True)
        return True

    def queue_state(self):
        """state.json, recreated as a fresh 'prepare' record (with a printed note) if it is missing,
        e.g. a queue folder created by an interrupted start. Never raises FileNotFoundError."""
        path=self.root/'state.json'
        if not path.exists():
            print(f"[v8] {self.binding.get('cohort','?')}: {path} was missing; recreating it (phase prepare). "
                  "Finished work is kept and re-checked; nothing is deleted.",flush=True)
            atomic(path,dict(phase='prepare',failure=None,actions=[dict(action='state_recreated',at=time.time())],failure_history=[]))
        return read(path)

    def fail(self,detail,task=None):
        state=self.queue_state()
        if not state['failure']:
            state['failure']=dict(detail=detail,task=task,at=time.time())
            atomic(self.root/'state.json',state)

    def register(self,task,phase):
        key=task_key(task);path=self.root/'jobs'/(key+'.json')
        definition=semantic_task(task)
        matrix=self.root/'tasks'/(key+'.json')
        if path.exists():
            job=read(path)
            if semantic_task(job['task'])!=definition or job['phase']!=phase:
                raise ValueError('Conflicting task definition for '+key)
            if not matrix.exists():
                atomic(matrix,dict(tasks=[job['task']],cohort_id=self.binding['cohort'],schema=SCHEMA))
            elif len(read(matrix).get('tasks',[]))!=1 or semantic_task(read(matrix)['tasks'][0])!=definition:
                raise ValueError('Conflicting immutable task matrix for '+key)
            return key
        if matrix.exists() and (len(read(matrix).get('tasks',[]))!=1 or semantic_task(read(matrix)['tasks'][0])!=definition):
            raise ValueError('Conflicting immutable task matrix for '+key)
        if not matrix.exists():
            atomic(matrix,dict(tasks=[task],cohort_id=self.binding['cohort'],schema=SCHEMA))
        atomic(path,dict(task=task,phase=phase,state='pending',attempts=[]))
        return key

    def prioritize_primary_results(self,runtime):
        """Record an explicit control-order change; preserve scientific binding, plans and live jobs."""
        state=self.queue_state()
        if not getattr(runtime,'primary_results_first',False):return state
        order=state.get('execution_order') or {};updated=False
        scope=([self.binding['cohort']] if getattr(runtime,'sequential_cohorts',()) else
               list(dict.fromkeys([*order.get('cohorts',[]),*getattr(runtime,'primary_cohort_scope',(self.binding['cohort'],))])))
        if getattr(runtime,'streaming_ready_tasks',False):
            if order.get('policy')!=CONCURRENT_ORDER or order.get('cohorts')!=scope:
                state.setdefault('actions',[]).append(dict(action='enable_ready_checkpoint_parallel_work',
                    previous_order=order,at=time.time(),policy=CONCURRENT_ORDER))
                state['execution_order']=dict(policy=CONCURRENT_ORDER,cohorts=scope)
                atomic(self.root/'state.json',state)
            return state
        if order.get('policy')!=PRIMARY_ORDER or order.get('cohorts')!=scope:
            state['execution_order']=dict(policy=PRIMARY_ORDER,cohorts=scope);updated=True
        phase=state['phase']
        if phase in ('extension','evaluation','finalize','complete') and not primary_results_ready(self.root.parent):
            if not self.held_leases() and (not state.get('failure') or aggregate_task_failure(state['failure'])):
                if state.get('failure'):
                    state.setdefault('failure_history',[]).append(state['failure']);state['failure']=None
                state.setdefault('actions',[]).append(dict(action='defer_extension_until_primary_results',
                    from_phase=phase,at=time.time(),policy=PRIMARY_ORDER))
                state['phase']='primary_evaluation';updated=True
        if updated:atomic(self.root/'state.json',state)
        return state

    def extension_blockers(self):
        state=self.queue_state();order=state.get('execution_order') or {}
        if order.get('policy')!=PRIMARY_ORDER:return []
        base=self.root.parent.parent
        return [c for c in order['cohorts'] if not primary_results_ready(base/c)]

    def tick(self,runtime,worker,stop,claim=True,*,offer=False):
        """Called with control lock held. Returns (state, claimed lease/task)."""
        if not (self.root/'binding.json').exists():
            raise BindingMismatch(f"{self.binding.get('cohort')}: this queue folder ({self.root}) was moved or removed "
                                  "while this worker was running; restart this worker (same command).")
        if read(self.root/'binding.json')!=self.binding:
            raise BindingMismatch(f"{self.binding.get('cohort')}: another node replaced this queue's code/Python record "
                                  "while this worker was running; restart this worker with the same bundle as the others.")
        self.observed_leases=[]   # task leases held by others in this inspection (shared with its 'waiting' result)
        self.refund_periodic_dump_crashes(worker)
        state=self.prioritize_primary_results(runtime)
        if isinstance(state.get('failure'),dict) and state['failure'].get('task'):
            # A still-running historical worker can publish its old cohort-wide
            # task exit after this node has already joined. Reuse the same
            # validated, task-only conversion; never retry the failed task.
            self.initialize();state=self.queue_state()
        aggregate=aggregate_task_failure(state.get('failure'))
        if state['failure'] and not aggregate:return 'failed',None
        phase=state['phase']
        if phase=='complete':return 'complete',None
        if phase=='prepare':
            return 'prepare',None
        if phase in CPU_PHASES:
            return phase,None
        if phase=='extension':
            blocked=self.extension_blockers()
            if blocked:
                if state.get('extension_waiting_for')!=blocked:
                    state['extension_waiting_for']=blocked;atomic(self.root/'state.json',state)
                    print(f'[{worker}] {self.binding["cohort"]}: extension deferred; matched-step results pending: '+', '.join(blocked),flush=True)
                self.observation=dict(checked_at=time.time(),phase=phase,extension_waiting_for=blocked)
                return 'waiting',None
            if state.pop('extension_waiting_for',None):atomic(self.root/'state.json',state)
        base_error=None;deferred_base=(getattr(runtime,'streaming_ready_tasks',False) and phase in ('extension','evaluation')
                                      and not (self.root.parent/'frozen/budget.json').exists())
        try:
            cached=self.root/'plans'/(phase+'.json')
            corrected=phase in getattr(runtime,'corrected_plans',())
            if deferred_base:tasks,sealed=[],False
            elif cached.exists():
                plan=read(cached);tasks,sealed=plan['tasks'],True
                if corrected and plan.get('planner')!=CORRECTED_PLANNER:
                    raise ValueError(f'{phase} plan was cached by this queue\'s original coordinator, whose evaluation '
                                     'planner predates the V8-016 budget-endpoint fix; it is not used. Inspect '
                                     f'{cached} and its jobs before deciding how to evaluate this queue.')
            else:
                tasks,sealed=runtime.plan(phase)
                if sealed and (tasks or phase=='extension'):
                    atomic(cached,dict(tasks=tasks,sealed=True,**({'planner':CORRECTED_PLANNER} if corrected else {})))
        except json.JSONDecodeError as exc:
            return self.metadata_wait(exc)
        except ValueError as exc:
            if not getattr(runtime,'streaming_ready_tasks',False) or phase not in ('extension','evaluation','primary_evaluation'):raise
            tasks,sealed=[],False;base_error=f'{type(exc).__name__}: {exc}'
        if state.get('base_planning_error')!=base_error:
            if base_error:state.setdefault('planning_failure_history',[]).append(dict(phase=phase,detail=base_error,at=time.time()))
            state['base_planning_error']=base_error;atomic(self.root/'state.json',state)
        job_phase='evaluation' if phase=='primary_evaluation' else phase
        selected={task_key(t) for t in tasks}
        for task in tasks:self.register(task,job_phase)
        side_keys=set();side_budget=None;side_rule=None
        if getattr(runtime,'streaming_ready_tasks',False):
            try:side=runtime.side_tasks()
            except json.JSONDecodeError as exc:
                return self.metadata_wait(exc)
            summary={k:v for k,v in side.items() if k!='tasks'}
            if state.get('ready_side_work')!=summary:
                state['ready_side_work']=summary;atomic(self.root/'state.json',state)
            side_budget=side.get('budget_sha256')
            side_rule=side.get('budget_rule_sha256')
            for item in side.get('tasks',[]):
                if item['phase'] not in ('extension','evaluation'):raise ValueError('Invalid side task phase')
                if not side_budget and (item['phase']=='extension' or not side_rule):
                    raise ValueError('Ready side work has no frozen budget or precommitted evaluation rule')
                side_keys.add(self.register(item['task'],item['phase']))
        self.metadata_error_wait=None
        jobs=[];side_rule_record=read(self.root/'budget-rule.json') if side_rule else None
        for path in sorted(self.root.glob('jobs/*.json')):
            job=read(path)
            # A cached, immutable evaluation is already a pinned original task.
            # A different endpoint's planner failure must not hide it from offers.
            if (getattr(runtime,'streaming_ready_tasks',False) and job['phase']=='evaluation'
                    and job['task'].get('kind')=='evaluate'):
                t=job['task'];rule=side_rule_record
                early=bool(rule and t.get('split')=='final' and t.get('completed_update')==rule['horizon']
                           and t.get('training_seed') in rule['seeds'])
                if side_budget or early:side_keys.add(path.stem)
            if (job['phase']==job_phase and (not (phase=='primary_evaluation' or deferred_base or base_error) or path.stem in selected)) or path.stem in side_keys:
                jobs.append((path,job))
        if not jobs and phase!='extension':
            if deferred_base or base_error:
                self.observation=dict(checked_at=time.time(),phase=phase,ready=0,leased=0,native_busy=0,
                                      dependencies=0,failed=0,planning_error=base_error,
                                      deferred_budget=deferred_base)
                return 'waiting',None
            raise ValueError('No tasks produced for '+phase+'; refusing to declare an empty experiment complete')
        pending=[];all_done=True;active=False;failed=[];later=False
        leased=native_count=dependencies=retry_wait=place_failed=0;self.observed_leases=[]
        for path,job in jobs:
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():
                all_done=False;active=True;leased+=1;self.observed_leases.append(path.stem);continue
            try:
                native_busy=runtime.native_busy(job['task'])
                active=active or native_busy
                done=False if native_busy else runtime.done(job['task'])
                if done:
                    if job['state']!='done':
                        job['state']='done';job['adopted_at']=time.time();atomic(path,job)
                    continue
                all_done=False
                if job['state']=='done':raise ValueError('Previously completed output disappeared or became inconsistent: '+path.stem)
                if native_busy:active=True;native_count+=1;continue
                if (native_failure_blocks(job) or self.retry_limit_reached(job)) and job['state']!='failed':
                    # Old controllers may have reset a failed job to pending/running.
                    # Only migrate it after its lease/native lock is free and outputs
                    # were checked; never interrupt a live task or discard a result.
                    job['state']='failed';atomic(path,job)
                if not self.retryable(job,state=state):
                    if job['state']=='failed':
                        failed.append(path.stem)
                        # Blocked for good is the same answer on every node; anything else (a crash place, a retry
                        # delay, no automatic retry here) may differ on another node and is never shared.
                        place_failed+=not (native_failure_blocks(job) or self.retry_limit_reached(job))
                    if self.retry_later(job,state):
                        later=True   # retry_wait: this GPU set may run it after its retry time (not a crash place)
                        retry_wait+=not native_failure_blocks(job,None if getattr(self,'readmitted',False) else self.place())
                    continue
                if not runtime.ready(job['task']):dependencies+=1;continue
                pending.append((path,job))
            finally:lease.close()
        self.observation=dict(checked_at=time.time(),phase=phase,ready=len(pending),leased=leased,
                              native_busy=native_count,dependencies=dependencies,failed=len(failed),place_failed=place_failed,
                              recovery_wait=later,retry_wait=retry_wait,planning_error=base_error,deferred_budget=deferred_base,
                              planning_errors=(state.get('ready_side_work') or {}).get('planning_errors'),
                              unsealed_plan=all_done and not sealed)
        if aggregate and (all_done or active or later or pending):
            # Only validated completed/active/eligible work invalidates an old
            # aggregate exhaustion record. Input/setup failures never reach here.
            state.setdefault('failure_history',[]).append(state['failure']);state['failure']=None
            atomic(self.root/'state.json',state)
        if all_done and sealed:
            state['phase']={'train':'freeze','primary_evaluation':'primary_finalize',
                            'extension':'evaluation','evaluation':'finalize'}[phase]
            atomic(self.root/'state.json',state)
            return 'advance',None
        if failed and not pending and not active and not later:
            state=self.queue_state();state.pop('retry_failed_requested_at',None)
            cap=f'at most {failure_limit(self.max_attempts)} failures over this task lifetime'
            state['failure']=dict(detail=f'{len(failed)} task(s) failed; no independent ready/running work remains. '
                                  'Failed attempts/checkpoints are preserved. Automatic recovery exhausted or disabled '
                                  f'({cap}, {self.retry_delay:g}s between attempts).',
                                  task=None,at=time.time())
            atomic(self.root/'state.json',state)
            return 'failed',None
        # Fresh ready work precedes failed retries, so a broken primary trajectory cannot
        # monopolize all workers when immediate restarts are enabled. Then fill trajectories.
        pending.sort(key=lambda item:(item[1]['state']=='failed',item[1]['task'].get('method_id')!='on_monitored',
                                      len(item[1]['attempts']),item[0].stem))
        if pending and offer:
            return 'ready',dict(phase=phase,side_budget_sha256=side_budget,side_rule_sha256=side_rule,
                                jobs=[dict(key=p.stem,phase=j['phase'],retry=j['state']=='failed',task=semantic_task(j['task'])) for p,j in pending])
        if pending and not claim:return 'gpu_busy',None

        return self.claim_offered(dict(phase=phase,side_budget_sha256=side_budget,side_rule_sha256=side_rule,jobs=[dict(key=p.stem,phase=j['phase'],retry=j['state']=='failed',task=semantic_task(j['task']))
                                                       for p,j in pending]),worker,stop)

    def claim_offered(self,offer,worker,stop):
        """Revalidate an inspected offer under control.lock; no Runtime RPC/model/ledger scans.

        Multi-cohort callers also hold the dispatch lock, solely for lease counts and this claim.
        Native execution locks and all scientific inputs are still checked by the original child.
        """
        if read(self.root/'binding.json')!=self.binding:raise BindingMismatch('Queue identity changed before claim')
        state=self.queue_state()
        if state.get('failure') or state['phase']!=offer['phase']:return 'stale',None
        if offer['phase']=='extension' and self.extension_blockers():return 'waiting',None
        if offer.get('side_budget_sha256'):
            budget=self.root.parent/'frozen/budget.json'
            if not budget.is_file() or hashlib.sha256(budget.read_bytes()).hexdigest()!=offer['side_budget_sha256']:
                return 'stale',None
        early_rule=None
        if offer.get('side_rule_sha256'):
            path=self.root/'budget-rule.json'
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=offer['side_rule_sha256']:return 'stale',None
            early_rule=read(path)
            if early_rule.get('policy')!=BUDGET_RULE or early_rule.get('binding_sha256')!=digest(self.binding):return 'stale',None
        parallel=state.get('execution_order',{}).get('policy')==CONCURRENT_ORDER
        phases=('train','extension','evaluation');items=offer['jobs']
        if parallel:
            cursor=state.get('ready_phase_cursor',0)
            items=[item for _,item in sorted(enumerate(items),key=lambda pair:(pair[1].get('retry',False),
                (phases.index(pair[1].get('phase','evaluation' if offer['phase']=='primary_evaluation' else offer['phase']))-cursor)%3,pair[0]))]
        for item in items:
            if stop.is_set():return 'stopped',None
            path=self.root/'jobs'/(item['key']+'.json');job=read(path)
            job_phase='evaluation' if offer['phase']=='primary_evaluation' else offer['phase']
            if parallel:
                job_phase=item.get('phase',job_phase)
                if job_phase not in (offer['phase'],'extension','evaluation'):raise ValueError('Invalid concurrent claim phase')
            if job['phase']!=job_phase or semantic_task(job['task'])!=item['task']:
                # The offer is older than this job record, or names side work this phase cannot claim here.
                # A stale offer is not a queue failure: raising here made the claim loop write pool.fail()
                # and every node left the queue (user 11-11: 'inspected task definition changed before
                # claim', 'state BLOCKED', '노드 다 날아감'). Skip it; the next inspection is current.
                continue
            if parallel and not offer.get('side_budget_sha256') and job_phase in ('extension','evaluation'):
                t=job['task']
                if (job_phase=='extension' or not early_rule or t.get('kind')!='evaluate' or t.get('split')!='final'
                        or t.get('completed_update')!=early_rule['horizon'] or t.get('training_seed') not in early_rule['seeds']):continue
            # A failed task is claimable only under the retry rules (--retry-failed, attempts, delay).
            if job['state']=='done' or not self.retryable(job,state=state):continue
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():continue
            attempt=dict(id=uuid.uuid4().hex,worker=worker,host=self.place()[0],gpus=list(self.place()[1]),
                         started_at=time.time(),pid=os.getpid(),
                         controller_sha256=self.controller_sha256,controller_policy=DISPATCH_POLICY,
                         task_entry_sha256=getattr(self,'task_entry_sha256',None),
                         periodic_stack_dump=getattr(self,'periodic_stack_dump','unknown'))
            if job.get('retry_epoch'):
                attempt.update(retry_epoch=dict(job['retry_epoch']),retry_budget_failures_before_start=retry_budget_failures(job))
            if job['state']=='running' and job['attempts']:
                previous=job['attempts'][-1];observed=time.time()
                previous['lease_released_without_worker_completion']=observed
                if not previous.get('outcome'):
                    previous.update(outcome='node_lost',node_loss_observed_at=observed,
                                    evidence='task lease free when replacement worker acquired it')
                attempt['resumed_from_attempt']=previous['id']
            elif job['state']=='interrupted' and job['attempts']:
                attempt['resumed_from_attempt']=job['attempts'][-1]['id']
            if job['state']=='failed':attempt['retry_of_failed_attempts']=counted_failures(job)
            job.pop('resume_not_before',None)
            job['attempts'].append(attempt);job['state']='running'
            try:
                atomic(path,job)
                if parallel:
                    state['ready_phase_cursor']=(phases.index(job['phase'])+1)%3
                    atomic(self.root/'state.json',state)
            except BaseException:lease.close();raise
            return 'claimed',(lease,path.stem,job['task'],attempt)
        return 'waiting',None

    def finish(self,key,rc,runtime,stop,node_error=None,preflight=None):
        """Keep original rank faults; refund node-local faults and intentional interruptions."""
        path=self.root/'jobs'/(key+'.json');job=read(path);attempt=job['attempts'][-1]
        attempt.update(exit_code=rc,ended_at=time.time());done=False
        if preflight is not None:
            attempt['preflight']=preflight
            if preflight.get('env'):attempt['env']=preflight['env']
        failure_path=self.root/'progress'/job['attempts'][-1]['id']/'child-failure.json'
        if failure_path.is_file():
            try:
                failure=read(failure_path)
                if failure.get('attempt_id')==job['attempts'][-1]['id']:
                    job['attempts'][-1]['child_failure']=failure
            except (OSError,ValueError,AttributeError) as exc:
                job['attempts'][-1]['error_record_read_error']=str(exc)
        if node_error is None:
            try:done=runtime.done(job['task'])
            except NodeFailure as exc:node_error=f'{type(exc).__name__}: {exc}'
            except Exception as exc:attempt['output_validation_error']=str(exc)
        termination=forced_termination(attempt,rc,self.root/'logs'/(attempt['id']+'.log'))
        if node_error is not None:
            attempt.update(outcome='node_failure',error=node_error);job['state']='interrupted'
        elif rc==0 and done:attempt['outcome']='done';job['state']='done'
        elif termination is not None:
            attempt.update(outcome='interrupted',termination=termination,
                           error='Forced process termination; origin unconfirmed; resume from last saved checkpoint')
            job.update(state='interrupted',resume_not_before=time.time()+self.retry_delay)
        elif (rc==75 or stop.is_set()) and not fatal_native_failure(dict(attempts=[attempt])):
            attempt['outcome']='interrupted';job['state']='interrupted'
        else:
            attempt.update(outcome='failed',error=f'task exit={rc}; output verified={done}'+
                           (f"; {attempt['output_validation_error']}" if attempt.get('output_validation_error') else ''))
            job['state']='failed'
            state=self.queue_state()
            state.setdefault('task_failures',[]).append(dict(detail=f'task exit={rc}; output verified={done}',
                                                           task=key,attempt=job['attempts'][-1]['id'],at=time.time()))
            atomic(self.root/'state.json',state)
        if refund_periodic_dump(self.root,job,attempt.get('worker')):
            print(f'[v8] {key}: SIGSEGV from the 30-min periodic stack dump; not counted as a failure or crash place; '
                  'the task resumes from its checkpoint',flush=True)
        if job['state'] in ('failed','interrupted'):self.arm_evaluation_recovery(job,attempt.get('worker'))
        if rc not in (0,75,None) or fatal_native_failure(dict(attempts=[attempt])):
            try:attempt.update(crash_fields(self.root,attempt,rc,job.get('task')))
            except Exception as exc:attempt['crash_record_error']=f'{type(exc).__name__}: {exc}'
            print('[v8] '+crash_line(attempt),flush=True)
        atomic(path,job)
        record_node_native_fault(self,runtime,attempt)
        return job['state']

    def status(self):
        problems=[]
        try:state=read(self.root/'state.json') if (self.root/'state.json').exists() else dict(phase='not_started')
        except (OSError,ValueError) as exc:
            state=dict(phase='unknown');problems.append(f"{self.root/'state.json'}: {exc.strerror if isinstance(exc,OSError) else 'unreadable JSON'}")
        counts={};owners=[]
        for path in self.root.glob('jobs/*.json'):
            try:job=read(path)
            except (OSError,ValueError) as exc:
                problems.append(f"{path}: {exc.strerror if isinstance(exc,OSError) else 'unreadable JSON'}");continue
            if not isinstance(job,dict) or not isinstance(job.get('task'),dict):
                problems.append(f'{path}: not a job record');continue
            job.setdefault('attempts',[])
            counts[job.get('state','?')]=counts.get(job.get('state','?'),0)+1
            try:
                lease=Lock(self.root/'leases'/(path.stem+'.lock'))
                if not lease.acquire():
                    last=job['attempts'][-1] if job['attempts'] else {}
                    row=dict(task=path.stem,worker=last.get('worker'),method=job['task'].get('method_id'),
                             seed=job['task'].get('training_seed'),kind=job['task'].get('kind'))
                    log=self.root/'logs'/(str(last.get('id'))+'.log')
                    if last.get('id') and log.exists():
                        row.update(log=str(log),log_written_seconds_ago=round(time.time()-log.stat().st_mtime))
                    owners.append(row)
                lease.close()
            except (OSError,NodeFailure) as exc:
                # Another account's lease file (read-only status run): report it, do not stop the status.
                problems.append(f"{self.root/'leases'/(path.stem+'.lock')}: lease state unknown ({getattr(exc,'strerror',None) or exc})")
        unreadable=problems+[str(p) for p in (self.root,self.root/'jobs',self.root/'leases',self.root/'workers')
                    if p.exists() and not os.access(p,os.R_OK|(os.X_OK if p.is_dir() else 0))]
        workers=[]
        for path in self.root.glob('workers/*.json'):
            try:
                beat=read(path)
                workers.append(dict(worker=beat.get('worker'),dispatch_policy=beat.get('dispatch_policy'),
                                    controller_sha256=beat.get('controller_sha256'),
                                    heartbeat_age_seconds=round(time.time()-beat['at']) if isinstance(beat.get('at'),(int,float)) else None))
            except (OSError,ValueError,AttributeError):unreadable.append(str(path))
        result=dict(phase=state['phase'],failure=state.get('failure'),known_task_counts=counts,leased_tasks=owners,
                    dispatch_workers=workers)
        for name in ('execution_order','primary_results_receipt_sha256','extension_waiting_for'):
            if name in state:result[name]=state[name]
        if unreadable:result['unreadable']=unreadable;result['note']='some queue records could not be read; counts are incomplete. Run --status as the account that runs the workers.'
        return result


def worker_loop(pool,runtime,worker,stop,*,poll=3,device_fds=(),retry=False,**policy):
    return worker_loop_many([(pool,runtime)],worker,stop,poll=poll,device_fds=device_fds,retry=retry,**policy)


NODE_LOCAL=(NodeFailure,PermissionError)  # local access faults; other OSErrors stay queue-visible
# User 11-16 'idle 고쳐': a node-local give-up (runtime/actor start, node verification, launch, queue access) is
# retried on this node after a bounded backoff instead of for the rest of the process. Never retried: this GPU
# set's native-crash quarantine and re-admission limits, a changed queue identity, queue-level failures.
GIVEUP_NEVER_RETRY=frozenset({'native crash quarantine','GPU re-admission','queue identity','extension deferred'})
GIVEUP_RETRY_CAP=3600.0


def giveup_retryable(kind,exc):
    # Only node-local faults (NodeFailure: dead/unstarted actor, timeouts, launch; PermissionError: local access).
    # A verification that answers 'these frozen inputs differ' (ValueError) is deterministic: it keeps exiting.
    if isinstance(exc,BindingMismatch) or kind in GIVEUP_NEVER_RETRY:return False
    return isinstance(exc,NODE_LOCAL)


IDLE_REPORT_SECONDS=float(os.environ.get('V8_IDLE_REPORT_SECONDS','30'))


def span_text(seconds):
    seconds=max(0,int(seconds))
    if seconds<60:return f'{seconds}s'
    if seconds<3600:return f'{seconds//60}m'
    return f'{seconds//3600}h {seconds%3600//60}m'


def observation_text(obs):
    """Plain words for why a cohort's last task inspection gave this node nothing (user 11-16 '주석이라도 잘 나오게')."""
    obs=dict(obs or {})
    if not obs:return 'tasks not inspected yet'
    if obs.get('metadata_error'):return 'a task record is being written by another node; reading it again shortly'
    if obs.get('extension_waiting_for'):
        return 'extension waits for the matched-step results of '+', '.join(map(str,obs['extension_waiting_for']))
    if obs.get('planning_error') or obs.get('planning_errors'):
        return 'planner reports an input problem: '+canonical(obs.get('planning_error') or obs['planning_errors'])[:120]
    def count(key):
        try:return int(obs.get(key) or 0)
        except (TypeError,ValueError):return 0
    ready,leased,native,deps,failed=(count(k) for k in ('ready','leased','native_busy','dependencies','failed'))
    bits=[f'{ready} ready' if ready else 'none ready']
    if leased:bits.append(f'{leased} running on other nodes')
    if native:bits.append(f'{native} wait for a process still using the same run folder')
    if deps:bits.append(f'{deps} waiting for parent checkpoints')
    if obs.get('deferred_budget'):bits.append('waiting for the frozen evaluation budget')
    if obs.get('recovery_wait'):bits.append('failed tasks wait for their retry time or another GPU host')
    if failed:bits.append(f'{failed} failed for good')
    if obs.get('unsealed_plan'):bits.append('all listed tasks are done; the planner has not added the next phase yet')
    if len(bits)==1 and not ready:bits.append('nothing else to do in this phase now')
    phase=obs.get('phase')
    return (f'{phase}: ' if phase else '')+', '.join(bits)


def giveup_retry_delay(n):
    """Seconds before retry n (1, 2, ...): V8_GIVEUP_RETRY_SECONDS (600), doubling, at most 3600."""
    try:base=float(os.environ.get('V8_GIVEUP_RETRY_SECONDS','600'))
    except ValueError:base=600.0
    if not math.isfinite(base) or base<=0:base=600.0
    base=max(base,.05);cap=max(base,GIVEUP_RETRY_CAP)
    return min(cap,base*2**min(max(int(n),1)-1,32))


def quiescent_failure(pairs):
    """No offers/scans/CPU stages remain: never idle forever beside recorded errors.

    Called outside dispatch/control locks. Observe leases and native execution locks
    before returning a worker-local failure; never mark another queue complete/failed.
    A subsequent launch may discover new work without resetting any task or receipt.
    """
    faults=[];snapshots=[]
    for pool,runtime in pairs:
        with pool.control() as control:
            if control is None:return None
            if read(pool.root/'binding.json')!=pool.binding:return None
            if pool.refund_periodic_dump_crashes(None):return None   # reopened work: not quiescent
            state=pool.queue_state()
            if state.get('failure'):faults.append(dict(cohort=pool.binding.get('cohort'),queue_failure=state['failure']))
            errors=dict((state.get('ready_side_work') or {}).get('planning_errors') or {})
            if state.get('base_planning_error'):errors['base']=state['base_planning_error']
            if errors:faults.append(dict(cohort=pool.binding.get('cohort'),planning_errors=errors))
            jobs=[read(p) for p in pool.root.glob('jobs/*.json')]
            if pool.held_leases():return None
            for job in jobs:
                if job.get('state')!='done' and (native_failure_blocks(job) or
                        (job.get('state')=='failed' and not pool.retryable(job,state=state))):
                    faults.append(dict(cohort=pool.binding.get('cohort'),task=task_key(job['task']),
                        failures=counted_failures(job),failure_limit=failure_limit(pool.max_attempts),
                        native_fault=fatal_native_failure(job)))
            snapshots.append((pool,jobs))
    if not faults:return None
    # Match original Runtime.native_busy without RPC or creating absent run folders.
    for pool,jobs in snapshots:
        for job in jobs:
            task=job['task']
            if task.get('kind')!='train':continue
            path=pool.root.parent/'runs'/task['run_id']/'.execution.lock'
            if not path.exists():continue
            lease=Lock(path)
            try:
                if not lease.acquire():return None
            finally:lease.close()
    return dict(reason='no eligible work remains beside failed tasks; worker exits instead of waiting',faults=faults)


def record_worker_retirement(pairs,worker,reason,faults=()):
    """Distinguish a blocked dispatcher exit from new crashes of this worker's tasks."""
    counts={};attempts=[];historical=0;unreadable=[]
    for pool,_ in pairs:
        for path in pool.root.glob('jobs/*.json'):
            try:
                job=read(path)
                if not isinstance(job,dict) or any(not isinstance(a,dict) for a in job.get('attempts',[])):
                    raise ValueError('invalid job/attempt record')
                historical+=counted_failures(job)
            except (OSError,ValueError,TypeError) as exc:
                unreadable.append(dict(path=str(path),error=str(exc)));continue
            for row in job.get('attempts',[]):
                if row.get('worker')!=worker:continue
                outcome=row.get('outcome')
                if outcome is None:outcome='failed' if row.get('exit_code') not in (None,0,75) else 'unrecorded'
                category=outcome
                if outcome=='failed':
                    messages=' '.join(str(f.get('message','')) for f in (row.get('child_failure') or {}).get('failures',[]))
                    if fatal_native_failure(dict(attempts=[row])):category='native_crash'
                    elif 'out of memory' in (messages+' '+str(row.get('error',''))).lower():category='out_of_memory'
                    elif row.get('exit_code')==124:category='progress_timeout'
                    else:category='task_error'
                counts[category]=counts.get(category,0)+1
                attempts.append(dict(cohort=pool.binding.get('cohort'),task=path.stem,attempt=row.get('id'),
                    category=category,exit_code=row.get('exit_code'),log=str(pool.root/'logs'/(str(row.get('id'))+'.log'))))
    current_failures=sum(counts.get(k,0) for k in ('native_crash','out_of_memory','progress_timeout','task_error'))
    previous_failures=historical-current_failures
    report=dict(worker=worker,at=time.time(),exit_code=1,exit_kind='blocked_work',reason=reason,faults=list(faults),
                current_worker_counts=counts,lifetime_task_failure_count=historical,
                previous_worker_failure_count=previous_failures,current_worker_attempts=attempts,
                unreadable_records=unreadable)
    for pool,_ in pairs:
        try:atomic(pool.root/'worker-exits'/(digest(worker)+'.json'),report)
        except (OSError,ValueError) as exc:print(f'[{worker}] cannot retain worker exit summary: {exc}',flush=True)
    print(f'[{worker}] WORKER EXIT: blocked_work; current worker outcomes='+canonical(counts)+
          f'; previous-worker failed attempts={previous_failures}; lifetime failed attempts={historical}; '
          'exit1 does not identify a new model crash; '+reason,flush=True)
    return report


class TaskScan:
    """One original actor's metadata scan, holding only its own cohort control lease."""
    def __init__(self,pool,runtime,worker,stop,fd,*,verify=False,retry=False,retain_offer=False):
        self.offer_control=os.fdopen(os.dup(fd),'rb') if fd is not None else None
        self.finished=threading.Event();self.result=None;self.error=None
        self.cancelled=threading.Event()
        self.started=self.reported=time.monotonic();self.kind='node verification' if verify else 'task inspection'
        def scan():
            try:
                if verify:
                    runtime.verify_node()
                    if self.cancelled.is_set() or stop.is_set():raise NodeFailure('node verification cancelled before queue initialization')
                    if self.offer_control is not None:
                        pool.initialize(retry=retry,worker=worker,recover=pool.controller_sha256)
                    else:
                        # Verified without the queue lock; take it only for the short initialize() write.
                        # Waiting for that lock is ordinary contention, not slow hashing: the verification
                        # limit stops here and the (longer) inspection limit covers the wait.
                        self.kind='queue initialization';self.started=time.monotonic()
                        while True:
                            with pool.control() as control:
                                if control is not None:
                                    pool.initialize(retry=retry,worker=worker,recover=pool.controller_sha256);break
                            if self.cancelled.is_set() or stop.is_set():
                                raise NodeFailure('node verification cancelled before queue initialization')
                            time.sleep(.05)
                    self.result=('verified',None)
                else:self.result=pool.tick(runtime,worker,stop,claim=False,offer=True)
            except BaseException as exc:
                self.error=exc
                if isinstance(exc,(KeyboardInterrupt,SystemExit)):stop.set()
                # Node-local faults (dead actor, local access) stop only this node: never a queue failure.
                if not verify and not stop.is_set() and not isinstance(exc,(BindingMismatch,*NODE_LOCAL)):
                    try:pool.fail(f'{type(exc).__name__}: {exc}')
                    except Exception as record_error:self.error=record_error
            finally:
                if self.offer_control is not None and not (retain_offer and self.error is None and self.result
                        and self.result[0]=='ready' and not self.cancelled.is_set() and not stop.is_set()):
                    self.offer_control.close()
                self.finished.set()
        self.thread=threading.Thread(target=scan,name='v8-scan-'+pool.binding.get('cohort','unknown'),daemon=True)
        try:self.thread.start()
        except BaseException:
            if self.offer_control is not None:self.offer_control.close()
            raise


def unfinished_cohort(pairs,order,*,check_leases=True):
    """Read shared completion, never infer it from no offers, progress %, or a failed job.

    Called again under the dispatch lock before every claim. Missing/unreadable state
    blocks the sequence; a complete queue with a live task lease still owns its turn.
    """
    base=pairs[0][0].root.parent.parent
    pools={pool.binding['cohort']:pool for pool,_ in pairs}
    for name in order:
        queue=base/name/'.multi-node'
        try:state=read(queue/'state.json')
        except FileNotFoundError:return name
        if state.get('phase')!='complete' or state.get('failure'):return name
        # Probe existing lease files only; do not create a missing cohort queue.
        if not check_leases:continue  # admission probes are serialized by the dispatch lock
        pool=pools.get(name)
        if pool is not None:
            if pool.held_leases():return name
        else:
            for path in queue.glob('leases/*.lock'):
                lease=Lock(path)
                try:
                    if not lease.acquire():return name
                finally:lease.close()
    return None


def worker_loop_many(pairs,worker,stop,*,poll=3,device_fds=(),retry=False,gpu_gate=None,
                     max_attempts=MAX_ATTEMPTS,retry_delay=RETRY_DELAY,automatic_retry=True,cohort_order=(),preflight=False,
                     not_served=None):
    # Only finished inspection leases enter this stack. A still-running scan
    # keeps its own lease until its thread exits, even during worker retirement.
    with ExitStack() as offer_cleanup:
        return _worker_loop_many(pairs,worker,stop,poll=poll,device_fds=device_fds,retry=retry,gpu_gate=gpu_gate,
            max_attempts=max_attempts,retry_delay=retry_delay,automatic_retry=automatic_retry,
            cohort_order=cohort_order,offer_cleanup=offer_cleanup,preflight=preflight,not_served=not_served)


def _worker_loop_many(pairs,worker,stop,*,poll,device_fds,retry,gpu_gate,
                     max_attempts,retry_delay,automatic_retry,cohort_order,offer_cleanup,preflight=False,not_served=None):
    """Schedule within cohort_order; an empty order retains archival fair dispatch."""
    if not pairs:raise ValueError('At least one cohort is required')
    if stop.is_set():return 75
    initialized=set();terminal={};offers={};scans={};last_report=0;reasons={};busy_report=0;setup_report=0
    shared_failures=set();recovery_checked={};claim_contention_since={};exit_recheck_started={};node_rechecks=set();control_waits={};offer_controls={}
    idle_state={'since':None,'reported':-float('inf')}  # this node has no task since (monotonic)
    retry_at={};giveups={};retry_checked=[0.0]  # index -> pending node-local retry; index -> give-ups since last served task
    def discard_offer(i):
        offers.pop(i,None)
        owned=offer_controls.pop(i,None)
        if owned is not None:owned.close()
    cursor=int(digest(worker)[:8],16)%len(pairs)
    roots=[p.root.parent.parent if p.root.name=='.multi-node' else p.root.parent for p,_ in pairs]
    dispatch_lock=Path(os.path.commonpath([str(p.resolve()) for p in roots]))/'.v8-dispatch.lock'
    controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    diagnostic=WorkerStatus(dispatch_lock.parent,worker,controller_sha256)
    cohort_order=tuple(cohort_order)
    sequence_report=None
    eligible=set()
    # Sequential order with one distinct rank per cohort: the claim order is the rank alone, so the per-claim
    # lease counts (one flock probe per task lease, under the one cross-cohort dispatch lock) are never used.
    distinct_ranks=bool(cohort_order) and len({cohort_order.index(p.binding.get('cohort')) if p.binding.get('cohort') in cohort_order
                                                else -1-i for i,(p,_) in enumerate(pairs)})==len(pairs)
    def retry_text(i):
        row=retry_at[i]
        return f"until {time.strftime('%H:%M',time.localtime(row['until']))} (retry {row['retry']}): {row['reason']}"
    def idle_cohort_text(i):
        pool,runtime=pairs[i];now=time.monotonic()
        if terminal.get(i)=='complete':return 'complete'
        if i in retry_at and terminal.get(i)=='failed':return 'NOT SERVING '+retry_text(i)[:200]
        if terminal.get(i)=='failed':
            why=str(reasons.get(cohort(pool),'gave up on this node'))[:200]
            return ('the queue has a recorded failure: ' if i in shared_failures else 'NOT SERVING: ')+why
        if i in scans:
            scan=scans[i];age=span_text(now-scan.started)
            if scan.kind=='node verification':return f'starting (node verification {age})'
            if scan.kind=='queue initialization':return f'starting (queue initialization {age})'
            return f'inspecting its tasks ({age})'
        if getattr(runtime,'pending_barrier',None) is not None:return 'CPU stage '+str(runtime.pending_barrier[0])+' running'
        if i not in initialized:
            if getattr(runtime,'channel',True) is None:return 'runtime not started yet (starts when earlier cohorts have nothing)'
            return 'starting ('+str(getattr(runtime,'startup_stage','runtime start'))+')'
        if i in offers:return 'ready task found, claiming it'
        text=observation_text(getattr(pool,'observation',None))
        if i in control_waits and not (getattr(pool,'observation',None) or {}).get('leased'):
            text=f'waiting {span_text(now-control_waits[i])} for its queue lock (another node is reading or writing it); last seen '+text
        if cohort_order and i not in eligible:text='earlier cohorts first; '+text
        return text
    def idle_reason():
        if idle_state['since'] is None:return None
        parts=[f'{cohort(pairs[i][0])} — {idle_cohort_text(i)}' for i in sorted(range(len(pairs)),key=lambda i:rank(pairs[i][0]))]
        parts+=[f'{name} — NOT SERVING: {str(why)[:120]}' for name,why in dict(not_served or {}).items()
                if name not in {cohort(p) for p,_ in pairs}]
        if idle_state.get('gpu_busy'):parts.insert(0,'GPUs busy before new work: '+str(idle_state['gpu_busy'])[:120])
        return f"IDLE {span_text(time.monotonic()-idle_state['since'])}: "+' · '.join(parts)
    def activity(i):
        pool,runtime=pairs[i]
        if i in retry_at and terminal.get(i)=='failed':return 'NOT SERVING '+retry_text(i)[:300]
        if terminal.get(i)=='failed':return 'not serving: '+str(reasons.get(cohort(pool),'gave up on this node'))[:300]
        if terminal.get(i)=='complete':return 'complete'
        if i in scans:return 'node verification' if scans[i].kind=='node verification' else 'task inspection'
        if getattr(runtime,'pending_barrier',None) is not None:return 'CPU stage'
        if i not in initialized:return 'starting runtime' if getattr(runtime,'channel',True) is not None else 'not started (later cohort)'
        if i in offers:return 'ready offer, claiming'
        if i in control_waits:return f'queue lock busy {time.monotonic()-control_waits[i]:.0f}s'
        return 'eligible, nothing ready' if i in eligible else 'waiting for its turn (earlier cohorts first)'
    def context():
        return dict(serves=[cohort(p) for i,(p,_) in sorted(enumerate(pairs),key=lambda x:rank(x[1][0]))
                            if terminal.get(i)!='failed'],
                    not_serving={**dict(not_served or {}),**{cohort(pairs[i][0]):(retry_text(i) if i in retry_at else
                                                              str(reasons.get(cohort(pairs[i][0]),'gave up')))[:300]
                                                              for i,v in terminal.items() if v=='failed'}},
                    cohorts={cohort(p):activity(i) for i,(p,_) in enumerate(pairs)},
                    retrying={cohort(pairs[i][0]):dict(until=row['until'],retry=row['retry'],kind=row['kind'],reason=row['reason'][:300])
                              for i,row in retry_at.items() if terminal.get(i)=='failed'},
                    **({'idle_reason':idle_reason()} if idle_state['since'] is not None else {}))
    diagnostic.context=context
    share_root=dispatch_lock.parent/'.v8-scan-share';used_shares={}
    def queue_marks(pool):
        try:return [(pool.root/'jobs').stat().st_mtime_ns,(pool.root/'state.json').stat().st_mtime_ns]
        except OSError:return None
    def share_scan(index,pool,scan,action,offer):
        if not cohort_order or SCAN_SHARE_SECONDS<=0 or action not in ('ready','waiting') or getattr(scan,'marks',None) is None:return
        observation=dict(getattr(pool,'observation',{}) or {})
        # 'waiting' is shared only when it holds for every node: failed tasks that are not blocked for good and
        # recovery waits depend on the inspecting node's place. Tasks blocked for good are the same everywhere and
        # are shared (else every idle node re-inspects the whole queue each poll, the 11-16 lock storm). Running
        # tasks are shared with their lease names; a reader inspects afresh as soon as one of those leases is free
        # (a dead node frees its lease without changing any queue file).
        if action=='waiting' and (any(observation.get(k) for k in ('ready','metadata_error','recovery_wait','place_failed'))
                                  or (observation.get('failed') and 'place_failed' not in observation)):return
        leases=list(getattr(pool,'observed_leases',None) or ()) if action=='waiting' else []
        if action=='waiting' and observation.get('leased') and len(leases)!=observation.get('leased'):return
        at=time.time()
        try:atomic(share_root/(cohort(pool)+'.json'),dict(schema=1,cohort=cohort(pool),binding_sha256=digest(pool.binding),
                   at=at,worker=worker,action=action,offer=offer if action=='ready' else None,
                   observation=observation,marks=scan.marks,leased_keys=leases))
        except (OSError,ValueError,TypeError):return  # a missing cache only means a full inspection
        if action=='ready':used_shares[index]=at  # this node already holds that offer
    def shared_scan(index,pool):
        """('ready', offer) / ('waiting', None) from another recent inspection, else None (inspect normally)."""
        if not cohort_order or SCAN_SHARE_SECONDS<=0:return None
        try:row=read(share_root/(cohort(pool)+'.json'))
        except (OSError,ValueError):return None
        if (not isinstance(row,dict) or row.get('binding_sha256')!=digest(pool.binding)
                or not isinstance(row.get('at'),(int,float)) or not 0<=time.time()-row['at']<SCAN_SHARE_SECONDS):return None
        if row.get('action')=='ready' and isinstance(row.get('offer'),dict) and used_shares.get(index)!=row['at']:
            used_shares[index]=row['at'];return 'ready',row['offer'],row
        if row.get('action')=='waiting' and row.get('marks') is not None and row['marks']==queue_marks(pool):
            observation=row.get('observation') or {}
            if observation.get('recovery_wait') or observation.get('place_failed'):return None
            if observation.get('failed') and 'place_failed' not in observation:return None   # written by an older node
            keys=row.get('leased_keys') or []
            if observation.get('leased') and len(keys)!=observation.get('leased'):return None
            for key in keys:   # every task seen running must still hold its lease
                lease=Lock(pool.root/'leases'/(str(key)+'.lock'))
                try:free=lease.acquire()
                except NodeFailure:return None
                if free:lease.close();return None
            return 'waiting',None,row
        return None
    # Spill (user 11-15): when every cohort up to `limit` has nothing this node can claim now,
    # the next cohort in cohort_order becomes eligible. Earlier cohorts keep claim priority; one
    # task at a time, so two cohorts never share this GPU group. Reset after each executed task.
    spill_rank=-1;spill_report=None;stall_since=time.monotonic()
    def rank(pool):
        name=cohort(pool)
        return cohort_order.index(name) if name in cohort_order else len(cohort_order)
    for pool,runtime in pairs:
        runtime.sequential_cohorts=cohort_order
        if cohort_order:runtime.primary_cohort_scope=(pool.binding['cohort'],)
        if getattr(runtime,'primary_results_first',False) and not getattr(runtime,'primary_cohort_scope',None):
            runtime.primary_cohort_scope=tuple(p.binding['cohort'] for p,_ in pairs)
        pool.retry_failed=retry;pool.max_attempts=max_attempts;pool.retry_delay=retry_delay
        pool.automatic_retry=automatic_retry
        pool.gpus=tuple(str(g) for g in getattr(runtime,'gpus',()) or ())  # the place recorded on each attempt
        runtime.stop_event=stop  # runtime actors report signals through the same stop event
        def heartbeat(task=None,pool=pool):
            atomic(pool.root/'workers'/(digest(worker)+'.json'),
                   dict(worker=worker,pid=os.getpid(),at=time.time(),task=task,
                        dispatch_policy=DISPATCH_POLICY,controller_sha256=controller_sha256))
        runtime.heartbeat=heartbeat
    def cohort(pool):
        return pool.binding.get('cohort','?')
    def first_unfinished():
        try:return unfinished_cohort(pairs,cohort_order,check_leases=False)
        except (OSError,ValueError):return None
    reserve_state={'text':None,'started':time.monotonic()}
    def reserve_hold(active):
        """Why this node stays with the first unfinished cohort (ACTIVE_RESERVE), or None when it may spill."""
        if not cohort_order or ACTIVE_RESERVE<=0 or active is None:return None
        mine=[i for i,(p,_) in enumerate(pairs) if cohort(p)==active and i not in terminal]
        if not mine:return None   # this node does not serve it now
        pool,runtime=pairs[mine[0]];obs=getattr(pool,'observation',None) or {}
        # A GPU set quarantined for this cohort could never take its orphan: it is no reserve. Give it up now, as
        # its first offer would, so it stops counting as an idle node of this cohort and another node holds instead.
        try:quarantine=node_native_hold(pool,runtime)
        except (OSError,ValueError,NodeFailure):quarantine=None
        if quarantine:
            give_up(mine[0],pool,'native crash quarantine',NodeFailure(quarantine));return None
        def count(key):
            try:return int(obs.get(key) or 0)
            except (TypeError,ValueError):return 0
        # A ready task seen in the last inspection counts too: this node may just have lost its claim to another node.
        running,retrying=count('leased')+count('native_busy')+count('ready'),count('retry_wait')
        if not (running or retrying):return None
        # Live idle workers of this cohort: fresh heartbeat and no task (a busy or starting node writes its task).
        fresh=max(30.0,4*poll);now=time.time();idle={worker}
        try:
            for path in (pool.root/'workers').glob('*.json'):
                try:row=read(path)
                except (OSError,ValueError):continue
                if (isinstance(row,dict) and row.get('worker') and row.get('task') is None
                        and isinstance(row.get('at'),(int,float)) and -5<=now-row['at']<=fresh):idle.add(str(row['worker']))
        except OSError:pass
        position=sorted(idle).index(worker)
        # Just started: the other nodes' heartbeats may not be in yet, so hold until they had time to write one.
        warming=time.monotonic()-reserve_state['started']<min(fresh,2*poll+5)
        if position>=ACTIVE_RESERVE and not warming:return None
        why=', '.join(t for t in (f'{running} running or ready' if running else '',f'{retrying} waiting for a retry' if retrying else '') if t)
        if position>=ACTIVE_RESERVE:
            return f'{active} has {why}; reading the other idle nodes before deciding whether this node may move on'
        return (f'reserve node {position+1} of {ACTIVE_RESERVE} for {active} ({why}; {len(idle)} idle node(s)); '
                'later cohorts are left to the other nodes')
    def give_up(index,pool,kind,exc):
        discard_offer(index)
        shared_failures.discard(index)  # node/source faults need fresh admission, not automatic rejoining
        terminal[index]='failed';reasons[cohort(pool)]=f'{type(exc).__name__}: {exc}'
        print(f'[{worker}] {cohort(pool)}: {kind} failed: {type(exc).__name__}: {exc}',flush=True)
        if giveup_retryable(kind,exc):
            giveups[index]=giveups.get(index,0)+1;delay=giveup_retry_delay(giveups[index])
            if cohort_order and cohort(pool)==first_unfinished():delay=min(delay,ACTIVE_GIVEUP_RETRY_CAP)
            retry_at[index]=dict(due=time.monotonic()+delay,until=time.time()+delay,retry=giveups[index],kind=kind,
                                 reason=f'{kind} failed: {type(exc).__name__}: {exc}')
            print(f'[v8] NODE {worker} NOT SERVING {cohort(pool)} {retry_text(index)}',flush=True)
            print(f'[{worker}] {cohort(pool)}: node-local fault; the shared queue was NOT marked failed and the other '
                  f'nodes continue. This node keeps serving its other cohorts and retries {cohort(pool)} in {delay:g}s '
                  '(restarting its runtime and verifying this node again).',flush=True)
            return
        retry_at.pop(index,None)
        print(f'[v8] NODE {worker} NOT SERVING {cohort(pool)} from now: {kind} failed: {type(exc).__name__}: {exc}',flush=True)
        if isinstance(exc,BindingMismatch):
            print(f'[{worker}] {cohort(pool)}: no work is taken for this cohort on this node; nothing was changed.',flush=True)
        elif isinstance(exc,NODE_LOCAL):
            print(f'[{worker}] {cohort(pool)}: this node stops serving {cohort(pool)}. The shared queue was NOT marked '
                  'failed and the other nodes continue. Fix this node (message above), then start the same command '
                  'here again.',flush=True)
        else:
            following='later cohorts remain queued' if cohort_order else 'the other cohort continues'
            print(f'[{worker}] {cohort(pool)}: no new work is taken for this cohort; {following}. '
                  f'Logs: {pool.root/"logs"}. Inspect the cause; the plain launcher recovers older-controller '
                  'failed tasks once after verification.',flush=True)
    def advance_barrier(pool,action):
        try:current=read(pool.root/'binding.json')
        except FileNotFoundError:current=None
        if current!=pool.binding:raise BindingMismatch('Queue identity changed or queue folder moved during CPU stage')
        state=pool.queue_state()
        state['actions'].append(dict(action=action,worker=worker,at=time.time()))
        if action=='primary_finalize':
            receipt=(pool.root/'primary-results.json').read_bytes()
            state['primary_results_receipt_sha256']=hashlib.sha256(receipt).hexdigest()
        state['phase']={'prepare':'train','freeze':'primary_evaluation' if state.get('execution_order',{}).get('policy') in (PRIMARY_ORDER,CONCURRENT_ORDER) else 'extension',
                        'primary_finalize':'extension','finalize':'complete'}[action]
        atomic(pool.root/'state.json',state)
        print(f'[{worker}] {cohort(pool)}: CPU stage finished; phase={state["phase"]}',flush=True)
    def barrier_pending():return any(getattr(r,'pending_barrier',None) is not None for _,r in pairs)
    def report_wait(active,claim_ok,busy):
        candidates=[i for i,(p,_) in enumerate(pairs) if cohort(p)==active] if cohort_order else [i for i in range(len(pairs)) if i not in terminal]
        if not candidates:return
        i=candidates[0]
        held_elsewhere=None
        if cohort_order and i not in scans and i not in offers and getattr(pairs[i][1],'pending_barrier',None) is None:
            # The node spilled to a later cohort and is starting, verifying, inspecting or claiming there: say so,
            # instead of the earlier full cohort's queue-lock wait (user 11-16 '전부 wait_queue_lock 에 걸려 있어').
            busy_later=[j for j in sorted(eligible,key=lambda j:rank(pairs[j][0]))
                        if j!=i and j not in terminal and rank(pairs[j][0])>rank(pairs[i][0])
                        and (j in scans or j in offers or j not in initialized
                             or getattr(pairs[j][1],'pending_barrier',None) is not None)]
            if busy_later:held_elsewhere=cohort(pairs[i][0]);i=busy_later[0]
        pool,runtime=pairs[i];name=cohort(pool)
        facts=dict(getattr(pool,'observation',{}));facts['order']=list(cohort_order)
        if held_elsewhere:facts['nothing_claimable_in']=held_elsewhere
        if not claim_ok and not update_held(busy):diagnostic.publish('WAIT_GPU',name,str(busy),**facts)
        elif not claim_ok and not barrier_pending():diagnostic.publish('WAIT_UPDATE',name,str(busy),**facts)
        elif i in scans:
            detail=getattr(runtime,'verification_stage',scans[i].kind) if scans[i].kind=='node verification' else getattr(runtime,'inspection_stage',scans[i].kind)
            diagnostic.publish('CHECKING',name,detail+'; no GPU task assigned',**facts)
        elif getattr(runtime,'pending_barrier',None) is not None:
            diagnostic.publish('CPU_STAGE',name,'CPU stage '+runtime.pending_barrier[0]+' pending',**facts)
        elif i in control_waits and not facts.get('ready') and facts.get('leased'):
            # The last inspection saw no ready task and every task held by other nodes: that is why this node has
            # nothing here, not the queue lock other nodes take while polling the full queue (user 11-16: 'qwen mbpp에는
            # 모든 노드가 할당되었다고 그래서 더 이상 못하는데 왜 여기 걸려 있는거냐').
            later=[cohort(p) for p,_ in pairs if cohort_order and cohort(p)!=name and rank(p)>rank(pool)]
            diagnostic.publish('WAIT_OTHER_WORKERS',name,f"all {facts.get('leased')} running task(s) of {name} are held by other nodes; "
                               +(f"this node also serves {', '.join(later)}" if later else 'no later cohort on this node'),**facts)
        elif i in control_waits:
            diagnostic.publish('WAIT_QUEUE_LOCK',name,'another worker holds queue control lock: '+str(pool.root/'control.lock'),
                               lock_wait_seconds=round(time.monotonic()-control_waits[i],1),**facts)
        elif i not in initialized:
            diagnostic.publish('STARTING',name,getattr(runtime,'startup_stage','node verification pending'),**facts)
        elif i in offers:diagnostic.publish('WAIT_CLAIM',name,'ready offer found; waiting to acquire or revalidate its lease',**facts)
        elif facts.get('metadata_error'):
            diagnostic.publish('WAIT_METADATA',name,'append-only JSONL is being written; bounded metadata retry: '+facts['metadata_error'],**facts)
        elif facts.get('planning_error') or facts.get('planning_errors'):
            diagnostic.publish('BLOCKED_INPUT',name,'planner reported unresolved input: '+canonical(facts.get('planning_error') or facts['planning_errors']),**facts)
        elif facts.get('leased') or facts.get('native_busy'):
            diagnostic.publish('WAIT_OTHER_WORKERS',name,'tasks hold execution leases; this node has no task',**facts)
        elif facts.get('dependencies'):
            diagnostic.publish('WAIT_CHECKPOINT',name,'parent checkpoint or evaluation snapshot is not ready',**facts)
        elif facts.get('deferred_budget'):
            diagnostic.publish('WAIT_BUDGET',name,'reference endpoint costs have not produced the frozen budget',**facts)
        elif facts.get('extension_waiting_for'):
            diagnostic.publish('WAIT_PRIMARY_RESULTS',name,'extension waits for matched-step result receipt: '+', '.join(facts['extension_waiting_for']),**facts)
        elif facts.get('recovery_wait'):
            diagnostic.publish('WAIT_RECOVERY',name,'failed work is waiting for retry eligibility or another GPU host',**facts)
        elif facts.get('unsealed_plan'):
            diagnostic.publish('WAIT_PLANNER',name,'observed jobs are done but original planner has not sealed this phase',**facts)
        else:diagnostic.publish('WAIT_UNEXPLAINED',name,'no ready task observed; no proven wait reason; inspect queue and task logs',**facts)
    def refresh_before_exit():
        """Recheck original readiness after the stale scan; never claim under this check."""
        refreshed=False
        for i,(pool,runtime) in enumerate(pairs):
            if (i not in initialized or (i in terminal and i not in shared_failures)
                    or getattr(runtime,'pending_barrier',None) is not None):continue
            try:
                with pool.control() as control:
                    if control is None:
                        since=exit_recheck_started.setdefault(i,time.monotonic())
                        if time.monotonic()-since<max(.05,min(poll,1)):
                            stop.wait(min(poll,.05));refreshed=True
                        continue  # finite final-check wait; never spin on a stale lock
                    exit_recheck_started.pop(i,None)
                    state=pool.queue_state()
                    if state.get('phase') not in PHASES:continue
                    if state.get('failure') and not aggregate_task_failure(state['failure']):continue
                    action,offer=pool.tick(runtime,worker,stop,claim=False,offer=True)
                    usable=action in ('ready','advance','complete') or (
                        i in terminal and action=='waiting' and not pool.queue_state().get('failure'))
                    if usable:
                        terminal.pop(i,None);shared_failures.discard(i);reasons.pop(cohort(pool),None)
                        if action=='ready':offers[i]=offer
                        elif action=='complete':terminal[i]='complete'
                        refreshed=True
            except Exception as exc:give_up(i,pool,'final readiness check',exc)
        return refreshed
    def retry_due():
        """Return given-up cohorts whose backoff has passed to service: recreate the runtime, verify again."""
        moved=False;now=time.monotonic()
        for i in tuple(retry_at):
            if terminal.get(i)!='failed':retry_at.pop(i);continue  # served again through another path
            pool,runtime=pairs[i]
            if now-retry_checked[0]>=max(.1,min(poll,5)):
                # Finished on other nodes meanwhile: never wait out the backoff for a complete cohort.
                try:done=read(pool.root/'state.json')
                except (OSError,ValueError):done={}
                if isinstance(done,dict) and done.get('phase')=='complete' and not done.get('failure'):
                    retry_at.pop(i);terminal[i]='complete';reasons.pop(cohort(pool),None);moved=True;continue
            if now<retry_at[i]['due']:continue
            row=retry_at.pop(i);terminal.pop(i,None);initialized.discard(i);discard_offer(i)
            reasons.pop(cohort(pool),None);control_waits.pop(i,None);claim_contention_since.pop(i,None)
            scan=scans.pop(i,None)
            if scan is not None:
                scan.cancelled.set()
                if scan.finished.is_set() and scan.offer_control is not None:scan.offer_control.close()
            print(f'[v8] NODE {worker} SERVING {cohort(pool)} again (retry {row["retry"]}): restarting its runtime '
                  f'after {row["reason"]}',flush=True)
            restart=getattr(runtime,'restart',None)
            if callable(restart):
                try:restart()
                except Exception as exc:give_up(i,pool,'runtime restart',NodeFailure(f'{type(exc).__name__}: {exc}'))
            moved=True
        if now-retry_checked[0]>=max(.1,min(poll,5)):retry_checked[0]=now
        return moved
    while not stop.is_set():
        progressed=False;dispatched=False;short_claim_wait=False;gate_busy=False
        active=None;active_rank=len(cohort_order);held=None
        if retry_at and retry_due():progressed=True
        if cohort_order:
            # Which cohort comes first, from the shared state files only (user 11-16 'dispatch lock queue lock 걸려
            # 있다고'). Taking the one cross-cohort dispatch lock here on every loop of every node, and probing every
            # task lease of finished cohorts under it, made all nodes queue behind each other once a node served
            # several cohorts. Lease probes and claims stay serialized under the dispatch lock in the claim step.
            active=unfinished_cohort(pairs,cohort_order,check_leases=False)
        if cohort_order:
            for i,(pool,_) in enumerate(pairs):
                if terminal.get(i)!='complete':continue
                try:state=read(pool.root/'state.json')
                except FileNotFoundError:state={}
                if state.get('phase')!='complete' or state.get('failure'):
                    # Another verified worker may reopen a historical phase.
                    # Local completion is only a cache: recheck identity/readiness
                    # normally, retaining finished outputs and failure budgets.
                    terminal.pop(i);discard_offer(i)
                    print(f'[{worker}] {cohort(pool)}: shared queue is no longer complete; '
                          'rechecking it before later cohorts',flush=True)
            if active!=sequence_report:
                print(f'[{worker}] cohort order: {" -> ".join(cohort_order)}; '
                      f'active={active or "all complete"}; when nothing here is claimable now this node '
                      'takes ready work from the next cohort in this order',flush=True)
                sequence_report=active
            available={cohort(pool) for pool,_ in pairs}
            active_rank=cohort_order.index(active) if active is not None else len(cohort_order)
            # Later cohorts this node can still serve (the active one may be unavailable or given up here).
            later=[i for i,(pool,_) in enumerate(pairs) if (i not in terminal or i in retry_at) and rank(pool)>active_rank]
            if active is not None and active not in available and not later:
                diagnostic.publish('BLOCKED',active,'required earlier runtime is unavailable; later cohorts remain queued',order=list(cohort_order))
                print(f'[{worker}] BLOCKED: {active} is unavailable on this node; '
                      'later cohorts are not started; queues/checkpoints retained',flush=True)
                return 1
            # Node-local give-ups only; a shared queue failure keeps waiting for recovery by another node.
            stuck=[i for i,(pool,_) in enumerate(pairs) if cohort(pool)==active and terminal.get(i)=='failed'
                   and i not in shared_failures and i not in retry_at]  # a pending retry waits instead
            if active is not None and stuck and not later:
                # Later cohorts are not eligible until the active one completes, so this node would only idle.
                why=reasons.get(active,'see the messages above')
                diagnostic.publish('BLOCKED',active,'this node cannot serve the active cohort: '+why,order=list(cohort_order))
                print(f'[{worker}] EXIT: this node cannot serve the active experiment {active} ({why}); later '
                      'experiments wait for it, so this worker exits instead of idling. Fix the cause and start the '
                      'same command again; queues and checkpoints are kept.',flush=True)
                return 1
            for i,row in retry_at.items():   # the first unfinished cohort is retried within ACTIVE_GIVEUP_RETRY_CAP
                if cohort(pairs[i][0])==active and row['due']-time.monotonic()>ACTIVE_GIVEUP_RETRY_CAP:
                    row['due']=time.monotonic()+ACTIVE_GIVEUP_RETRY_CAP;row['until']=time.time()+ACTIVE_GIVEUP_RETRY_CAP
            limit=max(active_rank,spill_rank)
            # A started scan or CPU stage keeps its cohort eligible until it is collected.
            eligible={i for i,(pool,runtime) in enumerate(pairs) if rank(pool)<=limit or i in scans
                      or getattr(runtime,'pending_barrier',None) is not None}
            # An offer can outlive its phase when another worker finishes the cohort, and an
            # ineligible offer would keep its retained control lock: drop both.
            for i in tuple(offers):
                if i not in eligible:discard_offer(i)
                elif time.time()-offers[i].get('shared_at',time.time())>SCAN_SHARE_SECONDS:discard_offer(i)  # inspect afresh
            if limit!=spill_report and limit>active_rank and limit<len(cohort_order):
                print(f'[{worker}] nothing claimable now in earlier cohorts; also serving {cohort_order[limit]} '
                      '(earlier cohorts keep priority; one task at a time on this GPU group)',flush=True)
                spill_report=limit
        else:eligible=set(range(len(pairs)))
        # A different verified worker may recover the shared queue while this
        # worker is waiting on the other cohort. Do not cache that old failure
        # forever. This never clears a failure or retries a job on its own.
        for index in tuple(shared_failures):
            if index not in eligible:continue
            now=time.monotonic()
            if now-recovery_checked.get(index,-float('inf'))<max(.1,min(poll,5)):continue
            recovery_checked[index]=now;pool,runtime=pairs[index]
            try:
                with pool.control() as control:
                    if control is None:continue
                    if read(pool.root/'binding.json')!=pool.binding:
                        raise BindingMismatch('Queue identity changed while waiting for shared recovery')
                    pool.refund_periodic_dump_crashes(worker)
                    state=pool.queue_state()
                    if index in initialized and pool.recover_budget_api_failure(controller_sha256,worker):state=pool.queue_state()
                    if state.get('failure'):continue
                    shared_failures.discard(index);reasons.pop(cohort(pool),None)
                    if state['phase']=='complete':terminal[index]='complete';continue
                    terminal.pop(index,None);initialized.discard(index);discard_offer(index)
                    print(f'[{worker}] {cohort(pool)}: shared queue recovered on another node; '
                          'rechecking original inputs before taking ready work',flush=True)
                    progressed=True
            except Exception as exc:give_up(index,pool,'queue recovery check',exc)
        claim_ok,busy=(True,None) if gpu_gate is None else gpu_gate()
        if claim_ok:
            held=update_hold()
            if held:claim_ok,busy=False,held
        if not claim_ok:
            for index in tuple(offer_controls):discard_offer(index)
        if not claim_ok and time.monotonic()-busy_report>=300:
            print(f'[{worker}] waiting before new GPU work: {busy}',flush=True);busy_report=time.monotonic()
        # Discovery/preparation is outside the global dispatch lock. Sequential
        # callers only prepare the current cohort; archival callers warm all of them.
        for index in [(cursor+i)%len(pairs) for i in range(len(pairs))]:
            if index not in eligible:continue
            if index in terminal:continue
            pool,runtime=pairs[index];action='waiting'
            if index in scans:
                scan=scans[index]
                if not scan.finished.is_set():
                    now=time.monotonic()
                    limit=(getattr(runtime,'verify_timeout',NODE_VERIFY_TIMEOUT) if scan.kind=='node verification'
                           else getattr(runtime,'scan_timeout',300))
                    if now-scan.started>=limit:
                        scan.cancelled.set();del scans[index]
                        detail=getattr(runtime,'verification_stage','checking node inputs') if scan.kind=='node verification' else getattr(runtime,'inspection_stage','original planner/ready/done metadata')
                        give_up(index,pool,scan.kind,NodeFailure(f'{scan.kind} timed out after {limit:g}s at {detail}; no GPU task claimed'))
                        if scan.kind!='node verification':
                            # The stuck inspection thread still holds this queue's control.lock (dup'd fd).
                            # Staying alive would keep every other node waiting on it; exiting releases it.
                            if scan.offer_control is not None:scan.offer_control.close()
                            print(f'[{worker}] EXIT: {cohort(pool)} {scan.kind} did not finish in {limit:g}s at {detail}; '
                                  'this worker exits so the queue lock is released for the other nodes. '
                                  'Start the same command again on this node.',flush=True)
                            diagnostic.publish('BLOCKED',cohort(pool),f'{scan.kind} timed out; worker exited to release the queue lock')
                            return 1
                        progressed=True;continue
                    if now-scan.reported>=30:
                        print(f'[{worker}] {cohort(pool)}: {scan.kind} pending {now-scan.started:.0f}s; '
                              'no cross-cohort dispatch lock held',flush=True);scan.reported=now
                    continue
                del scans[index]
                if isinstance(scan.error,QueueReset):progressed=True;continue
                if scan.error is not None:
                    give_up(index,pool,scan.kind,scan.error);progressed=True;continue
                action,offer=scan.result
                if scan.kind=='task inspection':share_scan(index,pool,scan,action,offer)
                if action=='verified':
                    initialized.add(index);progressed=True
                    if claim_ok:
                        try:
                            recheck_crashed_node(pool,runtime,device_fds,worker,diagnostic);node_rechecks.add(index)
                        except Exception as exc:give_up(index,pool,'GPU re-admission',exc)
                if action=='ready':
                    offers[index]=offer
                    if not scan.offer_control.closed:
                        offer_controls[index]=offer_cleanup.enter_context(scan.offer_control)
                        control_waits.pop(index,None)
                if action in ('advance','ready'):progressed=True
                if action in ('failed','complete'):
                    terminal[index]=action
                    if action=='failed':
                        shared_failures.add(index)
                        reasons[cohort(pool)]=canonical(pool.queue_state().get('failure'))
                continue
            if index in offers:continue
            if hasattr(runtime,'startup_ready') and not runtime.startup_ready():
                try:runtime.heartbeat('runtime_initialization')
                except (OSError,NodeFailure):pass  # reported by the next queue access below
                continue
            if getattr(runtime,'pending_barrier',None) is not None:
                result=runtime.poll_barrier(stop)
                if result is None:continue
                stage,held,error=result
                try:
                    if stop.is_set():return 75
                    if error is not None:raise error
                    advance_barrier(pool,stage);progressed=True
                except BindingMismatch as exc:give_up(index,pool,'queue identity',exc)
                except NODE_LOCAL as exc:give_up(index,pool,'node runtime',exc)
                except Exception as exc:
                    pool.fail(f'{type(exc).__name__}: {exc}');give_up(index,pool,'CPU stage',exc)
                finally:os.close(held)
                continue
            if index not in initialized and getattr(runtime,'independent_startup',False):
                # Hashing this node's own inputs needs no queue lock. Holding control.lock here made every
                # starting node verify one after another behind it (WAIT_QUEUE_LOCK) and time out.
                scans[index]=TaskScan(pool,runtime,worker,stop,None,verify=True,retry=retry)
                continue
            if index in initialized:
                shared=shared_scan(index,pool)
                if shared is not None:
                    action,offer,row=shared;control_waits.pop(index,None)
                    pool.observation=dict(row.get('observation') or {},shared_from=row.get('worker'),
                                          shared_age_seconds=round(time.time()-row['at'],1))
                    if action=='ready':offers[index]=dict(offer,shared_at=row['at']);progressed=True  # claim_offered revalidates it
                    try:runtime.heartbeat(None)
                    except (OSError,NodeFailure):pass
                    continue
            try:
                control_context=pool.control();control=control_context.__enter__()
            except NODE_LOCAL as exc:
                give_up(index,pool,'queue access',exc);progressed=True;continue
            try:
                if control is None:control_waits.setdefault(index,time.monotonic())
                else:control_waits.pop(index,None)
                if control is not None:
                    if index not in initialized:
                        # A mismatched joining node must not poison healthy workers: no pool.fail here.
                        if getattr(runtime,'independent_startup',False):
                            scans[index]=TaskScan(pool,runtime,worker,stop,control.fd,verify=True,retry=retry)
                            continue
                        try:runtime.verify_node();pool.initialize(retry=retry,worker=worker,recover=pool.controller_sha256)
                        except QueueReset:
                            progressed=True;continue  # re-lock control.lock of the fresh directory first
                        except Exception as exc:
                            give_up(index,pool,'node check',exc);progressed=True;continue
                        initialized.add(index)
                    try:
                        # Inspect only phase metadata here. Re-scanning the whole DAG on every
                        # idle node keeps control.lock busy before the shared claim gate, and
                        # repeats the same RPC/filesystem work immediately before the real tick.
                        try:current=read(pool.root/'binding.json')
                        except FileNotFoundError:current=None
                        if current!=pool.binding:
                            # Moved/replaced by another node's --new-queue: never write this worker's failure there.
                            raise BindingMismatch('Queue identity changed while this worker was running' if current else
                                                  f'This queue folder ({pool.root}) was moved or replaced while this worker '
                                                  'was running; restart this worker (same command).')
                        state=pool.prioritize_primary_results(runtime)
                        if isinstance(state.get('failure'),dict) and state['failure'].get('task'):
                            pool.initialize();state=pool.queue_state()
                        action='failed' if state.get('failure') else state['phase']
                        if action in PHASES:action='waiting'
                        if state['phase'] in PHASES and (not state.get('failure') or aggregate_task_failure(state['failure'])):
                            action='waiting'
                            if getattr(runtime,'independent_startup',False):
                                marks=queue_marks(pool)
                                scans[index]=TaskScan(pool,runtime,worker,stop,control.fd,
                                    retain_offer=bool(cohort_order and claim_ok and index in node_rechecks))
                                scans[index].marks=marks
                            else:
                                action,offer=pool.tick(runtime,worker,stop,claim=False,offer=True)
                                if action=='ready':offers[index]=offer
                            # Original actor scans never hold the cross-cohort dispatch lock.
                        if action in CPU_PHASES:
                            diagnostic.publish('CPU_STAGE',cohort(pool),'CPU stage '+action,phase=action,order=list(cohort_order))
                            print(f'[{worker}] {cohort(pool)}: CPU stage {action} started',flush=True)
                            if getattr(runtime,'independent_startup',False):
                                runtime.begin_barrier(action,control.fd,stop,device_fds);action='waiting'
                            else:
                                runtime.barrier(action,control.fd,stop,device_fds)
                                if stop.is_set():return 75
                                advance_barrier(pool,action);action='advance'
                    except BindingMismatch as exc:
                        # This is now somebody else's queue. Do not write our failure into its state.
                        give_up(index,pool,'queue identity',exc);progressed=True;continue
                    except NODE_LOCAL as exc:
                        if stop.is_set():return 75
                        give_up(index,pool,'node runtime',exc);progressed=True;continue
                    except Exception as exc:
                        if stop.is_set():return 75
                        if (pool.root/'state.json').exists():pool.fail(f'{type(exc).__name__}: {exc}')
                        give_up(index,pool,'stage',exc);progressed=True;continue
            finally:control_context.__exit__(None,None,None)
            try:runtime.heartbeat(None)
            except NODE_LOCAL as exc:
                give_up(index,pool,'queue access',exc);progressed=True;continue
            except (FileNotFoundError,FileExistsError,NotADirectoryError) as exc:
                moved=BindingMismatch(f'{cohort(pool)}: this queue folder ({pool.root}) was moved or replaced while this worker '
                                      f'was running ({type(exc).__name__}); restart this worker (same command).')
                give_up(index,pool,'queue identity',moved);progressed=True;continue
            if action=='failed':
                state=pool.queue_state()
                reasons[cohort(pool)]=canonical(state.get('failure'))
                print(f'[{worker}] {cohort(pool)}: queue has a recorded failure: {reasons[cohort(pool)]}. '
                      f'Inspect the recorded error and logs at {pool.root/"logs"}.',flush=True)
            if action in ('complete','failed'):
                if action=='failed':shared_failures.add(index)
                terminal[index]=action;progressed=True;continue
            if action=='advance':progressed=True
        if claim_ok:
            for i,(pool,runtime) in enumerate(pairs):
                if i not in eligible or i not in initialized or i in terminal or i in node_rechecks or i in scans:continue
                if getattr(runtime,'pending_barrier',None) is not None:continue
                node_rechecks.add(i)
                try:recheck_crashed_node(pool,runtime,device_fds,worker,diagnostic)
                except Exception as exc:
                    give_up(i,pool,'GPU re-admission',exc);progressed=True
        pending_setup=[]
        for index,(pool,runtime) in enumerate(pairs):
            if index not in eligible:continue
            if index in terminal:continue
            # Original Runtime actors prepare independently. A blocked Qwen setup must
            # not keep an otherwise verified OLMo node from claiming ready work.
            if getattr(runtime,'independent_startup',False):continue
            if index not in initialized:
                pending_setup.append(cohort(pool)+' (node verification)');continue
            state=read(pool.root/'state.json') if (pool.root/'state.json').exists() else {}
            if not state.get('failure') and state.get('phase','prepare')=='prepare':pending_setup.append(cohort(pool))
        claimed=None;index=None
        if pending_setup:
            if time.monotonic()-setup_report>=30:
                print(f'[{worker}] waiting for cohort setup: {", ".join(pending_setup)}; no GPU jobs claimed until healthy setup is ready',flush=True)
                setup_report=time.monotonic()
        elif claim_ok and len(terminal)<len(pairs):
            gate=Lock(dispatch_lock)
            try:
                try:gate_held=gate.acquire()
                except NodeFailure as exc:
                    for i,(p,_) in enumerate(pairs):
                        if i not in terminal:give_up(i,p,'queue access',exc)
                    gate_held=False;progressed=True
                if not gate_held and offers:gate_busy=True  # another node is claiming; this offer keeps its lease
                if gate_held:
                    # Kernel-held task leases, not heartbeat age or stale running labels, are
                    # authoritative. Claim while this same shared lock remains held.
                    live={i:0 if distinct_ranks else len(pool.held_leases()) for i,(pool,_) in enumerate(pairs)
                          if i in initialized and i not in terminal and i in offers and i in eligible}
                    # Sequential: the earliest cohort in the order first; archival: fewest live leases.
                    order=sorted(live,key=lambda i:((rank(pairs[i][0]) if cohort_order else 0),live[i],(i-cursor)%len(pairs)))
                    for candidate in order:
                        pool,runtime=pairs[candidate]
                        if cohort_order and rank(pool)>active_rank:
                            first=[i for i,(p,_) in enumerate(pairs) if cohort(p)==active and i not in terminal]
                            if first and (first[0] in scans or first[0] not in initialized):
                                break   # the first unfinished cohort is still being inspected or started here: wait for it
                            held=reserve_hold(active)
                            if held:
                                for other in tuple(offers):
                                    if rank(pairs[other][0])>active_rank:discard_offer(other)
                                spill_rank=-1;break
                        if cohort_order and any(scan.kind=='task inspection' and rank(pairs[j][0])<rank(pool)
                                                and time.monotonic()-scan.started<SPILL_LOCK_GRACE for j,scan in scans.items()):
                            short_claim_wait=True;break  # an earlier cohort is being inspected right now: it may offer work
                        try:
                            hold=node_native_hold(pool,runtime)
                            if hold:
                                discard_offer(candidate)
                                give_up(candidate,pool,'native crash quarantine',NodeFailure(hold))
                                progressed=True;continue
                            # Another node's --new-queue may be replacing this folder right now: the next
                            # phase-1 pass reports the moved/replaced queue; never recreate its state here.
                            try:state=read(pool.root/'state.json')
                            except FileNotFoundError:
                                discard_offer(candidate);continue
                            if state.get('phase') not in PHASES or state.get('failure'):
                                discard_offer(candidate);continue
                            # A ready inspector retains this exact control lease.
                            # Reuse it instead of letting another whole DAG scan
                            # overtake the claim between close() and reacquisition.
                            owned=offer_controls.pop(candidate,None)
                            control_context=owned if owned is not None else pool.control()
                            control=control_context.__enter__()
                        except NODE_LOCAL as exc:
                            give_up(candidate,pool,'queue access',exc);progressed=True;continue
                        try:
                            if control is None:
                                # Give short metadata writes a chance to preserve model balance.
                                # A prolonged scan must not idle this GPU when another cohort is ready.
                                since=claim_contention_since.setdefault(candidate,time.monotonic())
                                control_waits.setdefault(candidate,since)
                                if time.monotonic()-since<.25:
                                    short_claim_wait=True;break
                                continue
                            claim_contention_since.pop(candidate,None)
                            control_waits.pop(candidate,None)
                            try:
                                action,claimed=pool.claim_offered(offers.pop(candidate),worker,stop)
                            except BindingMismatch as exc:
                                give_up(candidate,pool,'queue identity',exc);progressed=True;continue
                            except NODE_LOCAL as exc:
                                if stop.is_set():return 75
                                give_up(candidate,pool,'node runtime',exc);progressed=True;continue
                            except Exception as exc:
                                if stop.is_set():return 75
                                pool.fail(f'{type(exc).__name__}: {exc}')
                                give_up(candidate,pool,'stage',exc);progressed=True;continue
                            if claimed:index=candidate;break
                            if action=='advance':progressed=True
                        finally:control_context.__exit__(None,None,None)
            finally:gate.close()
        # GPU execution, admission and CPU barriers NEVER hold the cross-cohort lock.
        if claimed is not None:
            idle_state['since']=None;idle_state['reported']=-float('inf')
            # Release every other retained offer (and its queue control lock) before a long task. A task
            # inspection of another cohort still in flight must not keep that cohort's control.lock either.
            for other in tuple(offers):discard_offer(other)
            for other,scan in tuple(scans.items()):
                if other==index or scan.kind!='task inspection':continue
                scan.cancelled.set()  # its thread releases the lease instead of retaining it for a claim
                if scan.finished.is_set():
                    if scan.offer_control is not None:scan.offer_control.close()
                    del scans[other]
            spill_rank=-1
            pool,runtime=pairs[index]
            lease,key,task,attempt=claimed;cursor=(index+1)%len(pairs)
            for other,(_,other_runtime) in enumerate(pairs):
                # Not an idle reserve node of any other cohort while this task runs (reserve_hold counts idle rows).
                if other in terminal:continue
                try:other_runtime.heartbeat(f'busy {cohort(pool)} {key}')
                except Exception:pass
            log=pool.root/'logs'/(attempt['id']+'.log')
            diagnostic.publish('ASSIGNED',cohort(pool),'task lease acquired; waiting for child work records',
                               task=key,kind=task['kind'],run_id=task.get('run_id'),seed=task.get('training_seed'),
                               method=task.get('method_id'),log=str(log),order=list(cohort_order))
            print(f'[{worker}] {time.strftime("%Y-%m-%d %H:%M:%S")} {cohort(pool)} {task["kind"]} '
                  f'{task.get("method_id",task.get("task_id"))} seed={task.get("training_seed")} log={log}',flush=True)
            node_error=None;checked=None
            try:
                try:
                    if preflight and not stop.is_set() and isinstance(getattr(runtime,'python',None),str):
                        # Before every launch (user 11-16 '환경 문제인지 torch 버전 문제인지'): the cohort's own Python,
                        # its versions against this queue and the model family, CUDA on these GPUs, free GPUs, /dev/shm.
                        diagnostic.publish('PREFLIGHT',cohort(pool),'checking Python/torch/CUDA/GPUs/shm before start',task=key)
                        checked=PREFLIGHT_CHECK(cohort(pool),runtime.python,list(getattr(runtime,'gpus',()) or ()),pool.binding)
                        print(f'[{worker}] '+preflight_line(checked),flush=True)
                        # Default: record only (user 11-16 'idle이 너무 잡히는데'). V8_PREFLIGHT=block refuses the launch,
                        # and even then the node only pauses (V8_PREFLIGHT_RETRY_SECONDS) and keeps serving.
                        if not checked.get('ok') and PREFLIGHT_MODE=='block':
                            raise PreflightFailed('preflight failed; task not started on this node, another node takes it: '+
                                                  '; '.join(checked.get('failures',[])))
                    rc=75 if stop.is_set() else runtime.execute(task,pool.root/'tasks'/(key+'.json'),lease.fd,stop,device_fds,attempt)
                except Exception as exc:
                    # Launch-side faults (admission, node verification, dead actor, missing interpreter) happen on
                    # this node before a training result exists: never a task failure or a queue failure.
                    rc=None;node_error=f'{type(exc).__name__}: {exc} (node {worker})'
                    print(f'[{worker}] launch failure on this node: {node_error}',flush=True)
                diagnostic.publish('RECORDING_RESULT',cohort(pool),'child exited; recording and validating its result',exit_code=rc,task=key)
                recorded=None
                while recorded is None:
                    try:
                        with pool.control() as control:
                            if control is not None:recorded=pool.finish(key,rc,runtime,stop,node_error=node_error,preflight=checked)
                    except NODE_LOCAL as exc:
                        node_error=node_error or f'{type(exc).__name__}: {exc}';recorded='unrecorded'
                        print(f'[{worker}] could not record the attempt ({exc}); the released lease lets another '
                              'node adopt or rerun it',flush=True)
                    if recorded is None:
                        diagnostic.publish('RECORDING_RESULT',cohort(pool),'child exited; waiting for queue control lock to record its result',exit_code=rc,task=key)
                        time.sleep(min(poll,.2))
            finally:lease.close()
            print(f'[{worker}] {time.strftime("%Y-%m-%d %H:%M:%S")} {cohort(pool)} task finished rc={rc} '
                  f'state={recorded} log={log}',flush=True)
            if node_error is not None and node_error.startswith('PreflightFailed'):
                # Not a reason to stop serving this cohort: the task is back in the queue for any node, this node
                # waits a little (the GPU may still be releasing memory) and claims again.
                print(f'[{worker}] preflight refused this launch; retrying in {PREFLIGHT_RETRY_SECONDS:g}s',flush=True)
                stop.wait(PREFLIGHT_RETRY_SECONDS)
            elif node_error is not None:give_up(index,pool,'node launch',NodeFailure(node_error))
            else:
                giveups.pop(index,None)  # served normally again: the next node-local give-up starts the backoff anew
                hold=node_native_hold(pool,runtime)
                if hold:give_up(index,pool,'native crash quarantine',NodeFailure(hold))
                elif fatal_native_failure(read(pool.root/'jobs'/(key+'.json'))):node_rechecks.discard(index)
            dispatched=True
        if cohort_order:
            current=unfinished_cohort(pairs,cohort_order,check_leases=False)
            failed=[i for i,(pool,_) in enumerate(pairs) if cohort(pool)==current and terminal.get(i)=='failed']
            if failed and not any(i not in terminal or i in retry_at for i in range(len(pairs))):
                if refresh_before_exit():continue
                diagnostic.publish('BLOCKED',current,reasons.get(current,'active cohort cannot proceed'),order=list(cohort_order))
                print(f'[{worker}] STOPPED: {current} is not complete; later cohorts remain queued; '
                      f'{reasons.get(current,"see queue logs")}',flush=True)
                record_worker_retirement(pairs,worker,'active sequential cohort unavailable on this node',
                                         [dict(cohort=current,detail=reasons.get(current))])
                return 1
        if not retry_at and (any(v=='failed' for v in terminal.values()) or any(getattr(r,'primary_unavailable_cohorts',()) for _,r in pairs)):
            deferred=[]
            for i,(pool,runtime) in enumerate(pairs):
                if i in terminal:continue
                if (not getattr(runtime,'primary_results_first',False)
                        or pool.queue_state()['phase']!='extension'
                        or not primary_results_ready(pool.root.parent) or not pool.extension_blockers()):break
                deferred.append(i)
            else:
                for i in deferred:
                    pool=pairs[i][0]
                    give_up(i,pool,'extension deferred',NodeFailure('matched-step results are saved; '
                        'extension still waits for unavailable cohorts: '+', '.join(pool.extension_blockers())))
        if len(terminal)==len(pairs) and not retry_at:
            failed=[cohort(pairs[i][0]) for i,v in terminal.items() if v=='failed']
            if failed and refresh_before_exit():continue
            unserved=(unfinished_cohort(pairs,cohort_order,check_leases=False) if cohort_order and not failed else None)
            if unserved is not None and unserved not in {cohort(p) for p,_ in pairs}:
                diagnostic.publish('BLOCKED',unserved,'required runtime is unavailable on this node',order=list(cohort_order))
                print(f'[{worker}] BLOCKED: every cohort this node can serve is done, but {unserved} is unavailable '
                      'on this node; queues/checkpoints retained',flush=True)
                return 1
            for name in failed:print(f'[{worker}] {name}: FAILED: {reasons.get(name,"see its .multi-node/state.json")}',flush=True)
            done=[cohort(pairs[i][0]) for i,v in terminal.items() if v=='complete']
            if done:print(f'[{worker}] complete: {", ".join(done)}',flush=True)
            if failed:record_worker_retirement(pairs,worker,'no eligible work after final readiness check',
                                               [dict(cohort=name,detail=reasons.get(name)) for name in failed])
            diagnostic.publish('BLOCKED' if failed else 'COMPLETE',None,'; '.join(failed) if failed else 'all selected cohorts finalized')
            return 1 if failed else 0
        idle=(not dispatched and not progressed and not offers and not scans and not pending_setup
              and all(i in initialized or i in terminal for i in eligible)
              and not any(getattr(r,'pending_barrier',None) is not None for _,r in pairs))
        # A short queue-lock wait is another node's metadata write, not proof the cohort has no work.
        lock_wait=any(i in eligible and time.monotonic()-since<SPILL_LOCK_GRACE for i,since in control_waits.items())
        if dispatched or offers:stall_since=time.monotonic()
        if cohort_order:
            # The spill timer starts only after this node finished starting and inspecting the first unfinished cohort.
            first=[i for i,(p,_) in enumerate(pairs) if cohort(p)==active and i not in terminal]
            if first and (first[0] in scans or first[0] not in initialized):stall_since=time.monotonic()
        stalled=cohort_order and not offers and time.monotonic()-stall_since>=SPILL_STALL_SECONDS
        if cohort_order and claim_ok and ((idle and not lock_wait) or stalled):
            following=[rank(p) for i,(p,_) in enumerate(pairs) if i not in terminal and rank(p)>limit]
            if following and not held:held=reserve_hold(active)
            if following and held:
                stall_since=time.monotonic();spill_rank=-1
            elif following:
                report_wait(active,claim_ok,busy)  # why the earlier cohort has nothing for this node
                if stalled and not idle:
                    print(f'[{worker}] no task from {cohort_order[limit] if limit<len(cohort_order) else "earlier cohorts"} for '
                          f'{time.monotonic()-stall_since:.0f}s (queue lock busy or nothing ready); also serving '
                          f'{cohort_order[min(following)]}',flush=True)
                stall_since=time.monotonic()
                spill_rank=min(following);continue  # strictly increases: bounded, never a busy loop
        if held!=reserve_state['text']:
            if held and not reserve_state['text']:print(f'[{worker}] {held}',flush=True)
            reserve_state['text']=held
        if idle and not retry_at and not held:  # a pending node-local retry is waited for, never a reason to exit
            if cohort_order:
                # Exit only when every cohort this node still serves is blocked by its own failures.
                inspected=[pair for i,pair in enumerate(pairs) if i not in terminal]
                blocked=None if not inspected else quiescent_failure(inspected) if len(inspected)==1 else (
                    None if any(quiescent_failure([pair]) is None for pair in inspected) else quiescent_failure(inspected))
            else:blocked=quiescent_failure(pairs)
            if blocked:
                if refresh_before_exit():continue
                blocked=quiescent_failure(inspected) if cohort_order else quiescent_failure(pairs)
                if not blocked:continue
                record_worker_retirement(pairs,worker,blocked['reason'],blocked['faults'])
                diagnostic.publish('BLOCKED',active,blocked['reason'],faults=blocked['faults'])
                print(f'[{worker}] STOPPED: {blocked["reason"]}; '+canonical(blocked['faults'])+
                      '; failed checkpoints/logs retained; no torchrun relaunch',flush=True)
                return 1
        if not dispatched:
            now=time.monotonic()
            if idle_state['since'] is None:idle_state['since']=now
            idle_state['gpu_busy']=None if claim_ok else busy
            if now-idle_state['since']>=IDLE_REPORT_SECONDS and now-idle_state['reported']>=IDLE_REPORT_SECONDS:
                idle_state['reported']=now
                try:print(f'[v8] NODE {worker} '+idle_reason(),flush=True)
                except Exception as exc:print(f'[v8] NODE {worker} IDLE (reason unavailable: {type(exc).__name__}: {exc})',flush=True)
        if not dispatched and not progressed:
            if not claim_ok and update_held(busy) and not barrier_pending():
                # The supervisor restarts this launch only after this record: written at the end of a loop pass that
                # claimed nothing because of the hold, with no CPU stage and no GPU re-admission in flight.
                diagnostic.publish('WAIT_UPDATE',active,str(busy),order=list(cohort_order))
            elif retry_at and len(terminal)==len(pairs):
                first=min(retry_at,key=lambda i:retry_at[i]['due'])
                diagnostic.publish('WAIT_RETRY',cohort(pairs[first][0]),'NOT SERVING '+cohort(pairs[first][0])+' '+retry_text(first))
            elif held:diagnostic.publish('WAIT_RESERVE',active,held,order=list(cohort_order))
            else:report_wait(active,claim_ok,busy)
            wait=min(poll,.05) if short_claim_wait else min(poll,.5) if gate_busy else poll
            if retry_at:wait=max(.01,min(wait,min(row['due'] for row in retry_at.values())-time.monotonic()))
            # A pending verification/inspection is collected as soon as it finishes, not one whole poll later
            # (each of startup -> verification -> inspection -> claim cost up to a poll on an idle node).
            pending=[scan for i,scan in scans.items() if i not in terminal]
            if pending and not any(scan.finished.is_set() for scan in pending):
                deadline=time.monotonic()+wait
                while (not stop.is_set() and time.monotonic()<deadline
                       and not any(scan.finished.is_set() for scan in pending)):
                    stop.wait(min(.05,max(0,deadline-time.monotonic())))
            else:stop.wait(min(wait,.05) if pending else wait)
    diagnostic.publish('STOPPED',None,'stop requested; checkpoints and queue retained')
    return 75


BASE_PACKAGES=('torch','transformers','peft','numpy','math-verify')
# Qwen seed runs pinned fla-core (srgc_qwen35.runtime_packages, checked again by cluster_child.prepare_runtime):
# without it transformers silently falls back to the slow torch DeltaNet kernel, so it is part of the identity.
QWEN_PACKAGES=('fla-core',)


def child_exit(code,exit_file):
    """The attempt's real exit code. torchrun reports a worker's exit 75 (stopped at a safe point, or the task is
    busy elsewhere) as its own exit 1; rank 0 records the real code in $V8_EXIT_FILE."""
    if code in (0,75):return code
    try:record=json.loads(Path(exit_file).read_text())
    except (OSError,ValueError):return code
    if isinstance(record,dict) and record.get('exit_code')==75 and record.get('status') in ('busy','interrupted'):
        print(f"[v8] child recorded exit 75 ({record.get('status')}); torchrun exit {code} is an interruption, "
              'not a task failure',flush=True)
        return 75
    return code


def python_identity(executable,extra=()):
    names=(*BASE_PACKAGES,*extra)
    code="""import importlib.metadata,json,sys
packages={}
for name in """+repr(names)+""":
 try:packages[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:packages[name]=None
print(json.dumps({'python_version':sys.version,'packages':packages}))
"""
    result=subprocess.run([executable,'-c',code],text=True,capture_output=True,timeout=30)
    if result.returncode:raise ValueError('Cannot inspect cohort Python: '+executable+': '+result.stderr.strip())
    return json.loads(result.stdout)


def cohort_python(cohort,fallback,environment=None):
    """Same precedence as the original launchers: run_srgc.sh (math PAIR_PYTHON, mbpp SWITCH_PYTHON)
    and run_srgc_qwen35.sh (QWEN_PYTHON first). The OLMo cohort never inherits a Qwen interpreter."""
    env=os.environ if environment is None else environment
    dataset=cohort.rsplit('_',1)[-1]
    names=['V8_PYTHON_'+cohort.upper()]
    if cohort.startswith('qwen_'):names.append('QWEN_PYTHON')
    names.append({'math':'PAIR_PYTHON','mbpp':'SWITCH_PYTHON'}.get(dataset,''))
    for name in names:
        if name and env.get(name):return env[name]
    return fallback


def gpu_utilisation(gpus):
    """'97%,96%,98%,95%' for the worker's GPUs, or 'unknown' (nvidia-smi missing/slow)."""
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,timeout=10).stdout
    except (OSError,subprocess.SubprocessError):
        return 'unknown'
    rows={}
    for line in out.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)==3:rows[parts[0]]=f'{parts[1]}%/{parts[2]}MiB'
    picked=[rows.get(str(g)) for g in gpus]
    return ','.join(x for x in picked if x) or 'unknown'


BUSY_MIB=4000  # the original SRGC worker's admission threshold (srgc_rebuttal/cluster.gpu_identity)


def gpu_inventory(gpus,run=subprocess.run):
    """Allocated GPUs from nvidia-smi (index, UUID, used/total MiB, name), or None when unavailable."""
    try:
        result=run(['nvidia-smi','--id='+','.join(gpus),'--query-gpu=index,uuid,memory.used,memory.total,name',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.SubprocessError):return None
    if result.returncode:return None
    rows=[[c.strip() for c in line.split(',',4)] for line in result.stdout.strip().splitlines() if line.strip()]
    if len(rows)!=len(gpus) or any(len(r)!=5 for r in rows):return None
    try:return [dict(index=r[0],uuid=r[1],used_mib=int(float(r[2])),total_mib=int(float(r[3])),name=r[4]) for r in rows]
    except ValueError:return None


def gpu_processes(run=subprocess.run):
    try:
        result=run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],
                   capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.SubprocessError):return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] if result.returncode==0 else []


def gpu_lock_root(environment=None):
    """The original SRGC workers' device-lease directory (scripts/run_srgc_rebuttal.py --node-lock-root):
    <SRGC_STORAGE_ROOT or OM_WORK/srgc-rebuttal>/gpu-node-locks on the group volume."""
    env=os.environ if environment is None else environment
    if env.get('V8_GPU_LOCK_ROOT'):return Path(env['V8_GPU_LOCK_ROOT'])
    group=Path(env.get('GROUP_VOLUME') or '/group-volume')
    work=Path(env.get('OM_WORK') or group/(env.get('OM_USER') or 'minsoo3.kim')/'offpolicy-misranking')
    return Path(env.get('SRGC_STORAGE_ROOT') or work/'srgc-rebuttal')/'gpu-node-locks'


def gpu_lease_names(gpus,inventory,host=None):
    """sha256(UUID).lock exactly like srgc_rebuttal.cluster.device_leases, so V8 and the original SRGC
    workers exclude each other; without nvidia-smi, (host, device id) keys on the same volume."""
    if inventory:return [(row['uuid'],hashlib.sha256(row['uuid'].encode()).hexdigest()+'.lock') for row in inventory]
    host=host or socket.gethostname()
    return [(f'{host}:{gpu}',digest([host,gpu])+'.lock') for gpu in gpus]


PREFLIGHT_POLICY='node-preflight-before-each-task-v1'
# Recorded (not part of the queue binding): native libraries whose ABI must match torch on this node.
PREFLIGHT_PACKAGES=('torch','transformers','peft','numpy','math-verify','fla-core','flash-linear-attention','triton',
                    'flash-attn','accelerate','tokenizers','safetensors','sympy','vllm','causal-conv1d',
                    'nvidia-nccl-cu12','nvidia-cudnn-cu12','nvidia-cublas-cu12','nvidia-cuda-runtime-cu12')
PREFLIGHT_ENV=('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','PYTHONHOME','CUDA_HOME','CUDA_PATH',
               'PYTORCH_CUDA_ALLOC_CONF','TORCH_NCCL_ASYNC_ERROR_HANDLING','TORCH_NCCL_BLOCKING_WAIT','TRITON_CACHE_DIR',
               'HF_HOME','TRANSFORMERS_CACHE','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM','VIRTUAL_ENV','CONDA_PREFIX')
_PREFLIGHT_SMOKE=r"""
import faulthandler,json,os,sys,time
faulthandler.enable(all_threads=True)
import importlib.metadata as md
out=dict(executable=sys.executable,python_version=sys.version,prefix=sys.prefix,packages={},imports={})
for name in json.loads(sys.argv[1]):
    try:out['packages'][name]=md.version(name)
    except Exception:out['packages'][name]=None
def libraries():
    found=set()
    try:
        for line in open('/proc/self/maps'):
            path=line.split()[-1] if line.strip() else ''
            if any(k in path for k in ('libnccl','libcudnn','libcublas','libcudart','libcuda.so','libnvrtc','libc10_cuda')):found.add(path)
    except OSError:pass
    return sorted(found)
try:
    import torch
    t=dict(version=torch.__version__,cuda=torch.version.cuda,cxx11_abi=bool(getattr(torch._C,'_GLIBCXX_USE_CXX11_ABI',False)),
           file=torch.__file__)
    try:t['cudnn']=torch.backends.cudnn.version()
    except Exception as exc:t['cudnn']='error: '+str(exc)
    try:
        v=torch.cuda.nccl.version();t['nccl']='.'.join(map(str,v)) if isinstance(v,tuple) else str(v)
    except Exception as exc:t['nccl']='error: '+str(exc)
    out['torch']=t
    if sys.argv[2]=='1':
        devices=[]
        out['cuda']=dict(available=torch.cuda.is_available(),device_count=torch.cuda.device_count(),devices=devices)
        for i in range(torch.cuda.device_count()):
            free,total=torch.cuda.mem_get_info(i)
            a=torch.ones((512,512),device=f'cuda:{i}',dtype=torch.float32)
            value=float((a@a).sum().item());torch.cuda.synchronize(i)
            devices.append(dict(index=i,name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
                                free_mib=free>>20,total_mib=total>>20,matmul_ok=value==512.0**3))
            del a
        out['libraries']=libraries()
except BaseException as exc:out['error']=f'{type(exc).__name__}: {exc}'
for name in json.loads(sys.argv[3]):
    try:__import__(name);out['imports'][name]='ok'
    except BaseException as exc:out['imports'][name]=f'{type(exc).__name__}: {exc}'
print('V8PREFLIGHT '+json.dumps(out),flush=True)
"""


def _version_tuple(text):
    parts=[]
    for piece in re.split(r'[.+-]',str(text or '')):
        if not piece.isdigit():break
        parts.append(int(piece))
    return tuple(parts)


def cohort_requirements(cohort):
    """model_matrix.py's loader minimums per model family: (package, minimum or None for 'installed', why)."""
    name=str(cohort or '').lower();rows=[('torch','2.3','requirements.txt torch>=2.3')]
    if name.startswith('olmo'):rows.append(('transformers','4.57.0','OLMo-3 needs transformers>=4.57'))
    if name.startswith('qwen'):
        rows+=[('transformers','5','Qwen3.5 needs transformers>=5 (qwen3_5 classes)'),
               ('fla-core',None,'Qwen3.5 DeltaNet kernels need fla-core (seed runs pinned 0.5.2)')]
    if name.startswith('gemma'):rows.append(('torch','2.5','Gemma needs torch>=2.5'))
    return rows


def cohort_imports(cohort):
    name=str(cohort or '').lower()
    return ['transformers','fla'] if name.startswith('qwen') else ['transformers']


def driver_info(run=subprocess.run):
    """(driver version, max CUDA version) from nvidia-smi, or (None, None)."""
    try:
        result=run(['nvidia-smi'],capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.SubprocessError):return None,None
    text=result.stdout or ''
    driver=re.search(r'Driver Version:\s*([\d.]+)',text);cuda=re.search(r'CUDA Version:\s*([\d.]+)',text)
    return (driver.group(1) if driver else None),(cuda.group(1) if cuda else None)


def gpu_health(gpus,run=subprocess.run):
    """Uncorrected volatile ECC errors and pending row remaps per allocated GPU (missing fields: None)."""
    rows={}
    for field in ('ecc.errors.uncorrected.volatile.total','remapped_rows.pending'):
        try:
            result=run(['nvidia-smi','--id='+','.join(gpus),'--query-gpu=index,'+field,'--format=csv,noheader,nounits'],
                       capture_output=True,text=True,timeout=30)
        except (OSError,subprocess.SubprocessError):continue
        if result.returncode:continue
        for line in result.stdout.strip().splitlines():
            parts=[x.strip() for x in line.split(',')]
            if len(parts)==2:rows.setdefault(parts[0],{})[field]=parts[1]
    return rows


def process_owner(pid):
    info=dict(pid=pid)
    try:info['cmdline']=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()[:300]
    except OSError:info['cmdline']=None
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('Uid:'):
                import pwd
                uid=int(line.split()[1])
                try:info['user']=pwd.getpwuid(uid).pw_name
                except KeyError:info['user']=str(uid)
            elif line.startswith('State:'):info['state']=line.split(':',1)[1].strip()
    except OSError:pass
    return info


def host_resources():
    import resource
    out={}
    try:
        st=os.statvfs('/dev/shm');out['shm_free_mib']=st.f_bavail*st.f_frsize>>20;out['shm_total_mib']=st.f_blocks*st.f_frsize>>20
    except OSError as exc:out['shm_error']=str(exc)
    try:
        info={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.split()[1:2] and l.split()[1].isdigit()}
        out['mem_available_mib']=info.get('MemAvailable',0)>>10;out['mem_total_mib']=info.get('MemTotal',0)>>10
    except (OSError,ValueError,IndexError) as exc:out['meminfo_error']=str(exc)
    limits={}
    for name in ('NOFILE','MEMLOCK','STACK','NPROC','CORE','AS'):
        key=getattr(resource,'RLIMIT_'+name,None)
        if key is None:continue
        soft,hard=resource.getrlimit(key)
        limits[name.lower()]=['unlimited' if v==resource.RLIM_INFINITY else v for v in (soft,hard)]
    out['ulimit']=limits
    try:out['load_average']=list(os.getloadavg())
    except OSError:pass
    return out


def child_like_environment(gpus):
    """The environment a task child gets for imports: this node's, the trainer roots first on PYTHONPATH."""
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus),PYTHONDONTWRITEBYTECODE='1')
    code=os.environ.get('V8_CODE_ROOT')
    if code:
        code=str(Path(code).resolve())
        env['PYTHONPATH']=os.pathsep.join(dict.fromkeys(filter(None,(code,code+'/src',*env.get('PYTHONPATH','').split(os.pathsep)))))
    return env


def node_preflight(cohort,python,gpus,expected=None,*,run=subprocess.run,inventory=gpu_inventory,processes=gpu_processes,
                   resources=host_resources,driver=driver_info,health=gpu_health,sleep=time.sleep,clock=time.monotonic):
    """Checks before one task launch on this node; never changes the queue. Returns the record that goes into the
    attempt ('preflight', with 'env'). ok=False means: do not start the task here (another node takes it)."""
    started=clock();failures=[];warnings=[];gpus=[str(g) for g in gpus]
    limit=float(os.environ.get('V8_PREFLIGHT_SMOKE_SECONDS','300'))
    env=dict(host=socket.gethostname(),gpus=gpus,cohort=cohort,executable=python,
             variables={k:os.environ.get(k) for k in PREFLIGHT_ENV if os.environ.get(k) is not None})
    # 1. GPUs: present, free (a leftover/foreign process waits up to V8_PREFLIGHT_GPU_WAIT seconds), healthy.
    wait=float(os.environ.get('V8_PREFLIGHT_GPU_WAIT','60'));rows=inventory(gpus);waited=0
    while rows and any(r['used_mib']>BUSY_MIB for r in rows) and waited<wait:
        sleep(5);waited+=5;rows=inventory(gpus)  # a just-finished task's memory is often released within seconds
    if rows is None:warnings.append('nvidia-smi did not report the allocated GPUs; GPU checks skipped')
    else:
        env['gpu_inventory']=rows
        uuids={r['uuid'] for r in rows};apps=[]
        for line in processes():
            parts=[x.strip() for x in line.split(',')]
            if len(parts)>=3 and parts[1] in uuids and parts[0].isdigit():
                apps.append(dict(process_owner(int(parts[0])),gpu_uuid=parts[1],used_mib=parts[2]))
        env['gpu_processes']=apps
        busy=[r for r in rows if r['used_mib']>BUSY_MIB]
        if busy:
            failures.append('GPUs still occupied before start: '+', '.join(f"GPU {r['index']} {r['used_mib']} MiB" for r in busy)+
                            '; processes: '+('; '.join(f"pid {a['pid']} {a.get('user','?')} {a['used_mib']} MiB {a.get('cmdline') or ''}"[:200] for a in apps)
                                             or 'none listed (another container or a dying process)'))
        elif apps:warnings.append(f'{len(apps)} process(es) hold small memory on these GPUs: '+
                                  ', '.join(f"pid {a['pid']} {a.get('user','?')}" for a in apps))
        for index,values in health(gpus).items():
            env.setdefault('gpu_health',{})[index]=values
            for field,value in values.items():
                if value.isdigit() and int(value)>0:
                    failures.append(f'GPU {index} reports {field}={value} (hardware memory errors; use another node)')
    d,cuda=driver();env.update(driver_version=d,driver_cuda=cuda)
    # 2. Host: /dev/shm (NCCL shared-memory transport), memory, limits.
    host=resources();env['host_resources']=host
    min_shm=float(os.environ.get('V8_PREFLIGHT_MIN_SHM_MIB','1024'))
    if 'shm_free_mib' in host and host['shm_free_mib']<min_shm:
        failures.append(f"/dev/shm has {host['shm_free_mib']} MiB free (< {min_shm:g}); NCCL shared-memory transport "
                        'can crash (SIGBUS/SIGSEGV). Free /dev/shm or start the container with a larger --shm-size')
    elif 'shm_free_mib' in host and host['shm_free_mib']<8192:warnings.append(f"/dev/shm only {host['shm_free_mib']} MiB free")
    min_mem=float(os.environ.get('V8_PREFLIGHT_MIN_MEM_MIB','8192'))
    if 'mem_available_mib' in host and host['mem_available_mib']<min_mem:
        failures.append(f"host MemAvailable {host['mem_available_mib']} MiB (< {min_mem:g}); model loading would be killed or crash")
    nofile=(host.get('ulimit') or {}).get('nofile')
    if nofile and nofile[0]!='unlimited' and nofile[0]<4096:warnings.append(f'ulimit -n soft limit {nofile[0]} (< 4096)')
    for name in ('LD_PRELOAD','LD_LIBRARY_PATH'):
        if os.environ.get(name):warnings.append(f'{name} is set ({os.environ[name][:200]}); it overrides the venv CUDA/NCCL libraries')
    # 3. The cohort's own Python: versions, CUDA init and one matmul on every allocated GPU (subprocess).
    smoke=dict(rc=None)
    try:
        result=run([python,'-c',_PREFLIGHT_SMOKE,json.dumps(PREFLIGHT_PACKAGES),'1' if rows else '0',
                    json.dumps(cohort_imports(cohort))],capture_output=True,text=True,timeout=limit,env=child_like_environment(gpus))
        smoke['rc']=result.returncode
        line=next((l for l in reversed(result.stdout.splitlines()) if l.startswith('V8PREFLIGHT ')),None)
        report=json.loads(line[len('V8PREFLIGHT '):]) if line else {}
        if result.returncode or not line:smoke['stderr_tail']=(result.stderr or '')[-4000:]
    except subprocess.TimeoutExpired as exc:
        report={};smoke.update(rc='timeout',stderr_tail=str(exc.stderr or '')[-4000:] if exc.stderr else '')
    except OSError as exc:report={};smoke.update(rc='not-started',error=str(exc))
    env.update({k:report[k] for k in ('python_version','prefix','packages','torch','cuda','libraries','imports') if k in report})
    env['smoke']=smoke
    if smoke['rc']=='timeout':failures.append(f'cohort Python {python} did not finish import torch/CUDA init within {limit:g}s')
    elif smoke['rc']=='not-started':failures.append(f'cohort Python {python} cannot be started: {smoke.get("error")}')
    elif smoke['rc']:
        rc=smoke['rc']
        try:what=signal.Signals(-rc).name if rc<0 else f'exit {rc}'
        except ValueError:what=f'exit {rc}'
        frames=crash_frames(smoke.get('stderr_tail'))
        failures.append(f'cohort Python {python} crashed ({what}) in import torch / CUDA init / one matmul before any task code'+
                        (': '+' <- '.join(frames[:4]) if frames else ''))
    if report.get('error'):failures.append(f"cohort Python {python}: {report['error']}")
    for name,state in (report.get('imports') or {}).items():
        if state!='ok':failures.append(f'cohort Python {python}: import {name} failed: {state}')
    cuda_report=report.get('cuda')
    if rows and cuda_report is not None:
        if not cuda_report.get('available') or cuda_report.get('device_count')!=len(gpus):
            failures.append(f"CUDA sees {cuda_report.get('device_count')} device(s) (available={cuda_report.get('available')}) "
                            f'for CUDA_VISIBLE_DEVICES={",".join(gpus)}')
        bad=[str(dv['index']) for dv in cuda_report.get('devices',[]) if not dv.get('matmul_ok')]
        if bad:failures.append('wrong matmul result on CUDA device(s) '+','.join(bad))
    torch_cuda=_version_tuple((report.get('torch') or {}).get('cuda'));driver_cuda=_version_tuple(cuda)
    if torch_cuda and driver_cuda:
        if torch_cuda[:1]>driver_cuda[:1]:
            failures.append(f'torch is built for CUDA {report["torch"]["cuda"]} but the driver {d} supports CUDA {cuda} only')
        elif torch_cuda[:2]>driver_cuda[:2]:
            warnings.append(f'torch CUDA {report["torch"]["cuda"]} is newer than the driver CUDA {cuda} (minor-version '
                            'compatibility; JIT/PTX kernels such as Triton may fail)')
    prefix=report.get('prefix') or ''
    foreign=[l for l in report.get('libraries') or [] if 'libcuda.so' not in l and prefix and not l.startswith(prefix)
             and '/site-packages/' not in l]
    if foreign:warnings.append('CUDA/NCCL libraries loaded from outside the venv: '+', '.join(foreign[:6]))
    # 4. Versions: what this queue was started with and what the model family needs.
    packages=report.get('packages') or {}
    if expected and report:
        if expected.get('python_version') and report.get('python_version')!=expected['python_version']:
            failures.append(f"Python {str(report.get('python_version')).split()[0]} differs from the queue's "
                            f"{str(expected['python_version']).split()[0]}")
        for name,version in (expected.get('packages') or {}).items():
            if name in packages and packages.get(name)!=version:
                failures.append(f'{name} {packages.get(name)} differs from the queue runtime {version}')
    if report:
        for name,minimum,why in cohort_requirements(cohort):
            have=packages.get(name)
            if have is None and name=='fla-core':have=packages.get('flash-linear-attention')
            if have is None:failures.append(f'{name} is not installed in {python} ({why})')
            elif minimum and _version_tuple(have)<_version_tuple(minimum):failures.append(f'{name} {have} < {minimum} ({why})')
    return dict(policy=PREFLIGHT_POLICY,ok=not failures,failures=failures,warnings=warnings,
                checked_at=time.time(),seconds=round(clock()-started,1),env=env)


def preflight_line(result):
    env=result.get('env') or {};torch=env.get('torch') or {}
    head=(f"PREFLIGHT {'ok' if result.get('ok') else 'FAILED'} host={env.get('host')} gpus={','.join(env.get('gpus') or [])} "
          f"python={env.get('executable')} ({str(env.get('python_version') or '?').split()[0]}) torch={torch.get('version')} "
          f"cuda={torch.get('cuda')} nccl={torch.get('nccl')} driver={env.get('driver_version')}/{env.get('driver_cuda')} "
          f"transformers={(env.get('packages') or {}).get('transformers')} shm_free={(env.get('host_resources') or {}).get('shm_free_mib')}MiB "
          f"({result.get('seconds')}s)")
    return '\n'.join([head,*('  FAIL: '+f for f in result.get('failures',[])),*('  warn: '+w for w in result.get('warnings',[]))])


class PreflightFailed(NodeFailure):
    """This node cannot start the task safely; the attempt is refunded and another node takes it."""


PREFLIGHT_MODE=os.environ.get('V8_PREFLIGHT','record').strip().lower()   # record (default) | block | 0 (off)
PREFLIGHT_RETRY_SECONDS=float(os.environ.get('V8_PREFLIGHT_RETRY_SECONDS','120'))
PREFLIGHT_CHECK=node_preflight  # module attribute so CPU tests can substitute the node


UPDATE_HOLD_PREFIX='code update pending ('


def update_held(busy):return isinstance(busy,str) and busy.startswith(UPDATE_HOLD_PREFIX)


def update_hold(environment=None):
    """Pushed code waits for this node's next task boundary (11-18): while the supervisor's hold file exists, this
    launch takes no new task; the running task is never touched. Ignored once that supervisor is gone (an orphaned
    child must not idle forever). Scheduling only: no queue, budget or task record changes."""
    path=(os.environ if environment is None else environment).get('V8_UPDATE_HOLD')
    if not path:return None
    try:record=json.loads(Path(path).read_text())
    except (OSError,ValueError):return None
    if not isinstance(record,dict) or record.get('supervisor_pid')!=os.getppid():return None
    return (UPDATE_HOLD_PREFIX+str(record.get('target','?'))[:7]+'): this node takes no new task; '
            'the V8 supervisor restarts it with the pushed code now that no task is running here')


class GpuGate:
    """Defer new GPU tasks while an allocated GPU already holds memory (a stale or foreign process)."""
    def __init__(self,gpus,*,limit=BUSY_MIB,ttl=30,inventory=gpu_inventory,processes=gpu_processes):
        self.gpus,self.limit,self.ttl=gpus,limit,ttl
        self.inventory,self.processes=inventory,processes
        self.checked=-1e9;self.value=(True,None)

    def __call__(self):
        # A finished/crashed task can release GPU memory immediately. Do not keep
        # an idle worker behind a stale30s busy reading after those processes exit.
        cache=self.ttl if self.value[0] else min(1.0,self.ttl)
        if time.monotonic()-self.checked<cache:return self.value
        self.checked=time.monotonic();rows=self.inventory(self.gpus)
        busy=[r for r in rows or [] if r['used_mib']>self.limit]
        if not busy:self.value=(True,None)
        else:
            detail=', '.join(f"GPU {r['index']} {r['used_mib']} MiB used" for r in busy)
            apps='; '.join(self.processes()) or 'nvidia-smi lists no compute process'
            self.value=(False,f'{detail} (> {self.limit} MiB before start; processes pid,gpu,MiB: {apps}). '
                              'Another job or a leftover process holds these GPUs; V8 starts when they are free.')
        return self.value


class Runtime:
    def __init__(self,runner,root,cohort,python,gpus):
        self.runner=Path(runner).resolve();self.root=Path(root).resolve()/cohort
        self.cohort=cohort;self.python=python;self.gpus=gpus
        self.primary_results_first=True;self.streaming_ready_tasks=True
        if not (self.runner/'cluster_child.py').is_file():raise ValueError('runner-root must contain cluster_child.py')
        if not os.environ.get('V8_CODE_ROOT'):raise ValueError('V8_CODE_ROOT must point to the frozen trainer checkout')
        sys.path.insert(0,str(self.runner))
        from switching import cluster,matrix,spec,coderepo
        self.cluster,self.mx,self.spec,self.code=cluster,matrix,spec,coderepo
        self.cfg=spec.resolve(cohort,role='rebuttal_bank')
        if len(gpus)!=self.cfg['runtime']['world_size']:raise ValueError('GPU count must match the frozen runtime world_size')
        files=sorted(list((self.runner/'switching').glob('*.py'))+list((self.runner/'configs').rglob('*.json'))+
                     [self.runner/'cluster_child.py',self.runner/'run_h100.sh'])
        environment_identity=python_identity(python,QWEN_PACKAGES if self.cfg['runtime'].get('profile')=='qwen' else ())
        self.binding=dict(schema=SCHEMA,cohort=cohort,root=str(self.root),**environment_identity,
                          coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          runner_sha256={str(p.relative_to(self.runner)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                          config_sha256=spec.config_hash(self.cfg),trainer_sha256=coderepo.implementation_digest(),world_size=len(gpus))
        self.environment=dict(os.environ,V8_ROOT=str(self.root.parent),V8_PYTHON=python,
                              PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                              TOKENIZERS_PARALLELISM='false')
        self.environment.setdefault('HF_ENDPOINT','https://hf-mirror.com')
        self.environment.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
        # Same child environment as the seed 5-9 SRGC workers (srgc_rebuttal/cluster.py child_environment).
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OPENBLAS_DEFAULT_NUM_THREADS',
                     'GOTO_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS',
                     'NUMEXPR_MAX_THREADS','OMP_THREAD_LIMIT','RAYON_NUM_THREADS'):
            self.environment[name]='1'
        self.environment.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING','1')
        self.environment.setdefault('NCCL_DEBUG','WARN')
        code=Path(os.environ['V8_CODE_ROOT']).resolve()
        self.environment['PYTHONPATH']=os.pathsep.join(filter(None,(str(code),str(code/'src'),self.environment.get('PYTHONPATH'))))
        self.admitted=None

    def input_identity(self):
        loc=read(self.root/'locator.json');records={}
        for seed in self.cfg['seeds']['training']:
            entry=loc['seeds'][str(seed)]
            metadata={k:entry.get(k) for k in ('model','model_revision','prefix_path','prefix_sha256','prefix_completed_updates','plan_path','plan_sha256','input_bundle')}
            hashes={}
            for key in ('plan_path','input_bundle'):
                if entry.get(key):
                    h=hashlib.sha256()
                    with open(entry[key],'rb') as handle:
                        for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
                    hashes[key]=h.hexdigest()
            if entry.get('plan_sha256') and hashes.get('plan_path')!=entry['plan_sha256']:
                raise ValueError('Source plan changed since locate for seed '+str(seed))
            records[str(seed)]=dict(metadata=metadata,file_sha256=hashes)
        return dict(trainer_sha256=loc['code']['implementation_sha256'],seeds=records)

    def freeze_inputs(self):
        path=self.root/'.multi-node/input-identity.json';value=self.input_identity()
        if path.exists() and read(path)!=value:raise ValueError('Frozen input identities changed')
        if not path.exists():atomic(path,value)

    def verify_node(self):
        if self.code.implementation_digest()!=self.binding['trainer_sha256']:
            raise ValueError('Frozen trainer files changed on this node')
        identity=self.root/'.multi-node/input-identity.json'
        if identity.exists() and read(identity)!=self.input_identity():
            raise ValueError('Source plan/bundle/prefix identity differs from this queue')
        loc=self.root/'locator.json'
        if loc.exists() and read(loc)['code']['implementation_sha256']!=self.binding['trainer_sha256']:
            raise ValueError('This node trainer digest differs from the verified seed-run locator')

    def cpu(self,stage,fd,stop,device_fds):
        command=['sh',str(self.runner/'run_h100.sh'),self.cohort,stage]
        log=self.root/'.multi-node'/'logs'/f'cpu-{stage}-{uuid.uuid4().hex}.log'
        rc=self.process(command,log,[fd,*device_fds],stop,cpu=True)
        if rc and not stop.is_set():raise RuntimeError(f'CPU stage {stage} failed exit={rc}; full output: {log}')

    def barrier(self,action,fd,stop,device_fds):
        if action=='prepare':
            for stage in ('check','locate','audit','profile'):
                self.cpu(stage,fd,stop,device_fds)
                if stop.is_set():return
            self.freeze_inputs();self.verify_node()
        elif action=='freeze':
            if (self.root/'.multi-node/budget-rule.json').exists() and not (self.root/'frozen/budget.json').exists():
                return freeze_precommitted_budget(self)
            if not (self.root/'frozen/budget.json').exists() and list((self.root/'evaluations').glob('*.json')):
                raise ValueError('Evaluation records exist before budget freeze; cannot certify outcome-independent budget selection')
            self.cpu('freeze-budget',fd,stop,device_fds)
        else:
            for stage in ('collect','analyze'):
                self.cpu(stage,fd,stop,device_fds)
                if stop.is_set():return
            if self.environment.get('EXPORT_CODE_ROOT'):self.cpu('history-export',fd,stop,device_fds)
            if action=='primary_finalize' and not stop.is_set():capture_primary_results(self)

    def locator(self):return self.cluster.ClusterLocator(self.root,self.cfg)

    def side_tasks(self):return ready_side_tasks(self)

    def plan(self,phase):
        if phase=='primary_evaluation':return primary_evaluation_tasks(self)
        loc=self.locator();seeds=self.cfg['seeds']['training'];tasks=[]
        if not seeds:raise ValueError('No training seeds configured')
        if phase=='train':
            trajectory,_=self.mx.build_stage('bank-trajectory',self.cfg,loc,self.root,seeds=seeds)
            tasks+=trajectory
            complete=all(self.done(t) and not self.native_busy(t) for t in trajectory)
            for stage in ('bank-branches','bank-extended','destination'):
                found,notes=self.mx.build_stage(stage,self.cfg,loc,self.root,seeds=seeds)
                if complete and any(n.get('status')=='blocked' for n in notes):
                    raise ValueError('Completed trajectory has unavailable required branch parents: '+canonical(notes))
                tasks+=found
            return tasks,complete
        budget=read(self.root/'frozen/budget.json')['value']
        if phase=='extension':
            tasks,notes=self.mx.build_stage('budget-extension',self.cfg,loc,self.root,seeds=seeds,budget=budget)
            if any(n.get('status')=='blocked' for n in notes):raise ValueError('Budget extension blocked: '+canonical(notes))
            return tasks,True
        from switching.bank import SeedBank
        class EvaluationBank(SeedBank):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);self.selected_endpoints={}

            def budget_endpoint(self,*args,**kwargs):
                selection=super().budget_endpoint(*args,**kwargs)
                if selection['status']=='selected':
                    endpoint=selection['endpoint']
                    self.selected_endpoints[(endpoint['run_id'],endpoint['completed_update'])]=endpoint
                return selection

        for seed in seeds:
            data=self.cluster.load_bundle(loc.loc['seeds'][str(seed)]['input_bundle'])
            run_ids=[p.parent.name for p in sorted((self.root/'runs').glob('*/manifest.json'))
                     if (read(p).get('training_seed'),read(p).get('cohort_id'),read(p).get('role'))==(seed,self.cohort,'rebuttal_bank')]
            tasks+=self.mx.eval_tasks(self.cfg,data,self.root,run_ids,split='final',updates={self.cfg['training']['matched_update_endpoint']})
            bank=EvaluationBank(self.root,self.cfg,self.cohort,seed,'rebuttal_bank',evals={})
            bank.policy_rows(budget=budget)
            # A budget endpoint can belong to the extension or pre-switch
            # trajectory, rather than the matched-horizon branch in row.run_id.
            for (run_id,update),endpoint in bank.selected_endpoints.items():
                found=self.mx.eval_tasks(self.cfg,data,self.root,[run_id],split='final',updates={update})
                if not found or any(t['snapshot_sha256']!=endpoint['snapshot_hash'] for t in found):
                    raise ValueError('Selected budget checkpoint is missing or changed: '+run_id+'@'+str(update))
                tasks+=found
        return tasks,True

    def ready(self,task):
        if task['kind']=='evaluate':return Path(task['snapshot_path']).is_file()
        parent=task['parent']
        # A saved immutable parent suffices; other nodes may keep training the
        # On trajectory while this branch trains on a different GPU group.
        return bool(parent.get('path')) and Path(parent['path']).is_file()

    def native_busy(self,task):
        if task['kind']!='train':return False
        path=self.root/'runs'/task['run_id']/'.execution.lock'
        if not path.exists():return False
        fd=os.open(path,os.O_RDWR)
        try:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return True
            return False
        finally:os.close(fd)

    def done(self,task):
        if task['kind']=='train':
            path=self.root/'runs'/task['run_id']/'manifest.json'
            if not path.exists():return False
            rec=read(path)
            if rec.get('config_hash')!=task['config_hash'] or any(rec.get(k)!=task.get(k) for k in ('run_id','training_seed','cohort_id','role','method_id')):
                raise ValueError('Existing run identity/config differs from task')
            return rec.get('status') in TERMINAL and rec.get('completed_updates',-1)>=task['horizon']
        path=self.root/'evaluations'/f"{task['snapshot_sha256'][:16]}__{task['evaluation_spec_hash'][:12]}.json"
        if not path.exists():return False
        rec=read(path);counts=rec.get('per_question_correct');k=task['responses']
        if (rec.get('checkpoint_hash')!=task['snapshot_sha256'] or rec.get('evaluation_spec_hash')!=task['evaluation_spec_hash']
                or rec.get('completed_update')!=task['completed_update'] or rec.get('split')!=task['split']
                or rec.get('responses_per_question')!=k or not isinstance(counts,dict) or not counts
                or rec.get('n_questions')!=len(counts) or any(type(v)is not int or not 0<=v<=k for v in counts.values())):
            raise ValueError('Existing evaluation record is inconsistent with its task')
        value=rec.get('reward_fraction')
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Existing evaluation reward is not a finite fraction')
        if self.mx.sha256_json(sorted(counts))!=task['question_ids_hash'] or abs(sum(counts.values())/(len(counts)*k)-rec['reward_fraction'])>1e-9:
            raise ValueError('Existing evaluation question set/counts disagree')
        return True

    def admit(self,device_fds):
        """Once per worker: the seed runs' four-GPU NCCL admission (fabric fallbacks, local model path)."""
        if self.admitted is not None:return self.admitted
        code=Path(os.environ['V8_CODE_ROOT']).resolve()
        if str(code) not in sys.path:sys.path.insert(0,str(code))
        try:
            from srgc_rebuttal.admission import admit
            from srgc_rebuttal.cluster import run_child
            from srgc_rebuttal.plan import load_plan
        except ImportError as exc:
            print(f'[v8] WARN NCCL admission unavailable in V8_CODE_ROOT ({exc}); launching without it',flush=True)
            self.admitted={};return self.admitted
        plan=None
        if self.cfg['runtime'].get('profile')!='qwen':
            loc=read(self.root/'locator.json');first=loc['seeds'][str(self.cfg['seeds']['training'][0])]
            plan=load_plan(Path(first['plan_path']))
        environment=dict(self.environment,CUDA_VISIBLE_DEVICES=','.join(self.gpus))
        before=dict(environment)
        directory=self.root/'.multi-node'/'admission'/f"{socket.gethostname()}-{'-'.join(self.gpus)}-{uuid.uuid4().hex[:8]}"
        directory.parent.mkdir(parents=True,exist_ok=True)
        print(f'[v8] {time.strftime("%H:%M:%S")} four-GPU NCCL admission on GPUs {",".join(self.gpus)} (log {directory}/preflight.log)',flush=True)
        try:
            report=admit(directory,environment,run_child,pass_fds=tuple(device_fds),plan=plan)
        except Exception as exc:
            raise RuntimeError(f'NCCL admission failed on GPUs {",".join(self.gpus)}: {exc}. Inspect {directory}/preflight.log') from exc
        added={k:v for k,v in environment.items() if before.get(k)!=v and k!='CUDA_VISIBLE_DEVICES'}
        self.environment.update(added)
        print(f'[v8] admission passed; overrides {sorted(added)}',flush=True)
        self.admitted=report;return report

    def fresh_admission(self,device_fds):
        self.verify_node();self.admitted=None
        return self.admit(device_fds)

    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        self.verify_node()
        self.admit(device_fds)
        command=[self.python,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(self.gpus)}','--max_restarts=0',
                 str(self.runner/'cluster_child.py'),'--cohort',self.cohort,'--matrix',str(matrix),
                 '--task',task.get('task_id') or task['run_id']]
        progress=self.root/'.multi-node'/'progress'/attempt['id']
        log=self.root/'.multi-node'/'logs'/(attempt['id']+'.log');exit_file=log.with_name(attempt['id']+'.exit.json')
        try:exit_file.unlink()
        except FileNotFoundError:pass
        rc=self.process(command,log,[fd,*device_fds],stop,label=task.get('task_id') or task['run_id'],
                        extra={'SRGC_PROGRESS_DIR':str(progress),'V8_EXIT_FILE':str(exit_file)})
        return child_exit(rc,exit_file)

    def process(self,command,log,fds,stop,*,cpu=False,extra=None,label=None):
        """Child output goes straight to its log file (a stalled terminal can no longer block the child);
        a separate thread relays the file to this terminal, and every V8_STATUS_SECONDS (60) a status line
        shows elapsed time, time since the child last wrote, GPU utilisation and the last log line."""
        log.parent.mkdir(parents=True,exist_ok=True)
        environment=dict(self.environment,CUDA_VISIBLE_DEVICES='' if cpu else ','.join(self.gpus),**(extra or {}))
        if extra and extra.get('SRGC_PROGRESS_DIR'):Path(extra['SRGC_PROGRESS_DIR']).mkdir(parents=True,exist_ok=True)
        interval=float(os.environ.get('V8_STATUS_SECONDS','60'))
        with log.open('ab') as handle:
            offset=handle.tell()
            process=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                                     pass_fds=tuple(fds),start_new_session=True)
        done=threading.Event();seen={'last':b'','changed':time.monotonic(),'size':offset}
        def relay():
            with log.open('rb') as reader:
                reader.seek(offset)
                while True:
                    block=reader.read(65536)
                    if block:
                        seen['changed']=time.monotonic();seen['size']+=len(block)
                        lines=[x for x in block.splitlines() if x.strip()]
                        if lines:seen['last']=lines[-1][-200:]
                        try:
                            stream=getattr(sys.stdout,'buffer',None)
                            if stream is None:sys.stdout.write(block.decode(errors='replace'));sys.stdout.flush()
                            else:stream.write(block);stream.flush()
                        except (BrokenPipeError,OSError):pass
                    elif done.is_set():break
                    else:time.sleep(.5)
        thread=threading.Thread(target=relay,daemon=True);thread.start()
        sent=False;last_heartbeat=0;started=time.monotonic();last_status=started
        while process.poll() is None:
            now=time.monotonic()
            if now-last_heartbeat>=10:
                # A heartbeat write failure (moved queue, full or read-only volume) must never abandon a running child.
                try:
                    if hasattr(self,'heartbeat'):self.heartbeat('child:'+str(process.pid))
                except Exception:pass
                last_heartbeat=now
            if not cpu and interval>0 and now-last_status>=interval:
                last_status=now
                print(f"[v8 {time.strftime('%H:%M:%S')}] {label or 'task'} running {int(now-started)//60}m; "
                      f"last output {int(now-seen['changed'])}s ago; GPU {gpu_utilisation(self.gpus)}; "
                      f"last: {seen['last'].decode(errors='replace')}",flush=True)
            if stop.wait(.25) and not sent:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                sent=True
        done.set();thread.join(timeout=10);return process.returncode


def main(argv=None,*,selected_cohorts=None,cohort_order=None,not_served=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort',nargs='?',default='all',help='default: finish olmo_math, qwen_mbpp, olmo_mbpp and qwen_math in order')
    parser.add_argument('--runner-root',type=Path,default=os.environ.get('V8_RUNNER_ROOT',str(Path.cwd()/'v8/experiments')))
    parser.add_argument('--root',type=Path,default=os.environ.get('V8_ROOT'))
    parser.add_argument('--python',default=os.environ.get('V8_PYTHON',sys.executable),help='fallback Python; cohort-specific environments take priority')
    parser.add_argument('--gpus',default=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3'))
    parser.add_argument('--poll-seconds',type=float,default=3)
    parser.add_argument('--retry-failed',action='store_true',
                        help='legacy explicit retry request; failed tasks already retry by default under '
                             '--max-attempts/--retry-delay; does not delete checkpoints or failure history')
    parser.add_argument('--max-attempts',type=int,default=MAX_ATTEMPTS,help='lifetime task failure cap; default2 (one retry); legacy0 also means2')
    parser.add_argument('--retry-delay',type=float,default=RETRY_DELAY,help='seconds before restart; default0 (immediate)')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--cohorts',default=os.environ.get('V8_COHORTS'),
                        help='comma-separated cohorts this node serves, in claim-priority order (launcher default: '
                             'the named cohort, then other cohorts with queues under V8_ROOT)')
    parser.add_argument('--new-queue',action='store_true',
                        help='after stopping the workers on ALL nodes: keep the existing queue as .multi-node.superseded-<time> '
                             'and start a fresh one (refused once any update has completed or a job lease is held)')
    args=parser.parse_args(argv)
    if args.root is None:parser.error('V8_ROOT or --root is required (shared by all nodes)')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.cohort):parser.error('cohort must be a simple configured identifier')
    listed=tuple(dict.fromkeys(([] if args.cohort=='all' else [args.cohort])+[c.strip() for c in (args.cohorts or '').split(',') if c.strip()]))
    cohorts=(tuple(selected_cohorts) if selected_cohorts is not None else listed if args.cohorts else
             DEFAULT_COHORTS if args.cohort=='all' else (args.cohort,))
    if not cohorts or len(set(cohorts))!=len(cohorts) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',c) for c in cohorts):
        parser.error('selected cohorts must be nonempty, distinct configured identifiers')
    if selected_cohorts is not None and args.cohort!='all' and cohorts!=(args.cohort,):
        parser.error('selected cohorts disagree with the requested cohort')
    if args.status:
        statuses={}
        for cohort in cohorts:
            queue=Path(args.root).resolve()/cohort/'.multi-node'
            statuses[cohort]=Pool(queue,{}).status() if queue.exists() else {'phase':'not_started'}
        print(json.dumps(statuses if args.cohort=='all' else statuses[args.cohort],indent=2));return 0
    if not args.python:parser.error('V8_PYTHON or --python is required')
    if not args.poll_seconds>0:parser.error('poll-seconds must be positive')
    if args.max_attempts<0 or not math.isfinite(args.retry_delay) or args.retry_delay<0:
        parser.error('--max-attempts must be >= 0 (legacy0 uses the default finite cap) and --retry-delay a finite number >= 0')
    gpus=[gpu.strip() for gpu in args.gpus.split(',')]
    if any(not x.strip() for x in gpus) or len(set(gpus))!=len(gpus):parser.error('GPU IDs must be nonempty and distinct')
    group=Path(os.environ.get('GROUP_VOLUME','/group-volume')).resolve()
    if not os.environ.get('V8_ALLOW_ANY_ROOT') and not Path(args.root).resolve().is_relative_to(group):
        parser.error(f'V8_ROOT={args.root} is not on the shared group volume GROUP_VOLUME={group}; all nodes must '
                     'share one V8_ROOT (default: $OM_WORK/v8)')
    locks=[];stop=threading.Event();previous={}
    try:
        inventory=gpu_inventory(gpus)
        if inventory is None:
            try:listing=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=30)
            except (OSError,subprocess.SubprocessError):listing=None
            if listing is not None and listing.returncode==0:
                # nvidia-smi works but does not report these four GPUs: a task would fail at rank startup.
                parser.error(f'this node does not expose the GPUs {",".join(gpus)} (CUDA_VISIBLE_DEVICES); '
                             'nvidia-smi -L reports:\n'+listing.stdout.strip())
            if not os.environ.get('V8_CPU_FIXTURE_NO_GPU'):
                # A login/CPU node would claim GPU work and fail every launch: refuse before any queue write.
                parser.error(f'nvidia-smi is not available or failed on this node, so the GPUs {",".join(gpus)} cannot '
                             'be verified. Start the V8 worker on an allocated 4x80 GB GPU node; no queue file was changed.')
        if inventory is not None:
            small=[r for r in inventory if r['total_mib']<75000]
            if small:parser.error('each task needs four full 80 GB GPUs; got '+', '.join(f"{r['index']}:{r['name']} {r['total_mib']} MiB" for r in small))
        lock_root=gpu_lock_root()
        try:lock_root.mkdir(parents=True,exist_ok=True)
        except OSError as exc:
            print(f'V8 queue error: cannot create the GPU lease directory {lock_root}: {exc.strerror}',file=sys.stderr);return 1
        for name,lease_name in sorted(gpu_lease_names(gpus,inventory),key=lambda item:item[1]):
            lock=Lock(lock_root/lease_name)
            if not lock.acquire():
                print(f'Another worker/child already owns an overlapping GPU group on this node (GPU {name} is leased '
                      f'in {lock_root}). Stop that SRGC/V8 worker or run this worker on other GPUs.',file=sys.stderr);return 75
            locks.append(lock)
        for signum in (signal.SIGINT,signal.SIGTERM):
            previous[signum]=signal.signal(signum,lambda *_:stop.set())
        pairs=[]
        for cohort in cohorts:
            runtime=Runtime(args.runner_root,args.root,cohort,cohort_python(cohort,args.python),gpus)
            pairs.append((Pool(runtime.root/'.multi-node',runtime.binding,new_queue=args.new_queue),runtime))
        worker=socket.gethostname()+'.'+str(os.getpid())
        print(f'[{worker}] V8 worker: cohorts={",".join(cohorts)} gpus={",".join(gpus)} '
              f'uuids={",".join(r["uuid"] for r in inventory) if inventory else "nvidia-smi unavailable"}',flush=True)
        print(f'[{worker}] V8_ROOT={Path(args.root).resolve()} trainer={os.environ.get("V8_CODE_ROOT")} '
              f'runner={Path(args.runner_root).resolve()}',flush=True)
        for pool,runtime in pairs:
            print(f'[{worker}] {runtime.cohort}: python={runtime.python} ({runtime.binding.get("python_version","?").split()[0]}) '
                  f'queue={pool.root} logs={pool.root/"logs"}',flush=True)
        print(f'[{worker}] progress: sh scripts/run_v8_nodes.sh --status   (from any node)',flush=True)
        return worker_loop_many(pairs,worker,stop,poll=args.poll_seconds,device_fds=tuple(lock.fd for lock in locks),
                                retry=args.retry_failed,gpu_gate=GpuGate(gpus) if inventory is not None else None,
                                max_attempts=args.max_attempts,retry_delay=args.retry_delay,
                                preflight=inventory is not None and os.environ.get('V8_PREFLIGHT','1')!='0',
                                cohort_order=cohorts if cohort_order is None else cohort_order,not_served=not_served)
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)
        for lock in reversed(locks):lock.close()


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as error:
        print(f'V8 queue error: {error}',file=sys.stderr);raise SystemExit(1)
