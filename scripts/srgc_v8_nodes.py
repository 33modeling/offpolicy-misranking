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
from contextlib import contextmanager
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
DISPATCH_POLICY = 'bounded-native-evaluation-recovery-v23'
EVAL_NATIVE_RECOVERY = 'evaluation-sigsegv-recovery-v1'
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
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as handle:
            handle.write(canonical(value)+'\n');handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


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


MAX_ATTEMPTS=2     # initial task fault plus one immediate retry, across all workers
RETRY_DELAY=0.0    # restart from the saved checkpoint without a cooldown


def failure_limit(value):
    """Legacy zero must not re-enable unlimited torchrun relaunches."""
    return value if type(value) is int and value>0 else MAX_ATTEMPTS


def counted_failures(job):
    """Original SRGC accounting: intentional interruptions and node-local launch faults are refunded."""
    count=0
    for attempt in job.get('attempts',[]):
        outcome=attempt.get('outcome')
        if outcome is None and attempt.get('exit_code') not in (None,0,75):outcome='failed'  # pre-outcome records
        count+=outcome=='failed'
    return count


def retry_budget_failures(job):
    """Lifetime task faults, except the single user-requested native evaluation replay.

    Historical retry_epoch records stay intact as evidence, but no longer grant
    another failure budget. Genuine node loss/intentional stops are still refunded.
    """
    recovery=evaluation_recovery_attempts(job)
    return counted_failures(dict(attempts=recovery)) if recovery is not None else counted_failures(job)


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


def native_failure_blocks(job):
    fault=fatal_native_failure(job)
    if not fault:return False
    replay=evaluation_recovery_attempts(job)
    if replay is None:return True
    # A replay crash, including one mislabelled as an interruption, never buys more work.
    return bool(fatal_native_failure(dict(attempts=replay))) or counted_failures(dict(attempts=replay))>=1


def fatal_native_failure(job):
    """A native crash is not repaired by another identical torchrun invocation."""
    for attempt in reversed(job.get('attempts',[])):
        rc=attempt.get('exit_code')
        if rc in (-4,-6,-7,-8,-11,132,134,135,136,139):
            return dict(signal=-rc if rc<0 else rc-128,attempt=attempt.get('id'))
        for failure in (attempt.get('child_failure') or {}).get('failures',[]):
            if failure.get('signal') in (4,6,7,8,11):
                return dict(signal=failure['signal'],attempt=attempt.get('id'))
    return None


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
        self.root.mkdir(parents=True,exist_ok=True)

    def retry_requested(self,state=None):
        """Task restarts admitted on this node; keep the existing shared retry marker."""
        return self.automatic_retry or self.retry_failed or bool((state or {}).get('retry_failed_requested_at'))

    def retry_limit_reached(self,job):
        if evaluation_recovery_attempts(job) is not None:return retry_budget_failures(job)>=1
        return retry_budget_failures(job)>=failure_limit(self.max_attempts)

    def retryable(self,job,now=None,state=None):
        """One immediate retry by default; repeated faults stay excluded across joins."""
        receipt=job.get('evaluation_native_recovery')
        if receipt and receipt.get('binding_sha256')!=digest(self.binding):return False
        if native_failure_blocks(job) or self.retry_limit_reached(job):return False
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
        if recover:
            self.recover_budget_api_failure(recover,worker)
            self.recover_failed(recover,worker)
        if self.automatic_retry:self.enable_automatic_retry(worker)
        state=read(state_path)
        if retry:
            # Queue-level failures (CPU stage, failed tasks blocking all work) are cleared here. Failed tasks keep
            # their attempt history; tick() retries them under the original SRGC --max-attempts/--retry-delay rules.
            self.retry_failed=True;state['retry_failed_requested_at']=time.time()
            if state['failure']:
                state['failure_history'].append(state['failure']);state['failure']=None
            atomic(state_path,state)

    def recover_budget_api_failure(self,controller,worker):
        """Verified recovery: fix the known API regression, or allow independent work beside a blocked eval plan."""
        if controller!=self.controller_sha256:return False
        state=self.queue_state();failure=state.get('failure')
        if state.get('phase') not in PHASES or not isinstance(failure,dict) or failure.get('task') is not None:
            return False
        detail=str(failure.get('detail',''))
        api=bool(re.search(r'''freeze_budget\(\) got an unexpected keyword argument ['"]expected_seeds['"]''',detail))
        evaluation='Selected budget checkpoint is missing or changed:' in detail
        if not api and not evaluation:return False
        state.setdefault('failure_history',[]).append(failure);state['failure']=None
        if evaluation:state['unresolved_evaluation_planner_failure']=failure
        state.setdefault('actions',[]).append(dict(action='recover_verified_legacy_budget_keyword_failure' if api else 'resume_independent_work_with_blocked_evaluation_plan',
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
            if job['state']!='failed' or already or native_failure_blocks(job) or self.retry_limit_reached(job):continue
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
        scope=list(dict.fromkeys([*order.get('cohorts',[]),*getattr(runtime,'primary_cohort_scope',(self.binding['cohort'],))]))
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
        except json.JSONDecodeError:
            # Only incomplete append-only metadata with an active writer is
            # transient. Broken completed metadata must fail visibly.
            active=False
            for path in self.root.glob('jobs/*.json'):
                probe=Lock(self.root/'leases'/(path.stem+'.lock'))
                if not probe.acquire():active=True
                probe.close()
            if active:return 'waiting',None
            raise ValueError('Invalid run metadata with no active writer')
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
            except json.JSONDecodeError:
                if self.held_leases():return 'waiting',None
                raise
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
            if deferred_base or base_error:return 'waiting',None
            raise ValueError('No tasks produced for '+phase+'; refusing to declare an empty experiment complete')
        pending=[];all_done=True;active=False;failed=[];later=False
        for path,job in jobs:
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():
                all_done=False;active=True;continue
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
                if native_busy:active=True;continue
                if (native_failure_blocks(job) or self.retry_limit_reached(job)) and job['state']!='failed':
                    # Old controllers may have reset a failed job to pending/running.
                    # Only migrate it after its lease/native lock is free and outputs
                    # were checked; never interrupt a live task or discard a result.
                    job['state']='failed';atomic(path,job)
                if not self.retryable(job,state=state):
                    if job['state']=='failed':failed.append(path.stem)
                    later=later or self.retry_later(job,state);continue
                if not runtime.ready(job['task']):continue
                pending.append((path,job))
            finally:lease.close()
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
                raise ValueError('Inspected task definition changed before claim: '+item['key'])
            if parallel and not offer.get('side_budget_sha256') and job_phase in ('extension','evaluation'):
                t=job['task']
                if (job_phase=='extension' or not early_rule or t.get('kind')!='evaluate' or t.get('split')!='final'
                        or t.get('completed_update')!=early_rule['horizon'] or t.get('training_seed') not in early_rule['seeds']):continue
            # A failed task is claimable only under the retry rules (--retry-failed, attempts, delay).
            if job['state']=='done' or not self.retryable(job,state=state):continue
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():continue
            attempt=dict(id=uuid.uuid4().hex,worker=worker,started_at=time.time(),pid=os.getpid(),
                         controller_sha256=self.controller_sha256,controller_policy=DISPATCH_POLICY)
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

    def finish(self,key,rc,runtime,stop,node_error=None):
        """Keep original rank faults; refund node-local faults and intentional interruptions."""
        path=self.root/'jobs'/(key+'.json');job=read(path);attempt=job['attempts'][-1]
        attempt.update(exit_code=rc,ended_at=time.time());done=False
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
        elif (rc==75 or stop.is_set()) and not fatal_native_failure(job):attempt['outcome']='interrupted';job['state']='interrupted'
        else:
            attempt.update(outcome='failed',error=f'task exit={rc}; output verified={done}'+
                           (f"; {attempt['output_validation_error']}" if attempt.get('output_validation_error') else ''))
            job['state']='failed'
            state=self.queue_state()
            state.setdefault('task_failures',[]).append(dict(detail=f'task exit={rc}; output verified={done}',
                                                           task=key,attempt=job['attempts'][-1]['id'],at=time.time()))
            atomic(self.root/'state.json',state)
        if job['state'] in ('failed','interrupted'):self.arm_evaluation_recovery(job,attempt.get('worker'))
        atomic(path,job)
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
    def __init__(self,pool,runtime,worker,stop,fd,*,verify=False,retry=False):
        held=os.dup(fd);self.finished=threading.Event();self.result=None;self.error=None
        self.started=self.reported=time.monotonic();self.kind='node verification' if verify else 'task inspection'
        def scan():
            try:
                if verify:
                    runtime.verify_node();pool.initialize(retry=retry,worker=worker,recover=pool.controller_sha256)
                    self.result=('verified',None)
                else:self.result=pool.tick(runtime,worker,stop,claim=False,offer=True)
            except BaseException as exc:
                self.error=exc
                if isinstance(exc,(KeyboardInterrupt,SystemExit)):stop.set()
                # Node-local faults (dead actor, local access) stop only this node: never a queue failure.
                if not verify and not stop.is_set() and not isinstance(exc,(BindingMismatch,*NODE_LOCAL)):
                    try:pool.fail(f'{type(exc).__name__}: {exc}')
                    except Exception as record_error:self.error=record_error
            finally:os.close(held);self.finished.set()
        self.thread=threading.Thread(target=scan,name='v8-scan-'+pool.binding.get('cohort','unknown'),daemon=True)
        try:self.thread.start()
        except BaseException:os.close(held);raise


def worker_loop_many(pairs,worker,stop,*,poll=3,device_fds=(),retry=False,gpu_gate=None,
                     max_attempts=MAX_ATTEMPTS,retry_delay=RETRY_DELAY,automatic_retry=True):
    """Original actors prepare independently; serialize lease-count comparison and task claim."""
    if not pairs:raise ValueError('At least one cohort is required')
    if stop.is_set():return 75
    initialized=set();terminal={};offers={};scans={};last_report=0;reasons={};busy_report=0;setup_report=0
    shared_failures=set();recovery_checked={};claim_contention_since={};exit_recheck_started={}
    cursor=int(digest(worker)[:8],16)%len(pairs)
    roots=[p.root.parent.parent if p.root.name=='.multi-node' else p.root.parent for p,_ in pairs]
    dispatch_lock=Path(os.path.commonpath([str(p.resolve()) for p in roots]))/'.v8-dispatch.lock'
    controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for pool,runtime in pairs:
        if getattr(runtime,'primary_results_first',False) and not getattr(runtime,'primary_cohort_scope',None):
            runtime.primary_cohort_scope=tuple(p.binding['cohort'] for p,_ in pairs)
        pool.retry_failed=retry;pool.max_attempts=max_attempts;pool.retry_delay=retry_delay
        pool.automatic_retry=automatic_retry
        runtime.stop_event=stop  # runtime actors report signals through the same stop event
        def heartbeat(task=None,pool=pool):
            atomic(pool.root/'workers'/(digest(worker)+'.json'),
                   dict(worker=worker,pid=os.getpid(),at=time.time(),task=task,
                        dispatch_policy=DISPATCH_POLICY,controller_sha256=controller_sha256))
        runtime.heartbeat=heartbeat
    def cohort(pool):
        return pool.binding.get('cohort','?')
    def give_up(index,pool,kind,exc):
        shared_failures.discard(index)  # node/source faults need fresh admission, not automatic rejoining
        terminal[index]='failed';reasons[cohort(pool)]=f'{type(exc).__name__}: {exc}'
        print(f'[{worker}] {cohort(pool)}: {kind} failed: {type(exc).__name__}: {exc}',flush=True)
        if isinstance(exc,BindingMismatch):
            print(f'[{worker}] {cohort(pool)}: no work is taken for this cohort on this node; nothing was changed.',flush=True)
        elif isinstance(exc,NODE_LOCAL):
            print(f'[{worker}] {cohort(pool)}: this node stops serving {cohort(pool)}. The shared queue was NOT marked '
                  'failed and the other nodes continue. Fix this node (message above), then start the same command '
                  'here again.',flush=True)
        else:
            print(f'[{worker}] {cohort(pool)}: no new work is taken for this cohort; the other cohort continues. '
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
    while not stop.is_set():
        progressed=False;dispatched=False;short_claim_wait=False
        # A different verified worker may recover the shared queue while this
        # worker is waiting on the other cohort. Do not cache that old failure
        # forever. This never clears a failure or retries a job on its own.
        for index in tuple(shared_failures):
            now=time.monotonic()
            if now-recovery_checked.get(index,-float('inf'))<max(.1,min(poll,5)):continue
            recovery_checked[index]=now;pool,runtime=pairs[index]
            try:
                with pool.control() as control:
                    if control is None:continue
                    if read(pool.root/'binding.json')!=pool.binding:
                        raise BindingMismatch('Queue identity changed while waiting for shared recovery')
                    state=pool.queue_state()
                    if index in initialized and pool.recover_budget_api_failure(controller_sha256,worker):state=pool.queue_state()
                    if state.get('failure'):continue
                    shared_failures.discard(index);reasons.pop(cohort(pool),None)
                    if state['phase']=='complete':terminal[index]='complete';continue
                    terminal.pop(index,None);initialized.discard(index);offers.pop(index,None)
                    print(f'[{worker}] {cohort(pool)}: shared queue recovered on another node; '
                          'rechecking original inputs before taking ready work',flush=True)
                    progressed=True
            except Exception as exc:give_up(index,pool,'queue recovery check',exc)
        claim_ok,busy=(True,None) if gpu_gate is None else gpu_gate()
        if not claim_ok and time.monotonic()-busy_report>=300:
            print(f'[{worker}] waiting before new GPU work: {busy}',flush=True);busy_report=time.monotonic()
        # Discovery/preparation is outside the global dispatch lock. Warm every cohort before
        # taking any GPU job, rather than filling one model while the other is still undiscovered.
        for index in [(cursor+i)%len(pairs) for i in range(len(pairs))]:
            if index in terminal:continue
            pool,runtime=pairs[index];action='waiting'
            if index in scans:
                scan=scans[index]
                if not scan.finished.is_set():
                    now=time.monotonic()
                    if now-scan.reported>=30:
                        print(f'[{worker}] {cohort(pool)}: {scan.kind} pending {now-scan.started:.0f}s; '
                              'no cross-cohort dispatch lock held',flush=True);scan.reported=now
                    continue
                del scans[index]
                if isinstance(scan.error,QueueReset):progressed=True;continue
                if scan.error is not None:
                    give_up(index,pool,scan.kind,scan.error);progressed=True;continue
                action,offer=scan.result
                if action=='verified':initialized.add(index);progressed=True
                if action=='ready':offers[index]=offer
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
            try:
                control_context=pool.control();control=control_context.__enter__()
            except NODE_LOCAL as exc:
                give_up(index,pool,'queue access',exc);progressed=True;continue
            try:
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
                                scans[index]=TaskScan(pool,runtime,worker,stop,control.fd)
                            else:
                                action,offer=pool.tick(runtime,worker,stop,claim=False,offer=True)
                                if action=='ready':offers[index]=offer
                            # Original actor scans never hold the cross-cohort dispatch lock.
                        if action in CPU_PHASES:
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
        pending_setup=[]
        for index,(pool,runtime) in enumerate(pairs):
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
                if gate_held:
                    # Kernel-held task leases, not heartbeat age or stale running labels, are
                    # authoritative. Claim while this same shared lock remains held.
                    live={i:len(pool.held_leases()) for i,(pool,_) in enumerate(pairs)
                          if i in initialized and i not in terminal and i in offers}
                    order=sorted(live,key=lambda i:(live[i],(i-cursor)%len(pairs)))
                    for candidate in order:
                        pool,runtime=pairs[candidate]
                        try:
                            # Another node's --new-queue may be replacing this folder right now: the next
                            # phase-1 pass reports the moved/replaced queue; never recreate its state here.
                            try:state=read(pool.root/'state.json')
                            except FileNotFoundError:
                                offers.pop(candidate,None);continue
                            if state.get('phase') not in PHASES or state.get('failure'):
                                offers.pop(candidate,None);continue
                            control_context=pool.control();control=control_context.__enter__()
                        except NODE_LOCAL as exc:
                            give_up(candidate,pool,'queue access',exc);progressed=True;continue
                        try:
                            if control is None:
                                # Give short metadata writes a chance to preserve model balance.
                                # A prolonged scan must not idle this GPU when another cohort is ready.
                                since=claim_contention_since.setdefault(candidate,time.monotonic())
                                if time.monotonic()-since<.25:
                                    short_claim_wait=True;break
                                continue
                            claim_contention_since.pop(candidate,None)
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
            pool,runtime=pairs[index]
            lease,key,task,attempt=claimed;cursor=(index+1)%len(pairs)
            log=pool.root/'logs'/(attempt['id']+'.log')
            print(f'[{worker}] {time.strftime("%Y-%m-%d %H:%M:%S")} {cohort(pool)} {task["kind"]} '
                  f'{task.get("method_id",task.get("task_id"))} seed={task.get("training_seed")} log={log}',flush=True)
            node_error=None
            try:
                try:
                    rc=75 if stop.is_set() else runtime.execute(task,pool.root/'tasks'/(key+'.json'),lease.fd,stop,device_fds,attempt)
                except Exception as exc:
                    # Launch-side faults (admission, node verification, dead actor, missing interpreter) happen on
                    # this node before a training result exists: never a task failure or a queue failure.
                    rc=None;node_error=f'{type(exc).__name__}: {exc} (node {worker})'
                    print(f'[{worker}] launch failure on this node: {node_error}',flush=True)
                recorded=None
                while recorded is None:
                    try:
                        with pool.control() as control:
                            if control is not None:recorded=pool.finish(key,rc,runtime,stop,node_error=node_error)
                    except NODE_LOCAL as exc:
                        node_error=node_error or f'{type(exc).__name__}: {exc}';recorded='unrecorded'
                        print(f'[{worker}] could not record the attempt ({exc}); the released lease lets another '
                              'node adopt or rerun it',flush=True)
                    if recorded is None:time.sleep(min(poll,.2))
            finally:lease.close()
            print(f'[{worker}] {time.strftime("%Y-%m-%d %H:%M:%S")} {cohort(pool)} task finished rc={rc} '
                  f'state={recorded} log={log}',flush=True)
            if node_error is not None:give_up(index,pool,'node launch',NodeFailure(node_error))
            dispatched=True
        if time.monotonic()-last_report>=60:
            try:print(f'[{worker}] '+canonical({p.binding.get('cohort','?'):p.status() for p,_ in pairs}),flush=True)
            except NODE_LOCAL as exc:print(f'[{worker}] status unavailable: {exc}',flush=True)
            last_report=time.monotonic()
        if any(v=='failed' for v in terminal.values()) or any(getattr(r,'primary_unavailable_cohorts',()) for _,r in pairs):
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
        if len(terminal)==len(pairs):
            failed=[cohort(pairs[i][0]) for i,v in terminal.items() if v=='failed']
            if failed and refresh_before_exit():continue
            for name in failed:print(f'[{worker}] {name}: FAILED: {reasons.get(name,"see its .multi-node/state.json")}',flush=True)
            done=[cohort(pairs[i][0]) for i,v in terminal.items() if v=='complete']
            if done:print(f'[{worker}] complete: {", ".join(done)}',flush=True)
            if failed:record_worker_retirement(pairs,worker,'no eligible work after final readiness check',
                                               [dict(cohort=name,detail=reasons.get(name)) for name in failed])
            return 1 if failed else 0
        if (not dispatched and not progressed and not offers and not scans and not pending_setup
                and all(i in initialized or i in terminal for i in range(len(pairs)))
                and not any(getattr(r,'pending_barrier',None) is not None for _,r in pairs)):
            blocked=quiescent_failure(pairs)
            if blocked:
                if refresh_before_exit():continue
                blocked=quiescent_failure(pairs)
                if not blocked:continue
                record_worker_retirement(pairs,worker,blocked['reason'],blocked['faults'])
                print(f'[{worker}] STOPPED: {blocked["reason"]}; '+canonical(blocked['faults'])+
                      '; failed checkpoints/logs retained; no torchrun relaunch',flush=True)
                return 1
        if not dispatched and not progressed:stop.wait(min(poll,.05) if short_claim_wait else poll)
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


def main(argv=None,*,selected_cohorts=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort',nargs='?',default='all',help='default: share olmo_math, qwen_mbpp, olmo_mbpp and qwen_math')
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
    parser.add_argument('--new-queue',action='store_true',
                        help='after stopping the workers on ALL nodes: keep the existing queue as .multi-node.superseded-<time> '
                             'and start a fresh one (refused once any update has completed or a job lease is held)')
    args=parser.parse_args(argv)
    if args.root is None:parser.error('V8_ROOT or --root is required (shared by all nodes)')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.cohort):parser.error('cohort must be a simple configured identifier')
    cohorts=tuple(selected_cohorts) if selected_cohorts is not None else DEFAULT_COHORTS if args.cohort=='all' else (args.cohort,)
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
                                max_attempts=args.max_attempts,retry_delay=args.retry_delay)
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)
        for lock in reversed(locks):lock.close()


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as error:
        print(f'V8 queue error: {error}',file=sys.stderr);raise SystemExit(1)
