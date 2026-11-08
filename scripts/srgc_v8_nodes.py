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
import json
import math
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

SCHEMA = 1
TERMINAL = {'completed_horizon', 'completed_budget'}
PHASES = ('train', 'extension', 'evaluation')


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
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.fd=os.open(self.path,os.O_CREAT|os.O_RDWR,0o660)
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


class QueueReset(Exception):
    """--new-queue replaced the queue directory; the caller must re-acquire control.lock in the new one."""


MISMATCH_HELP=('This queue was started with a different bundle. Either keep using that bundle on every node, or stop '
               'the workers on ALL nodes and run `sh scripts/run_v8_nodes.sh --new-queue` once on one node (allowed '
               'only while no update has completed).')


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
    def __init__(self,root,binding,*,new_queue=False):
        self.root=Path(root);self.binding=binding;self.new_queue=new_queue
        self.root.mkdir(parents=True,exist_ok=True)

    def held_leases(self):
        """Jobs whose lease a live worker or its training child still holds."""
        held=[]
        for path in sorted((self.root/'leases').glob('*.lock')):
            lease=Lock(path)
            if lease.acquire():lease.close()
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
        os.replace(self.root,kept);self.root.mkdir(parents=True)
        atomic(self.root/'binding.json',self.binding)
        print(f'[v8] {cohort}: --new-queue kept the previous queue unchanged as {kept} and started a fresh queue in '
              f'{self.root}. Restart every other node with this same bundle (plain command, without --new-queue).',flush=True)
        raise QueueReset(str(kept))

    @contextmanager
    def control(self):
        lock=Lock(self.root/'control.lock')
        try:
            yield lock if lock.acquire() else None
        finally:lock.close()

    def initialize(self,retry=False,worker=None):
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
                        f"{self.binding.get('cohort')}: {MISMATCH_HELP} (Worker/runner/config/trainer identity differs "
                        f"from this queue: {', '.join(binding_changes(old,self.binding)[:12])}; record: {path})")
        else:atomic(path,self.binding)
        state_path=self.root/'state.json'
        if not state_path.exists():atomic(state_path,dict(phase='prepare',failure=None,actions=[],failure_history=[]))
        state=read(state_path)
        if retry and state['failure']:
            state['failure_history'].append(state['failure']);state['failure']=None
            for file in self.root.glob('jobs/*.json'):
                job=read(file)
                if job['state']=='failed':
                    lease=Lock(self.root/'leases'/(file.stem+'.lock'))
                    if not lease.acquire():raise ValueError('Failed job is still leased; cannot retry it yet')
                    lease.close();job['state']='pending';atomic(file,job)
            atomic(state_path,state)

    def fail(self,detail,task=None):
        state=read(self.root/'state.json')
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

    def tick(self,runtime,worker,stop,claim=True):
        """Called with control lock held. Returns (state, claimed lease/task)."""
        if read(self.root/'binding.json')!=self.binding:
            raise BindingMismatch(f"{self.binding.get('cohort')}: another node replaced this queue's code/Python record "
                                  "while this worker was running; restart this worker with the same bundle as the others.")
        state=read(self.root/'state.json')
        if state['failure']:return 'failed',None
        phase=state['phase']
        if phase=='complete':return 'complete',None
        if phase=='prepare':
            return 'prepare',None
        if phase in ('freeze','finalize'):
            return phase,None
        try:
            cached=self.root/'plans'/(phase+'.json')
            if cached.exists():
                plan=read(cached);tasks,sealed=plan['tasks'],True
            else:
                tasks,sealed=runtime.plan(phase)
                if sealed and (tasks or phase=='extension'):
                    atomic(cached,dict(tasks=tasks,sealed=True))
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
        for task in tasks:self.register(task,phase)
        jobs=[]
        for path in sorted(self.root.glob('jobs/*.json')):
            job=read(path)
            if job['phase']==phase:jobs.append((path,job))
        if not jobs and phase!='extension':
            raise ValueError('No tasks produced for '+phase+'; refusing to declare an empty experiment complete')
        pending=[];all_done=True
        for path,job in jobs:
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():
                all_done=False;continue
            try:
                native_busy=runtime.native_busy(job['task'])
                done=False if native_busy else runtime.done(job['task'])
                if done:
                    if job['state']!='done':
                        job['state']='done';job['adopted_at']=time.time();atomic(path,job)
                    continue
                all_done=False
                if job['state']=='done':raise ValueError('Previously completed output disappeared or became inconsistent: '+path.stem)
                if native_busy or not runtime.ready(job['task']):continue
                pending.append((path,job))
            finally:lease.close()
        if all_done and sealed:
            state['phase']={'train':'freeze','extension':'evaluation','evaluation':'finalize'}[phase]
            atomic(self.root/'state.json',state)
            return 'advance',None
        # Fill trajectories first, then checkpoint-ready branches. Ties rotate
        # toward tasks with fewer launch attempts so a busy task cannot starve others.
        pending.sort(key=lambda item:(item[1]['task'].get('method_id')!='on_monitored',len(item[1]['attempts']),item[0].stem))
        if pending and not claim:return 'gpu_busy',None
        for path,job in pending:
            if stop.is_set():return 'stopped',None
            lease=Lock(self.root/'leases'/(path.stem+'.lock'))
            if not lease.acquire():continue
            attempt=dict(id=uuid.uuid4().hex,worker=worker,started_at=time.time(),pid=os.getpid())
            if job['state']=='running' and job['attempts']:
                job['attempts'][-1]['lease_released_without_worker_completion']=time.time()
            job['attempts'].append(attempt);job['state']='running';atomic(path,job)
            return 'claimed',(lease,path.stem,job['task'],attempt)
        return 'waiting',None

    def finish(self,key,rc,runtime,stop):
        path=self.root/'jobs'/(key+'.json');job=read(path)
        job['attempts'][-1].update(exit_code=rc,ended_at=time.time())
        try:done=runtime.done(job['task'])
        except Exception as exc:
            done=False;job['attempts'][-1]['output_validation_error']=str(exc)
        if rc==0 and done:job['state']='done'
        elif rc==75 or stop.is_set():job['state']='interrupted'
        else:
            job['state']='failed'
            self.fail(f'task exit={rc}; output verified={done}',key)
        atomic(path,job)

    def status(self):
        state=read(self.root/'state.json') if (self.root/'state.json').exists() else dict(phase='not_started')
        counts={};owners=[]
        for path in self.root.glob('jobs/*.json'):
            job=read(path);counts[job['state']]=counts.get(job['state'],0)+1
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
        return dict(phase=state['phase'],failure=state.get('failure'),known_task_counts=counts,leased_tasks=owners)


def worker_loop(pool,runtime,worker,stop,*,poll=3,device_fds=(),retry=False):
    return worker_loop_many([(pool,runtime)],worker,stop,poll=poll,device_fds=device_fds,retry=retry)


def worker_loop_many(pairs,worker,stop,*,poll=3,device_fds=(),retry=False,gpu_gate=None):
    """One GPU group takes the next ready job from any cohort, without fixed shares.

    A cohort whose setup or task fails is recorded and left alone; the worker keeps
    serving the other cohort and returns 1 at the end (never a silent success).
    gpu_gate() -> (ok, message) defers new GPU claims while the GPUs are in use.
    """
    if not pairs:raise ValueError('At least one cohort is required')
    initialized=set();terminal={};last_report=0;reasons={};busy_report=0
    cursor=int(digest(worker)[:8],16)%len(pairs)
    for pool,runtime in pairs:
        def heartbeat(task=None,pool=pool):
            atomic(pool.root/'workers'/(digest(worker)+'.json'),
                   dict(worker=worker,pid=os.getpid(),at=time.time(),task=task))
        runtime.heartbeat=heartbeat
    def cohort(pool):
        return pool.binding.get('cohort','?')
    def give_up(index,pool,kind,exc):
        terminal[index]='failed';reasons[cohort(pool)]=f'{type(exc).__name__}: {exc}'
        print(f'[{worker}] {cohort(pool)}: {kind} failed: {type(exc).__name__}: {exc}',flush=True)
        print(f'[{worker}] {cohort(pool)}: no new work is taken for this cohort; the other cohort continues. '
              f'Logs: {pool.root/"logs"}. After fixing the cause, run the same command with --retry-failed.',flush=True)
    while not stop.is_set():
        progressed=False;dispatched=False
        claim_ok,busy=(True,None) if gpu_gate is None else gpu_gate()
        if not claim_ok and time.monotonic()-busy_report>=300:
            print(f'[{worker}] waiting before new GPU work: {busy}',flush=True);busy_report=time.monotonic()
        for index in [(cursor+i)%len(pairs) for i in range(len(pairs))]:
            if index in terminal:continue
            pool,runtime=pairs[index];claimed=None;action='waiting'
            with pool.control() as control:
                if control is not None:
                    if index not in initialized:
                        # A mismatched joining node must not poison healthy workers: no pool.fail here.
                        try:runtime.verify_node();pool.initialize(retry=retry,worker=worker)
                        except QueueReset:
                            progressed=True;continue  # re-lock control.lock of the fresh directory first
                        except Exception as exc:
                            give_up(index,pool,'node check',exc);progressed=True;continue
                        initialized.add(index)
                    try:
                        action,claimed=pool.tick(runtime,worker,stop,claim_ok)
                        if action in ('prepare','freeze','finalize'):
                            print(f'[{worker}] {cohort(pool)}: CPU stage {action} started',flush=True)
                            runtime.barrier(action,control.fd,stop,device_fds)
                            if stop.is_set():return 75
                            state=read(pool.root/'state.json')
                            state['actions'].append(dict(action=action,worker=worker,at=time.time()))
                            state['phase']={'prepare':'train','freeze':'extension','finalize':'complete'}[action]
                            atomic(pool.root/'state.json',state);action='advance'
                            print(f'[{worker}] {cohort(pool)}: CPU stage finished; phase={state["phase"]}',flush=True)
                    except Exception as exc:
                        if stop.is_set():return 75
                        if (pool.root/'state.json').exists():pool.fail(f'{type(exc).__name__}: {exc}')
                        give_up(index,pool,'stage',exc);progressed=True;continue
            runtime.heartbeat(claimed[2].get('run_id') if claimed else None)
            if action=='failed':
                state=read(pool.root/'state.json')
                reasons[cohort(pool)]=canonical(state.get('failure'))
                print(f'[{worker}] {cohort(pool)}: queue has a recorded failure: {reasons[cohort(pool)]}. '
                      f'Fix it, then run the same command with --retry-failed.',flush=True)
            if action in ('complete','failed'):
                terminal[index]=action;progressed=True;continue
            if action=='advance':progressed=True
            if not claimed:continue
            lease,key,task,attempt=claimed;cursor=(index+1)%len(pairs)
            log=pool.root/'logs'/(attempt['id']+'.log')
            print(f'[{worker}] {time.strftime("%Y-%m-%d %H:%M:%S")} {cohort(pool)} {task["kind"]} '
                  f'{task.get("method_id",task.get("task_id"))} seed={task.get("training_seed")} log={log}',flush=True)
            try:
                try:
                    rc=75 if stop.is_set() else runtime.execute(task,pool.root/'tasks'/(key+'.json'),lease.fd,stop,device_fds,attempt)
                except Exception as exc:
                    rc=1;print(f'[{worker}] launch failure: {type(exc).__name__}: {exc}',flush=True)
                while True:
                    with pool.control() as control:
                        if control is not None:
                            pool.finish(key,rc,runtime,stop);break
                    time.sleep(min(poll,.2))
            finally:lease.close()
            print(f'[{worker}] {time.strftime("%Y-%m-%d %H:%M:%S")} {cohort(pool)} task finished rc={rc} log={log}',flush=True)
            dispatched=True;break
        if time.monotonic()-last_report>=60:
            print(f'[{worker}] '+canonical({p.binding.get('cohort','?'):p.status() for p,_ in pairs}),flush=True)
            last_report=time.monotonic()
        if len(terminal)==len(pairs):
            failed=[cohort(pairs[i][0]) for i,v in terminal.items() if v=='failed']
            for name in failed:print(f'[{worker}] {name}: FAILED: {reasons.get(name,"see its .multi-node/state.json")}',flush=True)
            done=[cohort(pairs[i][0]) for i,v in terminal.items() if v=='complete']
            if done:print(f'[{worker}] complete: {", ".join(done)}',flush=True)
            return 1 if failed else 0
        if not dispatched and not progressed:stop.wait(poll)
    return 75


def python_identity(executable):
    code="""import importlib.metadata,json,sys
packages={}
for name in ('torch','transformers','peft','numpy','math-verify'):
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
        if time.monotonic()-self.checked<self.ttl:return self.value
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
        if not (self.runner/'cluster_child.py').is_file():raise ValueError('runner-root must contain cluster_child.py')
        if not os.environ.get('V8_CODE_ROOT'):raise ValueError('V8_CODE_ROOT must point to the frozen trainer checkout')
        sys.path.insert(0,str(self.runner))
        from switching import cluster,matrix,spec,coderepo
        self.cluster,self.mx,self.spec,self.code=cluster,matrix,spec,coderepo
        self.cfg=spec.resolve(cohort,role='rebuttal_bank')
        if len(gpus)!=self.cfg['runtime']['world_size']:raise ValueError('GPU count must match the frozen runtime world_size')
        files=sorted(list((self.runner/'switching').glob('*.py'))+list((self.runner/'configs').rglob('*.json'))+
                     [self.runner/'cluster_child.py',self.runner/'run_h100.sh'])
        environment_identity=python_identity(python)
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
            if not (self.root/'frozen/budget.json').exists() and list((self.root/'evaluations').glob('*.json')):
                raise ValueError('Evaluation records exist before budget freeze; cannot certify outcome-independent budget selection')
            self.cpu('freeze-budget',fd,stop,device_fds)
        else:
            for stage in ('collect','analyze'):
                self.cpu(stage,fd,stop,device_fds)
                if stop.is_set():return
            if self.environment.get('EXPORT_CODE_ROOT'):self.cpu('history-export',fd,stop,device_fds)

    def locator(self):return self.cluster.ClusterLocator(self.root,self.cfg)

    def plan(self,phase):
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
        return self.process(command,self.root/'.multi-node'/'logs'/(attempt['id']+'.log'),[fd,*device_fds],stop,
                            extra={'SRGC_PROGRESS_DIR':str(progress)},label=task.get('task_id') or task['run_id'])

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
                if hasattr(self,'heartbeat'):self.heartbeat('child:'+str(process.pid))
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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort',nargs='?',default='all',help='default: automatically share both olmo_math and qwen_mbpp')
    parser.add_argument('--runner-root',type=Path,default=os.environ.get('V8_RUNNER_ROOT',str(Path.cwd()/'v8/experiments')))
    parser.add_argument('--root',type=Path,default=os.environ.get('V8_ROOT'))
    parser.add_argument('--python',default=os.environ.get('V8_PYTHON',sys.executable),help='fallback Python; cohort-specific environments take priority')
    parser.add_argument('--gpus',default=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3'))
    parser.add_argument('--poll-seconds',type=float,default=3)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--new-queue',action='store_true',
                        help='after stopping the workers on ALL nodes: keep the existing queue as .multi-node.superseded-<time> '
                             'and start a fresh one (refused once any update has completed or a job lease is held)')
    args=parser.parse_args(argv)
    if args.root is None:parser.error('V8_ROOT or --root is required (shared by all nodes)')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.cohort):parser.error('cohort must be a simple configured identifier')
    cohorts=('olmo_math','qwen_mbpp') if args.cohort=='all' else (args.cohort,)
    if args.status:
        statuses={}
        for cohort in cohorts:
            queue=Path(args.root).resolve()/cohort/'.multi-node'
            statuses[cohort]=Pool(queue,{}).status() if queue.exists() else {'phase':'not_started'}
        print(json.dumps(statuses if args.cohort=='all' else statuses[args.cohort],indent=2));return 0
    if not args.python:parser.error('V8_PYTHON or --python is required')
    if not args.poll_seconds>0:parser.error('poll-seconds must be positive')
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
                                retry=args.retry_failed,gpu_gate=GpuGate(gpus) if inventory is not None else None)
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)
        for lock in reversed(locks):lock.close()


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as error:
        print(f'V8 queue error: {error}',file=sys.stderr);raise SystemExit(1)
