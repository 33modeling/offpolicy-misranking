"""Bounded V8 Ctrl-C handling without changing trainer/runtime queue identities."""
import os
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from v8_processes import TAG,select,progress_every,stop,stop_grace,table
from v8_probe import read,root,snapshot,tail,worker_status
from v8_control_output import emit,run_control
from v8_progress_text import describe,scheduler_text


def work_events(rows,launch,fallback):
    """Read the active launch's actual per-attempt directory, never another launch's progress."""
    tasks=[row for row in rows.values() if row['env'].get(TAG)==launch and row.get('kind')=='task' and row.get('state')!='Z']
    if not tasks:return []  # launch/control processes do not prove a training child is still alive
    directories={}
    for row in tasks:
        directory=Path(row['env'].get('SRGC_PROGRESS_DIR') or fallback)
        started=row.get('started_at')
        if not isinstance(started,(int,float)):
            try:started=time.time()-time.clock_gettime(time.CLOCK_BOOTTIME)+float(row['start'])/os.sysconf('SC_CLK_TCK')
            except (KeyError,ValueError,TypeError,AttributeError,OSError):started=None
        directories.setdefault(directory,[]).append(started)
    parts=[]
    for directory,starts in sorted(directories.items()):
        valid_starts=[value for value in starts if value is not None]
        for rank in range(4):
            event=read(directory/f'rank-{rank}.json')
            if isinstance(event,dict) and isinstance(event.get('updated'),(int,float)):
                if valid_starts and event['updated']<min(valid_starts):continue
                age=time.time()-event['updated']
                kind='stage entered' if event.get('progress_kind')=='entered_only' else 'last new token' if event.get('progress_kind')=='returned_decode_steps' else 'last work record'
                state='NO_PROGRESS' if age>=120 else 'STARTING_TASK' if event.get('progress_kind')=='entered_only' else 'RUNNING'
                stamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(event['updated']))
                parts.append(f'state={state} | GPU process {rank} | {describe(event.get("stage","?"),event)} | {kind} {stamp} ({age:.0f}s ago)')
    return parts


def before_model_work(rows,launch,base,console,host=None):
    """Use the existing queue receipts/logs like SRGC NODE/BLOCKED status, never infer preparation."""
    host=host or socket.gethostname()
    owned={str(pid):row for pid,row in rows.items() if row['env'].get(TAG)==launch}
    bases={Path(row['env']['V8_ROOT']) for row in owned.values() if row['env'].get('V8_ROOT')} or {Path(base)}
    lines=[]
    for folder in sorted(bases):
        diagnoses=[]
        for pid in owned:
            record=worker_status(folder,host+'.'+pid)
            if record and record.get('launch_id')==launch:
                diagnoses.append(record)
        if diagnoses:
            record=max(diagnoses,key=lambda r:r['at'])
            lines.append('NODE '+host+': '+scheduler_text(record,time.time()))
            if record.get('log'):lines.append('TASK LOG: '+record['log'])
            continue
        for item in snapshot(folder,host):
            label=item['cohort'];failure=item.get('failure')
            for error in item.get('record_errors',[]):lines.append('STATUS ERROR '+label+': '+error)
            if failure:
                detail=failure.get('detail',str(failure)) if isinstance(failure,dict) else str(failure)
                lines.append(f'BLOCKED {label}: {detail}; queue={folder/label/".multi-node/state.json"}')
            mine=[job for job in item['local_jobs'] if job.get('worker','').rsplit('.',1)[-1] in owned]
            for job in mine:
                age=max(0,time.time()-job['attempt_started']) if isinstance(job.get('attempt_started'),(int,float)) else None
                status='NO_PROGRESS' if age is not None and age>=120 else 'ASSIGNED'
                lines.append(f'ASSIGNED {label} seed={job.get("seed")} method={job.get("method")}; '
                             f'state={status}; no model token/update record for this launch yet'
                             +(f' ({age:.0f}s since assignment)' if age is not None else '')+
                             f'; run state={job["run_manifest"]}; log={job["log"]}')
                if job['last_log_lines']:lines.append('TASK OUTPUT: '+job['last_log_lines'][-1])
            if not mine and not failure:
                counts=', '.join(f'{k}={v}' for k,v in sorted(item['recorded_job_counts'].items())) or 'no recorded jobs'
                lines.append(f'NOT ASSIGNED {label}: queue phase={item["phase"]}; recorded queue jobs: {counts}')
    recent=[line.strip() for line in tail(console,8192) if line.strip() and 'known_task_counts' not in line]
    if recent:lines.append('LATEST LAUNCHER OUTPUT: '+recent[-1][:1200])
    return lines or ['No task/progress evidence is readable for this launch; current execution phase is unknown.']


def describe_processes(remaining):
    kinds={}
    for row in remaining.values():kinds[row.get('kind') or 'control']=kinds.get(row.get('kind') or 'control',0)+1
    names={'task':'training/evaluation','cpu':'CPU stage','control':'dispatcher/actor'}
    return ', '.join(f'{count} {names.get(kind,kind)}' for kind,count in sorted(kinds.items()))


def main():
    args=sys.argv[1:];bootstrap=Path(__file__).with_name('v8_bootstrap.py')
    if any(a in ('--status','--help','-h') for a in args):
        os.execv(sys.executable,[sys.executable,str(bootstrap),*args])
    launch=uuid.uuid4().hex;env=dict(os.environ,**{TAG:launch})
    progress=root(env)/'.node-progress'/socket.gethostname()/launch
    env['SRGC_PROGRESS_DIR']=str(progress)
    stopped=threading.Event();force=threading.Event();finished=threading.Event();disconnected=threading.Event();received=[]
    child=None;relay_thread=None
    def interrupt(sig,*_):
        if stopped.is_set():force.set()
        else:received.append(sig);stopped.set()
    previous={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    previous[signal.SIGHUP]=signal.signal(signal.SIGHUP,lambda *_:disconnected.set())
    def report():
        while not finished.wait(15):
            rows=table();parts=work_events(rows,launch,progress)
            if not parts:
                try:parts=before_model_work(rows,launch,root(env),log)
                except (OSError,ValueError,TypeError,KeyError) as exc:
                    parts=[f'STATUS ERROR: cannot read this launch state: {type(exc).__name__}: {exc}; console log={log}']
            try:print('[V8 STATUS] '+' | '.join(parts),flush=True)
            except (OSError,ValueError):return  # disconnected console must not stop its independent child
    def relay(log):
        with log.open('rb') as reader:
            while True:
                block=reader.read(65536)
                if not block and finished.is_set():
                    block=reader.read(65536)
                    if not block:return
                if block:
                    try:
                        stream=getattr(sys.stdout,'buffer',None)
                        if stream is None:sys.stdout.write(block.decode(errors='replace'));sys.stdout.flush()
                        else:stream.write(block);stream.flush()
                    except (OSError,ValueError):return
                else:finished.wait(.05)
    try:
        if stopped.is_set():return 128+received[0]
        directory=Path(tempfile.gettempdir())/f'v8-console-{os.getuid()}'
        directory.mkdir(mode=0o700,exist_ok=True)
        if directory.is_symlink() or directory.stat().st_uid!=os.getuid() or directory.stat().st_mode&0o077:
            raise ValueError('V8 console directory must be private and owned by this user')
        log=directory/(launch+'.log');env['V8_CONSOLE_LOG']=str(log)
        with log.open('xb') as handle:
            child=subprocess.Popen([sys.executable,str(bootstrap),*args],env=env,start_new_session=True,
                                   stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
        relay_thread=threading.Thread(target=relay,args=(log,),daemon=True);relay_thread.start()
        threading.Thread(target=report,daemon=True).start()
        while child.poll() is None:
            if disconnected.is_set():
                disconnected.clear()
                try:
                    progress.mkdir(parents=True,exist_ok=True)
                    with (progress/'session-disconnect.jsonl').open('a') as output:
                        output.write(json.dumps(dict(at=time.time(),signal='SIGHUP',action='continue_current_launch',
                                                     supervisor_pid=os.getpid(),console_log=str(log)))+'\n')
                except OSError:pass  # losing a terminal or metadata write cannot cancel the learner
                emit(f'[v8] terminal disconnected (SIGHUP); this launch continues. Console log: {log}')
            if stopped.wait(.1):
                grace=stop_grace()
                emit(f'[v8 stop] stopping this launch: SIGTERM sent; waiting up to {grace:g}s (V8_STOP_WAIT_SECONDS) '
                     'for a running task to stop at its next safe point, record exit 75 and exit (resumable from its '
                     'last saved update; a rank blocked in a collective is ended by the child watchdog). '
                     'Press Ctrl-C again to kill everything immediately.')
                def waiting(remaining,elapsed):
                    events=work_events(table(),launch,progress)
                    emit(f'[v8 stop] still waiting {elapsed:.0f}/{grace:g}s for {describe_processes(remaining)} process(es) to '
                         'reach a safe update boundary'+(f'; {" | ".join(events)}' if events else '')+
                         '; Ctrl-C again kills now')
                remaining=stop(select(table(),tag=launch),tag=launch,grace=grace,force=force,progress=waiting,every=progress_every())
                if remaining:emit('[v8 stop] processes still in kernel wait: '+str(remaining),error=True)
                return 128+received[0]
        # No stdout pipe join: descendants cannot trap the controller in read().
        remaining=stop(select(table(),tag=launch),tag=launch,grace=1)
        if remaining:emit('[v8 stop] remaining owned processes: '+str(remaining),error=True)
        return 3 if remaining else child.returncode if child.returncode>=0 else 128-child.returncode
    finally:
        finished.set()
        leftovers=select(table(),tag=launch)
        if leftovers:stop(leftovers,tag=launch,grace=0 if stopped.is_set() else 1)
        if relay_thread:relay_thread.join(timeout=.2)
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':run_control(main)
