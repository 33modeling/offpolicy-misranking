"""Bounded V8 Ctrl-C handling without changing trainer/runtime queue identities."""
import os
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
from v8_probe import read,root,snapshot,tail
from v8_control_output import emit,run_control
from v8_progress_text import describe


def work_events(rows,launch,fallback):
    """Read the active launch's actual per-attempt directory, never another launch's progress."""
    directories={Path(row['env']['SRGC_PROGRESS_DIR']) for row in rows.values()
                 if row['env'].get(TAG)==launch and row['env'].get('SRGC_PROGRESS_DIR')}
    parts=[]
    for directory in sorted(directories or {Path(fallback)}):
        for rank in range(4):
            event=read(directory/f'rank-{rank}.json')
            if isinstance(event,dict) and isinstance(event.get('updated'),(int,float)):
                age=time.time()-event['updated']
                kind='stage entered' if event.get('progress_kind')=='entered_only' else 'last new token' if event.get('progress_kind')=='returned_decode_steps' else 'last work record'
                parts.append(f'GPU process {rank} | {describe(event.get("stage","?"),event)} | {kind} {age:.0f}s ago')
    return parts


def before_model_work(rows,launch,base,console,host=None):
    """Use the existing queue receipts/logs like SRGC NODE/BLOCKED status, never infer preparation."""
    host=host or socket.gethostname()
    owned={str(pid):row for pid,row in rows.items() if row['env'].get(TAG)==launch}
    bases={Path(row['env']['V8_ROOT']) for row in owned.values() if row['env'].get('V8_ROOT')} or {Path(base)}
    lines=[]
    for folder in sorted(bases):
        for item in snapshot(folder,host):
            label=item['cohort'];failure=item.get('failure')
            for error in item.get('record_errors',[]):lines.append('STATUS ERROR '+label+': '+error)
            if failure:
                detail=failure.get('detail',str(failure)) if isinstance(failure,dict) else str(failure)
                lines.append(f'BLOCKED {label}: {detail}; queue={folder/label/".multi-node/state.json"}')
            mine=[job for job in item['local_jobs'] if job.get('worker','').rsplit('.',1)[-1] in owned]
            for job in mine:
                lines.append(f'ASSIGNED {label} seed={job.get("seed")} method={job.get("method")}; '
                             f'no model token/update record for this launch yet; run state={job["run_manifest"]}; log={job["log"]}')
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
    stopped=threading.Event();force=threading.Event();finished=threading.Event();received=[]
    child=None;relay_thread=None
    def interrupt(sig,*_):
        if stopped.is_set():force.set()
        else:received.append(sig);stopped.set()
    previous={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    def report():
        while not finished.wait(15):
            rows=table();parts=work_events(rows,launch,progress)
            if not parts:
                try:parts=before_model_work(rows,launch,root(env),log)
                except (OSError,ValueError,TypeError,KeyError) as exc:
                    parts=[f'STATUS ERROR: cannot read this launch state: {type(exc).__name__}: {exc}; console log={log}']
            print('[Training status] '+' | '.join(parts),flush=True)
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
