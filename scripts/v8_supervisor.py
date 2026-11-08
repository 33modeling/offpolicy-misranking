"""Bounded V8 Ctrl-C handling without changing trainer/runtime queue identities."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from v8_processes import TAG,select,stop,table
from v8_probe import read,root


def work_events(rows,launch,fallback):
    """Read the active launch's actual per-attempt directory, never another launch's progress."""
    directories={Path(row['env']['SRGC_PROGRESS_DIR']) for row in rows.values()
                 if row['env'].get(TAG)==launch and row['env'].get('SRGC_PROGRESS_DIR')}
    parts=[]
    for directory in sorted(directories or {Path(fallback)}):
        for rank in range(4):
            event=read(directory/f'rank-{rank}.json')
            if isinstance(event,dict) and isinstance(event.get('updated'),(int,float)):
                parts.append(f'rank{rank}={event.get("stage","?")} event_age={time.time()-event["updated"]:.0f}s')
    return parts


def main():
    args=sys.argv[1:];bootstrap=Path(__file__).with_name('v8_bootstrap.py')
    if any(a in ('--status','--help','-h') for a in args):
        os.execv(sys.executable,[sys.executable,str(bootstrap),*args])
    launch=uuid.uuid4().hex;env=dict(os.environ,**{TAG:launch})
    progress=root(env)/'.node-progress'/socket.gethostname()/launch
    env['SRGC_PROGRESS_DIR']=str(progress)
    stopped=threading.Event();force=threading.Event();finished=threading.Event();received=[]
    child=None
    def interrupt(sig,*_):
        if stopped.is_set():force.set()
        else:received.append(sig);stopped.set()
    previous={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    def report():
        while not finished.wait(15):
            parts=work_events(table(),launch,progress)
            print('[v8 status] '+(' | '.join(parts) if parts else 'startup: no rank work event recorded yet'),flush=True)
    try:
        if stopped.is_set():return 128+received[0]
        child=subprocess.Popen([sys.executable,str(bootstrap),*args],env=env,start_new_session=True)
        threading.Thread(target=report,daemon=True).start()
        while child.poll() is None:
            if stopped.wait(.1):
                print('[v8 stop] stopping this launch; grace10s, second Ctrl-C requests immediate cleanup',flush=True)
                remaining=stop(select(table(),tag=launch),tag=launch,grace=10,force=force)
                if remaining:print('[v8 stop] processes still in kernel wait: '+str(remaining),file=sys.stderr,flush=True)
                return 128+received[0]
        # No stdout pipe join: descendants cannot trap the controller in read().
        remaining=stop(select(table(),tag=launch),tag=launch,grace=1)
        if remaining:print('[v8 stop] remaining owned processes: '+str(remaining),file=sys.stderr,flush=True)
        return 3 if remaining else child.returncode if child.returncode>=0 else 128-child.returncode
    finally:
        finished.set()
        leftovers=select(table(),tag=launch)
        if leftovers:stop(leftovers,tag=launch,grace=0 if stopped.is_set() else 1)
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':raise SystemExit(main())
