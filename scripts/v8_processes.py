"""Identify and stop only owned V8 launches, including ranks in new sessions."""
import math
import os
from pathlib import Path
import signal
import time

TAG='V8_LAUNCH_ID'
MARKERS={'srgc_v8_nodes.py','cluster_child.py','v8_dispatch.py','v8_runtime_worker.py'}
TASK_MARKERS={'cluster_child.py','v8_task_entry.py'}
DEFAULT_STOP_GRACE=900.0  # long enough for one training update to reach its safe stop boundary


def stop_grace(environment=None):
    """Seconds a stopping launch waits after SIGTERM before SIGKILL (V8_STOP_WAIT_SECONDS, default 900).

    Not V8_STOP_GRACE_SECONDS: that one is the training child's own watchdog for ranks blocked in a
    collective (cluster_child.py, default 7 s) and is inherited by the child unchanged."""
    value=(os.environ if environment is None else environment).get('V8_STOP_WAIT_SECONDS','')
    try:grace=float(value) if value.strip() else DEFAULT_STOP_GRACE
    except ValueError:grace=DEFAULT_STOP_GRACE
    return grace if math.isfinite(grace) and grace>=0 else DEFAULT_STOP_GRACE


def progress_every(environment=None):
    value=(os.environ if environment is None else environment).get('V8_STOP_PROGRESS_SECONDS','')
    try:every=float(value) if value.strip() else 30.0
    except ValueError:every=30.0
    return every if math.isfinite(every) and every>0 else 30.0


def table(proc=Path('/proc')):
    rows={}
    try:entries=list(Path(proc).iterdir())
    except OSError:return rows
    for entry in entries:
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            fields=(entry/'stat').read_text().rsplit(') ',1)[1].split()
            args=(entry/'cmdline').read_bytes().split(b'\0')
            env={}
            for item in (entry/'environ').read_bytes().split(b'\0'):
                key,sep,value=item.partition(b'=')
                if sep and key in (b'V8_ROOT',b'SRGC_PROGRESS_DIR',TAG.encode()):env[key.decode()]=value.decode(errors='replace')
            names={Path(a.decode(errors='replace')).name for a in args if a}
            kind='task' if names&TASK_MARKERS else 'cpu' if 'run_h100.sh' in names else None
            rows[int(entry.name)]=dict(ppid=int(fields[1]),start=fields[19],state=fields[0],env=env,
                                      v8=bool(names&MARKERS),kind=kind)
        except (OSError,ValueError,IndexError):continue
    return rows


def select(rows,*,root=None,tag=None):
    selected={pid for pid,row in rows.items() if
              (row['env'].get(TAG)==tag if tag else
               (row['v8'] or row['env'].get(TAG)) and row['env'].get('V8_ROOT')==str(root))}
    while True:
        added={pid for pid,row in rows.items() if row['ppid'] in selected}
        if added<=selected:break
        selected|=added
    return {pid:rows[pid]['start'] for pid in selected if pid!=os.getpid()}


def send(identities,sig,*,rows=None,killer=os.kill):
    rows=table() if rows is None else rows
    for pid,start in identities.items():
        if pid in rows and rows[pid]['start']==start and rows[pid]['state']!='Z':
            try:killer(pid,sig)
            except ProcessLookupError:pass


def alive(identities,rows=None):
    rows=table() if rows is None else rows
    return {pid:start for pid,start in identities.items() if pid in rows and rows[pid]['start']==start and rows[pid]['state']!='Z'}


def stop(identities,*,tag=None,root=None,grace=10,force=None,progress=None,every=30):
    """Bounded cleanup: SIGTERM, wait up to grace seconds (or until force is set), then SIGKILL.
    progress(remaining, elapsed) is called every `every` seconds while waiting. SIGKILL-pending kernel
    waits are reported, never called stopped."""
    known=dict(identities);started=time.monotonic();killed=False;kill_at=None;notified=set();reported=started
    while True:
        rows=table()
        if tag:known.update(select(rows,tag=tag))
        elif root is not None:known.update(select(rows,root=root))
        remaining=alive(known,rows)
        if not remaining:return []
        now=time.monotonic()
        if not killed and (now-started>=grace or force is not None and force.is_set()):
            send(remaining,signal.SIGKILL,rows=rows);killed=True;kill_at=now
        elif not killed:
            fresh={p:s for p,s in remaining.items() if (p,s) not in notified}
            send(fresh,signal.SIGTERM,rows=rows);notified.update(fresh.items())
            if progress is not None and now-reported>=every:
                reported=now
                try:progress({p:rows[p] for p in remaining if p in rows},now-started)
                except Exception:pass  # reporting must never block cleanup
        else:send(remaining,signal.SIGKILL,rows=rows)
        if killed and now-kill_at>=2:return sorted(alive(known))
        time.sleep(.05)
