"""Identify and stop only owned V8 launches, including ranks in new sessions."""
import os
from pathlib import Path
import signal
import time

TAG='V8_LAUNCH_ID'
MARKERS={'srgc_v8_nodes.py','cluster_child.py'}


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
            rows[int(entry.name)]=dict(ppid=int(fields[1]),start=fields[19],state=fields[0],env=env,
                                      v8=any(Path(a.decode(errors='replace')).name in MARKERS for a in args if a))
        except (OSError,ValueError,IndexError):continue
    return rows


def select(rows,*,root=None,tag=None):
    selected={pid for pid,row in rows.items() if
              (row['env'].get(TAG)==tag if tag else row['v8'] and row['env'].get('V8_ROOT')==str(root))}
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


def stop(identities,*,tag=None,grace=10,force=None):
    """Bounded cleanup. SIGKILL-pending kernel waits are reported, never called stopped."""
    known=dict(identities);started=time.monotonic();killed=False;kill_at=None;notified=set()
    while True:
        rows=table()
        if tag:known.update(select(rows,tag=tag))
        remaining=alive(known,rows)
        if not remaining:return []
        now=time.monotonic()
        if not killed and (now-started>=grace or force is not None and force.is_set()):
            send(remaining,signal.SIGKILL,rows=rows);killed=True;kill_at=now
        elif not killed:
            fresh={p:s for p,s in remaining.items() if (p,s) not in notified}
            send(fresh,signal.SIGTERM,rows=rows);notified.update(fresh.items())
        else:send(remaining,signal.SIGKILL,rows=rows)
        if killed and now-kill_at>=2:return sorted(alive(known))
        time.sleep(.05)
