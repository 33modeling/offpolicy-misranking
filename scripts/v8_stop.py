"""Stop this node's V8 job without deleting any checkpoint or queue record."""
import os
from pathlib import Path
import sys
import signal
import threading
from v8_probe import active_roots,root
from v8_processes import select,stop,table
from v8_control_output import emit,run_control


def main():
    roots=active_roots();preferred=root(os.environ)
    chosen=preferred if preferred in roots or not roots else roots[0] if len(roots)==1 else None
    if chosen is None:
        emit('Several independent V8 roots exist on this node; no process was signalled.',error=True);return 2
    targets=select(table(),root=chosen)
    if not targets:emit('No matching V8 process on this node.');return 0
    emit('Stopping only this node V8 PIDs: '+str(sorted(targets)))
    force=threading.Event()
    previous={sig:signal.signal(sig,lambda *_:force.set()) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:remaining=stop(targets,root=chosen,grace=10,force=force)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    if remaining:
        emit('SIGKILL pending / still alive: '+str(remaining),error=True);return 3
    emit('V8 processes stopped. Existing checkpoints and queue files preserved.');return 0


if __name__=='__main__':run_control(main)
