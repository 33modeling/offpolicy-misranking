"""Stop this node's V8 job without deleting any checkpoint or queue record."""
import os
from pathlib import Path
import sys
import signal
import threading
from v8_probe import active_roots,root
from v8_processes import select,progress_every,stop,stop_grace,table
from v8_control_output import emit,run_control


def main():
    roots=active_roots();preferred=root(os.environ)
    chosen=preferred if preferred in roots or not roots else roots[0] if len(roots)==1 else None
    if chosen is None:
        emit('Several independent V8 roots exist on this node; no process was signalled.',error=True);return 2
    targets=select(table(),root=chosen)
    if not targets:emit('No matching V8 process on this node.');return 0
    grace=stop_grace()
    emit('Stopping only this node V8 PIDs: '+str(sorted(targets))+f'. Waiting up to {grace:g}s (V8_STOP_WAIT_SECONDS) for '
         'a running task to stop at its next safe point and record exit 75; Ctrl-C kills immediately.')
    force=threading.Event()
    previous={sig:signal.signal(sig,lambda *_:force.set()) for sig in (signal.SIGINT,signal.SIGTERM)}
    def waiting(remaining,elapsed):
        emit(f'still waiting {elapsed:.0f}/{grace:g}s for {len(remaining)} process(es) to reach a safe update boundary; '
             'Ctrl-C kills now')
    try:remaining=stop(targets,root=chosen,grace=grace,force=force,progress=waiting,every=progress_every())
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    if remaining:
        emit('SIGKILL pending / still alive: '+str(remaining),error=True);return 3
    emit('V8 processes stopped. Existing checkpoints and queue files preserved.');return 0


if __name__=='__main__':run_control(main)
