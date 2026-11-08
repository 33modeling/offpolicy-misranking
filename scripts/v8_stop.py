"""Stop this node's V8 job without deleting any checkpoint or queue record."""
import os
from pathlib import Path
import sys
import signal
import threading
from v8_probe import active_roots,root
from v8_processes import select,stop,table


def main():
    roots=active_roots();preferred=root(os.environ)
    chosen=preferred if preferred in roots or not roots else roots[0] if len(roots)==1 else None
    if chosen is None:
        print('Several independent V8 roots exist on this node; no process was signalled.',file=sys.stderr);return 2
    targets=select(table(),root=chosen)
    if not targets:print('No matching V8 process on this node.');return 0
    print('Stopping only this node V8 PIDs: '+str(sorted(targets)),flush=True)
    force=threading.Event()
    previous={sig:signal.signal(sig,lambda *_:force.set()) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:remaining=stop(targets,grace=10,force=force)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    if remaining:
        print('SIGKILL pending / still alive: '+str(remaining),file=sys.stderr);return 3
    print('V8 processes stopped. Existing checkpoints and queue files preserved.',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
