"""Best-effort console messages must never block process cleanup or control exit."""
import os
import sys
import threading
import time

_pending=[]


def emit(message,*,error=False):
    done=threading.Event()
    def write():
        try:print(message,file=sys.stderr if error else sys.stdout,flush=True)
        except (OSError,ValueError):pass
        finally:done.set()
    _pending.append(done)
    threading.Thread(target=write,daemon=True).start()


def exit_control(code):
    # Python's final stdout flush can itself hang on a stalled terminal or on a daemon
    # relay holding its buffer lock. All child cleanup happens before this bounded exit.
    deadline=time.monotonic()+.2
    for done in _pending:done.wait(max(0,deadline-time.monotonic()))
    os._exit(code)


def run_control(main):
    try:code=main()
    except KeyboardInterrupt:code=130
    except Exception as exc:
        emit(f'V8 control error: {type(exc).__name__}: {exc}',error=True);code=2
    exit_control(code)
