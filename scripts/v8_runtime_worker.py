"""Private socketpair actor for one exact historical Runtime/Python; no listener or model changes."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
from v8_rpc import send,receive


def main():
    channel=socket.socket(fileno=int(sys.argv[1]));plan=json.loads(Path(sys.argv[2]).read_text())
    cohort=sys.argv[3];entry=next(x for x in plan['entries'] if x['cohort']==cohort)
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    runtime=None
    try:
        spec=importlib.util.spec_from_file_location('_original_v8_coordinator',entry['script'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        runtime=module.Runtime(entry['runner'],plan['root'],cohort,sys.executable,plan['gpus'])
        # Keep the original child/config/engine untouched. Add diagnostics through an
        # explicit control entry point whose hash is recorded with the dispatcher policy.
        if hasattr(runtime,'process'):
            original_process=runtime.process
            def process(command,log,fds,stop,**kwargs):
                command=list(command)
                if not kwargs.get('cpu'):
                    index=next((i for i,arg in enumerate(command) if Path(str(arg)).name=='cluster_child.py'),None)
                    if index is not None:
                        child=command[index]
                        command[index:index+1]=[str(Path(__file__).with_name('v8_task_entry.py')),child]
                        kwargs['extra']=dict(kwargs.get('extra') or {},
                            V8_ORIGINAL_CHILD_SHA256=runtime.binding['runner_sha256']['cluster_child.py'])
                return original_process(command,log,fds,stop,**kwargs)
            runtime.process=process
        def heartbeat(task=None):
            module.atomic(runtime.root/'.multi-node/workers'/(module.digest(plan['worker'])+'.json'),
                          dict(worker=plan['worker'],pid=os.getpid(),at=module.time.time(),task=task,
                               dispatch_policy=plan.get('dispatch_policy'),controller_sha256=plan.get('controller_sha256')))
        runtime.heartbeat=heartbeat
        send(channel,dict(ok=True,binding=runtime.binding))
    except Exception as exc:
        send(channel,dict(ok=False,type=type(exc).__name__,error=str(exc)));return 2
    while True:
        request,fds=receive(channel)
        if request is None:
            for fd in fds:os.close(fd)
            break
        try:
            method=request['method'];args=request.get('args',[])
            if method=='barrier':
                runtime.barrier(args[0],fds[0],stop,fds[1:]);result=None
            elif method=='execute':
                task,matrix,attempt=args
                result=runtime.execute(task,Path(matrix),fds[0],stop,fds[1:],attempt)
            elif method in ('verify_node','plan','done','ready','native_busy'):
                result=getattr(runtime,method)(*args)
            else:raise ValueError('unsupported runtime actor method')
            reply=dict(ok=True,result=result,interrupted=stop.is_set())
        except Exception as exc:reply=dict(ok=False,type=type(exc).__name__,error=str(exc),interrupted=stop.is_set())
        finally:
            for fd in fds:os.close(fd)
        send(channel,reply)
    channel.close();return 0


if __name__=='__main__':raise SystemExit(main())
