"""Private socketpair actor for one exact historical Runtime/Python; no listener or model changes."""
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import threading
from v8_rpc import send,receive


def activate_trainer(environment):
    """The frozen actor lives outside the checkout; use the already selected trainer root."""
    value=environment.get('V8_CODE_ROOT')
    if not value:return
    code=Path(value).resolve()
    if not (code/'srgc_rebuttal/srgc.py').is_file():
        raise ValueError(f'Original trainer package is missing at {code}/srgc_rebuttal; queue unchanged')
    for directory in (code/'src',code):
        path=str(directory)
        if path in sys.path:sys.path.remove(path)
        sys.path.insert(0,path)
    return code


def run_srgc_child(runtime,command,log,fds,stop,*,extra=None,label=None,interval=5):
    """Use the existing SRGC lifecycle/progress watchdog; supervisor already spools its console."""
    code=activate_trainer(runtime.environment)
    from srgc_rebuttal.cluster import run_child
    from srgc_rebuttal.progress import signature
    environment=dict(runtime.environment,CUDA_VISIBLE_DEVICES=','.join(runtime.gpus),**(extra or {}))
    if code:
        paths=[str(code),str(code/'src'),*environment.get('PYTHONPATH','').split(os.pathsep)]
        environment['PYTHONPATH']=os.pathsep.join(dict.fromkeys(path for path in paths if path))
    directory=Path(environment['SRGC_PROGRESS_DIR']);directory.mkdir(parents=True,exist_ok=True)
    stall=float(environment.get('V8_TASK_STALL_SECONDS','1800'))
    if not math.isfinite(stall) or stall<=0:raise ValueError('V8_TASK_STALL_SECONDS must be finite and positive')
    print(f'RUN {runtime.cohort}:{label or "task"}; log={log}; '
          f'SRGC progress watchdog={stall:g}s; no completed optimizer update is implied',flush=True)
    try:
        return run_child(command,Path(log),environment,pass_fds=tuple(fds),
                         heartbeat=lambda pid:runtime.heartbeat('child:'+str(pid)),should_stop=stop.is_set,
                         interval=interval,progress=lambda:signature(directory),stall_seconds=stall)
    except TimeoutError as exc:
        with Path(log).open('a') as handle:handle.write(f'TIMEOUT: {exc}\n')
        print(f'TIMEOUT {runtime.cohort}:{label or "task"}: {exc}; checkpoints retained',flush=True)
        return 124
    except KeyboardInterrupt:
        stop.set();return 75


def main():
    channel=socket.socket(fileno=int(sys.argv[1]));plan=json.loads(Path(sys.argv[2]).read_text())
    cohort=sys.argv[3];entry=next(x for x in plan['entries'] if x['cohort']==cohort)
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    runtime=None
    try:
        activate_trainer(os.environ)
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
                        kwargs.pop('cpu',None)
                        return run_srgc_child(runtime,command,log,fds,stop,**kwargs)
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
