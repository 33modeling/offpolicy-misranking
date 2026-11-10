"""Private socketpair actor for one exact historical Runtime/Python; no listener or model changes."""
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sys
import threading
import uuid
import types
from v8_rpc import send,receive


def child_exit(code,exit_file):
    """Same rule as srgc_v8_nodes.child_exit (kept here: this actor also runs from a minimal frozen copy).
    torchrun reports a worker's exit 75 (stopped at a safe point / busy elsewhere) as exit 1; rank 0 records it."""
    if code in (0,75):return code
    try:record=json.loads(Path(exit_file).read_text())
    except (OSError,ValueError):return code
    if isinstance(record,dict) and record.get('exit_code')==75 and record.get('status') in ('busy','interrupted'):
        print(f"[v8] child recorded exit 75 ({record.get('status')}); torchrun exit {code} is an interruption, "
              'not a task failure',flush=True)
        return 75
    return code


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


def child_result(rc,directory,environment,log,label,*,log_offset=0):
    """A real rank fault wins over an interruption marker; never turn errors into success."""
    if not rc:return rc
    failures=[]
    for path in sorted(directory.glob('rank-*-error.json')):
        try:
            row=json.loads(path.read_text())
            if row.get('invocation')!=environment['V8_FAILURE_TOKEN']:continue
            error=row['error'];message=error.get('message',{})
            if isinstance(message,str):message=dict(message=message)
            failures.append(dict(rank=row['rank'],message=message.get('message','Unknown rank error'),
                traceback=message.get('extraInfo',{}).get('py_callstack'),error_file=str(path),original_error=error))
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
            print(f'Cannot read rank error {path}: {exc}; original process exit={rc}',flush=True)
    # Keep fatal signals even beside another rank's Python exception. TERM/INT may be cleanup.
    numbers=set()
    try:
        with Path(log).open('rb') as handle:
            handle.seek(max(log_offset,Path(log).stat().st_size-131072))
            tail=handle.read(131072).decode(errors='replace')
        numbers={int(n) for n in re.findall(r'exitcode\s*:\s*-(\d+)',tail)}-{2,15}
    except OSError as exc:print(f'Cannot inspect torchrun failure output: {exc}',flush=True)
    if type(rc)is int and rc<0 and -rc not in (2,15):numbers.add(-rc)
    retained=set()
    for meta_path in sorted(directory.glob('native-fault-rank-*.json')):
        try:
            meta=json.loads(meta_path.read_text())
            if meta.get('invocation')!=environment['V8_FAILURE_TOKEN']:continue
            path=(directory/meta['log']).resolve()
            if path.parent!=directory.resolve():continue
            with path.open('rb') as handle:
                stack=handle.read(65536).decode(errors='replace')
                if path.stat().st_size>65536:
                    handle.seek(max(65536,path.stat().st_size-65536))
                    stack+='\n[bounded native trace: middle omitted]\n'+handle.read(65536).decode(errors='replace')
            fatal={'Segmentation fault':11,'Aborted':6,'Bus error':7,'Illegal instruction':4,'Floating point exception':8}
            found=next((n for text,n in fatal.items() if 'Fatal Python error: '+text in stack and n in numbers),None)
            if found is None:continue
            stage=None;progress=directory/f'rank-{meta["rank"]}.json'
            if progress.exists():
                event=json.loads(progress.read_text())
                if isinstance(event,dict) and event.get('pid')==meta['pid']:stage=event
            name=signal.Signals(found).name
            failures.append(dict(rank=meta['rank'],pid=meta['pid'],signal=found,
                message=f'{name} (exit -{found}): native crash; last stage={stage.get("stage") if stage else "unrecorded"}; '
                        f'rank stack={path}; underlying cause unconfirmed',error_file=str(path),
                native_fault_log=str(path),native_invocation=meta['invocation'],native_python_stack=stack[:12000],last_progress=stage))
            retained.add(found)
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:print(f'Cannot read native fault evidence {meta_path}: {exc}',flush=True)
    for number in sorted(numbers-retained):
        try:name=signal.Signals(number).name
        except ValueError:name='signal '+str(number)
        failures.append(dict(rank=None,message=f'Torchrun/launcher reported {name} (exit -{number}); '
            'no matching rank fault dump found; underlying cause unconfirmed',signal=number,error_file=str(log)))
    if failures:
        failure_rc=1 if rc==75 else rc
        report=dict(schema=1,attempt_id=directory.name,exit_code=failure_rc,original_exit_code=rc,task=label,failures=failures)
        try:
            target=directory/'child-failure.json';temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(report)+'\n');os.replace(temporary,target)
        except OSError as exc:print(f'Cannot retain child failure summary: {exc}; log={log}',flush=True)
        for failure in failures:
            text=f'RANK FAILURE rank={failure["rank"]}: {failure["message"]}; traceback={failure["error_file"]}'
            if failure.get('native_python_stack'):
                text+='\nNATIVE STACK (original fault-handler dump, bounded):\n'+failure['native_python_stack'][:6000]
            print(text,flush=True)
            try:
                with Path(log).open('a') as handle:handle.write(text+'\n')
            except OSError as exc:print(f'Cannot append rank failure to {log}: {exc}',flush=True)
        return failure_rc
    marker=Path(environment['V8_EXIT_FILE'])
    try:
        row=json.loads(marker.read_text())
        if (type(row.get('exit_code')) is int and row['exit_code']==75
                and row.get('status') in ('busy','interrupted')
                and (not label or not row.get('task') or row['task']==label)):
            print(f'TASK {row["status"].upper()}: {label}; original child exit75, '
                  f'torchrun exit={rc}; checkpoints retained',flush=True)
            return 75
    except FileNotFoundError:pass
    except (OSError,ValueError,TypeError,AttributeError) as exc:
        print(f'Cannot read task exit marker {marker}: {exc}; retaining process exit={rc}',flush=True)
    return rc


def run_srgc_child(runtime,command,log,fds,stop,*,extra=None,label=None,interval=5):
    """Use the existing SRGC lifecycle/progress watchdog; supervisor already spools its console."""
    code=activate_trainer(runtime.environment)
    from srgc_rebuttal.cluster import run_child
    from srgc_rebuttal.progress import signature
    environment=dict(runtime.environment,CUDA_VISIBLE_DEVICES=','.join(runtime.gpus),**(extra or {}))
    environment['PYTHONFAULTHANDLER']='1'
    if code:
        paths=[str(code),str(code/'src'),*environment.get('PYTHONPATH','').split(os.pathsep)]
        environment['PYTHONPATH']=os.pathsep.join(dict.fromkeys(path for path in paths if path))
    directory=Path(environment['SRGC_PROGRESS_DIR']);directory.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex
    environment.update(V8_FAILURE_TOKEN=token,V8_EXIT_FILE=str(directory/('exit-'+token+'.json')))
    stall=float(environment.get('V8_TASK_STALL_SECONDS','1800'))
    if not math.isfinite(stall) or stall<=0:raise ValueError('V8_TASK_STALL_SECONDS must be finite and positive')
    print(f'RUN {runtime.cohort}:{label or "task"}; log={log}; '
          f'SRGC progress watchdog={stall:g}s; no completed optimizer update is implied',flush=True)
    try:
        log_offset=Path(log).stat().st_size if Path(log).exists() else 0
        rc=run_child(command,Path(log),environment,pass_fds=tuple(fds),
                         heartbeat=lambda pid:runtime.heartbeat('child:'+str(pid)),should_stop=stop.is_set,
                         interval=interval,progress=lambda:signature(directory),stall_seconds=stall)
        return child_result(rc,directory,environment,log,label,log_offset=log_offset)
    except TimeoutError as exc:
        with Path(log).open('a') as handle:handle.write(f'TIMEOUT: {exc}\n')
        print(f'TIMEOUT {runtime.cohort}:{label or "task"}: {exc}; checkpoints retained',flush=True)
        return 124
    except KeyboardInterrupt:
        stop.set();return 75


def control_overrides(runtime,source):
    """Run the queue's original numerical Runtime with the current, bounded control code.

    Process handling is control, not science: archived coordinators c44f6d9/8cebbaa/375cd4e/cbd68a4 read
    child stdout through a PIPE and join it without a limit (V8-019), so a detached rank could keep the
    actor -- and the task lease -- blocked forever. The current Runtime.process writes child output to the
    log file and never joins a pipe; command, environment and file descriptors are unchanged.
    c44f6d9 plans evaluation before the V8-016 fix (matched-horizon run instead of the budget endpoint's
    actual run); only that evaluation planning is replaced by the current planner."""
    import srgc_v8_nodes as control  # the frozen control copy next to this file
    overrides=[];corrected=[]
    if hasattr(runtime,'process'):
        runtime.process=types.MethodType(control.Runtime.process,runtime);overrides.append('process:current-bounded')
    if "row['budget_endpoint_update']" in source and 'selected_endpoints' not in source:
        evaluation_original_plan=runtime.plan
        def plan(phase):
            if phase=='evaluation':return control.Runtime.plan(runtime,phase)
            return evaluation_original_plan(phase)
        runtime.plan=plan;overrides.append('plan.evaluation:current-v8-016');corrected.append('evaluation')
    if all(hasattr(runtime,name) for name in ('cfg','mx','cluster','cpu','locator')):
        runtime.primary_results_first=True;runtime.streaming_ready_tasks=True
        runtime.side_tasks=types.MethodType(control.ready_side_tasks,runtime)
        primary_original_plan=runtime.plan;primary_original_barrier=runtime.barrier
        def primary_plan(phase):
            if phase=='primary_evaluation':return control.primary_evaluation_tasks(runtime)
            return primary_original_plan(phase)
        def primary_barrier(action,*args):
            if action=='primary_finalize':return control.Runtime.barrier(runtime,action,*args)
            root=getattr(runtime,'root',None)
            if (action=='freeze' and root is not None and (Path(root)/'.multi-node/budget-rule.json').exists()
                    and not (Path(root)/'frozen/budget.json').exists()):return control.freeze_precommitted_budget(runtime)
            return primary_original_barrier(action,*args)
        runtime.plan=primary_plan;runtime.barrier=primary_barrier
        overrides.extend(('plan.primary:matched-horizon-first','barrier.primary:collect-and-preserve',
                          'plan.side:ready-eval-extension-original-budget'))
    return overrides,corrected


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
        overrides,corrected=control_overrides(runtime,Path(entry['script']).read_text())
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
        send(channel,dict(ok=True,binding=runtime.binding,overrides=overrides,corrected_plans=corrected,
                          primary_results_first=getattr(runtime,'primary_results_first',False),
                          streaming_ready_tasks=getattr(runtime,'streaming_ready_tasks',False)))
    except Exception as exc:
        try:send(channel,dict(ok=False,type=type(exc).__name__,error=str(exc)))
        except OSError:pass  # the dispatcher already left (stop or startup timeout)
        return 2
    while True:
        # The dispatcher closing its end (normal exit, Ctrl-C, or with an unread reply such as a finished
        # CPU-stage result) is the end of this actor, never a traceback on the console.
        try:request,fds=receive(channel)
        except (ConnectionResetError,BrokenPipeError,EOFError):break
        if request is None:
            for fd in fds:os.close(fd)
            break
        try:
            method=request['method'];args=request.get('args',[])
            if method=='barrier':
                runtime.barrier(args[0],fds[0],stop,fds[1:]);result=None
            elif method=='fresh_admission':
                runtime.verify_node()
                if not hasattr(runtime,'admit'):raise ValueError('Original runtime has no GPU admission verifier')
                runtime.admitted=None
                result=runtime.admit(fds)
            elif method=='execute':
                task,matrix,attempt=args
                result=runtime.execute(task,Path(matrix),fds[0],stop,fds[1:],attempt)
            elif method in ('verify_node','plan','done','ready','native_busy','side_tasks'):
                result=getattr(runtime,method)(*args)
            else:raise ValueError('unsupported runtime actor method')
            reply=dict(ok=True,result=result,interrupted=stop.is_set())
        except Exception as exc:
            reply=dict(ok=False,type=type(exc).__name__,error=str(exc),interrupted=stop.is_set())
            if isinstance(exc,json.JSONDecodeError):
                from srgc_v8_nodes import metadata_error_path
                reply.update(filename=metadata_error_path(exc),line=exc.lineno,column=exc.colno)
        finally:
            for fd in fds:os.close(fd)
        try:send(channel,reply)
        except (ConnectionResetError,BrokenPipeError):break
    channel.close();return 0


if __name__=='__main__':raise SystemExit(main())
