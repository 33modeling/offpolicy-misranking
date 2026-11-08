"""Versioned fair dispatcher over exact per-cohort numerical runtimes; uses only local socketpairs."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import srgc_v8_nodes as nodes
from v8_rpc import send,receive


class RuntimeProxy:
    def __init__(self,entry,plan,planfile):
        self.cohort=entry['cohort'];self.python=entry['python'];self.runner=Path(entry['runner'])
        self.root=Path(plan['root'])/self.cohort;self.error=None;self.process=None
        self.binding=dict(schema=1,cohort=self.cohort,root=str(self.root))
        self.channel,child=socket.socketpair()
        try:
            self.process=subprocess.Popen([self.python,str(Path(__file__).with_name('v8_runtime_worker.py')),
                str(child.fileno()),str(planfile),self.cohort],env=os.environ,pass_fds=(child.fileno(),),stdin=subprocess.DEVNULL)
            child.close();reply,fds=receive(self.channel)
            for fd in fds:os.close(fd)
            if reply is None:raise RuntimeError('runtime actor exited during initialization')
            if not reply.get('ok'):raise ValueError(reply.get('error','actor initialization failed'))
            self.binding=reply['binding']
        except Exception as exc:self.error=f'{type(exc).__name__}: {exc}'
        finally:child.close()

    def call(self,method,*args,fds=(),stop=None):
        if self.error:raise ValueError(self.error)
        send(self.channel,dict(method=method,args=args),fds)
        reply,received=receive(self.channel)
        for fd in received:os.close(fd)
        if reply is None:raise RuntimeError(f'{self.cohort}: runtime actor exited without reply')
        if reply.get('interrupted'):
            if stop is not None:stop.set()
            else:raise KeyboardInterrupt()
        if not reply.get('ok'):
            if reply.get('type')=='JSONDecodeError':raise json.JSONDecodeError(reply['error'],'',0)
            raise ValueError(f'{self.cohort}: {reply.get("type")}: {reply.get("error")}')
        return reply.get('result')

    def verify_node(self):return self.call('verify_node')
    def plan(self,phase):return self.call('plan',phase)
    def ready(self,task):return self.call('ready',task)
    def native_busy(self,task):return self.call('native_busy',task)
    def done(self,task):return self.call('done',task)
    def barrier(self,action,fd,stop,device_fds):return self.call('barrier',action,fds=(fd,*device_fds),stop=stop)
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        return self.call('execute',task,str(matrix),attempt,fds=(fd,*device_fds),stop=stop)

    def close(self):
        self.channel.close()
        if self.process is not None:
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:self.process.kill()


def main():
    planfile=Path(sys.argv[1]);plan=json.loads(planfile.read_text());actors=[]
    plan['worker']=socket.gethostname()+'.'+str(os.getpid())
    # The actor receives this private per-launch plan, with paths already frozen by bootstrap.
    runfile=planfile.with_name(planfile.stem+'-'+str(os.getpid())+'.json')
    runfile.write_text(json.dumps(plan))
    entries={entry['cohort']:entry for entry in plan['entries']}
    controls={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ('v8_dispatch.py','v8_runtime_worker.py','v8_rpc.py','srgc_v8_nodes.py')}
    def factory(runner,root,cohort,python,gpus):
        plan['root']=str(Path(root).resolve());plan['gpus']=gpus
        runfile.write_text(json.dumps(plan))
        actor=RuntimeProxy(entries[cohort],plan,runfile);actors.append(actor)
        if not actor.error:
            record=dict(schema=1,policy='global-live-task-leases-v1',controller_source=controls,
                        original_runtime_binding=actor.binding,numerical_runtime_changed=False)
            receipt=Path(root)/'.v8-dispatch-policies'/(cohort+'-'+nodes.digest(record)+'.json')
            if receipt.exists():
                if nodes.read(receipt)!=record:raise ValueError('dispatch policy receipt changed')
            else:nodes.atomic(receipt,record)
        return actor
    original=nodes.Runtime;nodes.Runtime=factory
    args=[];i=0
    while i<len(plan['argv']):
        token=plan['argv'][i]
        if token in ('--root','--runner-root','--python','--gpus','--poll-seconds'):
            if token!='--runner-root':args.extend(plan['argv'][i:i+2])
            i+=2;continue
        if token.startswith('-') and not token.startswith('--runner-root='):args.append(token)
        i+=1
    if len(entries)==1:args.insert(0,next(iter(entries)))
    for cohort,error in plan['errors'].items():print(f'V8 startup: {cohort} blocked: {error}',file=sys.stderr,flush=True)
    try:
        result=nodes.main(args)
        return result or (1 if plan['errors'] else 0)
    finally:
        for actor in actors:actor.close()
        nodes.Runtime=original


if __name__=='__main__':raise SystemExit(main())
