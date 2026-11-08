"""Local runtime actors, inherited leases and old-binding preservation; never start GPU training."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import threading

import pytest

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    return importlib.import_module('v8_dispatch'),importlib.import_module('v8_rpc')


def test_large_messages_and_transferred_lease_stay_valid_after_sender_closes(tmp_path,modules):
    dispatch,rpc=modules;left,right=socket.socketpair()
    lease=dispatch.nodes.Lock(tmp_path/'lease.lock');assert lease.acquire()
    payload={'body':'x'*(1024*1024)};result=[]
    sender=threading.Thread(target=lambda:rpc.send(left,payload,[lease.fd]));sender.start()
    value,fds=rpc.receive(right);sender.join(timeout=3)
    try:
        assert value==payload and len(fds)==1 and not sender.is_alive()
        lease.close();probe=dispatch.nodes.Lock(tmp_path/'lease.lock')
        assert not probe.acquire()
        os.close(fds.pop());assert probe.acquire();probe.close()
    finally:
        lease.close()
        for fd in fds:os.close(fd)
        left.close();right.close()


FAKE=r'''
import fcntl,hashlib,json,os,sys
from pathlib import Path
TAG=__TAG__
class Runtime:
 def __init__(self,runner,root,cohort,python,gpus):
  self.root=Path(root)/cohort;self.cohort=cohort
  self.binding=dict(schema=1,cohort=cohort,root=str(self.root),revision=TAG,
   coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python_path=sys.executable)
 def verify_node(self):pass
 def barrier(self,action,fd,stop,device_fds):
  self.root.mkdir(parents=True,exist_ok=True)
  if action=='prepare':
   p=self.root/'prefix.pt'
   if not p.exists():p.write_bytes(b'prefix fixture')
 def plan(self,phase):
  if phase=='extension':return [],True
  task=dict(run_id='fixture',method_id='on_monitored',training_seed=5,cohort_id=self.cohort,
   role='CPU control fixture',horizon=75,config_hash='fixture',parent={'path':str(self.root/'prefix.pt')})
  if phase=='train':task['kind']='train'
  else:task.update(kind='evaluate',snapshot_path=str(self.root/'snapshots/step-000075.pt'),snapshot_sha256='a'*64,evaluation_spec_hash='b'*64)
  return [task],True
 def ready(self,task):return Path(task.get('snapshot_path') or task['parent']['path']).is_file()
 def native_busy(self,task):return False
 def done(self,task):return (self.root/('finished-'+task['kind'])).exists()
 def execute(self,task,matrix,fd,stop,device_fds,attempt):
  expected=Path(matrix).parent.parent/'leases'/(Path(matrix).stem+'.lock')
  assert os.fstat(fd).st_ino==expected.stat().st_ino,'task lease descriptor not transferred'
  global_lock=os.open(self.root.parent/'.v8-dispatch.lock',os.O_RDWR)
  fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.close(global_lock)
  if task['kind']=='train':
   folder=self.root/'snapshots';folder.mkdir(exist_ok=True)
   for update in (25,50,75):
    p=folder/f'step-{update:06d}.pt'
    if not p.exists():p.write_bytes(('simulated checkpoint '+str(update)).encode())
   (self.root/'latest.pt').write_bytes(b'simulated latest75')
  (self.root/('finished-'+task['kind'])).write_text('CPU control fixture, not a scientific result')
  return 0
'''


@pytest.mark.parametrize('partial',[False,True])
def test_existing_cohort_bindings_and_checkpoints_survive_new_dispatcher(tmp_path,modules,monkeypatch,partial):
    dispatch,rpc=modules;root=tmp_path/'experiments';entries=[];bindings={};checkpoints={}
    alias=tmp_path/'other-python';alias.symlink_to(sys.executable)
    for i,cohort in enumerate(('olmo_math','qwen_mbpp')):
        if partial and i==0:continue
        source=tmp_path/(cohort+'.py');source.write_text(FAKE.replace('__TAG__',repr('old-'+cohort)))
        entries.append(dict(cohort=cohort,script=str(source),runner=str(REPO/'v8/experiments'),python=sys.executable if i==0 else str(alias)))
    plan=dict(root=str(root),gpus=['0','1','2','3'],worker='fixture.1',entries=entries,argv=['--gpus','0,1,2,3'],
              errors={'olmo_math':'simulated invalid input'} if partial else {})
    planfile=tmp_path/'plan.json';planfile.write_text(json.dumps(plan))
    for entry in entries:
        actor=dispatch.RuntimeProxy(entry,plan,planfile)
        try:
            assert actor.error is None
            pool=dispatch.nodes.Pool(root/entry['cohort']/'.multi-node',actor.binding);pool.initialize()
            bindings[pool.root/'binding.json']=(pool.root/'binding.json').read_bytes()
            checkpoint=root/entry['cohort']/'snapshots/step-000025.pt';checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'existing optimizer RNG rewards checkpoint25');checkpoints[checkpoint]=checkpoint.read_bytes()
        finally:actor.close()
    monkeypatch.setenv('V8_ROOT',str(root));monkeypatch.setenv('V8_ALLOW_ANY_ROOT','1')
    monkeypatch.setattr(sys,'argv',['v8_dispatch.py',str(planfile)])
    inventory=[dict(index=str(i),uuid='CPU-fixture-'+str(i),total_mib=80000,used_mib=0,name='CPU fixture') for i in range(4)]
    monkeypatch.setattr(dispatch.nodes,'gpu_inventory',lambda _:inventory)
    monkeypatch.setattr(dispatch.nodes,'gpu_lock_root',lambda:tmp_path/'GPU-leases')
    monkeypatch.setattr(dispatch.nodes,'GpuGate',lambda _:lambda:(True,None))
    assert dispatch.main()==(1 if partial else 0)
    assert all(path.read_bytes()==data for path,data in bindings.items())
    assert all(path.read_bytes()==data for path,data in checkpoints.items())
    for entry in entries:
        queue=root/entry['cohort']/'.multi-node'
        assert dispatch.nodes.read(queue/'state.json')['phase']=='complete'
        assert len(list((root/entry['cohort']/'snapshots').glob('*.pt')))==3
    receipts=list((root/'.v8-dispatch-policies').glob('*.json'))
    assert len(receipts)==len(entries)
    assert all(json.loads(p.read_text())['numerical_runtime_changed'] is False for p in receipts)


@pytest.mark.parametrize('cohort',['olmo_math','qwen_mbpp'])
def test_actor_uses_real_frozen_runtime_identity_without_loading_models(tmp_path,modules,cohort):
    dispatch,rpc=modules
    from scripts.v8_queue_runtime import selections
    plan=selections(REPO,dict(V8_ROOT=str(tmp_path/'experiments'),V8_RUNNER_ROOT=str(REPO/'v8/experiments'),
                             V8_PYTHON=sys.executable),[cohort])[0]
    entry=dict(cohort=cohort,script=str(plan['script']),runner=str(plan['runner']),python=sys.executable)
    conf=dict(root=str(tmp_path/'experiments'),gpus=['0','1','2','3'],worker='fixture.1',entries=[entry])
    file=tmp_path/'plan.json';file.write_text(json.dumps(conf))
    previous=os.environ.get('V8_CODE_ROOT');os.environ['V8_CODE_ROOT']=str(REPO)
    actor=dispatch.RuntimeProxy(entry,conf,file)
    try:
        assert actor.error is None
        assert actor.binding['coordinator_sha256']==hashlib.sha256(Path(entry['script']).read_bytes()).hexdigest()
        actor.verify_node()
        assert actor.done(dict(kind='train',run_id='absent')) is False
    finally:
        actor.close()
        if previous is None:os.environ.pop('V8_CODE_ROOT',None)
        else:os.environ['V8_CODE_ROOT']=previous
