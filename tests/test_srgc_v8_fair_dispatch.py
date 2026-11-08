"""Simultaneous CPU worker arrivals; held file leases stand in for long GPU tasks."""
import collections
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time

from scripts import srgc_v8_nodes as nodes


class Runtime:
    def __init__(self,root,cohort,stop,blocked=False):
        self.root=Path(root);self.cohort=cohort;self.stop=stop;self.blocked=blocked
        self.binding=dict(schema=1,cohort=cohort,code='fair-test')
    def verify_node(self):pass
    def barrier(self,action,*_):
        if action=='prepare':
            if self.cohort=='olmo_math':time.sleep(.15)
            nodes.atomic(self.root/(self.cohort+'-prepared'),{'prepared':True})
    def plan(self,phase):
        assert phase=='train'
        return [dict(kind='train',run_id=self.cohort+'-'+str(i),method_id='on_monitored',training_seed=5,
                     config_hash=nodes.digest([self.cohort,i]),parent={},horizon=275) for i in range(20)],True
    def ready(self,task):return not self.blocked
    def done(self,task):return False
    def native_busy(self,task):return False
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        if not (self.root/'olmo_math-prepared').exists():nodes.atomic(self.root/'early-gpu-claim',{'bad':True})
        nodes.atomic(self.root/'claims'/attempt['worker'],dict(cohort=self.cohort,run_id=task['run_id']))
        while not self.stop.wait(.01):pass
        return 75


def node(root,name,stop,blocked=False):
    pairs=[]
    for cohort in ('olmo_math','qwen_mbpp'):
        runtime=Runtime(root,cohort,stop,blocked=blocked and cohort=='olmo_math')
        pairs.append((nodes.Pool(Path(root)/cohort/'.multi-node',runtime.binding),runtime))
    run=nodes.worker_loop_many
    if os.environ.get('V8_FAIR_BASELINE'):
        namespace=dict(vars(nodes));exec(Path(os.environ['V8_FAIR_BASELINE']).read_text(),namespace)
        run=namespace['worker_loop_many']
    code=run(pairs,name,stop,poll=.003)
    nodes.atomic(Path(root)/'exits'/name,{'code':code})


def biased_names(count):
    return [f'node-{i}' for i in range(1000) if int(nodes.digest(f'node-{i}')[:8],16)%2==1][:count]


def wait_claims(root,count):
    deadline=time.monotonic()+15
    while len(list((root/'claims').glob('*')))<count and time.monotonic()<deadline:time.sleep(.01)
    rows=[nodes.read(p) for p in (root/'claims').glob('*')]
    assert len(rows)==count
    assert len({(r['cohort'],r['run_id']) for r in rows})==count
    return collections.Counter(r['cohort'] for r in rows)


def cleanup(processes,stop):
    stop.set()
    for process in processes:
        process.join(5)
        if process.is_alive():process.kill();process.join();raise AssertionError('CPU worker failed to stop')
        assert process.exitcode==0


def test_ten_simultaneous_qwen_biased_nodes_and_four_late_nodes_are_balanced(tmp_path):
    ctx=mp.get_context('fork');stop=ctx.Event();processes=[];names=biased_names(14)
    keep=tmp_path/'keep/step-000025.pt';keep.parent.mkdir();keep.write_bytes(b'optimizer RNG reward checkpoint sentinel')
    before=hashlib.sha256(keep.read_bytes()).hexdigest()
    try:
        for name in names[:10]:
            process=ctx.Process(target=node,args=(tmp_path,name,stop));process.start();processes.append(process)
        assert wait_claims(tmp_path,10)=={'olmo_math':5,'qwen_mbpp':5}
        assert not (tmp_path/'early-gpu-claim').exists()
        for name in names[10:]:
            process=ctx.Process(target=node,args=(tmp_path,name,stop));process.start();processes.append(process)
        assert wait_claims(tmp_path,14)=={'olmo_math':7,'qwen_mbpp':7}
    finally:cleanup(processes,stop)
    assert hashlib.sha256(keep.read_bytes()).hexdigest()==before


def test_new_workers_prefer_olmo_when_ten_legacy_qwen_jobs_hold_leases(tmp_path):
    ctx=mp.get_context('fork');stop=ctx.Event();processes=[];leases=[]
    for cohort in ('olmo_math','qwen_mbpp'):
        runtime=Runtime(tmp_path,cohort,stop);pool=nodes.Pool(tmp_path/cohort/'.multi-node',runtime.binding);pool.initialize()
        nodes.atomic(pool.root/'state.json',dict(phase='train',failure=None,actions=[],failure_history=[]))
        nodes.atomic(tmp_path/(cohort+'-prepared'),{'prepared':True})
    for i in range(10):
        lease=nodes.Lock(tmp_path/'qwen_mbpp/.multi-node/leases'/('legacy-'+str(i)+'.lock'))
        assert lease.acquire();leases.append(lease)
    try:
        for name in biased_names(4):
            process=ctx.Process(target=node,args=(tmp_path,name,stop));process.start();processes.append(process)
        assert wait_claims(tmp_path,4)=={'olmo_math':4}
    finally:
        cleanup(processes,stop)
        for lease in leases:lease.close()


def test_no_ready_olmo_work_allows_qwen_to_borrow_free_workers(tmp_path):
    ctx=mp.get_context('fork');stop=ctx.Event();processes=[]
    try:
        for name in biased_names(4):
            process=ctx.Process(target=node,args=(tmp_path,name,stop,True));process.start();processes.append(process)
        assert wait_claims(tmp_path,4)=={'qwen_mbpp':4}
    finally:cleanup(processes,stop)
