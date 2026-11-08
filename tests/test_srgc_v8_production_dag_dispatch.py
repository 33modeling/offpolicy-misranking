"""Real five-seed v8 DAG + CPU lease holders; no GPU, model, optimizer or scientific results."""
import collections
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

from scripts import srgc_v8_nodes as nodes

REPO=Path(__file__).resolve().parents[1]


class ProductionPlan(nodes.Runtime):
    def __init__(self,root,cohort,stop):
        sys.path.insert(0,str(REPO/'v8/experiments'))
        from switching import matrix,spec
        self.root=Path(root)/cohort;self.cohort=cohort;self.stop=stop;self.mx=matrix
        self.cfg=spec.resolve(cohort,role='rebuttal_bank')
        self.binding=dict(schema=1,cohort=cohort,scope='production DAG CPU fixture')
    def verify_node(self):pass
    def barrier(self,*args):pass
    def locator(self):
        root=self.root
        class Locator:
            interval=25
            def prefix(self,seed):
                return dict(kind='prefix',path=str(root/f'prefix-{seed}.pt'),sha256=nodes.digest(seed),
                            completed_update=25,run_id=None)
        return Locator()
    def execute(self,task,matrix,fd,stop,device_fds,attempt):
        nodes.atomic(self.root.parent/'claims'/attempt['worker'],
                     dict(cohort=self.cohort,run_id=task['run_id'],method=task['method_id']))
        stop.wait(20)
        return 75


def worker(root,name,stop):
    pairs=[]
    for cohort in ('olmo_math','qwen_mbpp'):
        runtime=ProductionPlan(root,cohort,stop)
        pairs.append((nodes.Pool(runtime.root/'.multi-node',runtime.binding),runtime))
    nodes.worker_loop_many(pairs,name,stop,poll=.01)


def wait_claims(root,count):
    end=time.monotonic()+15
    while time.monotonic()<end:
        rows=[nodes.read(p) for p in (root/'claims').glob('[!.]*')]
        if len(rows)==count:return rows
        time.sleep(.01)
    raise AssertionError(f'only {len(rows)}/{count} production DAG tasks were claimed')


def test_twenty_initial_jobs_and_ten_new_checkpoint_branches_fill_thirty_nodes(tmp_path):
    ctx=mp.get_context('fork');stop=ctx.Event();processes=[]
    for cohort in ('olmo_math','qwen_mbpp'):
        folder=tmp_path/cohort;folder.mkdir()
        for seed in range(5,10):(folder/f'prefix-{seed}.pt').write_bytes(b'CPU parent fixture, not weights')
    try:
        for i in range(24):
            p=ctx.Process(target=worker,args=(tmp_path,'node-'+str(i),stop));p.start();processes.append(p)
        rows=wait_claims(tmp_path,20)
        assert collections.Counter((r['cohort'],r['method']) for r in rows)=={
            ('olmo_math','on_monitored'):5,('olmo_math','bank_sr_t025'):5,
            ('qwen_mbpp','on_monitored'):5,('qwen_mbpp','bank_sr_t025'):5}
        # Keep all parents training; only append newly available step50 snapshot metadata.
        for cohort in ('olmo_math','qwen_mbpp'):
            runtime=ProductionPlan(tmp_path,cohort,stop)
            tasks,_=runtime.mx.build_stage('bank-trajectory',runtime.cfg,runtime.locator(),runtime.root)
            for task in tasks:
                directory=runtime.root/'runs'/task['run_id'];directory.mkdir(parents=True)
                (directory/'step-50.pt').write_bytes(b'CPU step50 fixture, not weights')
                (directory/'snapshots.jsonl').write_text(nodes.canonical(dict(completed_update=50,path='step-50.pt',
                    sha256=nodes.digest(task['run_id']),identity=dict(state_hash='fixture')))+'\n')
        for i in range(24,30):
            p=ctx.Process(target=worker,args=(tmp_path,'node-'+str(i),stop));p.start();processes.append(p)
        rows=wait_claims(tmp_path,30)
        assert collections.Counter(r['cohort'] for r in rows)=={'olmo_math':15,'qwen_mbpp':15}
        assert sum(r['method']=='bank_sr_t050' for r in rows)==10
        assert len({r['run_id'] for r in rows})==30
    finally:
        stop.set()
        for p in processes:
            p.join(4)
            if p.is_alive():p.kill();p.join();raise AssertionError('CPU dispatch worker failed to stop')
            assert p.exitcode==0


def test_five_legacy_qwen_jobs_do_not_leave_all_new_nodes_idle_after_a_task_failure(tmp_path):
    ctx=mp.get_context('fork');stop=ctx.Event();processes=[];legacy=[];failed_runs=set()
    for cohort in ('olmo_math','qwen_mbpp'):
        folder=tmp_path/cohort;folder.mkdir()
        for seed in range(5,10):(folder/f'prefix-{seed}.pt').write_bytes(b'CPU dependency fixture')
        runtime=ProductionPlan(tmp_path,cohort,stop)
        pool=nodes.Pool(folder/'.multi-node',runtime.binding);pool.initialize()
        state=nodes.read(pool.root/'state.json');state['phase']='train';nodes.atomic(pool.root/'state.json',state)
        tasks,_=runtime.plan('train')
        failed=next(t for t in tasks if t['method_id']=='bank_sr_t025')
        for task in tasks:
            key=pool.register(task,'train')
            if cohort=='qwen_mbpp' and task['method_id']=='on_monitored':
                held=nodes.Lock(pool.root/'leases'/(key+'.lock'));assert held.acquire();legacy.append(held)
        key=nodes.task_key(failed);job=nodes.read(pool.root/'jobs'/(key+'.json'))
        job['state']='failed';job['attempts']=[dict(id='legacy-failure',exit_code=1)]
        nodes.atomic(pool.root/'jobs'/(key+'.json'),job);failed_runs.add(failed['run_id'])
        pool.fail('task exit=1; output verified=False',key)
    try:
        assert len(legacy)==5
        for i in range(12):
            p=ctx.Process(target=worker,args=(tmp_path,'new-node-'+str(i),stop));p.start();processes.append(p)
        rows=wait_claims(tmp_path,12)
        counts=collections.Counter(r['cohort'] for r in rows)
        assert counts['olmo_math']>=5 and counts['qwen_mbpp']>=1
        assert not (failed_runs & {r['run_id'] for r in rows})
        assert len({r['run_id'] for r in rows})==12
    finally:
        stop.set()
        for p in processes:
            p.join(4)
            if p.is_alive():p.kill();p.join();raise AssertionError('CPU worker failed to stop')
            assert p.exitcode==0
        for held in legacy:held.close()
