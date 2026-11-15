"""Node preflight before each task launch and the crash fields of each failed attempt; CPU fixtures only
(fake nvidia-smi rows, fake smoke results). No real GPU, model or checkpoint is touched."""
import json
import sys
import threading
import types

import srgc_v8_nodes as nodes
from test_srgc_v8_retry_debug import prepared

GPUS=['0','1','2','3']
GOOD_PACKAGES={'torch':'2.7.1+cu126','transformers':'4.57.1','peft':'0.17.1','numpy':'2.2.6','math-verify':'0.9.0'}


def rows(used=0):
    return [dict(index=g,uuid='GPU-'+g,used_mib=used,total_mib=81559,name='NVIDIA H100 80GB HBM3') for g in GPUS]


def smoke(report=None,rc=0,stderr=''):
    report=dict(dict(python_version='3.12.3 (main)',prefix='/venv',packages=dict(GOOD_PACKAGES),imports={'transformers':'ok'},
                     torch=dict(version='2.7.1+cu126',cuda='12.6',nccl='2.26.2'),
                     cuda=dict(available=True,device_count=4,devices=[dict(index=i,matmul_ok=True) for i in range(4)]),
                     libraries=['/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2']),**(report or {}))
    def run(command,**kwargs):
        assert command[1]=='-c' and kwargs['env']['CUDA_VISIBLE_DEVICES']=='0,1,2,3'
        return types.SimpleNamespace(returncode=rc,stdout='' if rc else 'V8PREFLIGHT '+json.dumps(report),stderr=stderr)
    return run


def check(cohort='olmo_math',expected=None,used=0,run=None,host=None,health=None,driver=('550.54.15','12.4')):
    return nodes.node_preflight(cohort,'/venv/bin/python',GPUS,expected if expected is not None else dict(python_version='3.12.3 (main)',packages=GOOD_PACKAGES),
        run=run or smoke(),inventory=lambda g:rows(used),processes=lambda:['4242, GPU-0, 30000'],
        resources=lambda:dict(dict(shm_free_mib=65536,mem_available_mib=900000,ulimit={'nofile':[1048576,1048576]}),**(host or {})),
        driver=lambda:driver,health=lambda g:health or {},sleep=lambda s:None)  # 'sleep' fake: the GPU wait costs no time


def test_healthy_node_passes_and_records_versions_and_host():
    result=check()
    assert result['ok'] and result['failures']==[] and result['policy']==nodes.PREFLIGHT_POLICY
    env=result['env']
    assert env['executable']=='/venv/bin/python' and env['torch']['version']=='2.7.1+cu126' and env['torch']['nccl']=='2.26.2'
    assert env['packages']['transformers']=='4.57.1' and env['driver_version']=='550.54.15' and env['gpus']==GPUS
    assert env['host_resources']['shm_free_mib']==65536 and 'PREFLIGHT ok' in nodes.preflight_line(result)


def test_queue_runtime_version_mismatch_fails():
    expected=dict(python_version='3.12.3 (main)',packages=dict(GOOD_PACKAGES,torch='2.6.0+cu124'))
    result=check(expected=expected)
    assert not result['ok'] and any('torch 2.7.1+cu126 differs from the queue runtime 2.6.0+cu124' in f for f in result['failures'])


def test_model_family_minimum_versions():
    old=smoke(dict(packages=dict(GOOD_PACKAGES,transformers='4.56.2')))
    assert any('OLMo-3 needs transformers>=4.57' in f for f in check(run=old,expected={})['failures'])
    qwen=check('qwen_math',run=smoke(dict(packages=dict(GOOD_PACKAGES,transformers='4.57.1'))),expected={})
    assert any('Qwen3.5 needs transformers>=5' in f for f in qwen['failures'])
    assert any('fla-core is not installed' in f for f in qwen['failures'])
    ok=check('qwen_math',run=smoke(dict(packages=dict(GOOD_PACKAGES,transformers='5.14.1',**{'fla-core':'0.5.2'}),
                                         imports={'transformers':'ok','fla':'ok'})),expected={})
    assert ok['ok'],ok['failures']


def test_smoke_segfault_in_cuda_init_is_a_node_failure_with_frames():
    stderr=('Fatal Python error: Segmentation fault\n\nCurrent thread 0x7f (most recent call first):\n'
            '  File "/venv/lib/python3.12/site-packages/torch/cuda/__init__.py", line 319 in _lazy_init\n'
            '  File "<string>", line 30 in <module>\n')
    result=check(run=smoke(rc=-11,stderr=stderr))
    [failure]=[f for f in result['failures'] if 'crashed' in f]
    assert 'SIGSEGV' in failure and '_lazy_init' in failure and result['env']['smoke']['rc']==-11


def test_occupied_gpu_lists_the_process_and_fails():
    result=check(used=30000)
    assert not result['ok'] and any('GPU 0 30000 MiB' in f and 'pid 4242' in f for f in result['failures'])
    assert result['env']['gpu_processes'][0]['pid']==4242


def test_small_shm_low_memory_ecc_and_old_driver_fail():
    assert any('/dev/shm has 64 MiB free' in f for f in check(host=dict(shm_free_mib=64))['failures'])
    assert any('MemAvailable 1000 MiB' in f for f in check(host=dict(mem_available_mib=1000))['failures'])
    assert any('ecc.errors.uncorrected.volatile.total=3' in f
               for f in check(health={'2':{'ecc.errors.uncorrected.volatile.total':'3'}})['failures'])
    old=check(run=smoke(dict(torch=dict(version='2.9.0+cu130',cuda='13.0'))),expected={},driver=('535.1','12.2'))
    assert any('built for CUDA 13.0' in f for f in old['failures'])
    newer=check(run=smoke(dict(torch=dict(version='2.7.1+cu128',cuda='12.8'))),expected={},driver=('535.1','12.2'))
    assert newer['ok'] and any('newer than the driver CUDA 12.2' in w for w in newer['warnings'])


def test_cuda_device_count_mismatch_fails():
    result=check(run=smoke(dict(cuda=dict(available=True,device_count=2,devices=[]))))
    assert any('CUDA sees 2 device(s)' in f for f in result['failures'])


def test_real_smoke_script_runs_in_this_python():
    """The smoke program itself (no torch here): versions are read and the missing torch is a failure."""
    result=nodes.node_preflight('olmo_math',sys.executable,GPUS,{},inventory=lambda g:None,processes=lambda:[],
                                driver=lambda:(None,None),health=lambda g:{},sleep=lambda s:None)
    assert result['env']['smoke']['rc']==0 and result['env']['executable']==sys.executable
    assert result['env']['python_version']==sys.version and 'torch' in result['env']['packages']
    try:import torch  # noqa: F401
    except ImportError:assert any('No module named' in f for f in result['failures'])


def test_preflight_never_takes_a_node_out_of_service(tmp_path,monkeypatch):
    """User 11-16 'idle이 너무 잡히는데', '실패하면 다시 시도안하고 그냥 idle 되는거 같음'.
    Default (record): a failed check is written into the attempt and the task still runs on this node.
    V8_PREFLIGHT=block: the launch is refused without counting a task failure, the node pauses and claims
    again itself (it is never given up for the cohort), and runs the task once the check passes."""
    def setup(sub):
        (tmp_path/sub).mkdir();pool,runtime,task,key,saved=prepared(tmp_path/sub);calls=[];finished=[]
        runtime.python='/venv/bin/python';runtime.gpus=GPUS
        runtime.done=lambda _:bool(finished)
        def execute(task,matrix,fd,stop,device_fds,attempt):
            calls.append(pool.host);finished.append(True);stop.set();return 0
        runtime.execute=execute
        return pool,runtime,key,calls
    def fake(results):
        def preflight(cohort,python,gpus,expected):
            ok=results.pop(0) if results else True
            return dict(policy=nodes.PREFLIGHT_POLICY,ok=ok,failures=[] if ok else ['/dev/shm has 64 MiB free (< 1024)'],
                        warnings=[],seconds=0.1,env=dict(host='node-a',gpus=gpus,executable=python,
                        torch=dict(version='2.7.1+cu126'),packages=dict(GOOD_PACKAGES)))
        return preflight
    # record (default): runs anyway, the failed check is kept in the attempt
    monkeypatch.setattr(nodes,'PREFLIGHT_MODE','record');monkeypatch.setattr(nodes,'PREFLIGHT_CHECK',fake([False]))
    pool,runtime,key,calls=setup('record');pool.host='node-a'
    nodes.worker_loop(pool,runtime,'node-a.1',threading.Event(),poll=.001,retry_delay=0,preflight=True)
    job=nodes.read(pool.root/'jobs'/(key+'.json'))
    assert calls==['node-a'] and job['state']=='done' and job['attempts'][0]['preflight']['ok'] is False
    # block: refused once, the same node retries after the pause and finishes the task
    monkeypatch.setattr(nodes,'PREFLIGHT_MODE','block');monkeypatch.setattr(nodes,'PREFLIGHT_RETRY_SECONDS',0)
    monkeypatch.setattr(nodes,'PREFLIGHT_CHECK',fake([False]))
    pool,runtime,key,calls=setup('block');pool.host='node-a'
    nodes.worker_loop(pool,runtime,'node-a.2',threading.Event(),poll=.001,retry_delay=0,preflight=True)
    job=nodes.read(pool.root/'jobs'/(key+'.json'))
    first=job['attempts'][0]
    assert first['outcome']=='node_failure' and 'PreflightFailed' in first['error'] and nodes.counted_failures(job)==0
    assert calls==['node-a'] and job['state']=='done' and job['attempts'][-1]['preflight']['ok']

def test_crash_records_signal_stage_frames_env_host_gpus(tmp_path,monkeypatch,capsys):
    pool,runtime,task,key,saved=prepared(tmp_path)
    runtime.python='/venv/bin/python';runtime.gpus=GPUS;runtime.done=lambda _:False
    stack=('Fatal Python error: Segmentation fault\n\nThread 0x02 (most recent call first):\n  File "/x/threading.py", line 1 in wait\n\n'
           'Current thread 0x01 (most recent call first):\n'
           '  File "/venv/site-packages/transformers/generation/utils.py", line 2500 in _sample\n'
           '  File "/code/srgc_rebuttal/torch_backend.py", line 310 in _rollout\n')
    def execute(task,matrix,fd,stop,device_fds,attempt):
        progress=pool.root/'progress'/attempt['id']
        nodes.atomic(progress/'rank-2.json',dict(stage='generation',updated=1e10))
        nodes.atomic(progress/'child-failure.json',dict(attempt_id=attempt['id'],exit_code=1,failures=[dict(
            rank=2,pid=9,signal=11,message='SIGSEGV',native_python_stack=stack,native_fault_log=str(progress/'n.log'),
            last_progress=dict(stage='generation'))]))
        (pool.root/'logs').mkdir(parents=True,exist_ok=True)
        (pool.root/'logs'/(attempt['id']+'.log')).write_text('[rank2]: exitcode  : -11\nChildFailedError\n')
        stop.set();return 1
    runtime.execute=execute
    monkeypatch.setattr(nodes,'PREFLIGHT_CHECK',lambda c,p,g,e:dict(ok=True,failures=[],warnings=[],seconds=0,
        env=dict(host='h100-a03',gpus=g,executable=p,torch=dict(version='2.7.1+cu126',cuda='12.6'),packages=dict(GOOD_PACKAGES))))
    pool.host='h100-a03'
    nodes.worker_loop(pool,runtime,'h100-a03.5',threading.Event(),poll=.001,retry_delay=0,preflight=True)
    attempt=nodes.read(pool.root/'jobs'/(key+'.json'))['attempts'][0]
    assert attempt['exit_code']==1 and attempt['signal']==11 and attempt['signal_name']=='SIGSEGV'
    assert attempt['host']=='h100-a03' and attempt['gpus']==GPUS and attempt['stage']=='generation'
    assert attempt['crashed_rank']==2 and attempt['crash_class']=='native_crash_unclassified'
    assert attempt['crash_frames'][0].startswith('_sample (') and 'torch_backend.py:310' in attempt['crash_frames'][1]
    assert attempt['env']['executable']=='/venv/bin/python' and attempt['env']['packages']['torch']=='2.7.1+cu126'
    out=capsys.readouterr().out
    assert 'CRASH exit=1 SIGSEGV host=h100-a03 gpus=0,1,2,3 stage=generation' in out and '_sample' in out


def test_sigabrt_after_nccl_watchdog_is_classified(tmp_path):
    root=tmp_path/'q';(root/'logs').mkdir(parents=True)
    (root/'logs'/'a1.log').write_text('[rank0]:[E] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=9)\n'
                                       "terminate called after throwing an instance of 'c10::DistBackendError'\nexitcode  : -6\n")
    fields=nodes.crash_fields(root,dict(id='a1',exit_code=-6),-6)
    assert fields['signal']==6 and fields['crash_class']=='nccl_watchdog_abort' and fields['stage'] is None
    assert any('Watchdog' in line for line in fields['crash_log_lines'])


def test_frames_fall_back_to_the_task_log_and_stack_file(tmp_path):
    root=tmp_path/'q';(root/'logs').mkdir(parents=True);(root/'stacks').mkdir()
    (root/'stacks'/'sr25-rank1.txt').write_text('== 10:00 pid=1\nFatal Python error: Aborted\n\nCurrent thread 0x1 (most recent call first):\n'
                                                '  File "/venv/torch/distributed/distributed_c10d.py", line 2501 in all_reduce\n')
    fields=nodes.crash_fields(root,dict(id='a2',exit_code=-6,started_at=0),-6,dict(run_id='sr25'))
    assert fields['crash_frames']==['all_reduce (/venv/torch/distributed/distributed_c10d.py:2501)']
    assert str(root/'stacks'/'sr25-rank1.txt') in fields['crash_evidence']
