"""Old child Context path: real strict verifier; injected CPU OOM; no GPU/model training."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO=Path(__file__).resolve().parents[1]


def entry(tmp_path,body):
    child=tmp_path/'cluster_child.py';result=tmp_path/'result.json'
    child.write_text('''import sys,json,types
from pathlib import Path
sys.path.insert(0,RUNNER)
from switching.runner import Context
def main():
'''.replace('RUNNER',repr(str(REPO/'v8/experiments')))+body)
    env=dict(os.environ,V8_CODE_ROOT=str(REPO),PYTHONPATH=str(REPO),SRGC_PROGRESS_DIR=str(tmp_path/'progress'),
             V8_ORIGINAL_CHILD_SHA256=hashlib.sha256(child.read_bytes()).hexdigest(),RANK='0',
             V8_FAILURE_TOKEN='test',V8_OOM_RETRY_SECONDS='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,str(REPO/'scripts/v8_task_entry.py'),str(child),str(result)],
                          env=env,text=True,capture_output=True,timeout=15)
    return result,tmp_path/'result.json'


def test_old_context_uses_existing_seed_fallback_for_strict_gold_failure(tmp_path):
    body=r''' mv=types.ModuleType('math_verify');mv.parse=lambda *a,**kw:[];mv.verify=lambda *a,**kw:True
 mv.ExprExtractionConfig=mv.LatexExtractionConfig=lambda:None;sys.modules['math_verify']=mv
 from srgc_rebuttal.run_experiment import math_reward
 record={'answer':r'$\boxed{\text{(C)}}$'}
 backend=types.SimpleNamespace(model=types.SimpleNamespace(generate=lambda **kw:None),verifier=math_reward,records={'q':record})
 ctx=Context(srgc=None,engine_class=None,backend=backend,data={},config=None,out_root=Path(sys.argv[1]).parent/'olmo_math')
 correct=backend.verifier(record,'Answer: (C)');wrong=backend.verifier(record,'Answer: (D)')
 Path(sys.argv[1]).write_text(json.dumps({'correct':correct,'wrong':wrong,'identity':ctx.verifier_identity}))
'''
    result,path=entry(tmp_path,body)
    assert result.returncode==0,result.stdout+result.stderr
    data=json.loads(path.read_text());assert (data['correct'],data['wrong'])==(1.,0.)
    assert data['identity']['tolerant_math_fallback'] is True


def test_old_scoring_oom_releases_exception_frame_and_retries_same_call_once(tmp_path):
    pytest.importorskip('torch')
    body=r''' import torch,weakref
 releases=[];torch.cuda.empty_cache=lambda:releases.append('released')
 class Holder:pass
 class Backend:
  device=types.SimpleNamespace(type='cuda');model=types.SimpleNamespace(generate=lambda **kw:None)
  def __init__(self):self.calls=[];self.failed=None
  def _rollout(self,prompt,responses,seed):
   self.calls.append([prompt,responses,seed])
   if len(self.calls)==1:
    tensor=Holder();self.failed=weakref.ref(tensor);raise torch.OutOfMemoryError('injected CUDA OOM')
   assert self.failed() is None,'OOM traceback still owns failed tensors'
   return [1,2],[1.,0.],1
  def score_gradients(self,ids,**kwargs):return self._rollout(ids[0],kwargs['responses'],kwargs['seed'])
 backend=Backend()
 ctx=Context(srgc=None,engine_class=None,backend=backend,data={},config=None,out_root=Path(sys.argv[1]).parent/'olmo_math')
 result=backend.score_gradients(['q'],responses=8,group_size=4,seed=17)
 assert backend.calls==[['q',8,17],['q',8,17]] and releases
 Path(sys.argv[1]).write_text(json.dumps({'calls':backend.calls,'result':result}))
'''
    result,path=entry(tmp_path,body)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(path.read_text())['calls']==[['q',8,17],['q',8,17]]


@pytest.mark.parametrize('kind',['oom','cuda-runtime'])
def test_only_one_oom_retry_and_no_retry_of_another_cuda_runtime_error(tmp_path,monkeypatch,kind):
    torch=pytest.importorskip('torch');monkeypatch.syspath_prepend(str(REPO/'scripts'))
    entry_module=importlib.import_module('v8_task_entry')
    import types
    calls=[]
    error=torch.OutOfMemoryError('CUDA OOM') if kind=='oom' else RuntimeError('RegisterCUDA SymIntArrayRef: original CUDA error')
    def rollout(*args,**kwargs):calls.append(args);raise error
    backend=types.SimpleNamespace(_rollout=rollout,device=types.SimpleNamespace(type='cpu'))
    ctx=types.SimpleNamespace(backend=backend)
    monkeypatch.setenv('V8_OOM_RETRY_SECONDS','0');monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path))
    entry_module.repair_context(ctx)
    with pytest.raises(type(error),match='CUDA'):backend._rollout('q',8,17)
    assert len(calls)==(2 if kind=='oom' else 1)


def test_empty_gold_and_custom_reward_are_not_silently_reinterpreted(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'));entry_module=importlib.import_module('v8_task_entry')
    from srgc_rebuttal.run_experiment import math_reward
    import types
    backend=types.SimpleNamespace(verifier=math_reward,records={'q':{'answer':''}})
    with pytest.raises(ValueError,match='empty gold'):entry_module.repair_context(types.SimpleNamespace(backend=backend))
    assert backend.verifier is math_reward
    custom=lambda record,response:.25
    backend=types.SimpleNamespace(verifier=custom,records={})
    entry_module.repair_context(types.SimpleNamespace(backend=backend))
    assert backend.verifier is custom


def test_existing_memory_guard_is_extended_to_train_without_repeating_updates(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'));entry_module=importlib.import_module('v8_task_entry')
    import types
    class Backend:
        _v8_protect=False
        def __init__(self):self.updates=0
        def _rollout(self,*args):return None
        def train(self):assert self._v8_protect;self.updates+=1;return 'same result'
    backend=Backend();monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path))
    entry_module.repair_context(types.SimpleNamespace(backend=backend))
    assert backend.train()=='same result' and backend.updates==1 and backend._v8_protect is False


def test_other_cuda_runtime_error_is_retained_with_context_and_not_silently_fixed(tmp_path):
    result,path=entry(tmp_path," raise RuntimeError('RegisterCUDA SymIntArrayRef: original CUDA diagnostic')\n")
    assert result.returncode!=0
    record=json.loads((tmp_path/'progress/cuda-failure-rank-0.json').read_text())
    assert record['type']=='RuntimeError' and 'original CUDA diagnostic' in record['message']
    assert 'RuntimeError: RegisterCUDA' in result.stderr
