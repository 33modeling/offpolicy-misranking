"""Generation diagnostics do not sample, shorten responses, or count heartbeats as work."""
import importlib
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

import pytest

from scripts.v8_generation_probe import DecodeObserver,observed_generate,install

REPO=Path(__file__).resolve().parents[1]


def fake_ids(length):return types.SimpleNamespace(shape=(2,length))


def test_observer_records_only_increasing_decode_steps_and_never_stops_or_uses_rng():
    rows=[];clock=[0.];rng=random.getstate()
    observer=DecodeObserver(lambda stage,**d:rows.append((stage,d)),0,7,3,interval=0,clock=lambda:clock[0])
    assert observer(fake_ids(4),None) is False
    clock[0]=1.;observer(fake_ids(4),None)
    assert len(rows)==1 and observer.last_advance==0
    observer(fake_ids(5),None);clock[0]=2.;observer(fake_ids(4),None)
    assert rows[-1][1]['decode_steps']==3 and rows[-1][1]['segment']==1
    assert random.getstate()==rng


def test_wrapper_keeps_existing_criteria_sampling_args_and_result(monkeypatch):
    fake=types.ModuleType('transformers');fake.StoppingCriteriaList=list
    monkeypatch.setitem(sys.modules,'transformers',fake)
    criterion=object();rows=[];result=object()
    kwargs=dict(input_ids=fake_ids(3),num_return_sequences=8,do_sample=True,max_new_tokens=2048,
                temperature=1.,top_p=1.,top_k=0,stopping_criteria=[criterion])
    def original(**actual):
        for key in kwargs:
            if key!='stopping_criteria':assert actual[key]==kwargs[key]
        assert actual['stopping_criteria'][0] is criterion
        assert actual['stopping_criteria'][1](fake_ids(4),None) is False
        return result
    wrapped=observed_generate(original,lambda stage,**d:rows.append((stage,d)),0,interval=0)
    assert wrapped(**kwargs) is result
    assert [stage for stage,_ in rows]==['generation_start','generation','generation_finished']
    assert kwargs['stopping_criteria']==[criterion]


@pytest.mark.parametrize('path_mode',['normal','unwritable'])
def test_hung_generate_fails_bounded_and_keeps_checkpoint_bytes(tmp_path,path_mode):
    keep=tmp_path/'step-000025.pt';keep.write_bytes(b'existing optimizer RNG rewards checkpoint')
    progress=tmp_path/'progress'
    if path_mode=='unwritable':progress.write_text('not a directory')
    code='''import sys,time,types
f=types.ModuleType('transformers');f.StoppingCriteriaList=list;sys.modules['transformers']=f
from v8_generation_probe import observed_generate
def blocked(**kw):time.sleep(20)
observed_generate(blocked,lambda *a,**kw:None,0,timeout=.1)(input_ids=types.SimpleNamespace(shape=(1,3)))
'''
    result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(REPO/'scripts'),
                           SRGC_PROGRESS_DIR=str(progress)),capture_output=True,text=True,timeout=3)
    assert result.returncode==70 and 'no returned decode step' in result.stderr
    assert keep.read_bytes()==b'existing optimizer RNG rewards checkpoint'
    if path_mode=='normal':assert 'blocked' in (progress/'generation-stall-rank0.txt').read_text()


def test_real_tiny_olmo_outputs_and_rng_match_original(tmp_path):
    python=os.environ.get('V8_TINY_TEST_PYTHON')
    if not python:pytest.skip('Set V8_TINY_TEST_PYTHON to the existing CPU torch/transformers environment')
    code='''import json,torch
from transformers import Olmo3Config,Olmo3ForCausalLM,StoppingCriteriaList,StoppingCriteria
from v8_generation_probe import observed_generate
torch.set_num_threads(1);torch.manual_seed(17)
c=Olmo3Config(vocab_size=16,hidden_size=16,intermediate_size=24,num_hidden_layers=4,num_attention_heads=2,
 num_key_value_heads=2,max_position_embeddings=64,pad_token_id=0,eos_token_id=None,layer_types=['full_attention']*4,attention_dropout=0.)
m=Olmo3ForCausalLM(c).eval();inputs=torch.tensor([[1,2,3]])
class ExistingStop(StoppingCriteria):
 def __call__(self,ids,scores,**kw):return torch.full((ids.shape[0],),ids.shape[-1]>=7,dtype=torch.bool)
kw=dict(input_ids=inputs,attention_mask=torch.ones_like(inputs),num_return_sequences=8,max_new_tokens=8,
 do_sample=True,temperature=1.,top_p=1.,top_k=0,pad_token_id=0,eos_token_id=None,
 stopping_criteria=StoppingCriteriaList([ExistingStop()]))
torch.manual_seed(29);expected=m.generate(**kw);rng=torch.get_rng_state().clone()
records=[];wrapped=observed_generate(m.generate,lambda stage,**d:records.append((stage,d)),0,interval=0)
torch.manual_seed(29);actual=wrapped(**kw)
assert torch.equal(expected,actual)
assert torch.equal(rng,torch.get_rng_state())
assert actual.shape==(8,7),actual.shape
assert records[-1][0]=='generation_finished' and records[-1][1]['decode_steps']==4
print(json.dumps(dict(test='tiny random Olmo3 CPU software control, not trained model results',outputs_equal=True,
 rng_equal=True,responses=8,new_tokens=4,existing_stop_preserved=True)))
'''
    result=subprocess.run([python,'-c',code],env=dict(os.environ,PYTHONPATH=str(REPO/'scripts'),CUDA_VISIBLE_DEVICES='',
                           OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    (tmp_path/'tiny-olmo-output.json').write_text(result.stdout)


def test_supervisor_shows_decode_counts_and_distinguishes_entered_event(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    from scripts import v8_supervisor
    event=dict(stage='generation',updated=time.time(),generation_call=2,sequence_tokens=91,decode_steps=88)
    (tmp_path/'rank-0.json').write_text(json.dumps(event))
    rows={42:dict(kind='task',state='S',env={'V8_LAUNCH_ID':'fixture','SRGC_PROGRESS_DIR':str(tmp_path)})}
    line=v8_supervisor.work_events(rows,'fixture',tmp_path)[0]
    assert 'Writing model answers' in line and 'sequence length 91 tokens including input' in line
    event.update(stage='generation_start',progress_kind='entered_only')
    (tmp_path/'rank-0.json').write_text(json.dumps(event))
    assert 'no newly decoded token recorded yet' in v8_supervisor.work_events(rows,'fixture',tmp_path)[0]


@pytest.mark.parametrize('has_progress',[False,True])
@pytest.mark.parametrize('valid_hash',[False,True])
def test_original_child_entry_keeps_hash_verification_and_supports_legacy_context(tmp_path,has_progress,valid_hash):
    child=tmp_path/'cluster_child.py';marker=tmp_path/'result.json'
    child.write_text('''import sys,types,json
from pathlib import Path
class Context:
 def __init__(self,backend):self.backend=backend
runner=types.ModuleType('switching.runner');runner.Context=Context
switching=types.ModuleType('switching');switching.__path__=[]
sys.modules['switching']=switching;sys.modules['switching.runner']=runner
progress=types.ModuleType('srgc_rebuttal.progress');progress.record=lambda *a,**kw:None
pkg=types.ModuleType('srgc_rebuttal');pkg.progress=progress
sys.modules['srgc_rebuttal']=pkg;sys.modules['srgc_rebuttal.progress']=progress
hf=types.ModuleType('transformers');hf.StoppingCriteriaList=list;sys.modules['transformers']=hf
def main():
 class Model:
  def generate(self,**kw):
   assert kw['stopping_criteria'][-1](types.SimpleNamespace(shape=(1,4)),None) is False
   return [1,2,3,4]
 c=Context(types.SimpleNamespace(model=Model()))
 result=c.backend.model.generate(input_ids=types.SimpleNamespace(shape=(1,3)),num_return_sequences=8,max_new_tokens=2048)
 Path(sys.argv[1]).write_text(json.dumps(dict(result=result,original_argv=sys.argv[2:])))
'''+('def install_progress(rank):pass\n' if has_progress else ''))
    expected=hashlib.sha256(child.read_bytes()).hexdigest() if valid_hash else 'f'*64
    result=subprocess.run([sys.executable,str(REPO/'scripts/v8_task_entry.py'),str(child),str(marker),'--task','original-id'],
        env=dict(os.environ,V8_ORIGINAL_CHILD_SHA256=expected,RANK='0'),capture_output=True,text=True,timeout=4)
    if valid_hash:
        assert result.returncode==0,result.stdout+result.stderr
        assert json.loads(marker.read_text())==dict(result=[1,2,3,4],original_argv=['--task','original-id'])
    else:
        assert result.returncode!=0 and not marker.exists()
        assert 'bytes differ' in result.stderr


def test_logs_report_actual_local_questions_purpose_and_optimizer_steps_without_changing_results(monkeypatch,tmp_path):
    rows=[];ledger=[]
    progress=types.ModuleType('srgc_rebuttal.progress');progress.record=lambda stage,**d:rows.append((stage,d))
    pkg=types.ModuleType('srgc_rebuttal');pkg.progress=progress
    monkeypatch.setitem(sys.modules,'srgc_rebuttal',pkg);monkeypatch.setitem(sys.modules,'srgc_rebuttal.progress',progress)
    hf=types.ModuleType('transformers');hf.StoppingCriteriaList=list;monkeypatch.setitem(sys.modules,'transformers',hf)
    class Model:
        def generate(self,**kw):
            assert kw['num_return_sequences']==8 and kw['max_new_tokens']==2048
            for size in (4,5):kw['stopping_criteria'][-1](fake_ids(size),None)
            return 'unchanged model result'
    class Backend:
        rank=1;world=2
        def __init__(self):
            self.model=Model();self.cost_meter=types.SimpleNamespace(sections=['validation'],event=dict(phase='selection'))
        def score_gradients(self,ids,*,responses,group_size,seed):
            result={}
            for q in ids[self.rank::self.world]:
                assert self.model.generate(input_ids=fake_ids(3),num_return_sequences=responses,max_new_tokens=2048)=='unchanged model result'
                progress.record('rollout',prompt=q,responses=responses)
                progress.record('gradient_scoring',prompt=q);result[q]='unchanged gradient'
            return result
        def train(self,ids,**kwargs):
            progress.record('policy_update');return dict(loss=0.25)
    class Context:
        def __init__(self,backend):
            self.backend=backend;self.out_root=tmp_path/'olmo_math';self.meter=types.SimpleNamespace(record=lambda _:None)
        def on_meter_event(self,event):ledger.append(event)
    mod=types.ModuleType('switching.runner');mod.Context=Context
    parent=types.ModuleType('switching');parent.__path__=[]
    monkeypatch.setitem(sys.modules,'switching',parent);monkeypatch.setitem(sys.modules,'switching.runner',mod)
    install(1);ctx=Context(Backend())
    assert ctx.backend.score_gradients(['q0','q1','q2'],responses=8,group_size=8,seed=7)=={'q1':'unchanged gradient'}
    events=[d for stage,d in rows if stage=='generation']
    assert events and events[0]['questions_total']==events[0]['question_index']==1
    assert events[0]['questions_answered']==events[0]['optimizer_steps_this_process']==0
    assert events[0]['purpose']=='score reference questions to rank training candidates'
    completed=[d for stage,d in rows if stage=='gradient_scoring'][-1]
    assert completed['questions_answered']==completed['gradient_scores_completed']==1
    assert ctx.backend.train(['q0','q1'],responses=8,seed=7,objective='grpo')==dict(loss=.25)
    assert rows[-1][1]['optimizer_steps_this_process']==1
    event=dict(phase='checkpoint_save',checkpoint=25,state='finished',wall_seconds=1.)
    ctx.meter.record(event)
    assert ledger==[event] and ledger[0] is event
