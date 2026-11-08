"""Exercise the actual archived TorchBackend; CPU tensors, not a GPU capacity claim."""
import gc
import importlib
import importlib.util
import json
from pathlib import Path
import types
import weakref

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def backend(check_liveness=True):
    torch=pytest.importorskip('torch')
    spec=importlib.util.spec_from_file_location('srgc_rebuttal._legacy_memory_test',
          ROOT/'scripts/frozen_srgc/pre_20261004/torch_backend.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from srgc_rebuttal.timing import torch_meter
    torch.manual_seed(17)
    model=torch.nn.Linear(1,1,bias=False);model.weight.requires_grad_(False)
    model.register_parameter('adapter',torch.nn.Parameter(torch.tensor(.25)))
    b=object.__new__(module.TorchBackend)
    b.model=model;b.device=torch.device('cpu');b.rank=0;b.world=1
    b.projection_dim=8;b.projection_seed=3;b.logprob_micro_batch=2
    b.train_parameters=[('adapter',model.adapter)];b.score_parameters=[('weight',model.weight)]
    b.optimizer=torch.optim.AdamW([model.adapter],lr=.001);b.cost_meter=torch_meter(cuda=False)
    b.held=[];b.calls=[]
    def rollout(prompt,responses,seed):
        b.calls.append(prompt);gc.collect()
        if check_liveness and prompt=='q2' and any(r() is not None for r in b.held):
            raise torch.OutOfMemoryError('previous prompt dense gradient tensors are still live')
        return [torch.tensor([1,i+2]) for i in range(responses)],np.array([0.,1.]*(responses//2)),1
    b._rollout=rollout
    b._logps_batch=lambda seqs,start:[(model.weight*s.sum()).reshape(1) for s in seqs]
    project=b._project
    def tracked(name,gradient):b.held.append(weakref.ref(gradient));return project(name,gradient)
    b._project=tracked
    return b,module


def test_repaired_real_archived_scorer_releases_previous_prompt_gradients(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));entry=importlib.import_module('v8_task_entry')
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path));monkeypatch.setenv('V8_OOM_RETRY_SECONDS','0')
    b,_=backend();ctx=types.SimpleNamespace(backend=b)
    entry.repair_context(ctx)
    result=b.score_gradients(['q1','q2'],responses=8,group_size=4,seed=17)
    assert list(result)==['q1','q2'] and b.calls==['q1','q2']
    assert all(r() is None for r in b.held)


def test_gradient_values_and_rng_match_the_existing_fixed_scorer(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from srgc_rebuttal.torch_backend import TorchBackend
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));entry=importlib.import_module('v8_task_entry')
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path))
    reference,_=backend(False)
    expected=TorchBackend.score_gradients(reference,['q1','q2'],responses=8,group_size=4,seed=17)
    rng=torch.get_rng_state().clone()
    b,_=backend();entry.repair_context(types.SimpleNamespace(backend=b))
    actual=b.score_gradients(['q1','q2'],responses=8,group_size=4,seed=17)
    assert all(np.array_equal(actual[k],expected[k]) for k in expected)
    assert torch.equal(torch.get_rng_state(),rng)


def test_snapshot_preserves_optimizer_values_without_deepcopy_of_live_tensors(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));entry=importlib.import_module('v8_task_entry')
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path))
    b,_=backend(False)
    b.model.adapter.grad=torch.ones_like(b.model.adapter);b.optimizer.step()
    expected=b.state_dict()
    original=entry.copy.deepcopy
    def no_tensor_copy(value,*args,**kwargs):
        assert not isinstance(value,(dict,list,tuple,torch.Tensor)), 'deepcopy would duplicate live optimizer tensors'
        return original(value,*args,**kwargs)
    entry.repair_context(types.SimpleNamespace(backend=b))
    with monkeypatch.context() as patch:
        patch.setattr(entry.copy,'deepcopy',no_tensor_copy)
        actual=b.state_dict()
    def equal_and_independent(left,right):
        if isinstance(left,torch.Tensor):
            assert right.device.type=='cpu' and torch.equal(left,right)
            assert left.data_ptr()!=right.data_ptr()
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:equal_and_independent(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            assert type(left) is type(right) and len(left)==len(right)
            for a,c in zip(left,right):equal_and_independent(a,c)
        else:assert left==right
    equal_and_independent(expected,actual)
    b.model.adapter.data.add_(1)
    b.load_state_dict(actual)
    assert torch.equal(b.model.adapter,expected['trainable']['adapter'])
    receipt=json.loads((tmp_path/'runtime-repair-rank-0.json').read_text())
    assert receipt['repairs']['dense_memory']['original_backend_sha256']==entry.LEGACY_DENSE_BACKEND_SHA256
    assert receipt['repairs']['dense_memory']['extra_collectives']==0


def test_real_protected_subclass_keeps_its_guard_and_releases_dense_tensors(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));monkeypatch.syspath_prepend(str(ROOT/'v8/experiments'))
    entry=importlib.import_module('v8_task_entry')
    child=importlib.import_module('cluster_child')
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path))
    b,_=backend();b.__class__=child.protected_backend_class(type(b))
    entry.repair_context(types.SimpleNamespace(backend=b))
    assert list(b.score_gradients(['q1','q2'],responses=8,group_size=4,seed=17))==['q1','q2']
    assert b._v8_protect is False and all(ref() is None for ref in b.held)


def test_shards_gather_once_and_zero_advantage_prompts_are_safe(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));entry=importlib.import_module('v8_task_entry')
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path))
    ids=['q'+str(i) for i in range(9)]
    for rank in range(4):
        b,module=backend(False);b.rank=rank;b.world=4
        rollout=b._rollout
        def zero(*args):
            seqs,rewards,start=rollout(*args)
            return seqs,np.zeros_like(rewards),start
        b._rollout=zero;gathers=[]
        def gather(parts,local):
            gathers.append(list(local))
            for i in range(4):parts[i]={key:np.zeros(8) for key in ids[i::4]}
            parts[rank]=local
        monkeypatch.setattr(module.dist,'all_gather_object',gather)
        entry.repair_context(types.SimpleNamespace(backend=b))
        result=b.score_gradients(ids,responses=8,group_size=4,seed=17)
        assert b.calls==ids[rank::4] and gathers==[ids[rank::4]]
        assert list(result)==[key for r in range(4) for key in ids[r::4]]
        assert all(np.array_equal(value,np.zeros(8)) for value in result.values())


def test_unrecognized_backend_and_custom_snapshot_override_are_preserved(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));entry=importlib.import_module('v8_task_entry')
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path))
    class Other:
        def score_gradients(self):return 'custom'
    assert entry.repair_dense_memory(Other()) is None
    b,_=backend(False)
    class Custom(type(b)):
        def state_dict(self):return {'custom-snapshot':True}
    b.__class__=Custom;entry.repair_context(types.SimpleNamespace(backend=b))
    assert b.state_dict()=={'custom-snapshot':True}
