"""Real tiny OLMo CPU forward/backward; no H100 capacity or learning-result claim."""
import importlib
import importlib.util
import os
from pathlib import Path
import types

import pytest

ROOT=Path(__file__).resolve().parents[1]


def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    if os.environ.get('V8_MEMORY_ENTRY'):
        spec=importlib.util.spec_from_file_location('v8_entry_before_olmo',os.environ['V8_MEMORY_ENTRY'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    return importlib.import_module('v8_task_entry')


def tiny_backend():
    torch=pytest.importorskip('torch')
    from transformers import Olmo3Config,Olmo3ForCausalLM
    from srgc_rebuttal.torch_backend import TorchBackend
    torch.manual_seed(17)
    cfg=Olmo3Config(vocab_size=48,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                    num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
                    attention_dropout=0,pad_token_id=0,eos_token_id=2,bos_token_id=1)
    cfg._attn_implementation='eager';model=Olmo3ForCausalLM(cfg).eval()
    for name,p in model.named_parameters():p.requires_grad_('q_proj.weight' in name)
    tokenizer=types.SimpleNamespace(pad_token_id=0,eos_token_id=2)
    return TorchBackend(model,tokenizer,{},lambda row,text:0.,projection_dim=8,logit_chunk_tokens=4)


def gradients_and_saved_storage(backend):
    import torch
    params=[p for _,p in backend.train_parameters]
    parameter_storage={p.untyped_storage().data_ptr() for p in backend.model.parameters()}
    stored={}
    def pack(t):
        storage=t.untyped_storage();ptr=storage.data_ptr()
        if ptr not in parameter_storage:stored[ptr]=storage.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        values=backend._logps_batch([torch.tensor([1,3,4,5,6,7,8,9]),torch.tensor([1,3,4,10,11,12])],3)
        loss=sum(v.sum() for v in values)
    grads=torch.autograd.grad(loss,params)
    return [v.detach().clone() for v in values],grads,sum(stored.values())


def test_checkpointed_real_olmo_preserves_outputs_gradients_rng_and_reduces_saved_activation_storage(tmp_path,monkeypatch):
    import torch
    before=tiny_backend();after=tiny_backend()
    expected,expected_gradients,baseline_storage=gradients_and_saved_storage(before)
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path));module=entry(monkeypatch)
    module.repair_context(types.SimpleNamespace(backend=after))
    assert after.model.is_gradient_checkpointing,'OLMo decoder activation checkpointing was not enabled'
    rng=torch.get_rng_state().clone()
    actual,gradients,storage=gradients_and_saved_storage(after)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(gradients,expected_gradients):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),rng) and after.model.training is False
    assert storage<baseline_storage,(storage,baseline_storage)


def test_generation_keeps_all_eight_answers_tokens_rng_and_eval_mode(tmp_path,monkeypatch):
    import torch
    before=tiny_backend();after=tiny_backend()
    kwargs=dict(input_ids=torch.tensor([[1,3,4,5]]),attention_mask=torch.ones(1,4,dtype=torch.long),
                num_return_sequences=8,max_new_tokens=6,do_sample=True,temperature=1.,top_p=1.,top_k=0,
                pad_token_id=0,eos_token_id=2,use_cache=True)
    torch.manual_seed(29);expected=before.model.generate(**kwargs);rng=torch.get_rng_state().clone()
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path));entry(monkeypatch).repair_context(types.SimpleNamespace(backend=after))
    torch.manual_seed(29);actual=after.model.generate(**kwargs)
    assert actual.shape[0]==8 and torch.equal(actual,expected)
    assert torch.equal(torch.get_rng_state(),rng) and after.model.training is False


def test_checkpoint_mode_restored_on_failure_and_nonzero_dropout_rejected(tmp_path,monkeypatch):
    module=entry(monkeypatch);backend=tiny_backend();monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path))
    def broken(*args):
        assert backend.model.training is True
        raise RuntimeError('injected logprob failure')
    backend._logps_batch=broken;module.repair_context(types.SimpleNamespace(backend=backend))
    with pytest.raises(RuntimeError,match='injected logprob'):backend._logps_batch([])
    assert backend.model.training is False
    other=tiny_backend();other.model.config.attention_dropout=.1
    with pytest.raises(ValueError,match='zero dropout'):module.repair_context(types.SimpleNamespace(backend=other))


def test_real_grpo_optimizer_update_matches_with_checkpointed_olmo(tmp_path,monkeypatch):
    import numpy as np
    import torch
    before=tiny_backend();after=tiny_backend()
    def rollout(prompt,responses,seed):
        return [torch.tensor([1,3,4,5,6,7,i+8]) for i in range(responses)],np.array([0.,1.]*(responses//2)),3
    before._rollout=rollout;after._rollout=rollout
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path));entry(monkeypatch).repair_context(types.SimpleNamespace(backend=after))
    expected=before.train(['q'],responses=8,objective='grpo',seed=17)
    actual=after.train(['q'],responses=8,objective='grpo',seed=17)
    assert actual==expected and after.model.training is False
    for a,b in zip(after.model.parameters(),before.model.parameters()):assert torch.equal(a,b)
    for k,state in before.optimizer.state_dict()['state'].items():
        for name,value in state.items():
            assert torch.equal(value,after.optimizer.state_dict()['state'][k][name])
