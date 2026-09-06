"""Chunked LM-head log-probs must equal the single-pass computation.

Covers the 2026-09-06 memory change in ``grads._token_logps_chunked`` and
``train_policy_grpo._response_logps_batch``: values, gradients through the
checkpointed chunks, and the unmerged-PEFT layout (``PeftModel.model`` is the
causal LM, not the bare transformer).

    PYTHONPATH=src python3 -m pytest tests/test_logit_chunking.py -q
"""

from __future__ import annotations

import os

import pytest
import torch

import grads
import train_policy_grpo as tp

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def _tiny_olmo3(seed: int = 0):
    torch.manual_seed(seed)
    cfg = transformers.Olmo3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        eos_token_id=1,
        bos_token_id=None,
        pad_token_id=0,
    )
    return transformers.Olmo3ForCausalLM(cfg).eval()


@pytest.fixture
def sequences() -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [torch.randint(2, 97, (n,)) for n in (7, 13, 22)]


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setenv("OM_LOGIT_CHUNK_TOKENS", "5")
    monkeypatch.setattr(grads, "LOGIT_CHUNK_TOKENS", 5)


def _grads_single_pass(model, sequences, monkeypatch):
    monkeypatch.setattr(grads, "LOGIT_CHUNK_TOKENS", 0)
    try:
        return grads._padded_token_logps(model, sequences)
    finally:
        monkeypatch.setattr(grads, "LOGIT_CHUNK_TOKENS", 5)


def test_grads_chunked_logps_match_single_pass(sequences, small_chunks, monkeypatch):
    model = _tiny_olmo3()
    chunked = grads._padded_token_logps(model, sequences)
    single = _grads_single_pass(model, sequences, monkeypatch)
    assert [t.numel() for t in chunked] == [n - 1 for n in (7, 13, 22)]
    for a, b in zip(chunked, single, strict=True):
        assert torch.allclose(a.detach(), b.detach(), atol=1e-6, rtol=0)


def test_grads_chunked_projected_gradient_matches_single_pass(
    sequences, small_chunks, monkeypatch
):
    model = _tiny_olmo3()
    params = grads.grad_params(model, 2)
    rows = [{"input_ids": s, "resp_start": 3} for s in sequences]
    weights = [torch.full((s.numel() - 3,), 0.7) for s in sequences]
    spec = grads.ProjectionSpec(dim=256)
    chunked = grads.prompt_gradient(model, params, rows, weights, spec, micro_batch=2)
    monkeypatch.setattr(grads, "LOGIT_CHUNK_TOKENS", 0)
    single = grads.prompt_gradient(model, params, rows, weights, spec, micro_batch=2)
    assert float(single.norm()) > 0
    assert torch.allclose(chunked, single, atol=1e-5 * max(1.0, float(single.norm())), rtol=0)


def test_training_chunked_response_logps_match_single_pass(sequences, small_chunks, monkeypatch):
    model = _tiny_olmo3()
    starts = [3, 5, 4]
    chunked = tp._response_logps_batch(model, sequences, starts, pad_token_id=0)
    monkeypatch.setenv("OM_LOGIT_CHUNK_TOKENS", "0")
    single = tp._response_logps_batch(model, sequences, starts, pad_token_id=0)
    assert [t.numel() for t in chunked] == [n - s for n, s in zip((7, 13, 22), starts, strict=True)]
    for a, b in zip(chunked, single, strict=True):
        assert torch.allclose(a.detach(), b.detach(), atol=1e-6, rtol=0)


def test_grads_chunked_path_handles_unmerged_peft_model(sequences, small_chunks, monkeypatch):
    model = peft.get_peft_model(
        _tiny_olmo3(),
        peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
    ).eval()
    chunked = grads._padded_token_logps(model, sequences)
    single = _grads_single_pass(model, sequences, monkeypatch)
    for a, b in zip(chunked, single, strict=True):
        assert torch.allclose(a.detach(), b.detach(), atol=1e-6, rtol=0)
