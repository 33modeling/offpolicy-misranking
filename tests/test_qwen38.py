"""Qwen replication contract and text-only gradient parameter selection."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grads import grad_params
from model_matrix import _load_config, _load_specs


def test_qwen_matrix_matches_primary_sampling():
    qwen = _load_config(ROOT / "configs/qwen38_27b_grpo.json")
    primary = _load_config(ROOT / "configs/olmo3_rlzero_h100.json")
    assert len(qwen["models"]) == 1
    spec = next(iter(_load_specs(ROOT / "configs/qwen38_27b_grpo.json").values()))
    assert spec["initialization"] == "posttrained"
    assert spec["repository"] == "Qwen/Qwen3.8-27B"
    assert "in_proj_qkv" in spec["lora_targets"]
    for key in primary["experiment"]:
        if key != "runtime":
            assert qwen["experiment"][key] == primary["experiment"][key], key


@pytest.mark.parametrize("multimodal", [False, True])
def test_ranking_freezes_vision_embedding_and_earlier_layers(multimodal):
    decoder = nn.Module()
    decoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(5)])
    decoder.norm = nn.LayerNorm(4)
    decoder.embed_tokens = nn.Embedding(8, 4)
    model = nn.Module()
    if multimodal:
        model.model = nn.Module()
        model.model.language_model = decoder
        model.model.visual = nn.Linear(4, 4)
    else:
        model.model = decoder
    model.lm_head = nn.Linear(4, 8)
    params = grad_params(model, 4)
    expected = list(decoder.layers[1:].parameters()) + list(decoder.norm.parameters())
    assert [id(p) for p in params] == [id(p) for p in expected]
    assert all(p.requires_grad == (id(p) in {id(x) for x in expected})
               for p in model.parameters())
    for invalid in (0, -1, 6):
        with pytest.raises(ValueError):
            grad_params(model, invalid)


def test_qwen_actual_tiny_multimodal_decoder_gradient(monkeypatch):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from grads import ProjectionSpec, prompt_gradient

    # Exercise the actual architecture on CPU with the upstream torch kernels;
    # the separate H100 preflight tests the CUDA FLA implementation.
    for name in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
                 "FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update"):
        monkeypatch.setattr(modeling, name, None)
    config = Qwen3_5Config(
        text_config={"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 4, "num_attention_heads": 2,
                     "num_key_value_heads": 1, "head_dim": 16,
                     "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                     "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                     "pad_token_id": 0},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 2, "out_hidden_size": 32},
    )
    model = Qwen3_5ForConditionalGeneration(config).eval()
    params = grad_params(model, 4)
    projection = prompt_gradient(
        model, params, [{"input_ids": torch.tensor([1, 2, 3, 4]), "resp_start": 2}],
        [torch.ones(2)], ProjectionSpec(dim=32), micro_batch=1,
    )
    assert torch.isfinite(projection).all()
    assert projection.abs().sum() > 0
