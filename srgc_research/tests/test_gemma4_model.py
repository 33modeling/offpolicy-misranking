"""Actual unified loading, soft-capped gradients, offline identity and recovery."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Gemma4UnifiedConfig,
    Gemma4UnifiedForCausalLM,
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedTextConfig,
    PreTrainedTokenizerFast,
)

from srgc_research.dispatch.gemma4 import adapter, model
from srgc_research.dispatch.gemma4.memory import GemmaBackend, bounded_generate


def tiny_config():
    return Gemma4UnifiedTextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=1, num_global_key_value_heads=1,
        head_dim=8, global_head_dim=8, layer_types=["sliding_attention", "full_attention"] * 3,
        sliding_window=8, attention_k_eq_v=True, final_logit_softcapping=2.0,
        rope_parameters={"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.5, "rope_theta": 10000.0},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
        bos_token_id=2, eos_token_id=1, pad_token_id=0,
    )


@pytest.fixture
def snapshot(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    path = tmp_path / "models/gemma-4-12B"
    path.mkdir(parents=True)
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, **{f"t{i}": i for i in range(4, 32)}}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=raw, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>").save_pretrained(path)
    full = Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(
        text_config=tiny_config(), vision_config={"mm_embed_dim": 16, "mm_posemb_size": 8}, audio_config={"audio_embed_dim": 8}))
    full.save_pretrained(path)
    expected = {}
    for name in model.OFFICIAL_FILES:
        data = (path / name).read_bytes()
        kind = "sha256" if name in {"tokenizer.json", "model.safetensors"} else "blob"
        digest = hashlib.sha256(data).hexdigest() if kind == "sha256" else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        expected[name] = (len(data), digest, kind)
    return path, full, expected, {"GROUP_VOLUME": str(tmp_path), "OM_WORK": str(tmp_path / "work")}


def forced_rollout(prompt_id, responses, seed):
    offset = int(prompt_id.removeprefix("p")) % 4
    sequences = [torch.tensor([2, 4, 5, 8 + offset, 9]) if index % 2 else torch.tensor([2, 4, 5, 12 + offset, 13, 14]) for index in range(responses)]
    return sequences, np.array([0.0, 1.0] * (responses // 2)), 3


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[2, 4, 5]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens.tolist()))


def make_backend(dtype=torch.float32):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    policy = adapter.attach_adapter(Gemma4UnifiedForCausalLM(tiny_config()).to(dtype), LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM", lora_dropout=0.0), get_peft_model)
    base = policy.get_base_model()
    base.generate = bounded_generate(base.generate, torch.device("cpu"))
    return GemmaBackend(policy, TinyTokenizer(), {f"p{i}": {"prompt": f"problem {i}", "answer": "9"} for i in range(4)}, lambda record, text: float("9" in text.split()), projection_dim=16, max_new_tokens=3, logit_chunk_tokens=2)


def test_full_checkpoint_text_loading_preserves_native_logits(snapshot):
    path, full, _, _ = snapshot
    text, tokenizer = model.load_text_model(path, "cpu")
    ids = torch.tensor([[2, 4, 5, 6]])
    with torch.no_grad():
        torch.testing.assert_close(text(ids, use_cache=False).logits, full(ids, use_cache=False).logits, rtol=1e-5, atol=1e-6)
    assert not any("embed_vision" in name or "embed_audio" in name for name, _ in text.named_parameters())
    assert tokenizer.encode("t4 t5") == [2, 4, 5]
    assert tokenizer.encode("t4 t5", add_special_tokens=False) == [4, 5]
    assert tokenizer("t4 t5")["input_ids"].tolist() == [[2, 4, 5]]
    assert (text.generation_config.top_k, text.generation_config.top_p, text.generation_config.eos_token_id) == (0, 1.0, 1)
    assert text.generation_config.suppress_tokens is None


def test_missing_text_weight_cannot_silently_initialize(snapshot):
    from safetensors.torch import load_file, save_file
    path, _, _, _ = snapshot
    tensors = load_file(path / "model.safetensors")
    tensors.pop("model.language_model.layers.0.self_attn.q_proj.weight")
    save_file(tensors, path / "model.safetensors")
    with pytest.raises(ValueError, match="Incomplete Gemma text weights"):
        model.load_text_model(path, "cpu")


def test_offline_verification_is_cached_and_never_writes_weights(snapshot):
    path, _, expected, env = snapshot
    before = {file: file.read_bytes() for file in path.iterdir()}
    with patch.object(model, "OFFICIAL_FILES", expected):
        found, identity = model.resolve_snapshot(env)
        assert found == path and len(identity) == 64
        with patch.object(model, "validate_tokenizer_metadata", side_effect=AssertionError("must use verification cache")):
            assert model.resolve_snapshot(env) == (found, identity)
        assert adapter.model_path(adapter.MODEL, adapter.REVISION, env) == str(path)
        env["SRGC_GEMMA_MODEL_SHA256"] = "a" * 64
        with pytest.raises(ValueError, match="weights differ"):
            adapter.model_path(adapter.MODEL, adapter.REVISION, env)
    assert {file: file.read_bytes() for file in path.iterdir()} == before


def test_reserialized_tokenizer_is_accepted_before_first_verification(snapshot):
    path, _, expected, env = snapshot
    metadata = json.loads((path / "tokenizer_config.json").read_text())
    (path / "tokenizer_config.json").write_text(json.dumps(metadata, sort_keys=True, indent=4))
    with patch.object(model, "OFFICIAL_FILES", expected):
        assert model.resolve_snapshot(env)[0] == path


@pytest.mark.parametrize("fault", ["weights", "missing", "wrong_eos", "changed_metadata", "quantized"])
def test_wrong_incomplete_or_changed_snapshot_is_rejected(snapshot, fault):
    path, _, expected, env = snapshot
    with patch.object(model, "OFFICIAL_FILES", expected):
        if fault in {"changed_metadata", "wrong_eos"}:
            if fault == "changed_metadata":
                model.resolve_snapshot(env)
            metadata = json.loads((path / "tokenizer_config.json").read_text())
            metadata.update({"model_max_length": 4096} if fault == "changed_metadata" else {"eos_token": "t4"})
            (path / "tokenizer_config.json").write_text(json.dumps(metadata))
        elif fault == "weights":
            (path / "model.safetensors").write_bytes(b"x" * (path / "model.safetensors").stat().st_size)
        elif fault == "missing":
            (path / "model.safetensors").unlink()
        else:
            config = json.loads((path / "config.json").read_text())
            config["quantization_config"] = {"bits": 4}
            (path / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError):
            model.resolve_snapshot(env)


def test_hub_discovery_retains_original_cache(snapshot):
    from srgc_research.dispatch.gemma4.storage import setup_storage
    path, _, _, env = snapshot
    cache = Path(env["GROUP_VOLUME"]) / "hub"
    target = cache / "models--google--gemma-4-12B/snapshots" / adapter.REVISION
    target.parent.mkdir(parents=True)
    path.rename(target)
    env["HF_HUB_CACHE"] = str(cache)
    setup_storage(Path(env["OM_WORK"]) / "srgc-rebuttal/gemma4-12b-pt-v1", env)
    assert model.find_snapshot(env) == target
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"


def test_missing_snapshot_never_downloads(tmp_path):
    with pytest.raises(ValueError, match="No model download"):
        model.find_snapshot({"GROUP_VOLUME": str(tmp_path)})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_logps_and_gradients_match_native_softcapped_forward(dtype):
    backend = make_backend(dtype)
    sequences, _, start = forced_rollout("p0", 8, 17)
    selected = [sequences[0], sequences[1]]
    actual = backend._logps_batch(selected, start)
    ids = torch.zeros((2, max(map(len, selected))), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(selected):
        ids[index, :len(sequence)], mask[index, :len(sequence)] = sequence, 1
    native = backend.model(ids, attention_mask=mask, use_cache=False).logits[:, :-1].float().log_softmax(-1)
    expected = [native[row, start-1:len(sequence)-1].gather(-1, sequence[start:, None]).squeeze(-1) for row, sequence in enumerate(selected)]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5)
    params = [parameter for _, parameter in backend.train_parameters]
    ga = torch.autograd.grad(sum(value.sum() for value in actual), params)
    gb = torch.autograd.grad(sum(value.sum() for value in expected), params)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=0.02 if dtype == torch.bfloat16 else 1e-5, atol=2e-3 if dtype == torch.bfloat16 else 1e-6)
    for index, layer in enumerate(backend.model.get_base_model().model.layers):
        names = {name for name, module in layer.self_attn.named_children() if hasattr(module, "lora_A")}
        assert names == ({"q_proj", "v_proj"} if index % 2 == 0 else {"q_proj", "k_proj"})


def test_generation_dense_scoring_grpo_and_exact_resume(tmp_path):
    backend = make_backend()
    first, rewards, _ = backend._rollout("p0", 8, 17)
    second, other, _ = backend._rollout("p0", 8, 17)
    assert len(first) == 8 and all(torch.equal(a, b) for a, b in zip(first, second))
    np.testing.assert_array_equal(rewards, other)
    backend._rollout = forced_rollout
    initial = copy.deepcopy(backend.state_dict())
    flags = [parameter.requires_grad for parameter in backend.model.parameters()]
    gradients = backend.score_gradients(["p0", "p1"], responses=8, group_size=4, seed=17)
    assert all(np.isfinite(value).all() and np.linalg.norm(value) > 0 for value in gradients.values())
    assert flags == [parameter.requires_grad for parameter in backend.model.parameters()]
    assert not backend.optimizer.state
    backend.train(["p0", "p1"], responses=8, objective="grpo", seed=17)
    trained = backend.state_dict()
    assert any(not torch.equal(value, initial["trainable"][name]) for name, value in trained["trainable"].items())
    torch.save(initial, tmp_path / "checkpoint.pt")
    backend.load_state_dict(torch.load(tmp_path / "checkpoint.pt", weights_only=False))
    backend.train(["p0", "p1"], responses=8, objective="grpo", seed=17)
    for name, value in backend.state_dict()["trainable"].items():
        torch.testing.assert_close(value, trained["trainable"][name], rtol=0, atol=0)


def test_oom_retries_all_responses_from_saved_rng():
    calls = []
    def generate(**kwargs):
        values = torch.randint(4, 30, (kwargs["num_return_sequences"], 3))
        calls.append(kwargs["num_return_sequences"])
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return values
    torch.manual_seed(17)
    expected = torch.cat([torch.randint(4, 30, (1, 3)) for _ in range(8)])
    torch.manual_seed(17)
    actual = bounded_generate(generate, torch.device("cpu"))(num_return_sequences=8, pad_token_id=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls[:2] == [2, 2] and calls[2:] == [1] * 8
