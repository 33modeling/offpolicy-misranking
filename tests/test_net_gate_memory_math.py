import copy
import os

import pytest

torch = pytest.importorskip("torch")

import low_order_backend as backend
import net_gate_memory_worker as memory


def olmo(dtype, *, use_cache=False):
    from peft import LoraConfig, get_peft_model
    from transformers import Olmo3Config, Olmo3ForCausalLM
    torch.manual_seed(19)
    config = Olmo3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                        attention_dropout=.25, use_cache=use_cache)
    config._attn_implementation = "eager"
    model = get_peft_model(Olmo3ForCausalLM(config).to(dtype), LoraConfig(
        r=2, lora_alpha=2, lora_dropout=.3, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.eval()
    rows = [{"input_ids": torch.arange(96).remainder(62) + 1, "resp_start": 5, "reward": 1},
            {"input_ids": torch.arange(64).remainder(62) + 1, "resp_start": 5, "reward": 0}]
    direction = {n: torch.randn_like(p) for n, p in backend.trainable_parameters(model).items()}
    return model, rows, direction


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("merged", [False, True])
def test_decoder_guard_prevents_537_token_cache_doubling(device, attention, merged):
    if device == "cuda" and "SWITCH_TEST_CUDA" not in os.environ:
        pytest.skip("explicit SWITCH_TEST_CUDA device required")
    if device == "cuda":
        torch.cuda.set_device(int(os.environ["SWITCH_TEST_CUDA"]))
    model, _, _ = olmo(torch.bfloat16 if device == "cuda" else torch.float32, use_cache=True)
    model = model.to(device)
    if merged:
        model = model.merge_and_unload().eval()
        model.requires_grad_(True)
    model.set_attn_implementation(attention)
    reference = copy.deepcopy(model)
    ids = (torch.arange(537, device=device) % 62 + 1).unsqueeze(0)
    mask = torch.ones_like(ids)
    expected = reference(ids, attention_mask=mask, use_cache=False).logits
    expected.float().square().mean().backward()
    memory.checkpoint_decoder_layers(model)
    # No caller-side use_cache override: the decoder must guard itself.
    result = model(ids, attention_mask=mask)
    result.logits.float().square().mean().backward()
    assert result.past_key_values is None
    torch.testing.assert_close(result.logits, expected)
    for (name, actual), (_, wanted) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if wanted.grad is not None:
            torch.testing.assert_close(actual.grad, wanted.grad, msg=name)
    assert model.config.use_cache is True
    assert not any(module.training for module in model.modules())
    decoder = model.get_base_model().model if hasattr(model, "get_base_model") else model.model
    forward = decoder.forward
    memory.checkpoint_decoder_layers(model)
    assert decoder.forward == forward
    with torch.no_grad():
        cached = model(ids[:, :5], use_cache=True).past_key_values
        generated = model.generate(ids[:, :5], max_new_tokens=2, do_sample=False, use_cache=True,
                                   return_dict_in_generate=True, pad_token_id=0, eos_token_id=None)
    assert cached.get_seq_length() == 5
    assert generated.sequences.shape == (1, 7) and generated.past_key_values is not None
    with pytest.raises(ValueError, match="past_key_values"):
        model(ids[:, :1], past_key_values=cached, use_cache=False)
    assert cached.get_seq_length() == 5


@pytest.mark.parametrize("logit_chunk_tokens", [0, 16])
def test_selected_prefix_gradient_with_default_kv_cache(logit_chunk_tokens, monkeypatch):
    import copy
    import grads

    monkeypatch.setattr(grads, "LOGIT_CHUNK_TOKENS", logit_chunk_tokens)
    model, rows, _ = olmo(torch.float32, use_cache=True)
    model = model.merge_and_unload().eval()
    reference = copy.deepcopy(model)
    reference.config.use_cache = False
    spec = grads.ProjectionSpec(dim=64)
    weights = [torch.full((row["input_ids"].numel() - row["resp_start"],), advantage)
               for row, advantage in zip(rows, [1., -1.], strict=True)]
    expected = grads.prompt_gradient(reference, grads.grad_params(reference, 2), rows, weights, spec,
                                     micro_batch=1)
    params = grads.grad_params(model, 2)
    memory.checkpoint_decoder_layers(model)
    actual = grads.prompt_gradient(model, params, rows, weights, spec, micro_batch=1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual.norm() > 0
    assert model.config.use_cache is True
    assert not any(module.training for module in model.modules())
    for actual_logps, expected_logps in zip(
        grads.sequence_logprobs_batch(model, rows, micro_batch=2),
        grads.sequence_logprobs_batch(reference, rows, micro_batch=2), strict=True,
    ):
        torch.testing.assert_close(actual_logps, expected_logps)
    with torch.no_grad():
        generated = model.generate(rows[0]["input_ids"][:5].unsqueeze(0), max_new_tokens=2,
                                   do_sample=False, use_cache=True, return_dict_in_generate=True,
                                   pad_token_id=0, eos_token_id=None)
    assert generated.sequences.shape == (1, 7)
    assert generated.past_key_values is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpointed_olmo_gradient_matches_eval_with_dropout_disabled(dtype):
    model, rows, direction = olmo(dtype)
    before = backend.exact_directional(model, rows, direction)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    assert memory.checkpoint_decoder_layers(model) == 4
    after = backend.exact_directional(model, rows, direction)
    assert torch.allclose(before, after, atol=1e-7, rtol=1e-5), (before, after)
    assert not any(module.training for module in model.modules())
    assert all(torch.equal(original[n], p) for n, p in model.named_parameters())
    forwards = [layer.forward for layer in model.get_base_model().model.layers]
    memory.checkpoint_decoder_layers(model)
    assert forwards == [layer.forward for layer in model.get_base_model().model.layers]


def test_decoder_checkpointing_reduces_saved_activation_storage():
    model, rows, _ = olmo(torch.float32)
    parameter_storage = {p.untyped_storage().data_ptr() for p in model.parameters()}
    def saved_bytes():
        saved = []
        def pack(tensor):
            if tensor.untyped_storage().data_ptr() not in parameter_storage:
                saved.append(tensor.numel() * tensor.element_size())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            values = backend.response_logps(model, rows[0])
        del values
        return sum(saved)
    original = saved_bytes()
    memory.checkpoint_decoder_layers(model)
    reduced = saved_bytes()
    assert reduced < original / 3, (original, reduced)


def test_unchanged_oom_is_not_retried_on_another_node(tmp_path, monkeypatch):
    import low_order_experiment as low
    core, base = memory.core, memory.base
    core.atomic_json(tmp_path / "experiment.json", {"derivative": "autograd"})
    binding = {"experiment_sha256": base.digest(tmp_path / "experiment.json"), "worker_sha256": base.digest(memory.HERE)}
    core.atomic_json(tmp_path / "memory-score-0.json", {"binding": binding, "oom": True, "prompt": 356})
    monkeypatch.setattr(low, "verify", lambda *a, **k: {"derivative": "autograd"})
    monkeypatch.setattr(memory, "install_backend", lambda *a: pytest.fail("known OOM reloaded model"))
    assert memory.worker(tmp_path, "score", 0) == 2
