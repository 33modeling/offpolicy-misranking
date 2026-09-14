import pytest

torch = pytest.importorskip("torch")

import low_order_backend as backend
import net_gate_memory_worker as memory


def olmo(dtype):
    from peft import LoraConfig, get_peft_model
    from transformers import Olmo3Config, Olmo3ForCausalLM
    torch.manual_seed(19)
    config = Olmo3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                        attention_dropout=.25, use_cache=False)
    config._attn_implementation = "eager"
    model = get_peft_model(Olmo3ForCausalLM(config).to(dtype), LoraConfig(
        r=2, lora_alpha=2, lora_dropout=.3, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.eval()
    rows = [{"input_ids": torch.arange(96).remainder(62) + 1, "resp_start": 5, "reward": 1},
            {"input_ids": torch.arange(64).remainder(62) + 1, "resp_start": 5, "reward": 0}]
    direction = {n: torch.randn_like(p) for n, p in backend.trainable_parameters(model).items()}
    return model, rows, direction


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
