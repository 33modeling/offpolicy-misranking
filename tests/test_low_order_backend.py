from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import low_order_backend as lb


class TinyLoRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([[.1, -.2, .3, -.1]]*4, dtype=torch.float64))
        self.disabled = False

    def forward(self, ids, attention_mask=None):
        weight = torch.zeros_like(self.lora_weight) if self.disabled else self.lora_weight
        return SimpleNamespace(logits=weight[ids])

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


def rows():
    return [{"input_ids": torch.tensor([1, 2, 2]), "resp_start": 1, "reward": 1},
            {"input_ids": torch.tensor([1, 3]), "resp_start": 1, "reward": 0}]


def test_directional_derivative_and_mean_response_length():
    model = TinyLoRA()
    direction = {"lora_weight": torch.tensor([[.3, -.1, .2, -.4]]*4, dtype=torch.float64)}
    exact = lb.exact_directional(model, rows(), direction)
    finite = lb.finite_directional(model, rows(), direction, .01, 1)
    assert torch.allclose(exact, finite, atol=1e-5, rtol=1e-4)
    logits = model.lora_weight[1].float()
    manual = direction["lora_weight"][1, 3]-(logits.softmax(0).double()*direction["lora_weight"][1]).sum()
    assert float(exact[1]) == pytest.approx(float(manual), abs=1e-7)


def test_perturbation_restores_exact_parameter_bits_even_on_error():
    model = TinyLoRA()
    original = model.lora_weight.detach().clone()
    direction = {"lora_weight": torch.ones_like(original)}
    with pytest.raises(RuntimeError), lb.perturbed(model, direction, .03):
        assert not torch.equal(original, model.lora_weight)
        raise RuntimeError("failed forward")
    assert torch.equal(original, model.lora_weight)


def test_calibration_requires_autograd_agreement(monkeypatch):
    model = TinyLoRA()
    direction = {"lora_weight": torch.tensor([[.3, -.1, .2, -.4]]*4, dtype=torch.float64)}
    result = lb.calibrate(model, rows(), direction, step=.02, micro_batch=1)
    assert result["step"] == .01
    monkeypatch.setattr(lb, "finite_directional", lambda *a: torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="calibration failed"):
        lb.calibrate(model, rows(), direction, step=.02, micro_batch=1)


def test_validation_direction_matches_sum_of_reward_gradients():
    model = TinyLoRA()
    partial = lb.validation_gradient(model, {0: rows(), 1: rows()})
    assert partial["prompts"] == 2
    assert partial["gradient_input_tokens"] == 10
    direction = lb.make_direction([partial])
    assert sum(float(x.square().sum()) for x in direction["direction"].values()) == pytest.approx(1.)
    assert all(p.grad is None for p in model.parameters())


def test_optimizer_mapping_validates_state_and_normalizes():
    partial = {"sums": {"lora_weight": torch.tensor([1., 2.])}, "prompts": 1}
    optimizer = {"param_groups": [{"params": [7], "betas": (.9, .99), "eps": 1e-8}],
                 "state": {7: {"step": 1, "exp_avg_sq": torch.tensor([.01, .04])}}}
    result = lb.make_direction([partial], optimizer)
    assert result["geometry"] == "frozen_adam_rms"
    assert torch.allclose(result["direction"]["lora_weight"], torch.ones(2)/2**.5)
    optimizer["state"][7]["exp_avg_sq"][0] = -1
    with pytest.raises(ValueError, match="second moment"):
        lb.make_direction([partial], optimizer)


def test_dense_or_low_precision_trainable_parameters_are_not_allowed():
    with pytest.raises(ValueError, match="LoRA"):
        lb.trainable_parameters(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="FP32"):
        lb.trainable_parameters(TinyLoRA().to(torch.bfloat16))


def test_zero_validation_signal_is_not_a_random_score():
    with pytest.raises(ValueError, match="zero"):
        lb.make_direction([{"sums": {"lora_weight": torch.zeros(2)}, "prompts": 1}])


def test_real_peft_lora_remains_unmerged_and_supports_directional_backend():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(19)
    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=32, attention_dropout=0., use_cache=False)
    config._attn_implementation = "eager"
    base = LlamaForCausalLM(config).cpu().float()
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.,
                                          target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.eval()
    partial = lb.validation_gradient(model, {0: rows()})
    saved = lb.make_direction([partial])
    direction = lb.device_direction(model, saved["direction"])
    original = {n: p.detach().clone() for n, p in lb.trainable_parameters(model).items()}
    exact = lb.exact_directional(model, rows(), direction)
    finite = lb.finite_directional(model, rows(), direction, .001, 1)
    assert torch.allclose(exact, finite, atol=2e-4, rtol=.03)
    assert all(torch.equal(original[n], p) for n, p in lb.trainable_parameters(model).items())
    with model.disable_adapter():
        assert model.base_model.disable_adapters is not None
    assert list(lb.trainable_parameters(model)) == list(original)
