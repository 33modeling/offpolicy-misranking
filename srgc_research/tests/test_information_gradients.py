"""BF16 base/FP32 LoRA gradients, real GRPO updates and exact restoration."""

import hashlib
from unittest.mock import patch

import numpy as np
import pytest
import torch

from srgc_rebuttal.tests.test_torch_backend import TinyTokenizer
from srgc_rebuttal.torch_backend import TorchBackend
from srgc_research import information
from srgc_research.backend import ResearchBackend
from srgc_research.dispatch.information_gradients import (
    aligned_problem_gradients,
    problem_gradient,
)
from srgc_research.tests.test_research import equal_tree


def precision_backend(base_dtype=torch.bfloat16, adapter_dtype=torch.float32):
    from peft import LoraConfig, get_peft_model
    from transformers import Olmo3Config, Olmo3ForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = Olmo3ForCausalLM(Olmo3Config(vocab_size=128, hidden_size=32, intermediate_size=96,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=96,
        pad_token_id=0, eos_token_id=127, attention_dropout=0., layer_types=["full_attention"] * 4)).to(base_dtype)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                            lora_dropout=0., task_type="CAUSAL_LM"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(adapter_dtype)
    tokenizer = TinyTokenizer()
    tokenizer.eos_token_id = 127
    backend = ResearchBackend(model, tokenizer, {f"p{i}": {"prompt": "x", "answer": "3"} for i in range(9)},
                              lambda *_: 0., projection_dim=16, max_new_tokens=80)

    def rollout(rid, responses, seed):
        generator = torch.Generator().manual_seed(seed + int(rid[1:]))
        sequences = [torch.randint(1, 127, (3 + length,), generator=generator)
                     for length in (5, 11, 17, 23, 31, 37, 43, 53)][:responses]
        return sequences, np.array([float(i % 2) for i in range(responses)]), 3
    backend.generate_test_rollout = rollout
    return backend


def batch(backend, prompts):
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        records = backend.collect([f"p{i}" for i in range(prompts)], 8, 29)
        probe = backend.frozen_probe(["p8"], responses=8, seed=31)
        gradient = information.probe_gradient(backend, probe)
    return records, probe, gradient


def test_reported_mixed_precision_error_reproduces_then_same_checkpoint_completes():
    backend = precision_backend()
    records, probe, gradient = batch(backend, 4)
    initial = backend.state_dict()
    original_clip = torch.nn.utils.clip_grad_norm_
    with pytest.raises(FloatingPointError, match="per-problem gradients do not reconstruct"):
        information.inspect_update(backend, records, seed=29, probe=probe, probe_loss_gradient=gradient)
    equal_tree(initial, backend.state_dict())
    assert torch.nn.utils.clip_grad_norm_ is original_clip and backend.replay is None
    with aligned_problem_gradients():
        result, tensors = information.inspect_update(backend, records, seed=29, probe=probe,
                                                     probe_loss_gradient=gradient)
    equal_tree(initial, backend.state_dict())
    with backend.replaying(records):
        backend.train(list(records), responses=8, objective="grpo", seed=29)
    equal_tree(tensors["backend_after"], backend.state_dict())
    backend.load_state_dict(initial)
    assert result["metrics"]["problem_gradient_reconstruction_error_norm"] <= result["metrics"][
        "problem_gradient_reconstruction_tolerance"]
    assert result["metrics"]["problem_gradient_definition"] == "actual-GRPO-backward-contributions"
    adapter = tensors["problem_gradient_adapter"]
    assert hashlib.sha256(adapter["source"].encode()).hexdigest() == adapter["sha256"]
    assert result["metrics"]["problem_gradient_adapter_sha256"] == adapter["sha256"]
    assert hashlib.sha256(adapter["measured_inspect_source"].encode()).hexdigest() == adapter["measured_inspect_sha256"]


@pytest.mark.parametrize("base_dtype,adapter_dtype,micro,prompts", [
    (torch.float32, torch.float32, 1, 3),
    (torch.bfloat16, torch.float32, 1, 4),
    (torch.bfloat16, torch.float32, 2, 3),
    (torch.bfloat16, torch.float32, 4, 4),
    (torch.bfloat16, torch.bfloat16, 2, 4),
    (torch.float16, torch.float32, 2, 4),
])
def test_ragged_response_micro_batches_reconstruct_the_real_update(base_dtype, adapter_dtype, micro, prompts):
    backend = precision_backend(base_dtype, adapter_dtype)
    backend.logprob_micro_batch = micro
    records, probe, gradient = batch(backend, prompts)
    initial = backend.state_dict()
    with aligned_problem_gradients():
        result, tensors = information.inspect_update(backend, records, seed=29, probe=probe,
                                                     probe_loss_gradient=gradient)
    equal_tree(initial, backend.state_dict())
    assert result["metrics"]["updates"] == 1
    reconstructed = torch.stack(list(tensors["problem_loss_gradients"].values())).mean(0)
    error = float((reconstructed - tensors["vectors"]["gradient_before_clip"]).double().norm())
    assert error <= result["metrics"]["problem_gradient_reconstruction_tolerance"]
    with backend.replaying(records):
        backend.train(list(records), responses=8, objective="grpo", seed=29)
    equal_tree(tensors["backend_after"], backend.state_dict())


def test_real_normalization_mismatch_still_fails_and_restores_weights():
    backend = precision_backend()
    records, probe, gradient = batch(backend, 4)
    initial = backend.state_dict()
    from srgc_research.dispatch import information_gradients
    original = information_gradients.BackwardContributions.gradients
    def wrong(*args, **kwargs):
        return {rid: 2 * gradient for rid, gradient in original(*args, **kwargs).items()}
    with aligned_problem_gradients(), patch.object(information_gradients.BackwardContributions, "gradients", wrong), \
            pytest.raises(FloatingPointError, match="per-problem gradients do not reconstruct.*error=.*tolerance="):
        information.inspect_update(backend, records, seed=29, probe=probe, probe_loss_gradient=gradient)
    equal_tree(initial, backend.state_dict())
    assert backend.replay is None


@pytest.mark.parametrize("reward", [0., 1.])
def test_zero_advantage_and_existing_gradients_are_preserved(reward):
    backend = precision_backend()
    records, _probe, _gradient = batch(backend, 4)
    selected = {"p0": (*records["p0"][:1], [reward] * 8, records["p0"][2])}
    initial = backend.state_dict()
    for _, parameter in backend.train_parameters:
        parameter.grad = torch.full_like(parameter, 2.)
    gradient = problem_gradient(backend, selected, batch_prompts=4)
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert all(torch.equal(p.grad, torch.full_like(p, 2.)) for _, p in backend.train_parameters)
    equal_tree(initial, backend.state_dict())


def test_probe_objective_is_preserved_and_patch_is_restored_on_failure():
    backend = precision_backend()
    _records, probe, expected = batch(backend, 4)
    original_probe, original_inspect = information.probe_gradient, information.inspect_update
    with pytest.raises(RuntimeError, match="stop"), aligned_problem_gradients():
        torch.testing.assert_close(information.probe_gradient(backend, probe), expected, rtol=0, atol=0)
        raise RuntimeError("stop")
    assert information.probe_gradient is original_probe and information.inspect_update is original_inspect


def test_nonrepeatable_backward_uses_the_actual_step_not_another_forward():
    backend = precision_backend()
    records, probe, gradient = batch(backend, 4)
    initial = backend.state_dict()
    original = backend._logps_batch
    calls = []

    def nonrepeatable(sequences, start):
        values = original(sequences, start)
        if torch.is_grad_enabled():
            calls.append(len(sequences))
            # Reproduce derivatives that differ between separate executions,
            # without changing forward log probabilities or the chosen data.
            scale = 1. + .03 * len(calls)
            values = [value.detach() + scale * (value - value.detach()) for value in values]
        return values

    # The immediately preceding fix still calculated derivatives separately.
    # Reproduce its failure, then replay this same checkpoint using capture.
    original_probe = information.probe_gradient
    def separate_probe(backend, selected, *, distributed=True):
        if distributed:
            return original_probe(backend, selected, distributed=True)
        return problem_gradient(backend, selected, batch_prompts=len(records))
    with patch.object(backend, "_logps_batch", side_effect=nonrepeatable), \
            patch.object(information, "probe_gradient", separate_probe), \
            pytest.raises(FloatingPointError, match="per-problem gradients do not reconstruct"):
        information.inspect_update(backend, records, seed=29, probe=probe, probe_loss_gradient=gradient)
    equal_tree(initial, backend.state_dict())
    calls.clear()
    with patch.object(backend, "_logps_batch", side_effect=nonrepeatable), aligned_problem_gradients():
        result, tensors = information.inspect_update(backend, records, seed=29, probe=probe,
                                                     probe_loss_gradient=gradient)
        assert len(calls) == 16  # Four problems, eight responses, micro-batch two.
        calls.clear()
        with backend.replaying(records):
            backend.train(list(records), responses=8, objective="grpo", seed=29)
        equal_tree(tensors["backend_after"], backend.state_dict())
    backend.load_state_dict(initial)
    assert result["metrics"]["problem_gradient_reconstruction_error_norm"] <= result["metrics"][
        "problem_gradient_reconstruction_tolerance"]


@pytest.mark.parametrize("fault", ["backward", "optimizer", "post-logps"])
def test_actual_capture_hooks_and_replay_are_removed_on_failure(fault):
    backend = precision_backend()
    records, probe, gradient = batch(backend, 4)
    initial = backend.state_dict()
    hooks_before = [len(parameter._backward_hooks or {}) for _, parameter in backend.train_parameters]
    original_rollout = backend._rollout
    if fault == "backward":
        original = backend._logps_batch
        def broken(*args, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError("injected backward failure")
            return original(*args, **kwargs)
        context = patch.object(backend, "_logps_batch", side_effect=broken)
    elif fault == "optimizer":
        original = backend.optimizer.step
        calls = []
        def broken(*args, **kwargs):
            calls.append(1)
            result = original(*args, **kwargs)
            if len(calls) == 2:
                raise RuntimeError("injected optimizer failure")
            return result
        context = patch.object(backend.optimizer, "step", side_effect=broken)
    else:
        original = information.log_probabilities
        calls = []
        def broken(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("injected post-logps failure")
            return original(*args, **kwargs)
        context = patch.object(information, "log_probabilities", side_effect=broken)
    with aligned_problem_gradients(), context, pytest.raises(RuntimeError, match="injected"):
        information.inspect_update(backend, records, seed=29, probe=probe, probe_loss_gradient=gradient)
    equal_tree(initial, backend.state_dict())
    assert backend.replay is None and backend._rollout == original_rollout
    assert [len(parameter._backward_hooks or {}) for _, parameter in backend.train_parameters] == hooks_before
