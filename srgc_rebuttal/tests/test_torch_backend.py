import copy
import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_HF = HAS_TORCH and all(importlib.util.find_spec(m) is not None for m in ("transformers", "peft"))

if HAS_TORCH:
    import torch
    from srgc_rebuttal.torch_backend import TorchBackend


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 7

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids.tolist()))


@unittest.skipUnless(HAS_HF, "requires optional PyTorch, transformers and peft")
class ModelBackendTests(unittest.TestCase):
    def setUp(self):
        from peft import LoraConfig, get_peft_model
        from transformers import Olmo3Config, Olmo3ForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = Olmo3ForCausalLM(Olmo3Config(vocab_size=8, hidden_size=16, intermediate_size=24,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, pad_token_id=0, eos_token_id=7,
            layer_types=["full_attention"] * 4, attention_dropout=0.0))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                               lora_dropout=0.0, task_type="CAUSAL_LM"))
        records = {f"p{i}": {"prompt": f"problem {i}", "answer": "3"} for i in range(6)}
        self.backend = TorchBackend(model, TinyTokenizer(), records,
            lambda record, response: float("3" in response.split()), projection_dim=16, max_new_tokens=2)

    def deterministic_rollout(self, prompt_id, responses, seed):
        # Mixed rewards within each 4-response subgroup and variable response lengths.
        sequences = [torch.tensor([1, 2, 3]) if i % 2 else torch.tensor([1, 2, 4, 5])
                     for i in range(responses)]
        return sequences, np.asarray([float(i % 2) for i in range(responses)]), 2

    def test_real_generation_response_count_and_reproducibility(self):
        sequences, rewards, start = self.backend._rollout("p0", 8, 19)
        second, rewards2, _ = self.backend._rollout("p0", 8, 19)
        self.assertEqual((len(sequences), start), (8, 2))
        self.assertTrue(all(3 <= len(s) <= 4 for s in sequences))
        np.testing.assert_array_equal(rewards, rewards2)
        for a, b in zip(sequences, second):
            torch.testing.assert_close(a, b)

    def test_measured_generation_scoring_training_and_evaluation_reconcile(self):
        meter = self.backend.cost_meter
        events = []
        meter.record = events.append
        for phase, operation in (
            ("selection", lambda: self.backend.score_gradients(["p0"], responses=8, group_size=4, seed=19)),
            ("training", lambda: self.backend.train(["p0"], responses=8, objective="grpo", seed=19)),
            ("evaluation", lambda: self.backend.evaluate(["p0"], responses=8, seed=19)),
        ):
            with meter.phase(phase):
                operation()
            end = events[-1]
            self.assertEqual(end["state"], "finished")
            self.assertEqual(end["counts"]["responses"], 8)
            self.assertGreater(end["stages"]["generation"]["rank_wall_seconds"], 0)
            self.assertGreater(end["stages"]["reward_verification"]["rank_wall_seconds"], 0)
            self.assertAlmostEqual(sum(s["rank_wall_seconds"] for s in end["stages"].values()), end["wall_seconds"])
            self.assertEqual(end["gpu_seconds"], 0)

    def test_dense_scoring_does_not_update_model_optimizer_or_trainability(self):
        self.backend._rollout = self.deterministic_rollout
        before = {n: p.detach().clone() for n, p in self.backend.model.named_parameters()}
        flags = [p.requires_grad for p in self.backend.model.parameters()]
        gradients = self.backend.score_gradients(["p0", "p1"], responses=8, group_size=4, seed=1)
        self.assertEqual(gradients["p0"].shape, (16,))
        self.assertGreater(np.linalg.norm(gradients["p0"]), 0)
        self.assertEqual(flags, [p.requires_grad for p in self.backend.model.parameters()])
        self.assertFalse(self.backend.optimizer.state)
        for n, p in self.backend.model.named_parameters():
            torch.testing.assert_close(before[n], p, rtol=0, atol=0)
        self.assertTrue(all("lora_" not in name for name, _ in self.backend.score_parameters))

    def test_training_updates_only_adapters_and_restores_optimizer_exactly(self):
        self.backend._rollout = self.deterministic_rollout
        frozen = {n: p.detach().clone() for n, p in self.backend.model.named_parameters() if not p.requires_grad}
        self.backend.train(["p0", "p1", "p2", "p3"], responses=8, objective="grpo", seed=5)
        saved = copy.deepcopy(self.backend.state_dict())
        self.backend.train(["p0", "p1", "p2", "p3"], responses=8, objective="grpo", seed=6)
        expected = self.backend.state_dict()
        self.backend.load_state_dict(saved)
        self.backend.train(["p0", "p1", "p2", "p3"], responses=8, objective="grpo", seed=6)
        for name, weight in self.backend.state_dict()["trainable"].items():
            torch.testing.assert_close(weight, expected["trainable"][name], rtol=0, atol=0)
        for name, p in self.backend.model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)

    def test_rloo_update_and_equal_reward_scoring(self):
        self.backend._rollout = self.deterministic_rollout
        metrics = self.backend.train(["p0", "p1", "p2", "p3"], responses=8, objective="rloo", seed=8)
        self.assertGreater(metrics["gradient_norm"], 0)
        def all_equal(prompt_id, responses, seed):
            sequences, _, start = self.deterministic_rollout(prompt_id, responses, seed)
            return sequences, np.ones(responses), start
        self.backend._rollout = all_equal
        result = self.backend.score_gradients(["p0"], responses=8, group_size=4, seed=1)
        np.testing.assert_array_equal(result["p0"], np.zeros(16))

    def test_nonfinite_training_gradient_never_reaches_optimizer(self):
        from unittest.mock import patch
        self.backend._rollout = self.deterministic_rollout
        parameter = self.backend.train_parameters[0][1]
        handle = parameter.register_hook(lambda g: torch.full_like(g, float("nan")))
        try:
            with patch.object(self.backend.optimizer, "step") as step:
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    self.backend.train(["p0"], responses=8, objective="grpo", seed=1)
                step.assert_not_called()
        finally:
            handle.remove()

    def test_nonfinite_scoring_restores_trainability_and_stops_ranking(self):
        from unittest.mock import patch
        self.backend._rollout = self.deterministic_rollout
        flags = [p.requires_grad for p in self.backend.model.parameters()]
        with patch.object(self.backend, "_project", return_value=np.full(16, np.nan)):
            with self.assertRaises(FloatingPointError):
                self.backend.score_gradients(["p0"], responses=8, group_size=4, seed=1)
        self.assertEqual(flags, [p.requires_grad for p in self.backend.model.parameters()])

    def test_microbatch_preserves_dense_gradient_and_training_update(self):
        self.backend._rollout = self.deterministic_rollout
        initial = copy.deepcopy(self.backend.state_dict())
        self.backend.logprob_micro_batch = 1
        serial = self.backend.score_gradients(["p0"], responses=8, group_size=4, seed=1)["p0"]
        self.backend.train(["p0", "p1", "p2", "p3"], responses=8, objective="grpo", seed=1)
        expected = self.backend.state_dict()
        self.backend.load_state_dict(initial)
        self.backend.logprob_micro_batch = 4
        batched = self.backend.score_gradients(["p0"], responses=8, group_size=4, seed=1)["p0"]
        np.testing.assert_allclose(batched, serial, rtol=1e-5, atol=1e-7)
        self.backend.train(["p0", "p1", "p2", "p3"], responses=8, objective="grpo", seed=1)
        for name, value in self.backend.state_dict()["trainable"].items():
            torch.testing.assert_close(value, expected["trainable"][name], rtol=1e-5, atol=1e-8)

    def test_microbatch_reduces_forward_calls_and_projects_once_per_prompt(self):
        from unittest.mock import patch
        self.backend._rollout = self.deterministic_rollout
        self.backend.logprob_micro_batch = 2
        with patch.object(self.backend, "_logps_batch", wraps=self.backend._logps_batch) as forward:
            with patch.object(self.backend, "_project", wraps=self.backend._project) as project:
                self.backend.score_gradients(["p0"], responses=8, group_size=4, seed=1)
                self.assertEqual(forward.call_count, 4)
                self.assertEqual(project.call_count, len(self.backend.score_parameters))

    def test_chunked_head_values_and_gradients_match_full_model_forward(self):
        sequences, _, start = self.deterministic_rollout("p0", 8, 1)
        self.backend.logit_chunk_tokens = 1
        params = [p for _, p in self.backend.train_parameters]
        actual = self.backend._logps_batch(sequences[:2], start)
        actual_grads = torch.autograd.grad(sum(x.sum() for x in actual), params)
        expected = []
        for sequence in sequences[:2]:
            logits = self.backend.model(input_ids=sequence[None], use_cache=False).logits[0, :-1].float()
            values = logits.log_softmax(-1).gather(-1, sequence[1:, None]).squeeze(-1)
            expected.append(values[start - 1:])
        expected_grads = torch.autograd.grad(sum(x.sum() for x in expected), params)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
