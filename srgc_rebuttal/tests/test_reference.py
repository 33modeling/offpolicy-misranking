import copy
import unittest

import numpy as np

from srgc_rebuttal.objectives import (CountSketch, grpo_advantages, grpo_loss,
                                        loo_advantages, rloo_loss, score_gradient)
from srgc_rebuttal.plan import DEFAULT_PLAN, load_plan, validate_inputs
from srgc_rebuttal.srgc import (Config, Engine, TemporalRule, cached_sr_set,
                                   cosine_scores, gradient_contrast)
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


class MathTests(unittest.TestCase):
    def test_full_sets_not_training_top_four_and_not_cosines(self):
        on, sr = [f"o{i}" for i in range(40)], [f"s{i}" for i in range(40)]
        gradients = {i: np.array([10., 0.]) if n < 4 else np.array([-2., 0.])
                     for n, i in enumerate(on)}
        gradients.update({i: np.array([1., 0.]) for i in sr})
        # On full mean=-0.8, SR mean=1, v=2 -> -3.6. Top4-only would be +18.
        self.assertAlmostEqual(gradient_contrast(on, sr, gradients, np.array([2., 0.])), -3.6)
        self.assertAlmostEqual(gradient_contrast(on, sr, gradients, np.array([6., 0.])), -10.8)

    def test_overlap_cancels_and_invalid_inputs_rejected(self):
        g = {"shared": np.array([1e10, 0.]), "a": np.array([3., 0.]), "b": np.array([1., 0.])}
        self.assertEqual(gradient_contrast(["shared", "a"], ["shared", "b"], g, np.array([2., 0.])), 2)
        with self.assertRaises(ValueError):
            gradient_contrast(["a"], ["b", "shared"], g, np.ones(2))
        with self.assertRaises(ValueError):
            gradient_contrast(["a", "a"], ["b", "shared"], g, np.ones(2))
        g["a"][0] = np.nan
        with self.assertRaises(ValueError):
            gradient_contrast(["a"], ["b"], g, np.ones(2))

    def test_zero_norm_cosine_and_cached_success_rate(self):
        np.testing.assert_array_equal(cosine_scores(np.array([[0., 0.], [2., 0.]]), np.array([1., 0.])), [0, 1])
        np.testing.assert_array_equal(cosine_scores(np.ones((2, 2)), np.zeros(2)), [0, 0])
        cache = {"balanced": [0, 1] * 4, "easy": [1] * 8, "hard": [0] * 8}
        self.assertEqual(cached_sr_set(list(cache), cache, 1, 3), ("balanced",))

    def test_loo_subgroups_and_equal_rewards(self):
        rewards = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
        np.testing.assert_allclose(loo_advantages(rewards, 4), [-1/3, -1/3, -1/3, 1, -1, 1/3, 1/3, 1/3])
        self.assertFalse(np.allclose(loo_advantages(rewards, 4), loo_advantages(rewards, 8)))
        np.testing.assert_array_equal(loo_advantages(np.ones(8), 4), np.zeros(8))

    def test_scoring_sums_tokens_and_training_normalization_differs(self):
        rewards = np.array([0., 1.])
        mask = np.array([[1, 0], [1, 1]], dtype=bool)
        gradients = np.array([[[1.], [999.]], [[2.], [3.]]])
        np.testing.assert_allclose(score_gradient(gradients, rewards, mask, 2), [2.])
        logps = np.array([[-1., 0.], [-2., -3.]])
        self.assertEqual(rloo_loss(logps, rewards, mask), 2.)
        # Finite-difference the pre-update GRPO objective: per-response token average.
        epsilon = 1e-6
        old = logps.copy()
        derivative = (grpo_loss(logps + epsilon * gradients[:, :, 0], old, rewards, mask) -
                      grpo_loss(logps - epsilon * gradients[:, :, 0], old, rewards, mask)) / (2 * epsilon)
        expected = -np.mean(grpo_advantages(rewards) * np.array([1., 2.5]))
        self.assertAlmostEqual(derivative, expected, places=7)

    def test_fixed_projection_is_linear(self):
        p = CountSketch(7, 3, seed=5)
        a, b = np.arange(7.), np.arange(7.)[::-1]
        np.testing.assert_allclose(p(a - b), p(a) - p(b))
        np.testing.assert_array_equal(p(a), CountSketch(7, 3, seed=5)(a))


class TemporalTests(unittest.TestCase):
    def decisions(self, values):
        rule = TemporalRule(25)
        result = []
        for i, value in enumerate(values, 1):
            result.append(rule.observe(i * 25, value))
        return result

    def test_two_negative(self):
        self.assertEqual(self.decisions([-1, -0.1]), [False, True])

    def test_three_check_sign_and_equality(self):
        self.assertEqual(self.decisions([-2, 1, -2]), [False, False, True])
        self.assertEqual(self.decisions([-2, 5, -2, -1]), [False, False, False, True])
        self.assertEqual(self.decisions([-2, 4, -2]), [False, False, False])
        self.assertEqual(self.decisions([-1, 0, -1]), [False, False, True])

    def test_not_cumulative_and_reset_after_nonnegative_third(self):
        self.assertEqual(self.decisions([1000, -1, -1]), [False, False, True])
        self.assertEqual(self.decisions([-10, 0, 0, -1]), [False] * 4)

    def test_missing_checks_and_invalid_values(self):
        rule = TemporalRule()
        self.assertFalse(rule.observe(25, -1))
        self.assertFalse(rule.observe(75, -1))
        with self.assertRaises(ValueError):
            rule.observe(75, -1)
        with self.assertRaises(ValueError):
            rule.observe(100, float("nan"))
        self.assertTrue(rule.observe(100, -1))
        with self.assertRaises(RuntimeError):
            rule.observe(125, -1)


class ScriptedBackend:
    gpu_count = 0

    def __init__(self):
        self.score_calls, self.training = [], []
        self.parameter, self.optimizer_steps = 0, 0

    def score_gradients(self, ids, **kwargs):
        self.score_calls.append((tuple(ids), dict(kwargs)))
        # The fixed SR first40 have higher gradients than other candidates.
        return {i: np.array([2. if i.startswith("s") else 0., 0.]) for i in ids}

    def train(self, ids, **kwargs):
        self.training.append((tuple(ids), kwargs))
        self.parameter += 1
        self.optimizer_steps += 1
        return {"sample_reward": 0.5}

    def state_dict(self):
        return {"parameter": self.parameter, "optimizer_steps": self.optimizer_steps}

    def load_state_dict(self, state):
        self.parameter, self.optimizer_steps = state["parameter"], state["optimizer_steps"]

    def synchronize(self):
        pass


class EngineTests(unittest.TestCase):
    def make_engine(self, arm="switch", step=25):
        ids = [f"s{i}" for i in range(40)] + [f"o{i}" for i in range(360)]
        cache = {i: ([0, 1] * 4 if i.startswith("s") else [1] * 8) for i in ids}
        backend = ScriptedBackend()
        engine = Engine(backend, ids, ["sv"], cache, arm=arm,
                        config=Config(projection_dim=2), step=step)
        return engine, backend

    def test_refresh_both_sets_every_25_updates_and_reuse_top_four(self):
        engine, backend = self.make_engine()
        record = engine.update()
        self.assertEqual(len(record["on_ids"]), 40)
        self.assertEqual(len(record["sr_ids"]), 40)
        self.assertEqual(len(record["train_ids"]), 4)
        self.assertLess(record["d"], 0)
        self.assertEqual(len(backend.score_calls), 2)  # union + validation, no extra D batch
        union, kwargs = backend.score_calls[0]
        self.assertEqual(set(union), set(record["on_ids"]) | set(record["sr_ids"]))
        self.assertEqual(len(union), len(set(union)))
        self.assertEqual(kwargs["responses"], 8)
        self.assertEqual(kwargs["group_size"], 4)
        self.assertEqual(backend.score_calls[1][1]["group_size"], 8)
        self.assertNotEqual(kwargs["seed"], backend.training[0][1]["seed"])
        self.assertEqual(backend.training[0][1]["responses"], 8)
        next_record = engine.update()
        self.assertIsNone(next_record["d"])
        self.assertEqual(len(backend.score_calls), 2)
        self.assertFalse(next_record["selection_refreshed"])
        self.assertEqual(record["train_ids"], next_record["train_ids"])
        engine.run_until(50)
        self.assertEqual(len(backend.score_calls), 2)
        self.assertTrue(all(r["train_ids"] == record["train_ids"] for r in engine.history))
        renewed = engine.update()
        self.assertEqual(len(backend.score_calls), 4)
        self.assertTrue(renewed["selection_refreshed"])
        self.assertNotEqual(renewed["on_ids"], record["on_ids"])
        self.assertAlmostEqual(record["d"], record["on_mean_validation_dot"] - record["sr_mean_validation_dot"])

    def test_midblock_resume_preserves_selection_and_avoids_extra_scoring(self):
        engine, backend = self.make_engine(arm="on_policy", step=0)
        engine.run_until(13)
        state = engine.state_dict()
        restored, other = self.make_engine(arm="on_policy", step=0)
        restored.load_state_dict(state)
        engine.run_until(27)
        restored.run_until(27)
        self.assertEqual([r["train_ids"] for r in engine.history],
                         [r["train_ids"] for r in restored.history])
        self.assertEqual(len(other.score_calls), 2)  # refresh at 25, never at resume 13
        bad = copy.deepcopy(state)
        bad["active_selection"] = None
        with self.assertRaisesRegex(ValueError, "saved selection"):
            restored.load_state_dict(bad)

    def test_refresh_schedule_and_post_prefix_cost_counts(self):
        prefix, _ = self.make_engine(arm="on_policy", step=0)
        prefix.run_until(25)
        fork, backend = self.make_engine(arm="on_policy")
        fork.load_state_dict(prefix.state_dict(), fork_arm="on_policy")
        fork.run_until(275)
        refreshes = [r["checkpoint"] for r in fork.history if r["selection_refreshed"]]
        self.assertEqual(refreshes, list(range(25, 275, 25)))
        self.assertEqual(len(backend.score_calls), 20)
        self.assertEqual(sum(r["selection_refreshed"] for r in prefix.history), 1)
        self.assertEqual(len(backend.training), 250)

    def test_switch_keeps_optimizer_and_stops_scoring(self):
        engine, backend = self.make_engine()
        engine.run_until(52)
        self.assertEqual(engine.switched_at, 50)
        self.assertEqual(backend.optimizer_steps, 27)
        calls = len(backend.score_calls)
        before = engine.costs["selection_wall_seconds"]
        record = engine.update()
        self.assertEqual(len(backend.score_calls), calls)
        self.assertEqual(engine.costs["selection_wall_seconds"], before)
        self.assertEqual(record["selector"], "sr")
        self.assertTrue(set(record["train_ids"]) <= set(engine.sr_ids))
        self.assertEqual(engine.costs["selection_gpu_seconds"], 0)

    def test_sr_and_random_have_no_gradient_scoring(self):
        for arm in ["sr", "random"]:
            engine, backend = self.make_engine(arm=arm)
            engine.run_until(27)
            self.assertEqual(backend.score_calls, [])
            self.assertEqual(len(backend.training), 2)

    def test_resume_and_shared_prefix_preserve_model_optimizer_and_window(self):
        engine, _ = self.make_engine(step=0, arm="on_policy")
        engine.run_until(25)
        state = engine.state_dict()
        fork, backend = self.make_engine()
        fork.load_state_dict(state, fork_arm="switch")
        self.assertEqual((fork.step, backend.parameter, backend.optimizer_steps), (25, 25, 25))
        self.assertEqual(fork.costs["training_wall_seconds"], 0)
        fork.update()
        checkpoint = fork.state_dict()
        restored, second = self.make_engine()
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.rule.window, fork.rule.window)
        self.assertEqual(restored.update()["train_ids"], fork.update()["train_ids"])
        self.assertEqual(second.state_dict(), fork.backend.state_dict())

    def test_toy_resume_matches_uninterrupted_training(self):
        features, answers, ids, val, _, cache = make_problem(7, candidates=12, validation=3, evaluation=3)
        cfg = Config(seed=7, scoring_prompts=4, training_prompts=2, projection_dim=8,
                     selection_interval=2, check_interval=2, first_check=2)
        def make():
            backend = ToyBackend(features, answers, projection_dim=8, seed=7)
            return Engine(backend, ids, val, cache, config=cfg)
        original, resumed = make(), make()
        original.run_until(3)
        resumed.load_state_dict(original.state_dict())
        original.run_until(8)
        resumed.run_until(8)
        np.testing.assert_array_equal(original.backend.weights, resumed.backend.weights)
        np.testing.assert_array_equal(original.backend.m, resumed.backend.m)
        def without_timers(history):
            return [{k: v for k, v in r.items() if not k.endswith("_gpu_seconds")} for r in history]
        self.assertEqual(without_timers(original.history), without_timers(resumed.history))


class PlanTests(unittest.TestCase):
    def test_extra_seeds_do_not_reuse_old_seeds_and_hold_rule_fixed(self):
        plan = load_plan(DEFAULT_PLAN)
        self.assertEqual(plan["seeds"], [5, 6, 7, 8, 9])
        self.assertEqual(plan["total_updates"], 275)
        self.assertEqual(plan["arms"], ["random", "sr", "on_policy", "switch"])

    def test_split_leakage_and_validation_membership(self):
        ids = [f"p{i}" for i in range(800)]
        data = {"schema": "srgc-inputs-v1", "records": {i: {"prompt": i, "question": i, "answer": "1"} for i in ids},
                "candidate_ids": ids[:400], "validation_pool_ids": ids[400:500],
                "ranking_validation_ids": ids[400:425], "evaluation_ids": ids[500:],
                "cached_rewards": {i: [0, 1] * 4 for i in ids[:400]}, "provenance": "test"}
        validate_inputs(data)
        bad = copy.deepcopy(data)
        bad["records"]["p500"]["question"] = "  p0 "
        with self.assertRaisesRegex(ValueError, "normalized"):
            validate_inputs(bad)
        bad = copy.deepcopy(data)
        bad["ranking_validation_ids"] = ["p0"]
        with self.assertRaisesRegex(ValueError, "ranking-validation"):
            validate_inputs(bad)


if __name__ == "__main__":
    unittest.main()
