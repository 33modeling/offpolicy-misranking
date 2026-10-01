import copy
import unittest
from unittest.mock import patch

import numpy as np

from srgc_rebuttal.objectives import (CountSketch, grpo_advantages, grpo_loss,
                                        loo_advantages, rloo_loss, score_gradient)
from srgc_rebuttal.plan import DEFAULT_PLAN, load_plan, validate_inputs
from srgc_rebuttal.srgc import (Config, Engine, TemporalRule, cached_sr_set,
                                   cosine_scores, gradient_contrast, stream_seed)
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


class MathTests(unittest.TestCase):
    def test_full_candidate_mean_not_selected_training_batch_and_not_cosines(self):
        on, sr = [f"o{i}" for i in range(40)], [f"s{i}" for i in range(40)]
        gradients = {i: np.array([10., 0.]) if n < 4 else np.array([-2., 0.])
                     for n, i in enumerate(on)}
        gradients.update({i: np.array([1., 0.]) for i in sr})
        # The full candidate mean gives -3.6; using only the selected four would give +18.
        self.assertAlmostEqual(gradient_contrast(on, sr, gradients, np.array([2., 0.])), -3.6)
        self.assertAlmostEqual(gradient_contrast(on, sr, gradients, np.array([6., 0.])), -10.8)
        self.assertAlmostEqual(gradient_contrast(on[:4], sr[:4], gradients, np.array([2., 0.])), 18)

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
        self.strengths = {}

    def score_gradients(self, ids, **kwargs):
        self.score_calls.append((tuple(ids), dict(kwargs)))
        # Higher-ranked SR prompts have larger gradients; all nonzero cosines tie.
        return {i: np.array([self.strengths.get(i, 1.), 0.]) for i in ids}

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
        backend.strengths = {i: float(len(ids) - n) for n, i in enumerate(engine.sr_ranked_ids)}
        return engine, backend

    def test_on_policy_never_scores_sr_comparison_prompts(self):
        engine, backend = self.make_engine(arm="on_policy")
        with patch.object(engine, "_sr_comparison", side_effect=AssertionError("no SR preview")), \
                patch("srgc_rebuttal.srgc.gradient_contrast", side_effect=AssertionError("no D")):
            engine.run_until(76)
        refreshes = [r for r in engine.history if r["selection_refreshed"]]
        self.assertEqual([r["checkpoint"] for r in refreshes], [25, 50, 75])
        for record, (ids, _) in zip(refreshes, backend.score_calls[::2], strict=True):
            self.assertEqual(tuple(record["on_ids"]), ids)
            self.assertEqual(len(ids), 40)
            self.assertEqual(record["sr_ids"], [])
            self.assertEqual(record["scored_distinct_prompts"], 40)
            self.assertIsNone(record["d"])

    def test_switch_scores_sr_only_when_a_check_is_due(self):
        engine, backend = self.make_engine(step=0)
        engine.config = Config(projection_dim=2, first_check=50, check_interval=50)
        with patch.object(engine, "_sr_comparison", wraps=engine._sr_comparison) as preview:
            engine.run_until(76)
        refreshes = [r for r in engine.history if r["selection_refreshed"]]
        self.assertEqual(preview.call_count, 1)
        for record, (ids, _) in zip(refreshes, backend.score_calls[::2], strict=True):
            if record["checkpoint"] == 50:
                self.assertEqual(len(record["sr_ids"]), 40)
                self.assertIsNotNone(record["d"])
                self.assertEqual(set(ids), set(record["on_ids"]) | set(record["sr_ids"]))
            else:
                self.assertEqual(record["sr_ids"], [])
                self.assertEqual(len(ids), 40)
                self.assertIsNone(record["d"])

    def test_legacy_sr_scoring_checkpoint_cannot_mix_with_corrected_run(self):
        engine, _ = self.make_engine(arm="on_policy")
        state = engine.state_dict()
        state["sampling_protocol"] = "random-candidate40-training4-contrast40-v2"
        with self.assertRaisesRegex(ValueError, "protocol"):
            engine.load_state_dict(state)

    def test_refresh_and_compare_40_candidates_and_40_sr_prompts(self):
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
        self.assertLessEqual(len(union), 80)
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
        selected_mean = np.mean([backend.strengths[i] for i in record["selected_on_ids"]])
        candidate_mean = np.mean([backend.strengths[i] for i in record["on_ids"]])
        self.assertAlmostEqual(record["on_mean_validation_dot"], candidate_mean)
        self.assertNotAlmostEqual(record["on_mean_validation_dot"], selected_mean)

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
        self.assertEqual(sum(len(ids) for ids, _ in backend.score_calls[::2]), 400)
        self.assertTrue(all(not r["sr_ids"] for r in fork.history if r["selection_refreshed"]))
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
        self.assertTrue(set(record["train_ids"]) <= set(engine.sr_ranked_ids))
        transition = next(r for r in engine.history if r["switched"])
        candidates = set(transition["training_candidate_ids"])
        self.assertEqual(len(candidates), 40)
        self.assertEqual(transition["train_ids"], [i for i in engine.sr_ranked_ids if i in candidates][:4])
        self.assertEqual(engine.costs["selection_gpu_seconds"], 0)

    def test_sr_and_random_have_no_gradient_scoring(self):
        for arm in ["sr", "random"]:
            engine, backend = self.make_engine(arm=arm)
            engine.run_until(27)
            self.assertEqual(backend.score_calls, [])
            self.assertEqual(len(backend.training), 2)

    def test_random_draws_distinct_40_then_random_4_from_full_pool_each_update(self):
        engine, backend = self.make_engine(arm="random")
        self.assertEqual(engine.random_ids, engine.candidates)
        engine.run_until(225)
        for record, (trained, kwargs) in zip(engine.history, backend.training, strict=True):
            step = record["checkpoint"]
            rng = np.random.default_rng(stream_seed(engine.config.seed, step, "candidate-draw"))
            expected_candidates = rng.choice(engine.candidates, 40, replace=False).tolist()
            self.assertEqual(record["training_candidate_ids"], expected_candidates)
            rng = np.random.default_rng(stream_seed(engine.config.seed, step, "random-batch-draw"))
            self.assertEqual(record["train_ids"], rng.choice(expected_candidates, 4, replace=False).tolist())
            self.assertEqual(len(set(record["training_candidate_ids"])), 40)
            self.assertTrue(set(trained) <= set(record["training_candidate_ids"]))
            self.assertEqual(list(trained), record["train_ids"])
            self.assertEqual(len(set(trained)), 4)
            self.assertEqual(record["sampling_pool_size"], 400)
            self.assertEqual(record["selector"], "random")
            self.assertFalse(record["selection_refreshed"])
            self.assertEqual(kwargs["responses"], 8)
        seen_candidates = {i for r in engine.history for i in r["training_candidate_ids"]}
        self.assertEqual(seen_candidates, set(engine.candidates))
        self.assertNotEqual(engine.history[0]["training_candidate_ids"], engine.history[1]["training_candidate_ids"])
        self.assertEqual(backend.score_calls, [])
        self.assertEqual(engine.costs["selection_wall_seconds"], 0)

    def test_random_resume_preserves_full_pool_and_per_update_draws(self):
        engine, _ = self.make_engine(arm="random")
        engine.run_until(38)
        restored, _ = self.make_engine(arm="random")
        restored.load_state_dict(engine.state_dict())
        engine.run_until(60)
        restored.run_until(60)
        self.assertEqual([r["train_ids"] for r in engine.history],
                         [r["train_ids"] for r in restored.history])
        self.assertEqual([r["training_candidate_ids"] for r in engine.history],
                         [r["training_candidate_ids"] for r in restored.history])
        self.assertEqual(engine.backend.state_dict(), restored.backend.state_dict())

    def test_random_fork_uses_full_pool_after_shared_on_policy_prefix(self):
        prefix, _ = self.make_engine(arm="on_policy", step=0)
        prefix.run_until(25)
        random, backend = self.make_engine(arm="random")
        random.load_state_dict(prefix.state_dict(), fork_arm="random")
        self.assertEqual(random.backend.state_dict(), prefix.backend.state_dict())
        random.run_until(50)
        self.assertGreater(len({i for r in random.history for i in r["train_ids"]}), 40)
        self.assertTrue(all(r["sampling_pool_size"] == 400 for r in random.history))
        self.assertEqual(backend.score_calls, [])

    def test_sr_draws_distinct_40_then_ranks_only_those_candidates(self):
        engine, _ = self.make_engine(arm="sr")
        engine.run_until(128)
        for record in engine.history:
            rng = np.random.default_rng(stream_seed(engine.config.seed, record["checkpoint"], "candidate-draw"))
            expected = rng.choice(engine.candidates, 40, replace=False).tolist()
            self.assertEqual(record["training_candidate_ids"], expected)
            self.assertEqual(len(set(expected)), 40)
            self.assertEqual(record["train_ids"], [i for i in engine.sr_ranked_ids if i in expected][:4])
            self.assertEqual(len(set(record["train_ids"])), 4)
            self.assertEqual(record["sampling_pool_size"], 400)
        self.assertTrue(any(r["train_ids"] != list(engine.sr_ranked_ids[:4]) for r in engine.history))
        self.assertNotEqual(engine.history[0]["training_candidate_ids"], engine.history[1]["training_candidate_ids"])

    def test_sr_fork_and_resume_preserve_candidate_draws_and_diagnostic_progress(self):
        prefix, _ = self.make_engine(arm="on_policy", step=0)
        prefix.run_until(25)
        used = set(prefix.used_training_ids)
        sr, _ = self.make_engine(arm="sr")
        sr.load_state_dict(prefix.state_dict(), fork_arm="sr")
        sr.run_until(37)
        self.assertEqual(sr.used_training_ids, used | {i for r in sr.history for i in r["train_ids"]})
        restored, _ = self.make_engine(arm="sr")
        restored.load_state_dict(sr.state_dict())
        sr.run_until(130)
        restored.run_until(130)
        self.assertEqual([r["train_ids"] for r in sr.history],
                         [r["train_ids"] for r in restored.history])
        self.assertEqual([r["training_candidate_ids"] for r in sr.history],
                         [r["training_candidate_ids"] for r in restored.history])
        self.assertEqual(sr.used_training_ids, restored.used_training_ids)
        self.assertEqual(sr.sampling_cycle, restored.sampling_cycle)

    def test_srgc_preview_excludes_trained_prompts_but_does_not_consume_comparison(self):
        engine, _ = self.make_engine(arm="switch")
        engine.used_training_ids = set(engine.sr_ranked_ids[:4])
        expected = list(engine.sr_ranked_ids[4:44])
        record = engine.update()
        self.assertEqual(record["sr_ids"], expected)
        self.assertEqual(engine.used_training_ids, set(engine.sr_ranked_ids[:4]) | set(record["train_ids"]))

    def test_training_candidate_draw_uses_full_pool_even_after_prior_training(self):
        for arm in ("random", "sr"):
            with self.subTest(arm=arm):
                engine, _ = self.make_engine(arm=arm)
                engine.used_training_ids = set(engine.candidates)
                record = engine.update()
                self.assertEqual(len(set(record["training_candidate_ids"])), 40)
                self.assertTrue(set(record["train_ids"]) <= set(record["training_candidate_ids"]))
                self.assertEqual(len(set(record["train_ids"])), 4)
                self.assertEqual(engine.sampling_cycle, 1)

    def test_diagnostic_partial_pool_preview_wraps_without_duplicate_or_consumption(self):
        engine, _ = self.make_engine()
        last = engine.sr_ranked_ids[-1]
        engine.used_training_ids = set(engine.candidates) - {last}
        original = set(engine.used_training_ids)
        comparison = engine._sr_comparison()
        self.assertEqual(comparison[0], last)
        self.assertEqual(len(set(comparison)), 40)
        self.assertEqual(engine.used_training_ids, original)

    def test_all_arms_share_the_same_random_candidate_draw_at_refresh(self):
        records = {}
        for arm in ("random", "sr", "on_policy", "switch"):
            engine, _ = self.make_engine(arm=arm)
            records[arm] = engine.update()
        candidates = records["on_policy"]["on_ids"]
        self.assertEqual(candidates, records["switch"]["on_ids"])
        for arm in ("random", "sr"):
            self.assertEqual(candidates, records[arm]["training_candidate_ids"])

    def test_global_no_repeat_checkpoint_cannot_resume_new_candidate_protocol(self):
        engine, _ = self.make_engine(arm="sr")
        state = engine.state_dict()
        state["sampling_protocol"] = "full-pool-without-replacement-candidate40-contrast-v1"
        with self.assertRaisesRegex(ValueError, "sampling protocol mismatch"):
            engine.load_state_dict(state)

    def test_failed_training_does_not_consume_prompts(self):
        engine, backend = self.make_engine(arm="sr")
        with patch.object(backend, "train", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                engine.update()
        self.assertEqual(engine.used_training_ids, set())
        self.assertEqual(engine.sampling_cycle, 0)
        self.assertEqual(engine.step, 25)

    def test_fixed_random_subset_checkpoint_cannot_silently_change_protocol(self):
        for arm in ("random", "on_policy"):
            with self.subTest(arm=arm):
                engine, backend = self.make_engine(arm=arm)
                legacy = engine.state_dict()
                legacy["random_ids"] = tuple(np.random.default_rng(engine.config.seed + 424243)
                                            .choice(engine.candidates, 40, replace=False))
                before = backend.state_dict()
                with self.assertRaisesRegex(ValueError, "Random sampling pool mismatch"):
                    engine.load_state_dict(legacy)
                self.assertEqual(backend.state_dict(), before)
                legacy.pop("sampling_protocol")
                with self.assertRaisesRegex(ValueError, "sampling protocol mismatch"):
                    engine.load_state_dict(legacy)

    def test_invalid_saved_sampling_progress_is_rejected(self):
        engine, backend = self.make_engine(arm="sr")
        original = engine.state_dict()
        for change in ({"used_training_ids": ["sv"]}, {"used_training_ids": ["s0", "s0"]},
                       {"sampling_cycle": -1}, {"sampling_cycle": True}):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "saved sampling progress"):
                    engine.load_state_dict({**original, **change})
                self.assertEqual(backend.state_dict(), original["backend"])

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
