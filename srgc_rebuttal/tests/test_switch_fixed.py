import unittest
from unittest.mock import patch

from scripts.srgc_sr_refresh import extra_arm, make_engine as make_extra_engine
from scripts.srgc_switch_fixed import SwitchFixedEngine, fixed_step_of
from srgc_rebuttal.srgc import Config
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


def make_engine(fixed_step=100, seed=3, arm="switch_fixed"):
    features, answers, candidates, validation, _, cache = make_problem(seed)
    backend = ToyBackend(features, answers, projection_dim=64, seed=seed)
    engine = SwitchFixedEngine(backend, candidates, validation, cache, arm=arm,
                               config=Config(seed=seed, projection_dim=64), fixed_step=fixed_step)
    return engine, backend


class SwitchFixedTest(unittest.TestCase):
    def test_on_policy_through_the_fixed_step_then_sr_without_any_decision(self):
        engine, backend = make_engine(100)
        with patch("srgc_rebuttal.srgc.gradient_contrast", side_effect=AssertionError("no SR-GC check")):
            engine.run_until(100)
            self.assertIsNone(engine.switched_at)
            self.assertTrue(all(r["selector"] == "on_policy" and r["d"] is None for r in engine.history))
            scoring_before = len(backend.score_calls)
            record = engine.update()  # check at 100 governs update 101, like adaptive Switch
            self.assertEqual(record["selector"], "sr")
            self.assertTrue(record["switched"])
            self.assertEqual(engine.switched_at, 100)
            self.assertEqual(len(backend.score_calls), scoring_before + 2)  # the step-100 refresh was scored
            engine.run_until(130)
            self.assertEqual(len(backend.score_calls), scoring_before + 2)  # no scoring after the transition
            self.assertTrue(all(r["selector"] == "sr" for r in engine.history[100:]))
            self.assertTrue(all(r["d"] is None for r in engine.history))
            for refresh in (r for r in engine.history if r["selection_refreshed"]):
                self.assertEqual(refresh["sr_ids"], [])
                self.assertEqual(refresh["scored_distinct_prompts"], 40)

    def test_transition_step_must_sit_on_a_refresh_boundary(self):
        with self.assertRaisesRegex(ValueError, "multiple"):
            make_engine(110)
        self.assertEqual(fixed_step_of("switch_fixed125"), 125)
        with self.assertRaises(ValueError):
            fixed_step_of("switch_repeat")
        self.assertEqual(extra_arm("switch_fixed75"), "switch_fixed75")

    def test_all_documented_boundaries_switch_on_the_next_update(self):
        for step in (25, 75, 100, 125, 200):
            with self.subTest(step=step):
                engine, backend = make_engine(step)
                engine.run_until(step)
                self.assertEqual(engine.history[-1]["selector"], "on_policy")
                self.assertIsNone(engine.switched_at)
                calls = len(backend.score_calls)
                self.assertEqual(engine.update()["selector"], "sr")
                self.assertEqual(engine.switched_at, step)
                self.assertEqual(len(backend.score_calls), calls + 2)
                self.assertEqual(engine.update()["selector"], "sr")
                self.assertEqual(len(backend.score_calls), calls + 2)

    def test_checkpoint_roundtrip_before_and_after_the_transition(self):
        engine, _ = make_engine(50)
        engine.run_until(30)
        state = engine.state_dict()
        resumed, _ = make_engine(50)
        resumed.load_state_dict(state)
        self.assertEqual((resumed.arm, resumed.step, resumed.switched_at), ("switch_fixed", 30, None))
        engine.run_until(60)
        state = engine.state_dict()
        resumed, _ = make_engine(50)
        resumed.load_state_dict(state)
        self.assertEqual((resumed.step, resumed.switched_at), (60, 50))
        self.assertEqual(resumed.update()["selector"], "sr")
        with self.assertRaisesRegex(ValueError, "fixed transition step"):
            other, _ = make_engine(100)
            other.load_state_dict(state)

    def test_boundary_resume_and_legacy_protocol_rejection(self):
        engine, _ = make_engine(50)
        engine.run_until(50)
        state = engine.state_dict()
        resumed, _ = make_engine(50)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.update()["selector"], "sr")
        self.assertEqual(resumed.switched_at, 50)
        self.assertEqual(engine.update()["train_ids"], resumed.history[-1]["train_ids"])
        state.pop("fixed_transition_protocol")
        with self.assertRaisesRegex(ValueError, "transition protocol changed"):
            resumed.load_state_dict(state)

    def test_matches_adaptive_switch_training_at_the_same_boundary(self):
        fixed, _ = make_engine(50)
        adaptive, _ = make_engine(50, arm="switch")
        with patch("srgc_rebuttal.srgc.gradient_contrast", return_value=-1.0):
            adaptive.run_until(52)
        fixed.run_until(52)
        self.assertEqual(fixed.switched_at, adaptive.switched_at)
        self.assertEqual([r["train_ids"] for r in fixed.history],
                         [r["train_ids"] for r in adaptive.history])

    def test_fork_from_prefix_and_registry(self):
        prefix, _ = make_engine(100, arm="on_policy")
        prefix.run_until(25)
        forked, _ = make_engine(100)
        forked.load_state_dict(prefix.state_dict(), fork_arm="switch_fixed")
        self.assertEqual((forked.arm, forked.step, forked.switched_at), ("switch_fixed", 25, None))
        self.assertEqual(forked.costs["training_gpu_seconds"], 0.0)
        features, answers, candidates, validation, _, cache = make_problem(3)
        data = {"candidate_ids": candidates, "ranking_validation_ids": validation, "cached_rewards": cache}
        backend = ToyBackend(features, answers, projection_dim=64, seed=3)
        engine, label = make_extra_engine("switch_fixed125", backend, data, Config(seed=3, projection_dim=64))
        self.assertIsInstance(engine, SwitchFixedEngine)
        self.assertEqual((label, engine.fixed_step), ("switch_fixed", 125))

    def test_recorded_arms_are_unchanged(self):
        engine, _ = make_engine(100, arm="switch")
        record = engine.update()
        self.assertEqual(record["selector"], "on_policy")
        self.assertNotIn("fixed_step", record)


if __name__ == "__main__":
    unittest.main()
