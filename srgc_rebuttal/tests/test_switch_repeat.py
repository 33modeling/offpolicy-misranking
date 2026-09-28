import unittest
from unittest.mock import patch

import numpy as np

from scripts.srgc_switch_repeat import SwitchRepeatEngine
from scripts.srgc_sr_refresh import make_engine as make_extra_engine
from srgc_rebuttal.srgc import Config, TemporalRule
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


def make_engine(seed=3, arm="switch_repeat"):
    features, answers, candidates, validation, _, cache = make_problem(seed)
    backend = ToyBackend(features, answers, projection_dim=64, seed=seed)
    engine = SwitchRepeatEngine(backend, candidates, validation, cache, arm=arm, config=Config(seed=seed, projection_dim=64))
    return engine, backend


def force_d(engine, values):
    """Make gradient_contrast return the given D values at successive checks."""
    sequence = iter(values)
    return patch("scripts.srgc_switch_repeat.gradient_contrast", side_effect=lambda *a, **k: next(sequence)), \
        patch("srgc_rebuttal.srgc.gradient_contrast", side_effect=lambda *a, **k: next(sequence))


class SwitchRepeatTest(unittest.TestCase):
    def test_starts_like_switch_and_goes_to_sr_on_two_negative_checks(self):
        engine, backend = make_engine()
        d_patch_local, d_patch_base = force_d(engine, [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
        with d_patch_local, d_patch_base:
            engine.run_until(25)
            self.assertEqual(engine.mode, "on")
            self.assertEqual([r["selector"] for r in engine.history][:2], ["on_policy", "on_policy"])
            engine.run_until(51)  # checks at 25 (-1) and 50 (-1): transition to SR after the step-50 check
            self.assertEqual(engine.mode, "sr")
            self.assertEqual(engine.transitions, [{"step": 50, "to": "sr"}])
            self.assertEqual(engine.switched_at, 50)
            self.assertEqual(engine.history[-1]["selector"], "sr")
            # In SR mode every check still scores 40 vs 40 and can come back.
            calls_before = len(backend.score_calls)
            engine.run_until(75)
            self.assertEqual(len(backend.score_calls), calls_before)  # no scoring between checks
            engine.run_until(101)  # checks at 75 (+1) and 100 (+1): back to On-policy after step 100
            self.assertEqual(engine.mode, "on")
            self.assertEqual(engine.transitions, [{"step": 50, "to": "sr"}, {"step": 100, "to": "on"}])
            self.assertEqual(engine.switched_at, 50)
            self.assertEqual(engine.history[-1]["selector"], "on_policy")
            self.assertEqual(engine.history[-1]["selection_step"], 100)  # trains the batch scored for the check
            self.assertTrue(set(engine.history[-1]["train_ids"]) <= set(engine.history[-1]["on_ids"]))
            engine.run_until(151)  # checks at 125 (-1) and 150 (-1): to SR again
            self.assertEqual(engine.transitions[-1], {"step": 150, "to": "sr"})

    def test_return_rule_mirrors_the_forward_rule(self):
        engine, _ = make_engine()
        d_local, d_base = force_d(engine, [-1.0, -1.0, 1.0, -0.5, 1.0])
        with d_local, d_base:
            engine.run_until(51)
            self.assertEqual(engine.mode, "sr")
            engine.run_until(126)  # +1, -0.5, +1 with positive sum -> back after the step-125 check
            self.assertEqual(engine.mode, "on")
            self.assertEqual(engine.transitions[-1], {"step": 125, "to": "on"})

    def test_checkpoint_roundtrip_in_both_modes(self):
        engine, _ = make_engine()
        d_local, d_base = force_d(engine, [-1.0, -1.0, 1.0, 1.0])
        with d_local, d_base:
            engine.run_until(30)
            self.assertEqual(engine.mode, "on")
            state = engine.state_dict()
            resumed, _ = make_engine()
            resumed.load_state_dict(state)
            self.assertEqual((resumed.mode, resumed.step, resumed.transitions), ("on", 30, []))
            engine.run_until(60)
            self.assertEqual(engine.mode, "sr")
            state = engine.state_dict()
            resumed, _ = make_engine()
            resumed.load_state_dict(state)
            self.assertEqual((resumed.mode, resumed.step, resumed.transitions), ("sr", 60, [{"step": 50, "to": "sr"}]))
            self.assertEqual(resumed.back_rule.window, engine.back_rule.window)

    def test_fork_from_prefix_and_registry(self):
        prefix, _ = make_engine(arm="on_policy")
        prefix.run_until(25)
        forked, _ = make_engine()
        forked.load_state_dict(prefix.state_dict(), fork_arm="switch_repeat")
        self.assertEqual((forked.arm, forked.mode, forked.step, forked.transitions), ("switch_repeat", "on", 25, []))
        self.assertGreater(forked.costs["sr_preparation_wall_seconds"], -1)
        features, answers, candidates, validation, _, cache = make_problem(3)
        data = {"candidate_ids": candidates, "ranking_validation_ids": validation, "cached_rewards": cache}
        backend = ToyBackend(features, answers, projection_dim=64, seed=3)
        engine, label = make_extra_engine("switch_repeat", backend, data, Config(seed=3, projection_dim=64))
        self.assertIsInstance(engine, SwitchRepeatEngine)
        self.assertEqual(label, "switch_repeat")

    def test_recorded_arms_are_unchanged(self):
        engine, backend = make_engine(arm="switch")
        record = engine.update()
        self.assertEqual(record["selector"], "on_policy")
        self.assertNotIn("mode", record)


if __name__ == "__main__":
    unittest.main()
