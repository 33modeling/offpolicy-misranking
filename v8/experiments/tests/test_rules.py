import math
import random
import unittest

from helpers import HAS_CODE, coderepo  # noqa: F401
from switching.rules import (FrozenWindow, SRGCController, normalize_observations, replay, should_switch,
                             subsample_cadence)

NEG = -1.0


class PureRuleTable(unittest.TestCase):
    """Plan Code §5.4 boundary table (pure judgement on the current contiguous history)."""
    TABLE = [([], False, False, False), ([-1], True, False, False), ([-1, -1], True, True, True),
             ([-1, 0, -1], True, False, True), ([-1, 2, -1], True, False, False), ([-1, 3, -1], True, False, False),
             ([0, 0], False, False, False)]

    def test_table(self):
        for hist, r1, r2, cur in self.TABLE:
            self.assertEqual(should_switch(hist, "R1"), r1, hist)
            self.assertEqual(should_switch(hist, "R2"), r2, hist)
            self.assertEqual(should_switch(hist, "current"), cur, hist)

    def test_negative_missing_negative_resets(self):
        c = SRGCController("current", 25, 25)
        self.assertEqual(c.observe(25, -1)["action"], "keep")
        c.observe(50, None, valid=False)  # measured but invalid -> reset
        self.assertEqual(c.observe(75, -1)["action"], "keep")
        r1 = SRGCController("one_negative", 25, 25)
        self.assertEqual(r1.observe(25, -1)["action"], "switch")
        r2 = SRGCController("two_negative", 25, 25)
        r2.observe(25, -1)
        r2.observe(50, float("nan"))
        self.assertEqual(r2.observe(75, -1)["action"], "keep")

    def test_gap_resets_controller(self):
        c = SRGCController("two_negative", 25, 25)
        c.observe(25, -1)
        self.assertEqual(c.observe(75, -1)["action"], "keep")  # 50 skipped -> no confirmation

    def test_three_negative(self):
        self.assertFalse(should_switch([-1, -1], "R4"))
        self.assertTrue(should_switch([-1, -1, -1], "R4"))
        self.assertFalse(should_switch([-1, 0, -1], "R4"))

    def test_zero_is_not_negative_and_equality_does_not_switch(self):
        self.assertFalse(should_switch([-1, 1, 0], "current"))
        self.assertFalse(should_switch([-1, 2, -1], "current"))  # sum exactly 0


class Controller(unittest.TestCase):
    def test_duplicate_delivery_counts_once(self):
        c = SRGCController("two_negative", 25, 25)
        c.observe(25, -1)
        d = c.observe(25, -1)
        self.assertTrue(d.get("duplicate"))
        self.assertEqual(c.history, [-1.0])

    def test_one_way(self):
        c = SRGCController("one_negative", 25, 25)
        self.assertEqual(c.observe(25, -1)["action"], "switch")
        self.assertEqual(c.observe(50, -1)["reason"], "already_transitioned")
        self.assertEqual(c.transition_update, 25)

    def test_state_round_trip(self):
        c = SRGCController("current", 25, 25)
        c.observe(25, -1)
        c.observe(50, 0.5)
        d = SRGCController.from_state(c.state_dict())
        self.assertEqual(d.observe(75, -1)["action"], c.observe(75, -1)["action"])

    def test_unscheduled_check_rejected(self):
        with self.assertRaises(ValueError):
            SRGCController("current", 25, 25).observe(30, -1)


class Replay(unittest.TestCase):
    def test_bridge_fires_before_r2(self):
        """Codex REPLAY-R2-CENSOR: current=75, R2=150 on D=[-1,+.5,-1,+.5,-1,-1]."""
        obs = list(zip(range(25, 175, 25), [-1, 0.5, -1, 0.5, -1, -1]))
        full = dict(last_scheduled_check=250)
        self.assertEqual(replay(obs, "current", **full)["switch_update"], 75)
        self.assertEqual(replay(obs, "R2", observed_through=150, **full)["switch_update"], 150)
        # A Switch run under the current rule stops checking at 75: R2 is censored, not not_triggered.
        r = replay(obs, "R2", observed_through=75, **full)
        self.assertEqual((r["status"], r["censored_after"]), ("censored", 75))

    def test_not_triggered_requires_full_log(self):
        obs = [(s, 1.0) for s in range(25, 275, 25)]
        self.assertEqual(replay(obs, "current", last_scheduled_check=250)["status"], "not_triggered")
        self.assertEqual(replay(obs[:5], "current", last_scheduled_check=250)["status"], "censored")

    def test_codex_missing_middle_check_is_unknown(self):
        r = replay([(25, -1), (75, -1), (100, -1)], "R2", last_scheduled_check=250)
        self.assertEqual((r["status"], r["observed_through"], r["censored_after"]), ("censored", 25, 25))

    def test_codex_only_last_check_is_censored(self):
        r = replay([(250, -1)], "current", last_scheduled_check=250)
        self.assertEqual(r["status"], "censored")
        self.assertEqual(r["observed_through"], 0)

    def test_codex_conflicting_duplicate_raises(self):
        with self.assertRaises(ValueError):
            replay([(25, -1), (25, 1), (50, -1)], "R2")
        self.assertEqual(normalize_observations([(25, -1), (25, -1)]), {25: -1.0})

    def test_non_integer_and_unscheduled_steps_rejected(self):
        with self.assertRaises(ValueError):
            replay([(25.0, -1)], "R1")
        with self.assertRaises(ValueError):
            replay([(30, -1)], "R1")

    def test_known_skipped_resets(self):
        r = replay([(25, -1), (75, -1)], "R2", known_skipped=[50], last_scheduled_check=100)
        self.assertEqual(r["status"], "censored")  # 100 unobserved, no trigger after the reset
        r = replay([(25, -1), (75, -1), (100, -1)], "R2", known_skipped=[50], last_scheduled_check=100)
        self.assertEqual(r["switch_update"], 100)

    def test_invalid_value_is_known_reset(self):
        r = replay([(25, -1), (50, None), (75, -1), (100, -1)], "R2", last_scheduled_check=100)
        self.assertEqual(r["switch_update"], 100)

    def test_cadence_50_subsample(self):
        obs = [(s, -1.0 if s in (50, 100) else 1.0) for s in range(25, 275, 25)]
        sub = subsample_cadence(obs, 50, 50)
        self.assertEqual([s for s, _ in sub], [50, 100, 150, 200, 250])
        self.assertEqual(replay(sub, "R2", first_check=50, interval=50, last_scheduled_check=250)["switch_update"], 100)


@unittest.skipUnless(HAS_CODE, "code repository not available")
class FrozenParity(unittest.TestCase):
    """The v8 'current' rule must equal the code repo TemporalRule on every sequence."""

    def test_random_sequences_match_temporal_rule(self):
        TemporalRule = coderepo.srgc().TemporalRule
        rng = random.Random(0)
        for trial in range(3000):
            steps, values = [], []
            s = 25
            for _ in range(rng.randint(1, 10)):
                steps.append(s)
                values.append(rng.choice([-1.0, -0.5, 0.0, 0.3, 1.0, 2.0, -2.0]))
                s += 25 * (1 if rng.random() > 0.15 else 2)  # occasional gap
            ref, mine = TemporalRule(25), SRGCController("current", 25, 25)
            ref_at = mine_at = None
            for st, v in zip(steps, values):
                if ref_at is None and ref.observe(st, v):
                    ref_at = st
                if mine_at is None and mine.observe(st, v)["action"] == "switch":
                    mine_at = st
                if ref_at is not None:
                    break
            self.assertEqual(ref_at, mine_at, (steps, values))

    def test_frozen_window_matches_pure_function(self):
        rng = random.Random(1)
        for _ in range(3000):
            vals = [rng.choice([-1.0, 0.0, 0.5, 1.0, -0.3]) for _ in range(rng.randint(1, 8))]
            w, fired = FrozenWindow(), None
            for i, v in enumerate(vals):
                if w.push(v):
                    fired = i
                    break
            pure = next((i for i in range(len(vals)) if should_switch(vals[:i + 1], "current")), None)
            self.assertEqual(fired, pure, vals)


if __name__ == "__main__":
    unittest.main()
