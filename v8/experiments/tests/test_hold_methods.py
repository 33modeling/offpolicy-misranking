"""V9 X8 hold methods (sr_hold, uniform_hold, bank_srhold): the batch chosen at a refresh is kept for the
refresh interval, the refresh draws the same 40 candidates as an Alignment refresh at that update, a resume in
the middle of a block keeps the batch, and the hashes of all earlier methods are unchanged."""

import os
import unittest

import numpy as np

os.environ.setdefault("V8_FSYNC", "0")
from helpers import HAS_CODE, toy_setup  # noqa: E402


@unittest.skipUnless(HAS_CODE, "code repository not available")
class HoldMethods(unittest.TestCase):
    def setUp(self):
        self.srgc, self.PE, self.data, self.cfg, self.backend, self.prefix = toy_setup(seed=5, prefix_updates=25)

    def engine(self, method_id, state=None, *, fork=True):
        from switching.engine import fork_arm_for
        from switching.methods import method
        m = method(method_id)
        e = self.PE(self.backend(), self.data["candidate_ids"], self.data["ranking_validation_ids"],
                    self.data["cached_rewards"], method=m, config=self.cfg)
        e.load_state_dict(state or self.prefix, fork_arm=fork_arm_for(m) if fork else None)
        return e

    def draw(self, step):
        rng = np.random.default_rng(self.srgc.stream_seed(self.cfg.seed, step, "candidate-draw"))
        return list(rng.choice(self.data["candidate_ids"], self.cfg.scoring_prompts, replace=False))

    def check_blocks(self, hist, first, interval):
        by = {}
        for r in hist:
            by.setdefault((r["checkpoint"] - first) // interval, []).append(r)
        for rows in by.values():
            ids = {tuple(r["train_ids"]) for r in rows}
            self.assertEqual(len(ids), 1, "batch changed inside a hold block")
            start = rows[0]["checkpoint"]
            self.assertEqual(start % interval, 0)
            self.assertTrue(all(r["hold_selection_step"] == start for r in rows))
            self.assertEqual(rows[0]["training_candidate_ids"], self.draw(start))
        self.assertGreater(len({tuple(r["train_ids"]) for r in hist}), 1, "batch never refreshed")

    def test_sr_hold_keeps_batch_and_uses_alignment_draw(self):
        e = self.engine("sr_hold025")
        hist = [e.update() for _ in range(75)]
        self.check_blocks(hist, 25, 25)
        for r in hist:
            if r["checkpoint"] % 25 == 0:  # SR top four of the refresh draw
                eligible = set(self.draw(r["checkpoint"]))
                want = [i for i in e.sr_ranked_ids if i in eligible][:self.cfg.training_prompts]
                self.assertEqual(r["train_ids"], want)

    def test_uniform_hold_keeps_batch(self):
        e = self.engine("uniform_hold025")
        self.check_blocks([e.update() for _ in range(50)], 25, 25)

    def test_resume_mid_block_keeps_batch(self):
        a = self.engine("sr_hold025")
        full = [a.update() for _ in range(40)]
        b = self.engine("sr_hold025")
        for _ in range(12):
            b.update()
        state = b.state_dict()
        c = self.engine("sr_hold025", state, fork=False)
        rest = [c.update() for _ in range(28)]
        self.assertEqual([r["train_ids"] for r in full[12:]], [r["train_ids"] for r in rest])

    def test_staged_hold_branch_from_on_trajectory(self):
        on = self.engine("on_continued")
        for _ in range(25):
            on.update()
        self.assertEqual(on.step, 50)
        e = self.engine("bank_srhold_t050", on.state_dict())
        self.assertEqual(e.phase, "sr")
        self.check_blocks([e.update() for _ in range(50)], 50, 25)

    def test_missing_batch_mid_block_fails(self):
        e = self.engine("sr_hold025")
        for _ in range(5):
            e.update()
        e.held = None
        with self.assertRaises(ValueError):
            e.update()

    def test_hashes_of_earlier_methods_unchanged(self):
        from switching.common import sha256_json
        from switching.methods import EXECUTION_FIELDS, equivalent_key, method, method_hash
        for mid in ("on_continued", "sr_early", "uniform_early", "fixed_sr_t050", "bank_sr_t025", "bank_sr_t100",
                    "srgc_current", "sr_early_refreshed_k025"):
            m = method(mid)
            self.assertNotIn("hold_interval", m)
            self.assertEqual(method_hash(m), sha256_json({k: m.get(k) for k in EXECUTION_FIELDS}))
        self.assertNotEqual(method_hash(method("sr_hold025")), method_hash(method("sr_early")))
        self.assertNotEqual(equivalent_key(method("sr_hold025"), start_update=25),
                            equivalent_key(method("sr_early"), start_update=25))
        self.assertNotEqual(equivalent_key(method("bank_srhold_t050"), start_update=50),
                            equivalent_key(method("bank_sr_t050"), start_update=50))


if __name__ == "__main__":
    unittest.main()
