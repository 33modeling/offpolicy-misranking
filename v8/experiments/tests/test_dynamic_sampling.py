"""V9 X11 dynamic sampling (dynsample_early): the training batch is the first four candidates of the update's
40-candidate draw whose current-policy rollouts have mixed rewards, the filter is charged to selection, a resume
gives the same batches, and the hashes of all earlier methods are unchanged."""

import os
import unittest

import numpy as np

os.environ.setdefault("V8_FSYNC", "0")
from helpers import HAS_CODE, toy_setup  # noqa: E402


@unittest.skipUnless(HAS_CODE, "code repository not available")
class DynamicSampling(unittest.TestCase):
    def setUp(self):
        self.srgc, self.PE, self.data, self.cfg, self.backend, self.prefix = toy_setup(seed=5, prefix_updates=25)

    def engine(self, state=None, *, fork=True):
        from switching.engine import fork_arm_for
        from switching.methods import method
        m = method("dynsample_early")
        e = self.PE(self.backend(), self.data["candidate_ids"], self.data["ranking_validation_ids"],
                    self.data["cached_rewards"], method=m, config=self.cfg)
        e.load_state_dict(state or self.prefix, fork_arm=fork_arm_for(m) if fork else None)
        return e

    def draw(self, step):
        rng = np.random.default_rng(self.srgc.stream_seed(self.cfg.seed, step, "candidate-draw"))
        return list(rng.choice(self.data["candidate_ids"], self.cfg.scoring_prompts, replace=False))

    def test_batch_is_first_mixed_candidates_in_draw_order(self):
        e = self.engine()
        hist = [e.update() for _ in range(30)]
        for r in hist:
            draw = self.draw(r["checkpoint"])
            self.assertEqual(r["training_candidate_ids"], draw)
            f = r["dynamic_filter"]
            means = f["scored_means"]
            self.assertEqual(list(means), draw[:f["scored_prompts"]])  # scanned in draw order, whole chunks
            self.assertEqual(f["scored_prompts"] % 8 == 0 or f["scored_prompts"] == len(draw), True)
            mixed = [i for i in draw if i in means and 0 < means[i] < 1][:self.cfg.training_prompts]
            self.assertEqual(r["train_ids"][:len(mixed)], mixed)
            self.assertEqual(f["kept_mixed"], len(mixed))
            self.assertEqual(f["kept_mixed"] + f["filled"], self.cfg.training_prompts)
            if f["kept_mixed"] == self.cfg.training_prompts:
                # stopped at the first chunk that completed the batch
                self.assertLess(f["scored_prompts"] - 8, draw.index(mixed[-1]) + 1)
        self.assertTrue(any(r["dynamic_filter"]["kept_mixed"] == 4 for r in hist))

    def test_filter_rollouts_are_selection_cost(self):
        events = []
        e = self.engine()
        meter = getattr(e.backend, "cost_meter", None)
        if meter is None:
            self.skipTest("toy backend without a cost meter")
        old = meter.record
        meter.record = lambda ev: (events.append(ev), old(ev))
        e.update()
        phases = {ev.get("phase") for ev in events if isinstance(ev, dict)}
        self.assertIn("selection", phases)

    def test_resume_gives_same_batches(self):
        a = self.engine()
        full = [a.update()["train_ids"] for _ in range(20)]
        b = self.engine()
        for _ in range(7):
            b.update()
        c = self.engine(b.state_dict(), fork=False)
        self.assertEqual(full[7:], [c.update()["train_ids"] for _ in range(13)])

    def test_hashes_of_earlier_methods_unchanged(self):
        from switching.common import sha256_json
        from switching.methods import EXECUTION_FIELDS, equivalent_key, method, method_hash
        for mid in ("on_continued", "sr_early", "uniform_early", "fixed_sr_t050", "bank_sr_t025", "srgc_current",
                    "sr_early_refreshed_k025", "sr_hold025", "uniform_hold025"):
            m = method(mid)
            self.assertNotIn("dynamic_filter", m)
            want = {k: m.get(k) for k in EXECUTION_FIELDS}
            if m.get("hold_interval"):
                want["hold_interval"] = m["hold_interval"]
            self.assertEqual(method_hash(m), sha256_json(want))
        self.assertNotEqual(method_hash(method("dynsample_early")), method_hash(method("uniform_early")))
        self.assertNotEqual(equivalent_key(method("dynsample_early"), start_update=25),
                            equivalent_key(method("uniform_early"), start_update=25))


if __name__ == "__main__":
    unittest.main()
