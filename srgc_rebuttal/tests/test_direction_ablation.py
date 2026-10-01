import unittest

import numpy as np

from scripts.srgc_direction_ablation import ARMS, MODES, DirectionAblationEngine, mode_of
from scripts.srgc_direction_records import DirectionRecordMixin
from scripts.srgc_sr_refresh import make_engine as make_extra_engine
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


class RecordedOnPolicy(DirectionRecordMixin, Engine):
    pass


class SeedRecordingBackend(ToyBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeds = []

    def score_gradients(self, ids, *, responses, group_size, seed):
        self.seeds.append(("score", tuple(ids), seed))
        return super().score_gradients(ids, responses=responses, group_size=group_size, seed=seed)

    def train(self, ids, *, responses, objective, seed):
        self.seeds.append(("train", seed))
        return super().train(ids, responses=responses, objective=objective, seed=seed)


class VectorKeepingEngine(DirectionAblationEngine):
    """Keeps the projected gradients of the last refresh so tests can recompute the ranking."""

    def _vectors(self, ids, group_size, purpose):
        received = super()._vectors(ids, group_size, purpose)
        setattr(self, f"kept_{purpose}", received)
        return received


def make(mode, cls=DirectionAblationEngine, arm="direction_ablation", seed=3):
    features, answers, candidates, validation, _, cache = make_problem(seed)
    backend = SeedRecordingBackend(features, answers, projection_dim=64, seed=seed)
    config = Config(seed=seed, projection_dim=64)
    if cls is RecordedOnPolicy:
        return cls(backend, candidates, validation, cache, arm=arm, config=config), backend
    return cls(backend, candidates, validation, cache, arm=arm, config=config, mode=mode), backend


class DirectionAblationTest(unittest.TestCase):
    def test_everything_but_the_ranking_matches_on_policy(self):
        reference, reference_backend = make(None, RecordedOnPolicy, arm="on_policy")
        reference.run_until(51)
        for mode in MODES:
            with self.subTest(mode=mode):
                engine, backend = make(mode)
                engine.run_until(51)
                refreshes = [r for r in engine.history if r["selection_refreshed"]]
                self.assertEqual([r["checkpoint"] for r in refreshes], [0, 25, 50])
                for mine, theirs in zip(refreshes, (r for r in reference.history if r["selection_refreshed"])):
                    self.assertEqual(mine["on_ids"], theirs["on_ids"])  # the candidate draw is stream-determined
                    self.assertEqual(mine["validation_ids"], theirs["validation_ids"])
                    self.assertEqual(mine["sr_ids"], [])
                    self.assertEqual(mine["scored_distinct_prompts"], 40)
                    self.assertNotIn("sr_mean_norm", mine)
                    if mine["checkpoint"] == 0:  # identical policy: identical true cosines
                        self.assertEqual(mine["sr_ids"], theirs["sr_ids"])
                        self.assertEqual(mine["ranking_scores"], theirs["ranking_scores"])
                    self.assertEqual(len(mine["selected_on_ids"]), 4)
                    self.assertTrue(set(mine["selected_on_ids"]) <= set(mine["on_ids"]))
                    self.assertIn("on_mean_cos", mine)
                    self.assertIn("on_top4_dot", mine)
                    self.assertEqual(mine["ablation"], mode)
                # Same scoring calls (ids and stream seeds) and same training stream seeds: only the trained ids differ.
                self.assertEqual([s for s in backend.seeds if s[0] == "score"],
                                 [s for s in reference_backend.seeds if s[0] == "score"])
                self.assertEqual([len(s[1]) for s in backend.seeds if s[0] == "score"][1::2],
                                 [len(s[1]) for s in reference_backend.seeds if s[0] == "score"][1::2])  # validation
                self.assertEqual([s for s in backend.seeds if s[0] == "train"],
                                 [s for s in reference_backend.seeds if s[0] == "train"])
                self.assertNotEqual([r["train_ids"] for r in refreshes],
                                    [r["train_ids"] for r in reference.history if r["selection_refreshed"]])
                self.assertTrue(all(r["selector"] == f"direction_{mode}" and r["d"] is None for r in engine.history))
                self.assertTrue(all(r["train_ids"] == refreshes[0]["train_ids"] for r in engine.history[:25]))

    def test_each_mode_ranks_by_what_it_says(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                engine, _ = make(mode, VectorKeepingEngine)
                record = engine.update()
                on_ids = record["on_ids"]
                stack = np.stack([engine.kept_selection[i] for i in on_ids])
                v = np.stack([engine.kept_validation[i] for i in engine.validation]).mean(axis=0)
                scores = np.asarray(record["ablation_scores"])
                if mode == "removed":
                    self.assertTrue(np.all(scores == 0))
                elif mode == "magnitude":
                    np.testing.assert_allclose(scores, np.linalg.norm(stack, axis=1))
                else:
                    self.assertFalse(np.allclose(scores, record["ranking_scores"]))
                    self.assertTrue(np.all(np.abs(scores) <= 1 + 1e-12))
                    other, _ = make(mode, VectorKeepingEngine)
                    self.assertEqual(other.update()["ablation_scores"], record["ablation_scores"])  # seeded
                chosen = record["selected_on_ids"]
                if mode != "removed":
                    worst_chosen = min(scores[on_ids.index(i)] for i in chosen)
                    best_left = max(scores[on_ids.index(i)] for i in on_ids if i not in chosen)
                    self.assertGreaterEqual(worst_chosen, best_left)
                    self.assertGreater(worst_chosen, -1.0)
                true_top4 = [on_ids[i] for i in np.argsort(-np.asarray(record["ranking_scores"]))[:4]]
                self.assertNotEqual(sorted(chosen), sorted(true_top4))
                trained = stack[[on_ids.index(i) for i in chosen]].mean(axis=0)
                self.assertAlmostEqual(record["on_top4_dot"], float(np.dot(v, trained)))

    def test_checkpoint_roundtrip_and_protocol_guards(self):
        engine, _ = make("magnitude")
        engine.run_until(7)
        state = engine.state_dict()
        self.assertEqual(state["ablation_mode"], "magnitude")
        resumed, backend = make("magnitude")
        resumed.load_state_dict(state)
        resumed.update()
        self.assertEqual(backend.score_calls, [])  # mid-block: no new refresh
        self.assertEqual(resumed.history[-1]["train_ids"], engine.history[-1]["train_ids"])
        with self.assertRaisesRegex(ValueError, "ablation mode differs"):
            other, _ = make("removed")
            other.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            resumed.load_state_dict({k: v for k, v in state.items() if k != "ablation_protocol"})
        stale = {**state, "active_selection": None}
        with self.assertRaisesRegex(ValueError, "selection block"):
            resumed.load_state_dict(stale)

    def test_fork_from_prefix_registry_and_unchanged_recorded_arms(self):
        prefix, _ = make(None, RecordedOnPolicy, arm="on_policy")
        prefix.run_until(25)
        forked, _ = make("replaced")
        forked.load_state_dict(prefix.state_dict(), fork_arm="direction_ablation")
        self.assertEqual((forked.arm, forked.step, forked.costs["training_gpu_seconds"]), ("direction_ablation", 25, 0.0))
        record = forked.update()
        self.assertTrue(record["selection_refreshed"])
        self.assertEqual(record["selection_step"], 25)
        features, answers, candidates, validation, _, cache = make_problem(3)
        data = {"candidate_ids": candidates, "ranking_validation_ids": validation, "cached_rewards": cache}
        backend = ToyBackend(features, answers, projection_dim=64, seed=3)
        for arm in ARMS:
            engine, label = make_extra_engine(arm, backend, data, Config(seed=3, projection_dim=64))
            self.assertIsInstance(engine, DirectionAblationEngine)
            self.assertEqual((label, engine.mode), ("direction_ablation", mode_of(arm)))
        with self.assertRaises(ValueError):
            mode_of("on_policy")
        with self.assertRaises(ValueError):
            make("sideways")
        plain, _ = make("removed", arm="switch")
        record = plain.update()
        self.assertEqual(record["selector"], "on_policy")
        self.assertNotIn("ablation", record)


if __name__ == "__main__":
    unittest.main()
