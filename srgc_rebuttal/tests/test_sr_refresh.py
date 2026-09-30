import unittest

import numpy as np

from scripts.srgc_sr_refresh import SCOPES, SRRefreshEngine, arm_name
from srgc_rebuttal.srgc import Config, Engine, stream_seed
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


class EvalToyBackend(ToyBackend):
    """ToyBackend plus the success-rate evaluation the refresh arm needs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_calls = []

    def evaluate(self, ids, *, seed, responses=8):
        self.evaluate_calls.append((tuple(ids), seed))
        return {i: float(self._rollout(i, responses, seed)[0].mean()) for i in ids}


def make_engine(scope="candidates", seed=3):
    features, answers, candidates, validation, _, cache = make_problem(seed)
    backend = EvalToyBackend(features, answers, projection_dim=64, seed=seed)
    config = Config(seed=seed, projection_dim=64)
    engine = SRRefreshEngine(backend, candidates, validation, cache, arm="sr_refresh", config=config,
                             refresh_scope=scope)
    return engine, backend


class SRRefreshTest(unittest.TestCase):
    def test_refresh_measures_fresh_success_rates_and_trains_the_four_closest_to_half(self):
        engine, backend = make_engine()
        record = engine.update()
        self.assertTrue(record["selection_refreshed"])
        self.assertEqual(len(record["refreshed_ids"]), 40)
        self.assertEqual(len(backend.evaluate_calls), 1)
        self.assertEqual(backend.evaluate_calls[0][0], tuple(record["refreshed_ids"]))
        self.assertEqual(backend.score_calls, [])  # no gradients: SR needs success rates only
        rates = record["refreshed_success_rates"]
        chosen = record["train_ids"]
        self.assertEqual(len(chosen), 4)
        worst_chosen = max(abs(rates[i] - 0.5) for i in chosen)
        best_left = min(abs(rates[i] - 0.5) for i in record["refreshed_ids"] if i not in chosen)
        self.assertLessEqual(worst_chosen, best_left)
        self.assertEqual(backend.train_calls[0], tuple(chosen))
        self.assertGreater(record["selection_gpu_seconds"] + engine.costs["selection_wall_seconds"], 0)

    def test_block_is_reused_until_the_next_refresh(self):
        engine, backend = make_engine()
        first = engine.update()
        engine.run_until(25)
        self.assertEqual(len(backend.evaluate_calls), 1)
        self.assertTrue(all(r["train_ids"] == first["train_ids"] for r in engine.history))
        renewed = engine.update()
        self.assertTrue(renewed["selection_refreshed"])
        self.assertEqual(len(backend.evaluate_calls), 2)
        self.assertNotEqual(renewed["refreshed_ids"], first["refreshed_ids"])

    def test_pool_scope_refreshes_every_candidate_and_saves_a_forty_prompt_block(self):
        engine, backend = make_engine("pool")
        record = engine.update()
        self.assertEqual(len(record["refreshed_ids"]), 400)
        self.assertEqual(len(record["ranked_ids"]), 40)
        self.assertEqual(len(engine.active_selection["on_ids"]), 40)
        self.assertTrue(set(record["train_ids"]) <= set(engine.active_selection["on_ids"]))
        self.assertEqual(arm_name("pool"), "sr_refresh-pool")
        self.assertEqual(arm_name("candidates"), "sr_refresh")

    def test_midblock_checkpoint_resumes_without_a_new_refresh(self):
        engine, backend = make_engine()
        engine.run_until(7)
        state = engine.state_dict()
        self.assertEqual(state["refresh_scope"], "candidates")
        resumed, resumed_backend = make_engine()
        resumed.load_state_dict(state)
        resumed.update()
        self.assertEqual(resumed_backend.evaluate_calls, [])
        self.assertEqual(resumed.history[-1]["train_ids"], engine.history[-1]["train_ids"])
        with self.assertRaisesRegex(ValueError, "refresh scope"):
            other, _ = make_engine("pool")
            other.load_state_dict(state)

    def test_fork_from_a_shared_prefix_starts_the_refresh_arm_at_the_prefix_step(self):
        prefix, _ = make_engine()
        prefix.arm = "on_policy"
        prefix.run_until(25)
        forked, backend = make_engine()
        forked.load_state_dict(prefix.state_dict(), fork_arm="sr_refresh")
        self.assertEqual((forked.arm, forked.step, forked.costs["training_gpu_seconds"]), ("sr_refresh", 25, 0.0))
        record = forked.update()
        self.assertTrue(record["selection_refreshed"])
        self.assertEqual(record["selection_step"], 25)

    def test_cached_scope_is_the_sr_rule_with_on_policy_batch_retention(self):
        engine, backend = make_engine("cached")
        features, answers, candidates, validation, _, cache = make_problem(3)
        sr = Engine(EvalToyBackend(features, answers, projection_dim=64, seed=3), candidates, validation, cache,
                    arm="sr", config=Config(seed=3, projection_dim=64))
        record, plain = engine.update(), sr.update()
        self.assertTrue(record["selection_refreshed"])
        self.assertEqual(record["refreshed_ids"], plain["training_candidate_ids"])  # same stream, same 40
        self.assertEqual(record["train_ids"], plain["train_ids"])  # same cached ranking of that draw
        self.assertEqual((backend.evaluate_calls, backend.score_calls), ([], []))  # no rollout, no gradient
        self.assertIsNone(record["refreshed_success_rates"])
        self.assertEqual((record["scored_distinct_prompts"], record["scoring_responses_per_prompt"]), (0, 0))
        self.assertEqual(record["selector"], "sr_hold")
        engine.run_until(25)
        sr.run_until(25)
        self.assertTrue(all(r["train_ids"] == record["train_ids"] for r in engine.history))
        self.assertGreater(len({tuple(r["train_ids"]) for r in sr.history}), 1)  # SR redraws every update
        renewed = engine.update()
        self.assertTrue(renewed["selection_refreshed"])
        self.assertEqual(backend.evaluate_calls, [])
        self.assertEqual(arm_name("cached"), "sr_hold")
        state = engine.state_dict()
        self.assertEqual(state["refresh_scope"], "cached")
        with self.assertRaisesRegex(ValueError, "refresh scope"):
            other, _ = make_engine("candidates")
            other.load_state_dict(state)

    def test_other_arms_are_unchanged(self):
        features, answers, candidates, validation, _, cache = make_problem(1)
        backend = EvalToyBackend(features, answers, projection_dim=64, seed=1)
        engine = SRRefreshEngine(backend, candidates, validation, cache, arm="sr", config=Config(seed=1, projection_dim=64))
        record = engine.update()
        self.assertEqual(record["selector"], "sr")
        self.assertEqual(backend.evaluate_calls, [])
        self.assertEqual(SCOPES, ("candidates", "pool", "cached"))


if __name__ == "__main__":
    unittest.main()
