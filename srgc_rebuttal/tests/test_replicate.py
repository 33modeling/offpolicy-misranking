import dataclasses
import unittest
from unittest.mock import patch

from scripts.srgc_replicate import (REPLICATE_PROTOCOL, ReplicateEngine, ReplicateSwitchFixedEngine,
                                    make_engine as make_replicate_engine, parse_replicate_arm, sampling_seed)
from scripts.srgc_sr_refresh import extra_arm, make_engine as make_extra_engine
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


class SeedRecordingBackend(ToyBackend):
    """ToyBackend that also records the stream seed of every scoring and training call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeds = []

    def score_gradients(self, ids, *, responses, group_size, seed):
        self.seeds.append(("score", seed))
        return super().score_gradients(ids, responses=responses, group_size=group_size, seed=seed)

    def train(self, ids, *, responses, objective, seed):
        self.seeds.append(("train", seed))
        return super().train(ids, responses=responses, objective=objective, seed=seed)


def problem(seed=3):
    features, answers, candidates, validation, _, cache = make_problem(seed)
    backend = SeedRecordingBackend(features, answers, projection_dim=64, seed=seed)
    return backend, candidates, validation, cache


def recorded(arm, seed=3):
    backend, candidates, validation, cache = problem(seed)
    return Engine(backend, candidates, validation, cache, arm=arm, config=Config(seed=seed, projection_dim=64)), backend


def replicate(arm, k, seed=3):
    backend, candidates, validation, cache = problem(seed)
    engine = ReplicateEngine(backend, candidates, validation, cache, arm=arm,
                             config=Config(seed=seed, projection_dim=64), replicate=k)
    return engine, backend


def prefix_state(seed=3, updates=25):
    engine, _ = recorded("on_policy", seed)
    engine.run_until(updates)
    return engine.state_dict()


class ReplicateNamingTest(unittest.TestCase):
    def test_arm_names_and_stream_seeds(self):
        self.assertEqual(parse_replicate_arm("replicate1-switch"), (1, "switch"))
        self.assertEqual(parse_replicate_arm("replicate12-switch_fixed200"), (12, "switch_fixed200"))
        self.assertIsNone(parse_replicate_arm("switch"))
        self.assertIsNone(parse_replicate_arm("replicate1-sr_refresh"))  # extra arms are not replicated
        with self.assertRaisesRegex(ValueError, "replicate 0"):
            parse_replicate_arm("replicate0-sr")
        seeds = {sampling_seed(base, k) for base in (5, 6, 7, 8, 9) for k in (1, 2, 3)}
        self.assertEqual(len(seeds), 15)
        self.assertNotIn(5, seeds)
        self.assertEqual(sampling_seed(5, 1), sampling_seed(5, 1))
        with self.assertRaises(ValueError):
            sampling_seed(5, 0)
        self.assertEqual(extra_arm("replicate2-sr"), "replicate2-sr")
        for name in ("replicate0-sr", "replicate1-sr_refresh", "replicate-sr"):
            with self.subTest(name=name), self.assertRaises(Exception):
                extra_arm(name)


class ReplicateEngineTest(unittest.TestCase):
    def test_only_the_post_fork_stream_changes(self):
        state = prefix_state()
        base, base_backend = recorded("switch")
        base.load_state_dict(state, fork_arm="switch")
        rep, rep_backend = replicate("switch", 1)
        rep.load_state_dict(state, fork_arm="switch")
        self.assertEqual(rep.sr_ranked_ids, base.sr_ranked_ids)  # the SR cache ranking is a prefix condition
        self.assertEqual(rep.validation, base.validation)
        self.assertEqual((rep.step, rep.costs["training_gpu_seconds"]), (25, 0.0))
        with patch("srgc_rebuttal.srgc.gradient_contrast", return_value=1.0):
            base.run_until(30)
            rep.run_until(30)
        self.assertNotEqual(base.history[0]["on_ids"], rep.history[0]["on_ids"])
        self.assertNotEqual(base_backend.seeds, rep_backend.seeds)
        self.assertEqual(len(base_backend.seeds), len(rep_backend.seeds))
        self.assertEqual(rep.config.seed, 3)  # the base config is restored after every update
        self.assertTrue(all(r["replicate"] == 1 for r in rep.history))
        saved = rep.state_dict()
        self.assertEqual(saved["config"]["seed"], 3)
        self.assertEqual(saved["replicate"], {"protocol": REPLICATE_PROTOCOL, "id": 1, "base_seed": 3,
                                              "sampling_seed": sampling_seed(3, 1)})

    def test_with_the_base_stream_the_mixin_reproduces_the_recorded_arm(self):
        state = prefix_state()
        base, base_backend = recorded("sr")
        base.load_state_dict(state, fork_arm="sr")
        rep, rep_backend = replicate("sr", 1)
        rep.load_state_dict(state, fork_arm="sr")
        rep._stream_config = dataclasses.replace(rep.config, seed=rep.config.seed)
        base.run_until(40)
        rep.run_until(40)
        self.assertEqual([r["train_ids"] for r in base.history], [r["train_ids"] for r in rep.history])
        self.assertEqual(base_backend.seeds, rep_backend.seeds)

    def test_paired_arms_of_one_replicate_share_the_stream_and_replicates_differ(self):
        state = prefix_state()
        sr1, sr1_backend = replicate("sr", 1)
        switch1, switch1_backend = replicate("switch", 1)
        sr2, _ = replicate("sr", 2)
        for engine in (sr1, switch1, sr2):
            engine.load_state_dict(state, fork_arm=engine.arm)
        with patch("srgc_rebuttal.srgc.gradient_contrast", return_value=1.0):
            sr1.run_until(26)
            switch1.run_until(26)
            sr2.run_until(26)
        # Both arms draw their 40 candidates at update 25 from the same stream.
        self.assertEqual(sr1.history[0]["training_candidate_ids"], switch1.history[0]["on_ids"])
        self.assertNotEqual(sr1.history[0]["training_candidate_ids"], sr2.history[0]["training_candidate_ids"])
        self.assertEqual([s for kind, s in sr1_backend.seeds if kind == "train"],
                         [s for kind, s in switch1_backend.seeds if kind == "train"])

    def test_checkpoints_stay_inside_their_replicate(self):
        state = prefix_state()
        rep, _ = replicate("switch", 1)
        rep.load_state_dict(state, fork_arm="switch")
        with patch("srgc_rebuttal.srgc.gradient_contrast", return_value=1.0):
            rep.run_until(30)
        saved = rep.state_dict()
        resumed, _ = replicate("switch", 1)
        resumed.load_state_dict(saved)
        self.assertEqual((resumed.step, resumed.replicate, resumed.sampling_seed), (30, 1, sampling_seed(3, 1)))
        with patch("srgc_rebuttal.srgc.gradient_contrast", return_value=1.0):
            self.assertEqual(resumed.update()["train_ids"], rep.update()["train_ids"])
        other, _ = replicate("switch", 2)
        with self.assertRaisesRegex(ValueError, "replicate id, sampling seed or protocol"):
            other.load_state_dict(saved)
        with self.assertRaisesRegex(ValueError, "not an independent replicate"):
            other.load_state_dict(recorded("switch")[0].state_dict() | {"step": 0})
        with self.assertRaisesRegex(ValueError, "forks from the shared prefix"):
            other.load_state_dict(saved, fork_arm="switch")
        plain, _ = recorded("switch")
        plain.load_state_dict(state, fork_arm="switch")
        plain.run_until(30)
        self.assertNotIn("replicate", plain.state_dict())

    def test_fixed_control_and_registry(self):
        state = prefix_state()
        backend, candidates, validation, cache = problem()
        fixed = ReplicateSwitchFixedEngine(backend, candidates, validation, cache, arm="switch_fixed",
                                           config=Config(seed=3, projection_dim=64), fixed_step=50, replicate=1)
        fixed.load_state_dict(state, fork_arm="switch_fixed")
        with patch("srgc_rebuttal.srgc.gradient_contrast", side_effect=AssertionError("no SR-GC check")):
            fixed.run_until(52)
        self.assertEqual(fixed.switched_at, 50)
        self.assertEqual([r["selector"] for r in fixed.history[-3:]], ["on_policy", "sr", "sr"])
        self.assertEqual(fixed.state_dict()["replicate"]["id"], 1)
        data = {"candidate_ids": candidates, "ranking_validation_ids": validation, "cached_rewards": cache}
        engine, label = make_replicate_engine(2, "switch_fixed100", backend, data, Config(seed=3, projection_dim=64))
        self.assertIsInstance(engine, ReplicateSwitchFixedEngine)
        self.assertEqual((label, engine.fixed_step, engine.replicate), ("switch_fixed", 100, 2))
        engine, label = make_extra_engine("replicate3-random", backend, data, Config(seed=3, projection_dim=64))
        self.assertIsInstance(engine, ReplicateEngine)
        self.assertEqual((label, engine.arm, engine.sampling_seed), ("random", "random", sampling_seed(3, 3)))


if __name__ == "__main__":
    unittest.main()
