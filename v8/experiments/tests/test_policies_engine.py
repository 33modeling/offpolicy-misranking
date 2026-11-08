import unittest

from helpers import HAS_CODE, toy_setup
from switching import snapshots
from switching.engine import fork_arm_for
from switching.methods import equivalent_key, method
from switching.policies import (BidirectionalPolicy, PlateauPolicy, SampledTimePolicy, TrainingRewardStats,
                                build_policy, replay_plateau)


class Plateau(unittest.TestCase):
    def stats(self, rewards, counts=None):
        s = TrainingRewardStats()
        for u, r in enumerate(rewards, start=1):
            c = (counts or [32] * len(rewards))[u - 1]
            s.add(u, r * c, c)
        return s

    def test_count_weighting(self):
        s = TrainingRewardStats()
        s.add(1, 1.0, 1)   # 1/1
        s.add(2, 0.0, 99)  # 0/99
        self.assertAlmostEqual(s.window_mean(0, 2), 1 / 100)

    def test_epsilon_is_fraction(self):
        flat = self.stats([0.5] * 100)
        p = PlateauPolicy(25, 0.005, 2, 25)
        self.assertIsNone(p.before_update(50, flat))
        d = p.before_update(75, flat)
        self.assertEqual(d["action"], "switch")  # two consecutive plateau checks at 50, 75
        rising = self.stats([0.3 + 0.001 * u for u in range(100)])  # +2.5 pp per 25 updates > 0.5 pp
        self.assertIsNone(PlateauPolicy(25, 0.005, 2, 25).before_update(50, rising))

    def test_duplicate_check_does_not_double_count(self):
        flat = self.stats([0.5] * 100)
        p = PlateauPolicy(25, 0.005, 2, 25)
        p.before_update(50, flat)
        p.before_update(50, flat)
        self.assertEqual(p.counter, 1)

    def test_missing_resets_confirmation(self):
        s = self.stats([0.5] * 100)
        s.rows[60] = (None, 0)
        p = PlateauPolicy(25, 0.005, 2, 25)
        p.before_update(50, s)
        self.assertIsNone(p.before_update(75, s))  # window (50,75] has a missing update
        self.assertEqual(p.counter, 0)

    def test_replay_censored_when_log_short(self):
        s = self.stats([0.3 + 0.01 * u for u in range(60)])
        self.assertEqual(replay_plateau(s, observed_through=60, last_check=250)["status"], "censored")


class Sampled(unittest.TestCase):
    def test_no_redraw_on_resume(self):
        p = SampledTimePolicy(100, "h", 7)
        q = SampledTimePolicy(125, "h", 7)
        with self.assertRaises(ValueError):
            q.load_state_dict(p.state_dict())

    def test_no_switch_mass(self):
        p = SampledTimePolicy(None, "h", 7)
        self.assertIsNone(p.before_update(100, None))


class Bidirectional(unittest.TestCase):
    def test_forward_reverse_with_dwell(self):
        b = BidirectionalPolicy("two_negative", 25, 25, min_dwell=50, margin=0.0)
        b.on_signal(25, -1)
        self.assertEqual(b.on_signal(50, -1)["action"], "switch_to_sr")
        self.assertEqual(b.on_signal(75, 1)["reason"], "min_dwell")
        b.on_signal(100, 1)
        self.assertEqual(b.on_signal(125, 1)["action"], "switch_to_on_policy")
        self.assertEqual(len(b.transitions), 2)

    def test_margin_band_is_no_evidence(self):
        b = BidirectionalPolicy("two_negative", 25, 25, margin=0.5)
        b.on_signal(25, -1)
        b.on_signal(50, -0.1)
        self.assertEqual(b.on_signal(75, -1)["action"], "keep")


class Methods(unittest.TestCase):
    def test_tau25_equals_sr_early_and_never_equals_on(self):
        self.assertEqual(equivalent_key(method("fixed_sr_t025"), start_update=25),
                         equivalent_key(method("sr_early"), start_update=25))
        self.assertEqual(equivalent_key(method("on_continued"), start_update=25),
                         equivalent_key(method("on_monitored"), start_update=25))
        self.assertNotEqual(equivalent_key(method("fixed_sr_t050"), start_update=25),
                            equivalent_key(method("sr_early"), start_update=25))

    def test_tuned_requires_frozen(self):
        with self.assertRaises(ValueError):
            method("fixed_sr_tuned")


@unittest.skipUnless(HAS_CODE, "code repository not available")
class EngineParity(unittest.TestCase):
    """Original method definitions are executed bit-identically (same code-repo computations)."""

    @classmethod
    def setUpClass(cls):
        cls.srgc, cls.PE, cls.data, cls.cfg, backend, cls.prefix = toy_setup()
        cls.backend = staticmethod(backend)

    def base(self, arm, H=275, engine=None, **kw):
        cls = engine or self.srgc.Engine
        e = cls(self.backend(), self.data["candidate_ids"], self.data["ranking_validation_ids"],
                self.data["cached_rewards"], arm=arm, config=self.cfg, **kw)
        e.load_state_dict(self.prefix, fork_arm=arm)
        e.run_until(H)
        return e

    def v8(self, mid, H=275, state=None, monitor=None):
        m = method(mid)
        e = self.PE(self.backend(), self.data["candidate_ids"], self.data["ranking_validation_ids"],
                    self.data["cached_rewards"], method=m, policy=build_policy(m["policy"]), config=self.cfg,
                    monitor=m["monitor"] if monitor is None else monitor)
        e.load_state_dict(state or self.prefix, fork_arm=fork_arm_for(m))
        e.run_until(H)
        return e

    def same(self, a, b):
        self.assertEqual([r["train_ids"] for r in a.history], [r["train_ids"] for r in b.history])
        self.assertEqual(snapshots.state_identity(a.state_dict())["model_hash"],
                         snapshots.state_identity(b.state_dict())["model_hash"])

    def test_arms(self):
        for arm, mid in [("switch", "srgc_current"), ("on_policy", "on_continued"), ("sr", "sr_early"),
                         ("random", "uniform_early")]:
            with self.subTest(arm=arm):
                self.same(self.base(arm), self.v8(mid))

    def test_switch_transition_update_matches(self):
        b, v = self.base("switch"), self.v8("srgc_current")
        self.assertEqual(b.switched_at, v.transition_update)
        self.assertEqual([(r["checkpoint"], r["d"]) for r in b.history if r["d"] is not None],
                         [(s["completed_update"], s["value"]) for s in v.signal_log])

    def test_monitor_does_not_perturb_training_and_reproduces_switch_d(self):
        on, mon, sw = self.base("on_policy"), self.v8("on_monitored"), self.base("switch")
        self.same(on, mon)
        d_sw = [(r["checkpoint"], r["d"]) for r in sw.history if r["d"] is not None]
        d_mon = [(s["completed_update"], s["value"]) for s in mon.signal_log]
        self.assertEqual(d_mon[:len(d_sw)], d_sw)
        self.assertEqual(len(d_mon), 10)  # checks 25..250 observed on the On trajectory

    def test_bank_branch_reproduces_switch_after_its_trigger(self):
        sw = self.base("switch")
        tau = sw.switched_at
        self.assertIsNotNone(tau)
        traj = self.v8("on_monitored", H=tau)
        branch = self.v8(f"bank_sr_t{tau:03d}", state=traj.state_dict())
        self.assertEqual([r["train_ids"] for r in sw.history if r["checkpoint"] >= tau],
                         [r["train_ids"] for r in branch.history])
        self.assertEqual(snapshots.state_identity(sw.state_dict())["model_hash"],
                         snapshots.state_identity(branch.state_dict())["model_hash"])

    def test_fixed_off_by_one(self):
        e = self.v8("fixed_sr_t050", H=60)
        sel = {r["completed_updates"]: r["selector"] for r in e.history}
        self.assertEqual(sel[50], "on_policy")   # update 50 is On
        self.assertEqual(sel[51], "sr")          # first SR update is tau+1
        self.assertFalse(any(r["checkpoint"] == 50 and r["selection_refreshed"] for r in e.history))

    def test_fixed_matches_code_repo_fixed_control_training(self):
        from scripts.srgc_switch_fixed import SwitchFixedEngine
        base = SwitchFixedEngine(self.backend(), self.data["candidate_ids"], self.data["ranking_validation_ids"],
                                 self.data["cached_rewards"], arm="switch_fixed", config=self.cfg, fixed_step=100)
        base.load_state_dict(self.prefix, fork_arm="switch_fixed")
        base.run_until(275)
        self.same(base, self.v8("fixed_sr_t100"))

    def test_two_negative_matches_code_repo_rule_engine(self):
        from scripts.srgc_switch_rules import SwitchRuleEngine
        for rule_arm, mid in (("switch_consecutive", "srgc_two_negative"), ("switch_single", "srgc_one_negative")):
            b = SwitchRuleEngine(self.backend(), self.data["candidate_ids"], self.data["ranking_validation_ids"],
                                 self.data["cached_rewards"], arm="switch_rule", config=self.cfg, rule_arm=rule_arm)
            b.load_state_dict(self.prefix, fork_arm="switch_rule")
            b.run_until(275)
            v = self.v8(mid)
            self.assertEqual(b.switched_at, v.transition_update)
            self.same(b, v)

    def test_no_switch_stays_on(self):
        m = method("srgc_three_negative")
        e = self.v8("srgc_three_negative")
        if e.transition_update is None:
            self.assertTrue(all(r["selector"] == "on_policy" for r in e.history))
        self.assertIn(m["policy"]["rule"], ("three_negative",))

    def test_uniform_destination_uses_random_batch_rule(self):
        e = self.v8("srgc_uniform_destination")
        tau = e.transition_update
        if tau is not None:
            post = [r for r in e.history if r["checkpoint"] >= tau]
            self.assertTrue(all(r["selector"] == "uniform" for r in post))

    def test_cache_refresh_failure_retains_previous(self):
        """Rank-symmetric failures after generation (commit/validation) keep the previous cache and retry."""
        from switching.cache import CacheStore
        from helpers import tmpdir
        m = method("sr_early_refreshed_k050")
        store = CacheStore(tmpdir() / "caches")
        calls = {"n": 0}
        original = store.commit

        def flaky(*args, **kw):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("injected commit failure")
            return original(*args, **kw)
        store.commit = flaky
        e = self.PE(self.backend(), self.data["candidate_ids"], self.data["ranking_validation_ids"],
                    self.data["cached_rewards"], method=m, policy=build_policy(m["policy"]), config=self.cfg,
                    cache_store=store, run_id="r")
        e.load_state_dict(self.prefix, fork_arm=fork_arm_for(m))
        initial = e.sr_ranked_ids
        e.run_until(76)
        self.assertEqual(e.sr_ranked_ids, initial)            # both attempts at 75 failed -> old cache kept
        self.assertEqual([c["event"] for c in e.cache_log], ["failed", "failed"])
        e.run_until(126)
        self.assertEqual(e.cache_log[-1]["event"], "committed")  # next boundary 125 succeeds
        self.assertEqual(e.cache_state["source_policy_update"], 125)

    def test_cache_refresh_generation_fault_propagates(self):
        """RT-01: a rank-local generation fault must end the process (a per-rank catch/retry leaves the other
        ranks inside the evaluation collective: deadlock until the NCCL timeout)."""
        m = method("sr_early_refreshed_k050")
        be = self.backend()

        def oom(ids, **kw):
            raise RuntimeError("CUDA out of memory (injected on this rank)")
        be.evaluate = oom
        e = self.PE(be, self.data["candidate_ids"], self.data["ranking_validation_ids"], self.data["cached_rewards"],
                    method=m, policy=build_policy(m["policy"]), config=self.cfg)
        e.load_state_dict(self.prefix, fork_arm=fork_arm_for(m))
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            e.run_until(76)
        self.assertEqual(e.cache_log, [])

    def test_cache_commit_runs_through_primary(self):
        """RT-11: only rank 0 commits a refreshed cache (every rank used to write it)."""
        from switching.cache import CacheStore
        from helpers import tmpdir
        m = method("sr_early_refreshed_k050")
        store = CacheStore(tmpdir() / "caches")
        state = {"inside": False, "commits": [], "primary_calls": 0}
        original = store.commit

        def commit(*args, **kw):
            state["commits"].append(state["inside"])
            return original(*args, **kw)
        store.commit = commit

        def primary(fn):
            state["primary_calls"] += 1
            state["inside"] = True
            try:
                return fn()
            finally:
                state["inside"] = False
        e = self.PE(self.backend(), self.data["candidate_ids"], self.data["ranking_validation_ids"],
                    self.data["cached_rewards"], method=m, policy=build_policy(m["policy"]), config=self.cfg,
                    cache_store=store, run_id="r", primary=primary)
        e.load_state_dict(self.prefix, fork_arm=fork_arm_for(m))
        e.run_until(76)
        self.assertEqual(state["commits"], [True])
        self.assertEqual(e.cache_log[-1]["event"], "committed")


if __name__ == "__main__":
    unittest.main()
