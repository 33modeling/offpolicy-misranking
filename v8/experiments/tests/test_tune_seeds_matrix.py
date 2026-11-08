import unittest

from helpers import HAS_CODE, HERE, coderepo, tmpdir
from switching.tune import frozen_tuning, loso, select_dev


class Loso(unittest.TestCase):
    R = {50: {5: 0.40, 6: 0.41, 7: 0.39, 8: 0.40, 9: 0.42},
         100: {5: 0.41, 6: 0.40, 7: 0.40, 8: 0.39, 9: 0.40},
         150: {5: 0.39, 6: 0.39, 7: 0.41, 8: 0.38, 9: 0.39}}

    def test_leave_one_out_selection(self):
        res = loso(self.R, [5, 6, 7, 8, 9])
        for s, v in res["per_seed"].items():
            others = [x for x in (5, 6, 7, 8, 9) if x != s]
            means = {t: sum(self.R[t][x] for x in others) / 4 for t in self.R}
            best = max(means.values())
            self.assertEqual(v["tau_hat"], min(t for t in means if abs(means[t] - best) < 1e-12))
            self.assertEqual(v["tuned_reward"], self.R[v["tau_hat"]][s])
        self.assertIn("not IID", res["dependence_note"])

    def test_ties_choose_earlier_tau(self):
        r = {50: {1: 0.5, 2: 0.5, 3: 0.5}, 100: {1: 0.5, 2: 0.5, 3: 0.5}}
        self.assertTrue(all(v["tau_hat"] == 50 for v in loso(r, [1, 2, 3])["per_seed"].values()))

    def test_missing_seed_reported_not_dropped_silently(self):
        r = {50: {1: 0.5, 2: None, 3: 0.4}, 100: {1: 0.6, 2: 0.5, 3: 0.4}}
        res = loso(r, [1, 2, 3])
        self.assertEqual(res["missing_seeds"], [2])
        self.assertEqual(res["complete_seeds"], [1, 3])

    def test_instability_flag(self):
        r = {50: {1: 0.9, 2: 0.1, 3: 0.1}, 100: {1: 0.1, 2: 0.9, 3: 0.9}}
        self.assertTrue(loso(r, [1, 2, 3])["tau_hat_unstable"])


class DevTuning(unittest.TestCase):
    rows = [{"role": "development", "tau": t, "reward_fraction": r, "continuation_gpu_seconds": c}
            for t, r, c in [(50, 0.400, 10), (50, 0.402, 10), (100, 0.401, 8), (100, 0.4025, 8), (150, 0.38, 6), (150, 0.38, 6)]]

    def test_tolerance_then_cost_then_smaller(self):
        r = select_dev(self.rows, candidates=[50, 100, 150], key="tau", tie_tolerance_pp=0.5)
        self.assertEqual(r["selected"], 100)  # 50 and 100 within 0.5 pp -> lower mean cost (100)
        r = select_dev(self.rows, candidates=[50, 100, 150], key="tau", tie_tolerance_pp=0.0)
        self.assertEqual(r["selected"], 100)  # strictly best mean

    def test_tolerance_required_and_dev_only(self):
        with self.assertRaises(ValueError):
            select_dev(self.rows, candidates=[50], key="tau", tie_tolerance_pp=None)
        with self.assertRaises(ValueError):
            select_dev(self.rows + [dict(self.rows[0], role="test")], candidates=[50], key="tau", tie_tolerance_pp=0.1)

    def test_frozen_tuning_test_access(self):
        f = frozen_tuning(cohort="c", dev_seeds=[15], dev_eval_hash="h", fixed={"selected": 100})
        self.assertEqual(f["test_access_status"], "clean")
        g = frozen_tuning(cohort="c", dev_seeds=[15], dev_eval_hash="h", fixed={"selected": 100}, test_runs_seen=["x"])
        self.assertEqual(g["test_access_status"], "contaminated")
        self.assertNotEqual(f["tuning_hash"], g["tuning_hash"])


class SeedsAndSchedules(unittest.TestCase):
    @unittest.skipUnless(HAS_CODE, "code repository not available")
    def test_registry_excludes_inspected_seeds(self):
        from switching.seeds import choose, registry
        rows, scanned = registry(coderepo.code_root(), HERE.parents[1])
        olmo = {r["training_seed"]: r for r in rows if r["cohort"] == "olmo_math"}
        self.assertTrue(all(not olmo[s]["eligible_as_fresh_test"] for s in range(10)))
        c = choose(rows, "olmo_math")
        self.assertEqual(c["test"], [10, 11, 12, 13, 14])
        self.assertEqual(c["development"], [15, 16, 17])
        self.assertFalse(set(c["test"]) & set(c["development"]))

    def test_tau_distribution_and_sampling_fixed_before_test(self):
        from switching.bank import sample_schedules, tau_distribution
        rows = [{"policy_id": "srgc_current", "role": "development", "seed": s, "status": st, "switch_update": u}
                for s, st, u in [(15, "triggered", 50), (16, "triggered", 50), (17, "not_triggered", None)]]
        d = tau_distribution(rows)
        self.assertEqual({x["switch_update"]: round(x["probability"], 6) for x in d["support"]}, {50: 0.666667, None: 0.333333})
        a = sample_schedules(d, [10, 11], draws=2)
        b = sample_schedules(d, [10, 11], draws=2)
        self.assertEqual(a, b)  # deterministic: resume never re-draws a different tau
        with self.assertRaises(ValueError):
            tau_distribution(rows + [{"policy_id": "srgc_current", "role": "development", "seed": 18,
                                      "status": "censored", "switch_update": None}])


@unittest.skipUnless(HAS_CODE, "code repository not available")
class Matrix(unittest.TestCase):
    def test_stable_ids_and_dedupe(self):
        from switching import matrix as mx, spec
        cfg = spec.resolve("olmo_math", role="rebuttal_bank")
        parent = {"kind": "prefix", "path": "/x/prefix.pt", "sha256": "ab" * 32, "completed_update": 25, "run_id": None}
        a = mx.train_task(cfg, role="test", seed=10, method_id="fixed_sr_t025", parent=parent, horizon=275, stage="p")
        b = mx.train_task(cfg, role="test", seed=10, method_id="sr_early", parent=parent, horizon=275, stage="p")
        a2 = mx.train_task(cfg, role="test", seed=10, method_id="fixed_sr_t025", parent=parent, horizon=275, stage="p")
        self.assertEqual(a["run_id"], a2["run_id"])
        out = mx.dedupe([a, b])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["aliases"], ["sr_early"])
        c = mx.train_task(cfg, role="test", seed=10, method_id="fixed_sr_t025", parent=dict(parent, sha256="cd" * 32),
                          horizon=275, stage="p")
        self.assertNotEqual(a["run_id"], c["run_id"])  # different parent state -> different run

    def test_config_validation_fails_fast(self):
        from switching import spec
        cfg = spec.resolve("olmo_math", role="test")
        res = spec.validate(cfg, mode="test")
        self.assertFalse(res["ok"])
        self.assertTrue(any("budget.value" in e for e in res["errors"]))
        self.assertTrue(spec.validate(spec.resolve("olmo_math", role="rebuttal_bank"), mode="rebuttal_bank")["ok"])

    def test_overrides_are_visible(self):
        from switching import spec
        cfg = spec.resolve("olmo_math", role="smoke", extra={"training": {"matched_update_endpoint": 30}})
        self.assertEqual(cfg["resolved_overrides"][0]["field"], "training.matched_update_endpoint")


if __name__ == "__main__":
    unittest.main()


class FrozenBudget(unittest.TestCase):
    rows = [{"policy_id": "srgc_current", "status": "triggered", "seed": s, "run_id": f"r{s}", "switch_update": 50,
             "continuation_gpu_seconds": 10.0 + s, "cold_start_gpu_seconds": None} for s in (5, 6, 7)]

    def test_rerun_reuses_original_and_mismatch_fails(self):
        import time
        from switching.freeze import freeze_budget, write_frozen_budget
        d = tmpdir() / "budget.json"
        first, how1 = write_frozen_budget(d, freeze_budget(self.rows))
        time.sleep(1.1)
        second, how2 = write_frozen_budget(d, freeze_budget(self.rows))
        self.assertEqual((how1, how2), ("frozen", "reused"))
        self.assertEqual(second["frozen_at"], first["frozen_at"])
        self.assertEqual(second["value"], first["value"])
        changed = [dict(r, continuation_gpu_seconds=r["continuation_gpu_seconds"] + 1) for r in self.rows]
        with self.assertRaises(ValueError):
            write_frozen_budget(d, freeze_budget(changed))

    def test_budget_never_reads_evaluations(self):
        from switching.freeze import freeze_budget
        with self.assertRaises(ValueError):
            freeze_budget(self.rows, files_read=["/x/olmo_math/evaluations/a.json"])
