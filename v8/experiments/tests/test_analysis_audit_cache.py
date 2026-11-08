import json
import unittest

from helpers import HAS_CODE, HERE, coderepo, tmpdir
from switching import analyze, report
from switching.cache import CacheStore, cache_key, refresh_due, rewards_from_means
from switching.runner import integer_counts


class Stats(unittest.TestCase):
    def test_paired_matches_scipy(self):
        from scipy import stats
        a = {5: 0.40, 6: 0.42, 7: 0.39, 8: 0.41, 9: 0.43}
        b = {5: 0.39, 6: 0.41, 7: 0.40, 8: 0.39, 9: 0.41}
        r = analyze.paired(a, b, [5, 6, 7, 8, 9])
        res = stats.ttest_rel([100 * a[s] for s in a], [100 * b[s] for s in b])
        self.assertAlmostEqual(r["p_two_sided"], res.pvalue, places=10)
        self.assertEqual((r["wins"], r["ties"], r["losses"]), (4, 0, 1))
        self.assertEqual(r["n_training_seeds"], 5)

    def test_missing_seed_is_shown(self):
        r = analyze.paired({5: 0.4, 6: None}, {5: 0.3, 6: 0.3}, [5, 6, 7])
        self.assertEqual(r["missing_seeds"], [6, 7])
        self.assertEqual(r["n_training_seeds"], 1)

    def test_stratified_sign_test_plan_numbers(self):
        """Rebuttal plan v2 §3.1: 25/38 wins -> p≈0.073 (two-sided exact binomial)."""
        from scipy import stats
        self.assertAlmostEqual(analyze.binom_two_sided(25, 38), stats.binomtest(25, 38, 0.5).pvalue, places=12)
        self.assertAlmostEqual(analyze.binom_two_sided(24, 38), stats.binomtest(24, 38, 0.5).pvalue, places=12)
        deltas = {"c1": {5: 1.0, 6: -1.0, 7: 0.0}, "c2": {5: 2.0, 6: 3.0}}
        r = analyze.stratified_sign_test(deltas)
        self.assertEqual((r["wins"], r["losses"], r["ties_excluded"]), (3, 1, 1))

    def test_loso_interval_withheld(self):
        rows = [{"policy_id": f"fixed_sr_t{t:03d}", "seed": s, "status": "triggered", "reward_fraction": 0.4 + 0.001 * (s + t / 50)}
                for t in (50, 100) for s in (5, 6, 7)] + \
               [{"policy_id": "srgc_current", "seed": s, "status": "triggered", "reward_fraction": 0.41} for s in (5, 6, 7)]
        r = analyze.loso_contrast(rows, [5, 6, 7])
        self.assertIsNone(r["primary_contrast"]["ci_low"])
        self.assertIn("not IID", r["primary_contrast"]["interval_method"])

    def test_bootstrap(self):
        self.assertIsNotNone(analyze.bootstrap_ci([1.0, 2.0, -0.5, 0.3]))

    def test_report_guard(self):
        with self.assertRaises(ValueError):
            report.guard("SR-GC is universally better", costs_partial=False, any_loss=False)
        with self.assertRaises(ValueError):
            report.guard("end-to-end savings of 20%", costs_partial=True, any_loss=False)
        report.guard("end-to-end savings of 20%", costs_partial=False, any_loss=False)


class Evaluation(unittest.TestCase):
    def test_codex_counts_never_invented(self):
        for bad in (-0.125, 1.125, 0.37, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                integer_counts({"q": bad}, 8)
        self.assertEqual(integer_counts({"a": 0.0, "b": 1.0, "c": 0.375}, 8), {"a": 0, "b": 8, "c": 3})


class Cache(unittest.TestCase):
    def test_key_changes_with_source_and_spec(self):
        base = dict(source_checkpoint_hash="m1", pool_ids=["a", "b"], prompt_hash="p", generation_spec="g",
                    verifier_hash="v", responses=8, rng_spec="r")
        k1 = cache_key(**base)["key"]
        self.assertNotEqual(k1, cache_key(**dict(base, source_checkpoint_hash="m2"))["key"])
        self.assertNotEqual(k1, cache_key(**dict(base, pool_ids=["a", "c"]))["key"])
        self.assertEqual(k1, cache_key(**dict(base, pool_ids=["b", "a"]))["key"])

    def test_atomic_commit_rejects_partial(self):
        store = CacheStore(tmpdir())
        info = cache_key(source_checkpoint_hash="m", pool_ids=["a", "b"], prompt_hash="p", generation_spec="g",
                         verifier_hash="v", responses=8, rng_spec="r")
        with self.assertRaises(ValueError):
            store.commit(info, {"a": [1] * 8}, ["a", "b"], source_policy_update=50)
        self.assertFalse(store.exists(info["key"]))
        store.commit(info, {"a": [1] * 8, "b": [0] * 8}, ["a", "b"], source_policy_update=50)
        self.assertEqual(store.load(info["key"])["manifest"]["n_completed"], 2)

    def test_refresh_cadence_plan_examples(self):
        due = lambda entry: [t for t in range(0, 300) if refresh_due(t, entry_update=entry, interval=50)]  # noqa: E731
        self.assertEqual(due(25)[:3], [75, 125, 175])
        self.assertEqual(due(100)[:3], [150, 200, 250])
        self.assertEqual([t for t in range(1, 160) if refresh_due(t, entry_update=25, interval=50, origin="global_clock")],
                         [50, 100, 150])

    def test_rewards_from_means(self):
        self.assertEqual(rewards_from_means({"a": 0.25}, 8)["a"].count(1.0), 2)
        with self.assertRaises(ValueError):
            rewards_from_means({"a": 0.3}, 8)


class Audit(unittest.TestCase):
    def test_paper_cells_from_v7_evidence(self):
        from switching.audit import paper_cells
        cells = paper_cells(HERE.parents[1])
        olmo = [c for c in cells if c["cohort"] == "olmo_math"]
        self.assertEqual(len(olmo), 20)
        self.assertTrue(all(c["responses"] == 2400 for c in olmo))
        qwen = [c for c in cells if c["cohort"] == "qwen_mbpp" and c["arm"] == "switch"]
        self.assertEqual(sorted(c["switch_update"] for c in qwen), [75, 100, 125, 200, 250])

    def test_audit_cell_statuses(self):
        from switching.audit import audit_cell
        root = tmpdir()
        seed_dir = root / "seed-5"
        seed_dir.mkdir()
        (seed_dir / "prefix-ready.json").write_text(json.dumps({"checkpoint_sha256": "p"}))
        per_q = {f"q{i}": (i % 9) / 8 for i in range(300)}
        reward = sum(per_q.values()) / 300
        ep = {"per_question_reward": per_q, "reward": reward, "total_updates": 275, "switched_at": 50,
              "prefix_checkpoint_sha256": "p", "cost_measurement_complete": True}
        (seed_dir / "switch-endpoint.json").write_text(json.dumps(ep))
        cell = {"cohort": "olmo_math", "seed": 5, "arm": "switch", "reward_fraction": reward,
                "precision": "exact_counts", "switch_update": 50}
        self.assertEqual(audit_cell(cell, root)["status"], "verified")
        self.assertEqual(audit_cell(dict(cell, reward_fraction=reward + 0.01), root)["status"], "inconsistent")
        self.assertEqual(audit_cell(dict(cell, arm="sr"), root)["status"], "missing_raw_evaluation")
        (seed_dir / "switch-endpoint.json").write_text(json.dumps(dict(ep, cost_measurement_complete=False)))
        self.assertEqual(audit_cell(cell, root)["status"], "partial_cost")
        (seed_dir / "switch-endpoint.json").write_text(json.dumps(dict(ep, prefix_checkpoint_sha256="other")))
        self.assertEqual(audit_cell(cell, root)["status"], "incomplete_provenance")

    @unittest.skipUnless(HAS_CODE, "code repository not available")
    def test_split_overlap_real_bundle(self):
        from switching.audit import split_overlap
        bundles = sorted(coderepo.code_root().glob("srgc_rebuttal/inputs/seed-5.json"))
        if not bundles:
            self.skipTest("no input bundle")
        r = split_overlap(bundles)[0]
        self.assertTrue(r["validate_inputs"], r["error"])
        self.assertEqual((r["ranking_vs_final_overlap"], r["dev_vs_final_overlap"], r["dev_vs_ranking_overlap"]), (0, 0, 0))
        self.assertEqual(r["n_dev_eval"], 50)


if __name__ == "__main__":
    unittest.main()
