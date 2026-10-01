import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import srgc_cache_analysis as analysis
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


def toy(seed=3):
    features, answers, candidates, validation, evaluation, cache = make_problem(seed)
    backend = ToyBackend(features, answers, projection_dim=64, seed=seed)
    return backend, candidates, validation, evaluation, cache


class CacheAnalysisTest(unittest.TestCase):
    def test_predicted_sr_schedule_matches_the_engine(self):
        backend, candidates, validation, _, cache = toy()
        config = Config(seed=3, projection_dim=64)
        engine = Engine(backend, candidates, validation, cache, arm="sr", config=config)
        engine.run_until(40)
        ranked, batches = analysis.sr_schedule(candidates, cache, 3, config, 0, 40)
        self.assertEqual(ranked, engine.sr_ranked_ids)
        self.assertEqual([list(batch) for _, batch in batches], [r["train_ids"] for r in engine.history])
        self.assertEqual([step for step, _ in batches], list(range(40)))

    def test_histogram_and_half_rate_counts(self):
        _, candidates, _, _, cache = toy()
        successes = analysis.successes_of(cache, 8)
        hist = analysis.histogram(successes, 8)
        self.assertEqual(sum(hist), 400)
        self.assertEqual(hist[4], sum(1 for i in candidates if sum(cache[i]) == 4))
        with self.assertRaisesRegex(ValueError, "binary rewards"):
            analysis.successes_of({"p": [0, 1, 2, 0, 0, 0, 0, 0]}, 8)
        for value in (0.5, 1.9, "1", True, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "binary rewards"):
                analysis.successes_of({"p": [value] + [0] * 7}, 8)

    def test_receipts_do_not_invent_zero_or_choose_a_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, step, counts in (("a", 0, {}), ("b", 1, {"zero_advantage_responses": 8}),
                                       ("c", 1, {"zero_advantage_responses": 16})):
                (root / f"{name}.json").write_text(json.dumps({"phase": "training", "state": "finished",
                                                              "checkpoint": step, "counts": counts}))
            warnings = []
            self.assertEqual(analysis.zero_advantage_prompts(root, 8, 4, warnings), {0: None, 1: None})
            self.assertEqual(len(warnings), 2)
            (root / "d.json").write_text(json.dumps({"phase": "training", "state": "finished",
                                                    "checkpoint": 2, "counts": {"zero_advantage_responses": 3}}))
            with self.assertRaisesRegex(ValueError, "invalid zero-advantage"):
                analysis.zero_advantage_prompts(root, 8, 4)

    def test_history_scope_identity_and_unknown_metrics(self):
        backend, candidates, validation, _, cache = toy()
        config = Config(seed=3, projection_dim=64)
        engine = Engine(backend, candidates, validation, cache, arm="sr", config=config)
        engine.run_until(30)
        bundle = {"candidate_ids": candidates, "cached_rewards": cache}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sr-progress.json"
            def report(history, seed=3):
                path.write_text(json.dumps({"seed": seed, "arm": "sr", "history": history}))
                return analysis.analyse_seed(3, bundle, config, 25, 35, directory)
            result = report(engine.history)
            self.assertEqual(result["sr_compared"], 5)
            self.assertEqual(result["arms"]["sr"]["excluded_prefix_updates"], 25)
            self.assertEqual(result["arms"]["sr"]["missing_updates"], 5)
            self.assertEqual(len(result["updates"]), 5)
            self.assertIsNone(report(engine.history[:25])["sr_mismatch"])
            rows = [dict(r, metrics={}) for r in engine.history]
            text = analysis.summarize([report(rows)])
            self.assertIn("0/0 measured (5 unknown)", text)
            for history, seed in ((engine.history + [engine.history[-1]], 3), (engine.history, 4),
                                  ([dict(engine.history[25], train_ids=["unknown"] + engine.history[25]["train_ids"][1:])], 3)):
                with self.assertRaises(ValueError):
                    report(history, seed)
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                analysis.recorded_arms(directory)

    def test_report_covers_cache_schedule_and_recorded_arms(self):
        backend, candidates, validation, evaluation, cache = toy()
        config = Config(seed=3, projection_dim=64)
        sr = Engine(backend, candidates, validation, cache, arm="sr", config=config)
        sr.run_until(30)
        on = Engine(ToyBackend(*make_problem(3)[:2], projection_dim=64, seed=3), candidates, validation, cache,
                    arm="switch", config=config)
        on.run_until(26)  # the step-25 check records the 40-prompt SR comparison set
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "seed-3"
            (folder / "replicate-1").mkdir(parents=True)
            (folder / "sr-progress.json").write_text(json.dumps({"history": sr.history}))
            (folder / "switch-progress.json").write_text(json.dumps({"history": on.history}))
            (folder / "replicate-1" / "switch-progress.json").write_text(json.dumps({"history": on.history[:5]}))
            report = analysis.analyse_seed(3, {"candidate_ids": candidates, "cached_rewards": cache}, config, 0, 30, folder)
            self.assertEqual(report["sr_mismatch"], 0)
            self.assertEqual(report["schedule"]["slots"], 120)
            self.assertEqual(sum(report["schedule"]["composition"]), 120)
            self.assertEqual(set(report["arms"]), {"sr", "switch", "replicate1-switch"})
            self.assertEqual(sum(report["arms"]["switch"]["composition"]), 26 * 4)
            self.assertEqual(sum(report["arms"]["switch"]["sr_comparison_composition"]), 40)  # one check at 25
            self.assertEqual(sum(report["arms"]["switch"]["candidate_composition"]), 80)
            self.assertNotIn("sr_comparison_composition", report["arms"]["replicate1-switch"])  # no check before 25
            self.assertLessEqual(report["schedule"]["half_trained"], report["exactly_half"])
            # Learning signal: every recorded update has a reward and gradient norm; receipts add saturated prompts.
            self.assertEqual(len([u for u in report["updates"] if u["arm"] == "sr"]), 30)
            self.assertTrue(all(u["gradient_norm"] is not None for u in report["updates"]))
            self.assertFalse(report["arms"]["sr"]["receipts"])
            receipts = folder / "cost-receipts" / "sr"
            receipts.mkdir(parents=True)
            for step, zero in ((0, 16), (1, 0), (26, 32)):
                (receipts / f"e{step}.json").write_text(json.dumps({"id": f"e{step}", "phase": "training", "checkpoint": step,
                    "gpu_count": 4, "state": "finished", "counts": {"zero_advantage_responses": zero, "responses": 32}}))
            (receipts / "open.json").write_text(json.dumps({"id": "open", "phase": "training", "checkpoint": 2,
                                                            "gpu_count": 4, "state": "started"}))
            report = analysis.analyse_seed(3, {"candidate_ids": candidates, "cached_rewards": cache}, config, 0, 30, folder)
            by_step = {u["step"]: u["zero_advantage_prompts"] for u in report["updates"] if u["arm"] == "sr"}
            self.assertEqual((by_step[0], by_step[1], by_step[26], by_step[2]), (2.0, 0.0, 4.0, None))
            blocks = report["arms"]["sr"]["signal"]
            self.assertEqual(sorted(blocks), [0, 25])
            self.assertAlmostEqual(blocks[0]["mean_saturated_prompts"], 1.0)
            self.assertAlmostEqual(blocks[25]["mean_saturated_prompts"], 4.0)
            self.assertTrue(report["arms"]["sr"]["receipts"])
            self.assertIn("saturated prompts per update 2.00", analysis.summarize([report]))
            # SR trains only prompts near 50% when enough exist: the trained mean sits closer to 50% than the pool mean.
            self.assertLess(abs(report["schedule"]["mean_trained_rate"] - 0.5), abs(report["mean_rate"] - 0.5) + 1e-9)
            summary = analysis.write_outputs([report], Path(directory) / "out")
            self.assertIn("exactly 4/8 (measured 50%)", summary)
            self.assertIn("recorded sr history vs predicted schedule: 0 mismatching", summary)
            self.assertIn("replicate1-switch", summary)
            rows = list(__import__("csv").DictReader((Path(directory) / "out" / "cache.csv").open()))
            self.assertEqual(len(rows), 400)
            self.assertEqual(sum(int(r["predicted_sr_train_count"]) for r in rows), 120)
            self.assertIn("learning signal: zero-gradient updates", summary)
            self.assertIn("saturated prompts per update", summary)
            updates = list(__import__("csv").DictReader((Path(directory) / "out" / "updates.csv").open()))
            self.assertEqual(len(updates), 30 + 26 + 5)
            # a wrong SR history is reported, not silently accepted
            broken = [dict(r, train_ids=list(reversed(r["train_ids"]))) for r in sr.history]
            (folder / "sr-progress.json").write_text(json.dumps({"history": broken}))
            self.assertGreater(analysis.analyse_seed(3, {"candidate_ids": candidates, "cached_rewards": cache},
                                                     config, 0, 30, folder)["sr_mismatch"], 0)

    def test_cli_with_an_input_bundle_and_with_a_plan_without_caches(self):
        _, candidates, validation, evaluation, cache = toy()
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle.json"
            bundle.write_text(json.dumps({"candidate_ids": candidates, "validation_pool_ids": validation,
                                          "evaluation_ids": evaluation, "cached_rewards": cache}))
            out = Path(directory) / "out"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                code = analysis.main(["--input", str(bundle), "--seed", "3", "--total", "60", "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertIn("SR schedule (predicted, updates 25-59, 140 training slots)", printed.getvalue())
            self.assertTrue((out / "composition.csv").exists())
            bundle.write_text(json.dumps({"candidate_ids": candidates, "cached_rewards": {}}))
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                analysis.main(["--input", str(bundle), "--out", str(out)])
            # The checked-in plans point at bundles without caches: every seed is skipped, nothing crashes.
            plan = Path(__file__).resolve().parents[1] / "experiments/additional_seeds.json"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                code = analysis.main(["--plan", str(plan), "--out", str(out / "plan")])
            self.assertEqual(code, 1)
            self.assertIn("skipped seeds without a complete cache: [5, 6, 7, 8, 9]", printed.getvalue())
            self.assertIn("no seed with a complete cache", printed.getvalue())

    def test_input_seed_inferred_and_conflicts_rejected(self):
        _, candidates, _, _, cache = toy(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            data = {"candidate_ids": candidates, "cached_rewards": cache, "provenance": {"experiment_seed": 5}}
            path.write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(analysis.main(["--input", str(path)]), 0)
            self.assertIn("seed 5", output.getvalue())
            with self.assertRaisesRegex(ValueError, "seed differs"):
                analysis.main(["--input", str(path), "--seed", "3"])
            data["cached_rewards"].pop(candidates[0])
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analysis.load_bundle(path)


if __name__ == "__main__":
    unittest.main()
