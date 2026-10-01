import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from scripts import srgc_direction_analysis as analysis
from scripts.srgc_direction_records import DirectionRecordMixin
from scripts.srgc_switch_repeat import SwitchRepeatEngine
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


class RecordedSwitch(DirectionRecordMixin, Engine):
    pass


def make(arm, cls, seed=3):
    features, answers, candidates, validation, _, cache = make_problem(seed)
    backend = ToyBackend(features, answers, projection_dim=64, seed=seed)
    return cls(backend, candidates, validation, cache, arm=arm, config=Config(seed=seed, projection_dim=64))


class DirectionRecordTest(unittest.TestCase):
    def test_check_record_carries_a_consistent_decomposition(self):
        engine = make("switch", RecordedSwitch)
        engine.run_until(25)
        record = engine.update()  # first check at update 25
        self.assertIsNotNone(record["d"])
        for key in ("validation_norm", "on_mean_norm", "sr_mean_norm", "on_mean_cos", "sr_mean_cos",
                    "on_top4_dot", "on_random4_expected_dot", "ranking_cos_std", "ranking_gap4"):
            self.assertIn(key, record)
        reconstructed = record["validation_norm"] * (record["on_mean_norm"] * record["on_mean_cos"]
                                                     - record["sr_mean_norm"] * record["sr_mean_cos"])
        self.assertAlmostEqual(reconstructed, record["d"], places=6)
        self.assertAlmostEqual(record["on_random4_expected_dot"], record["on_mean_validation_dot"], places=6)
        self.assertGreaterEqual(record["on_top4_dot"] + 1e-9, min(record["on_random4_expected_dot"], record["on_top4_dot"]))
        self.assertGreaterEqual(record["ranking_gap4"], 0.0)
        plain = engine.update()
        self.assertFalse(plain["selection_refreshed"])
        self.assertNotIn("on_mean_cos", plain)

    def test_on_policy_refresh_without_a_check_is_decomposed_too(self):
        engine = make("on_policy", RecordedSwitch)
        record = engine.update()
        self.assertIsNone(record["d"])
        self.assertIn("on_mean_cos", record)
        self.assertIn("ranking_gap4", record)
        self.assertEqual(record["sr_ids"], [])
        self.assertNotIn("sr_mean_norm", record)
        self.assertNotIn("sr_mean_cos", record)

    def test_switch_repeat_records_in_both_modes(self):
        engine = make("switch_repeat", SwitchRepeatEngine)
        engine.run_until(26)
        first_check = next(r for r in engine.history if r.get("d") is not None)
        self.assertIn("on_mean_cos", first_check)
        engine.mode, engine.transitions = "sr", [{"step": 25, "to": "sr"}]
        engine.run_until(51)
        sr_check = next(r for r in engine.history if r["checkpoint"] == 50)
        self.assertEqual(sr_check["selector"], "sr")
        self.assertIn("on_mean_cos", sr_check)
        self.assertAlmostEqual(sr_check["validation_norm"] * (sr_check["on_mean_norm"] * sr_check["on_mean_cos"]
                               - sr_check["sr_mean_norm"] * sr_check["sr_mean_cos"]), sr_check["d"], places=6)


class DirectionAnalysisTest(unittest.TestCase):
    def test_fixed_diagnostic_requires_finite_terms_and_never_replaces_a_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "seed-5"
            folder.mkdir()
            base = {"selection_refreshed": True, "checkpoint": 25, "d": None,
                    "validation_norm": 2.0, "on_mean_norm": 3.0, "sr_mean_norm": 4.0,
                    "on_mean_cos": 0.5, "sr_mean_cos": 0.25}
            path = folder / "switch_fixed200-progress.json"
            path.write_text(json.dumps({"history": [base, {**base, "checkpoint": 50, "d": -9.0},
                {**base, "checkpoint": 75, "sr_mean_norm": float("nan")},
                {**base, "checkpoint": 100, "sr_mean_cos": None},
                {**base, "checkpoint": 125, "sr_mean_norm": 0.0}]}))
            rows = analysis.refresh_rows(root)
            self.assertEqual([r["d"] for r in rows], [1.0, -9.0, None, None, 3.0])
            self.assertEqual(rows[1]["d_source"], "decision")
            self.assertEqual(rows[-1]["d_source"], "reconstructed_diagnostic")
            self.assertIn("+0.000(0.000)", analysis.summarize([rows[-1]]))

    def test_on_policy_without_sr_gradients_has_no_reconstructed_d(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = make("on_policy", RecordedSwitch)
            engine.run_until(51)
            folder = root / "seed-3"
            folder.mkdir()
            path = folder / "on_policy-progress.json"
            path.write_text(json.dumps({"history": engine.history}))
            before = path.read_bytes()
            rows = analysis.refresh_rows(root)
            for row in rows:
                self.assertIsNone(row["d_source"])
                self.assertIsNone(row["d"])
                self.assertIsNone(row["sr_dot"])
                self.assertIsNotNone(row["on_mean_cos"])
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(record["d"] is None for record in engine.history))
            self.assertEqual(path.read_bytes(), before)
            legacy = {"history": [{"selection_refreshed": True, "checkpoint": 75, "d": None}]}
            path.write_text(json.dumps(legacy))
            self.assertIsNone(analysis.refresh_rows(root)[0]["d"])

    def test_csv_summary_and_legacy_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, cls in ((5, RecordedSwitch), (6, RecordedSwitch)):
                engine = make("switch", cls, seed=seed)
                engine.run_until(51)
                folder = root / f"seed-{seed}"
                folder.mkdir()
                (folder / "switch-progress.json").write_text(json.dumps({"history": engine.history}))
            legacy = root / "legacy.txt"
            legacy.write_text("state,step,d_a,d_b,d,upper,confirmed_sr,status\n"
                              "s3-t25,25,-1.3,5.4,2.0,19.2,False,measured\n"
                              "s3-t25,300,,,,,,projection_missing\n")
            out = root / "out"
            with redirect_stdout(io.StringIO()) as printed:
                code = analysis.main(["--root", str(root), "--legacy-d", str(legacy), "--out", str(out)])
            self.assertEqual(code, 0)
            with (out / "direction.csv").open() as handle:
                rows = list(__import__("csv").DictReader(handle))
            recorded = [r for r in rows if r["source"] == "recorded"]
            self.assertEqual({r["seed"] for r in recorded}, {"5", "6"})
            self.assertTrue(all(r["on_mean_cos"] for r in recorded if r["step"] in {"25", "50"}))
            legacy_rows = [r for r in rows if r["source"] == "legacy.txt"]
            self.assertEqual([(r["seed"], r["step"], r["d"]) for r in legacy_rows], [("3", "25", "2.0")])
            summary = (out / "summary.txt").read_text()
            self.assertIn("switch", summary)
            self.assertIn("recorded-on-policy-path", summary)
            self.assertIn("refresh rows", printed.getvalue())

    def test_direction_ablations_and_replicates_are_separate_series(self):
        from scripts.srgc_direction_ablation import DirectionAblationEngine
        from scripts.srgc_replicate import ReplicateEngine
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "seed-5"
            (folder / "replicate-1").mkdir(parents=True)
            features, answers, candidates, validation, _, cache = make_problem(5)
            config = Config(seed=5, projection_dim=64)
            ablation = DirectionAblationEngine(ToyBackend(features, answers, projection_dim=64, seed=5), candidates,
                                               validation, cache, arm="direction_ablation", config=config, mode="magnitude")
            ablation.run_until(26)
            (folder / "direction_magnitude-progress.json").write_text(json.dumps({"history": ablation.history}))
            replicate = ReplicateEngine(ToyBackend(features, answers, projection_dim=64, seed=5), candidates, validation,
                                        cache, arm="switch", config=config, replicate=1)
            replicate.run_until(26)
            (folder / "replicate-1" / "switch-progress.json").write_text(json.dumps({"history": replicate.history}))
            (folder / "sr_refresh-progress.json").write_text(json.dumps({"history": []}))
            rows = analysis.refresh_rows(root)
            self.assertEqual({(r["arm"], r["seed"]) for r in rows}, {("direction_magnitude", 5), ("replicate1-switch", 5)})
            self.assertTrue(all(r["on_mean_cos"] is not None for r in rows))
            self.assertEqual([r["selector"] for r in rows if r["arm"] == "direction_magnitude"], ["direction_magnitude"] * 2)
            self.assertEqual({r["arm"] for r in analysis.refresh_rows(root, ["replicate1-switch"])}, {"replicate1-switch"})


if __name__ == "__main__":
    unittest.main()
