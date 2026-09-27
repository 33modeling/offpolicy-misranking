import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_rebuttal.cost_report import seed_costs
from srgc_rebuttal.timing import CostMeter, StageTimer, aggregate_ranks, invocation
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class TimingTests(unittest.TestCase):
    def test_live_costs_survive_partial_cache_phase_without_double_counting(self):
        from srgc_rebuttal.runtime import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "seed-5.json"
            clock = Clock()
            ledger = PhaseLedger(bundle.with_suffix(".cache") / "cost-receipts")
            meter = CostMeter(rank=0, local_gpu_count=1, clock=clock, record=ledger.record)
            meter.begin_phase("cache_generation", gpu_count=1)
            with meter.stage("generation"):
                clock.value += 3
                meter.count("prompts")
            with meter.stage("reward_verification"):
                clock.value += 2
            snapshot = meter.live_snapshot()
            atomic_json(bundle.with_suffix(".cache") / "live-costs/attempt-rank-1.json", snapshot)
            report = seed_costs(root / "runs", bundle, ["random"])
            self.assertEqual(report["cache_live_rank_costs"][0]["local_gpu_seconds"], 5)
            self.assertEqual(snapshot["stages"]["generation"]["wall_seconds"], 3)
            self.assertEqual(snapshot["stages"]["reward_verification"]["wall_seconds"], 2)
            self.assertFalse(snapshot["additive_to_phase_totals"])
            self.assertIsNone(report["experiment_accounting"]["cache_inclusive_gpu_seconds"])
            self.assertFalse(ledger.totals()["complete"])
            meter.end_phase()
            self.assertEqual(ledger.totals()["total_gpu_seconds"], 5)

    def test_nested_stages_are_exclusive(self):
        clock = Clock()
        meter = StageTimer(clock=clock)
        meter.begin()
        with meter.section("validation"), meter.stage("scoring"):
            clock.value = 1
            with meter.stage("generation"):
                clock.value = 4
            clock.value = 6
            meter.count("responses", 8)
        clock.value = 8
        local = meter.finish(1)
        report = aggregate_ranks([local])
        self.assertEqual(report["stages"]["validation.generation"]["gpu_seconds"], 3)
        self.assertEqual(report["stages"]["validation.scoring"]["gpu_seconds"], 3)
        self.assertEqual(report["stages"]["unattributed_and_wait"]["gpu_seconds"], 2)
        self.assertEqual(sum(s["gpu_seconds"] for s in report["stages"].values()), 8)
        self.assertEqual(report["counts"]["validation.responses"], 8)

    def test_uneven_ranks_sum_local_stages_not_maxima(self):
        def part(seconds, generation, backward):
            return {"wall_seconds": seconds, "local_gpu_count": 1,
                    "stages": {"generation": {"wall_seconds": generation, "calls": 1},
                               "backward": {"wall_seconds": backward, "calls": 1}}, "counts": {"prompts": 1}}
        report = aggregate_ranks([part(10, 8, 1), part(10, 1, 8)])
        self.assertEqual(report["gpu_seconds"], 20)
        self.assertEqual(report["stages"]["generation"]["gpu_seconds"], 9)
        self.assertEqual(report["stages"]["backward"]["gpu_seconds"], 9)
        self.assertEqual(report["counts"]["prompts"], 2)

    def test_cpu_time_is_preserved_without_claiming_gpu_time(self):
        report = aggregate_ranks([{"wall_seconds": 2, "local_gpu_count": 0,
                                  "stages": {"check": {"wall_seconds": 1, "calls": 1}}, "counts": {}}])
        self.assertEqual(report["gpu_seconds"], 0)
        self.assertEqual(report["stages"]["check"]["rank_wall_seconds"], 1)

    def test_rejects_nested_double_counted_and_nonfinite_records(self):
        for duration in (11, float("nan"), -1):
            with self.assertRaises(ValueError):
                aggregate_ranks([{"wall_seconds": 10, "local_gpu_count": 1,
                                 "stages": {"bad": {"wall_seconds": duration, "calls": 1}}, "counts": {}}])

    def test_durable_phases_retries_and_interrupted_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PhaseLedger(Path(directory))
            clock = Clock()
            meter = CostMeter(local_gpu_count=1, clock=clock, record=ledger.record)
            for _ in range(2):
                with meter.phase("selection", 25, 1):
                    with meter.stage("generation"):
                        clock.value += 3
            result = ledger.totals()
            self.assertEqual(result["total_gpu_seconds"], 6)
            self.assertEqual(result["exclusive_stages"]["selection.generation"]["gpu_seconds"], 6)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with meter.phase("training", 25, 1):
                    raise RuntimeError("interrupted")
            self.assertIsNone(ledger.totals()["total_gpu_seconds"])
            self.assertFalse(ledger.totals()["complete"])

    def test_invocation_exception_does_not_publish_zero_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PhaseLedger(Path(directory))
            with self.assertRaises(RuntimeError):
                with invocation(ledger, CostMeter(), 0):
                    raise RuntimeError("failed")
            self.assertFalse(ledger.totals()["complete"])

    def test_absent_costs_remain_unknown_until_endpoint_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def report():
                return seed_costs(root, root / "input.json", ["random"])["arms"]["random"]
            self.assertIsNone(report()["continuation_phases_gpu_seconds"])
            meter = CostMeter(record=PhaseLedger(root / "cost-receipts/random").record)
            with meter.phase("training"):
                pass
            self.assertIsNone(report()["continuation_phases_gpu_seconds"])
            (root / "random-endpoint.json").write_text(json.dumps({"cost_measurement_complete": True}))
            self.assertIsNone(report()["continuation_phases_gpu_seconds"])
            with invocation(PhaseLedger(root / "invocations/random"), meter, 0):
                pass
            self.assertEqual(report()["selection_gpu_seconds"], 0)
            self.assertEqual(report()["continuation_phases_gpu_seconds"], 0)

    def test_metered_engine_charges_only_actual_refreshes_and_keeps_training(self):
        features, answers, candidates, validation, _, cache = make_problem(candidates=80, validation=2, evaluation=1)
        for arm, selections in (("on_policy", 2), ("sr", 0), ("random", 0)):
            backend = ToyBackend(features, answers, projection_dim=16)
            events = []
            backend.cost_meter = CostMeter(record=events.append)
            engine = Engine(backend, candidates, validation, cache, arm=arm, config=Config(projection_dim=16))
            engine.run_until(26)
            completed = [e for e in events if e["state"] == "finished"]
            self.assertEqual(sum(e["phase"] == "selection" for e in completed), selections)
            self.assertEqual(sum(e["phase"] == "training" for e in completed), 26)
            self.assertEqual(len(backend.score_calls), selections * 2)

    def test_cost_report_checks_identity_without_matching_manifest_only_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"seed": 5, "input_sha256": "input", "plan_sha256": "plan", "implementation_sha256": "code"}
            (root / "run.json").write_text(json.dumps({**expected, "status": "complete"}))
            endpoint_path = root / "sr-endpoint.json"
            endpoint = {**expected, "cost_measurement_complete": True}
            endpoint_path.write_text(json.dumps(endpoint))
            meter = CostMeter(record=PhaseLedger(root / "cost-receipts/sr").record)
            with meter.phase("training"):
                pass
            with invocation(PhaseLedger(root / "invocations/sr"), meter, 0):
                pass
            self.assertTrue(seed_costs(root, root / "input.json", ["sr"])["arms"]["sr"]["complete"])
            endpoint_path.write_text(json.dumps({**endpoint, "implementation_sha256": "other"}))
            with self.assertRaisesRegex(ValueError, "identity"):
                seed_costs(root, root / "input.json", ["sr"])

    def test_py_launcher_help_does_not_require_model_or_launch_workers(self):
        script = Path(__file__).resolve().parents[2] / "scripts/run_srgc_rebuttal.py"
        for action in ("run", "cache", "costs", "plan", "summary", "cluster"):
            result = subprocess.run([sys.executable, str(script), action, "--help"],
                                    cwd="/tmp", capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
