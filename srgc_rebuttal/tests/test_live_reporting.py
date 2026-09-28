import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from srgc_rebuttal import reports
from srgc_rebuttal.cluster import publish_reports
from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_rebuttal.plan import digest, input_path
from srgc_rebuttal.runtime import atomic_json, code_digest, lease
from srgc_rebuttal.tests.test_cluster import write_inputs
from srgc_rebuttal.tests import test_hardening as hardening


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
spec = importlib.util.spec_from_file_location("srgc_live_status", SCRIPTS / "srgc_live_status.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class LiveStatusTests(unittest.TestCase):
    def fixture(self, root, arm="prefix"):
        queue = TaskQueue(write_inputs(root))
        queue.bind()
        key = f"seed-5.{arm}"
        started = time.time() - 10
        atomic_json(queue.directory / "tasks" / f"{key}.json", {
            "task": key, "status": "running", "attempt_id": "current-attempt", "attempt": 2,
            "started": started, "host": "h100-node"})
        scope = "shared-prefix" if arm == "prefix" else arm
        phases = queue.root / "seed-5/cost-receipts" / scope
        return queue, key, phases, started

    def phase(self, phases, phase, checkpoint, state="started", *, name="event", modified=None):
        path = phases / f"{name}.json"
        atomic_json(path, {"id": name, "phase": phase, "checkpoint": checkpoint, "state": state,
                           "gpu_count": 4, **({"gpu_seconds": 4, "wall_seconds": 1} if state == "finished" else {})})
        if modified is not None:
            os.utime(path, (modified, modified))
        return path

    def snapshot_row(self, queue, key):
        report = live.snapshot(reports.snapshot, queue.plan_path)
        return report, next(row for row in report["tasks"] if row["task"] == key)

    def test_live_prefix_current_completed_saving_and_restore_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            queue, key, phases, _ = self.fixture(Path(directory))
            with lease(queue.directory / "leases" / f"{key}.lock"):
                for phase, checkpoint, state, current, completed in (
                    ("startup", None, "started", None, None),
                    ("selection", 0, "started", 1, 0),
                    ("selection", 0, "finished", 1, 0),
                    ("training", 0, "started", 1, 0),
                    ("training", 0, "finished", None, 1),
                    ("checkpoint_save", 1, "started", None, 1),
                    ("checkpoint_load", 5, "finished", None, 5),
                    ("training", 5, "started", 6, 5),
                    ("checkpoint_save", 25, "finished", None, 25),
                ):
                    with self.subTest(phase=phase, state=state, step=checkpoint):
                        self.phase(phases, phase, checkpoint, state)
                        report, row = self.snapshot_row(queue, key)
                        self.assertEqual(report["errors"], [])
                        self.assertEqual(row["total_steps"], 25)
                        self.assertEqual((row["current_step"], row["completed_steps"]), (current, completed))
                        self.assertEqual(row["status"], "running")
                        text = live.render(report)
                        self.assertIn("running now:", text)
                        self.assertIn("node h100-node", text)
                        if current == 1:
                            self.assertIn("1/25", text)
                            self.assertIn("0/25", text)

    def test_continuation_and_evaluation_do_not_confuse_275_with_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            queue, key, phases, _ = self.fixture(Path(directory), "switch")
            with lease(queue.directory / "leases" / f"{key}.lock"):
                self.phase(phases, "training", 26)
                _, row = self.snapshot_row(queue, key)
                self.assertEqual((row["current_step"], row["completed_steps"], row["total_steps"]), (27, 26, 275))
                self.phase(phases, "evaluation", 275)
                _, row = self.snapshot_row(queue, key)
                self.assertIsNone(row["current_step"])
                self.assertEqual(row["completed_steps"], 275)
                self.assertEqual(row["status"], "running")
                self.assertIsNone(row["reward_percent"])

    def test_retry_ignores_previous_attempt_steps_and_failed_work_is_not_live(self):
        with tempfile.TemporaryDirectory() as directory:
            queue, key, phases, started = self.fixture(Path(directory))
            self.phase(phases, "training", 20, "finished", name="previous", modified=started - 1)
            with lease(queue.directory / "leases" / f"{key}.lock"):
                _, row = self.snapshot_row(queue, key)
                self.assertIsNone(row["completed_steps"])
                self.phase(phases, "training", 5, name="resumed")
                _, row = self.snapshot_row(queue, key)
                self.assertEqual((row["current_step"], row["completed_steps"]), (6, 5))
            report, row = self.snapshot_row(queue, key)
            self.assertEqual(row["status"], "recoverable")
            self.assertIsNone(row["current_step"])
            self.assertEqual(row["last_observed_step"], 6)
            self.assertIn("last step 6/25", live.render(report))

    def test_invalid_phase_does_not_crash_or_invent_a_step(self):
        with tempfile.TemporaryDirectory() as directory:
            queue, key, phases, _ = self.fixture(Path(directory))
            self.phase(phases, "training", 500)
            with lease(queue.directory / "leases" / f"{key}.lock"):
                report, row = self.snapshot_row(queue, key)
                self.assertIsNone(row["current_step"])
                self.assertIsNone(row["completed_steps"])
                self.assertTrue(report["errors"])
                self.assertIn("ERROR", live.render(report))
                (phases / "event.json").write_text("broken JSON")
                report, _ = self.snapshot_row(queue, key)
                self.assertIn("JSONDecodeError", report["errors"][-1])

    def test_completed_prefix_and_cache_counts_are_separate_from_training(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            hardening.ReportTests().endpoint(queue, 5, "sr")
            report = live.snapshot(reports.snapshot, queue.plan_path)
            prefix = next(r for r in report["tasks"] if r["task"] == "seed-5.prefix")
            self.assertEqual(prefix["completed_steps"], 25)
            cache = next(r for r in report["tasks"] if r["task"] == "seed-5.cache")
            self.assertNotIn("current_step", cache)
            self.assertRegex(live.render(report), r"(?m)^\s*5\s+done\s+done\s")
            self.assertEqual(code_digest(), "1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a")

    def test_status_hook_restored_after_exit(self):
        original_snapshot, original_render = reports.snapshot, reports.render
        with patch.object(reports, "main", side_effect=SystemExit(0)):
            with self.assertRaises(SystemExit):
                live.main()
        self.assertIs(reports.snapshot, original_snapshot)
        self.assertIs(reports.render, original_render)

    def test_watch_preserves_json_export_and_stops_without_changing_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = write_inputs(root, pending=True)
            original_snapshot, original_render = reports.snapshot, reports.render
            with patch("sys.argv", ["status", "status", "--plan", str(plan), "--watch", ".01",
                                    "--json", "--output", str(root / "status.txt")]), \
                    patch("sys.stdout", new_callable=io.StringIO) as output, \
                    patch.object(reports.time, "sleep", side_effect=KeyboardInterrupt):
                live.main()
            value = json.loads(output.getvalue())
            self.assertEqual(value["errors"], [])
            self.assertIn("SRGC math", (root / "status.txt").read_text())
            self.assertFalse((root / "runs/.queue/protocol.json").exists())
            self.assertIs(reports.snapshot, original_snapshot)
            self.assertIs(reports.render, original_render)


def completed_phase(directory, name, gpu_seconds, checkpoint=None):
    ledger = PhaseLedger(directory)
    event = {"id": name, "phase": name, "checkpoint": checkpoint, "gpu_count": 4, "state": "started"}
    ledger.record(event)
    ledger.record({**event, "state": "finished", "gpu_seconds": gpu_seconds, "wall_seconds": gpu_seconds / 4})


class ResultsVerificationTests(unittest.TestCase):
    def complete_seed(self, queue, seed):
        folder = queue.root / f"seed-{seed}"
        for arm in queue.plan["arms"]:
            hardening.ReportTests().endpoint(queue, seed, arm)
            path = folder / f"{arm}-endpoint.json"
            endpoint = json.loads(path.read_text())
            values = {"training": 12, "preparation": 1, "evaluation": 2, "checkpoint_save": 3, "startup": 2}
            if arm in {"on_policy", "switch"}:
                values["selection"] = 8
            endpoint.update(cost_measurement_complete=True,
                costs={"selection_gpu_seconds": values.get("selection", 0),
                       "sr_preparation_gpu_seconds": 1,
                       **{f"{phase}_gpu_seconds": seconds for phase, seconds in values.items()}})
            atomic_json(path, endpoint)
            for phase, seconds in values.items():
                completed_phase(folder / "cost-receipts" / arm, phase, seconds)
            completed_phase(folder / "invocations" / arm, "session", sum(values.values()) + 2)
        completed_phase(folder / "cost-receipts/shared-prefix", "training", 5)
        completed_phase(folder / "invocations/prefix", "session", 7)
        bundle = input_path(queue.plan_path, queue.plan, seed)
        atomic_json(bundle.with_suffix(".cache") / "cost-summary.json", {
            "bundle_sha256": digest(bundle), "complete": True,
            "invocations": {"total_gpu_seconds": 2}})

    def test_all_20_results_cost_totals_zero_stages_means_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = TaskQueue(write_inputs(root))
            queue.bind()
            for seed in queue.plan["seeds"]:
                self.complete_seed(queue, seed)
            report = reports.snapshot(queue.plan_path, include_costs=True)
            self.assertEqual(report["errors"], [])
            self.assertTrue(report["complete"])
            for row in report["tasks"]:
                if row["arm"] in queue.plan["arms"]:
                    self.assertEqual(row["reward_percent"], 50)
                    self.assertTrue(row["cost_complete"])
                    selection = 8 if row["arm"] in {"on_policy", "switch"} else 0
                    self.assertEqual(row["selection_gpu_seconds"], selection)
                    self.assertEqual(row["selection_training_preparation_gpu_seconds"], selection + 13)
                    self.assertEqual(row["continuation_phases_gpu_seconds"], selection + 20)
                    self.assertEqual(row["checkpoint_gpu_seconds"], 3)
            for cost in report["costs"]:
                self.assertEqual(cost["experiment_accounting"]["experiment_including_cache_gpu_seconds"], 113)
            for stats in report["arm_statistics"].values():
                self.assertEqual(stats["completed_seeds"], 5)
                self.assertEqual(stats["mean_reward_percent"], 50)
            reports.export(report, root / "home/results.txt")
            rows = list(csv.DictReader(io.StringIO((queue.root / "results.csv").read_text())))
            self.assertEqual(len(rows), 20)
            self.assertEqual(json.loads((queue.root / "results.json").read_text()), report)
            self.assertEqual((root / "home/results.txt").read_text(), (queue.root / "results.txt").read_text())
            publish_reports(queue)
            summary = json.loads((queue.root / "results-summary.json").read_text())
            comparison = json.loads((queue.root / "cost-comparison.json").read_text())
            self.assertEqual(len(summary["per_seed"]), 5)
            self.assertEqual(summary["paired_seed_statistics"]["switch_minus_sr_pp"]["mean"], 0)
            self.assertEqual(comparison["arm_statistics"]["switch"]["mean_gpu_seconds"], 21)

    def test_interrupted_cost_is_null_not_zero_and_other_seed_results_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = TaskQueue(write_inputs(root))
            queue.bind()
            self.complete_seed(queue, 5)
            self.complete_seed(queue, 6)
            ledger = PhaseLedger(queue.root / "seed-5/cost-receipts/sr")
            ledger.record({"id": "interrupted", "phase": "training", "checkpoint": 25,
                           "gpu_count": 4, "state": "started"})
            report = reports.snapshot(queue.plan_path, include_costs=True)
            self.assertEqual(report["errors"], [])
            row = next(r for r in report["tasks"] if r["task"] == "seed-5.sr")
            self.assertEqual(row["reward_percent"], 50)
            self.assertFalse(row["cost_complete"])
            self.assertIsNone(row["selection_gpu_seconds"])
            self.assertIsNone(row["training_gpu_seconds"])
            self.assertIsNone(report["arm_statistics"]["sr"]["mean_reward_percent"])
            reports.export(report, root / "partial.txt")
            rows = list(csv.DictReader(io.StringIO((queue.root / "results.csv").read_text())))
            self.assertEqual(next(r for r in rows if r["seed"] == "5" and r["arm"] == "sr")["training_gpu_seconds"], "")
            self.assertEqual(next(r for r in rows if r["seed"] == "6" and r["arm"] == "sr")["training_gpu_seconds"], "12.0")

    def test_malformed_endpoint_and_cost_do_not_prevent_partial_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = TaskQueue(write_inputs(root))
            queue.bind()
            self.complete_seed(queue, 5)
            self.complete_seed(queue, 6)
            (queue.root / "seed-5/sr-endpoint.json").write_text("bad endpoint")
            (queue.root / "seed-6/cost-receipts/sr/training.json").write_text("bad cost")
            report = reports.snapshot(queue.plan_path, include_costs=True)
            self.assertGreaterEqual(len(report["errors"]), 2)
            self.assertFalse(report["complete"])
            reports.export(report, root / "errors.txt")
            self.assertEqual(len(list(csv.DictReader(io.StringIO((queue.root / "results.csv").read_text())))), 20)
            self.assertIn("ERROR", (root / "errors.txt").read_text())
            self.assertEqual(next(r for r in report["tasks"] if r["task"] == "seed-6.switch")["reward_percent"], 50)

    def test_math_and_mbpp_status_results_cli_without_gpu_packages_and_no_input_changes(self):
        wrapper = ("import runpy, sys; from pathlib import Path; "
                   "sys.modules.update(torch=None, transformers=None, peft=None); "
                   "sys.argv=sys.argv[1:]; sys.path.insert(0,str(Path(sys.argv[0]).parent)); "
                   "runpy.run_path(sys.argv[0],run_name='__main__')")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ("math", "mbpp"):
                folder = root / dataset
                folder.mkdir()
                path = write_inputs(folder, pending=True)
                plan = json.loads(path.read_text())
                if dataset == "mbpp":
                    plan.update(dataset="mbpp", verifier="srgc_rebuttal.verifiers:code_reward")
                    atomic_json(path, plan)
                before = {p.name: p.read_bytes() for p in folder.glob("*.json")}
                for action in ("status", "results"):
                    output = folder / "home" / f"{action}.txt"
                    result = subprocess.run([sys.executable, "-c", wrapper, str(SCRIPTS / "run_srgc_rebuttal.py"),
                        action, "--dataset", dataset, "--plan", str(path), "--json", "--output", str(output)],
                        cwd="/tmp", text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(len(report["tasks"]), 30)
                    self.assertFalse(report["complete"])
                    self.assertEqual(report["errors"], [])
                    if action == "status":
                        self.assertIn("SRGC ", output.read_text())
                    else:
                        self.assertEqual(len(list(csv.DictReader(io.StringIO((folder / "runs/results.csv").read_text())))), 20)
                self.assertEqual({p.name: p.read_bytes() for p in folder.glob("*.json")}, before)
                self.assertFalse((folder / "runs/.queue/protocol.json").exists())


if __name__ == "__main__":
    unittest.main()
