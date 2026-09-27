import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from srgc_rebuttal.admission import admit
from srgc_rebuttal.cluster import child_environment, gpu_identity, run_child, worker
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.reports import export, snapshot
from srgc_rebuttal.runtime import Busy, atomic_json, lease
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs
from srgc_rebuttal.timing import CostMeter, StageTimer


class HardeningTests(unittest.TestCase):
    def test_h100_admission_rejects_wrong_hardware_busy_and_aliased_devices(self):
        def query(name="NVIDIA H100 80GB HBM3", used=0, memory=81559, alias=False):
            return SimpleNamespace(stdout="\n".join(f"GPU-{0 if alias else i}, {used}, {name}, {memory}" for i in range(4)))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            for response, error in ((query("NVIDIA A100"), ValueError), (query(memory=40000), ValueError),
                                    (query(used=5000), Busy), (query(alias=True), ValueError)):
                with patch("srgc_rebuttal.cluster.subprocess.run", return_value=response), self.assertRaises(error):
                    gpu_identity()
            with patch("srgc_rebuttal.cluster.subprocess.run", return_value=query()):
                self.assertEqual(len(gpu_identity()[1]), 4)

    def test_existing_nccl_preflight_is_required_and_runtime_fallback_is_inherited(self):
        with tempfile.TemporaryDirectory() as directory, patch("srgc_rebuttal.admission.packages", return_value={"torch": "fixture"}), \
                patch("srgc_rebuttal.admission.verifier_environment"):
            root = Path(directory) / "admission"
            environment = {}
            def run(command, log, env, **kwargs):
                self.assertTrue(command[1].endswith("scripts/selection_nccl_preflight.py"))
                self.assertEqual(kwargs["timeout"], 600)
                atomic_json(root / "node-preflight/node/admission.json", {"state": "passed",
                    "overrides": {"NCCL_NVLS_ENABLE": "0"}, "allocated_gpu_seconds": 12})
                return 0
            report = admit(root, environment, run)
            self.assertEqual(environment["NCCL_NVLS_ENABLE"], "0")
            self.assertEqual(report["allocated_gpu_seconds"], 12)
            with self.assertRaisesRegex(RuntimeError, "admission failed"):
                admit(Path(directory) / "failed", {}, lambda *a, **kw: 78)

    def test_failed_admission_never_claims_a_training_task(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = write_inputs(Path(directory), pending=True)
            args = SimpleNamespace(plan=plan, node_lock_root=None, worker_id="test")
            with patch("srgc_rebuttal.cluster.gpu_identity", return_value=("0,1,2,3", ("a", "b", "c", "d"))), \
                    patch("srgc_rebuttal.cluster.admit", side_effect=RuntimeError("bad NCCL")), \
                    patch("srgc_rebuttal.cluster.run_worker") as run:
                with self.assertRaisesRegex(RuntimeError, "bad NCCL"):
                    worker(args)
            run.assert_not_called()
            queue = TaskQueue(plan)
            self.assertFalse(list((queue.directory / "tasks").glob("*.json")))
            self.assertEqual(json.loads((queue.directory / "workers/test.json").read_text())["status"], "failed")

    def test_timeout_and_stall_kill_only_the_owned_child_heartbeat_is_not_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in ({"timeout": .05}, {"progress": lambda: (), "stall_seconds": .05}):
                pids = []
                with self.assertRaises(TimeoutError):
                    run_child([sys.executable, "-c", "import time; time.sleep(20)"], Path(directory) / "task.log",
                              child_environment(), heartbeat=pids.append, interval=.01, **options)
                self.assertGreater(len(pids), 1)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pids[-1], 0)

    def test_exited_launcher_does_not_leave_its_child_running(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pid"
            code = ("import os, pathlib, time; pid=os.fork(); "
                    f"pathlib.Path({str(path)!r}).write_text(str(pid)) if pid else time.sleep(20)")
            self.assertEqual(run_child([sys.executable, "-c", code], Path(directory) / "task.log",
                                      child_environment(), interval=.01), 0)
            pid = int(path.read_text())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists() or stat.read_text().split()[2] == "Z":
                    break
                time.sleep(.01)
            else:
                self.fail("orphaned child still running after launcher exit")

    def test_dependencies_wait_for_launcher_and_cost_receipts_to_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            with queue.claim() as task:
                finish_fake(queue, task)
                self.assertFalse(queue.ready(Task(task.seed, "switch")))
                self.assertEqual(next(r for r in queue.status() if r["task"] == task.key)["status"], "running")
                queue.finish(task, 0)
            self.assertTrue(queue.ready(Task(task.seed, "switch")))

    def test_abandoned_attempts_are_bounded_and_keep_their_history(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            for _ in range(2):
                with queue.claim(max_attempts=2) as task:
                    self.assertEqual(task, Task(5, "prefix"))
            with queue.claim(max_attempts=2) as task:
                self.assertEqual(task, Task(6, "prefix"))
            self.assertEqual(next(r for r in queue.status(max_attempts=2) if r["task"] == "seed-5.prefix")["status"], "attempts_exhausted")
            self.assertEqual(len(list((queue.directory / "attempts").glob("*.json"))), 1)

    def test_original_exception_is_not_masked_by_cleanup_synchronization(self):
        calls = []
        def sync():
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError("secondary CUDA sync error")
        meter = StageTimer(synchronize=sync)
        meter.begin()
        with self.assertRaisesRegex(ValueError, "original"):
            with meter.stage("generation"):
                raise ValueError("original OOM or verifier error")
        self.assertFalse(meter.stack)
        self.assertEqual(len(calls), 1)


class ReportTests(unittest.TestCase):
    def endpoint(self, queue, seed, arm):
        folder = queue.root / f"seed-{seed}"
        atomic_json(folder / "run.json", {**queue.identities[seed], "status": "running"})
        finish_fake(queue, Task(seed, "prefix"))
        prefix = json.loads((folder / "prefix-ready.json").read_text())
        bundle = json.loads((queue.plan_path.parent / f"inputs-{seed}.json").read_text())
        atomic_json(folder / f"{arm}-endpoint.json", {**queue.identities[seed], "arm": arm,
            "total_updates": 275, "shared_prefix_updates": 25, "prefix_checkpoint_sha256": prefix["checkpoint_sha256"],
            "reward": .5, "per_question_reward": {key: .5 for key in bundle["evaluation_ids"]},
            "switched_at": 75 if arm == "switch" else None, "cost_measurement_complete": False})

    def test_partial_results_export_all_20_arms_and_do_not_average_missing_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = TaskQueue(write_inputs(root))
            queue.bind()
            self.endpoint(queue, 5, "switch")
            report = snapshot(queue.plan_path, include_costs=True)
            self.assertEqual(report["errors"], [])
            self.assertFalse(report["complete"])
            self.assertEqual(report["arm_statistics"]["switch"]["completed_seeds"], 1)
            self.assertIsNone(report["arm_statistics"]["switch"]["mean_reward_percent"])
            export(report, root / "home/results.txt")
            rows = list(csv.DictReader(io.StringIO((queue.root / "results.csv").read_text())))
            self.assertEqual(len(rows), 20)
            row = next(r for r in rows if r["seed"] == "5" and r["arm"] == "switch")
            self.assertEqual(row["reward_percent"], "50.0")
            self.assertEqual(row["selection_training_preparation_gpu_seconds"], "")
            self.assertTrue((root / "home/results.json").exists())

    def test_bad_file_and_new_code_do_not_hide_other_seed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            self.endpoint(queue, 5, "switch")
            self.endpoint(queue, 6, "switch")
            (queue.root / "seed-5/switch-endpoint.json").write_text("broken")
            atomic_json(queue.directory / "workers/bad.json", {})
            with patch("srgc_rebuttal.reports.code_digest", return_value="new code"):
                report = snapshot(queue.plan_path, include_costs=True)
            self.assertTrue(report["warnings"])
            self.assertTrue(report["errors"])
            self.assertEqual(next(r for r in report["tasks"] if r["task"] == "seed-6.switch")["reward_percent"], 50)

    def test_live_endpoint_is_not_reported_as_complete_before_worker_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            self.endpoint(queue, 5, "switch")
            with lease(queue.directory / "leases/seed-5.switch.lock"):
                row = next(r for r in snapshot(queue.plan_path)["tasks"] if r["task"] == "seed-5.switch")
            self.assertEqual(row["status"], "running")
            self.assertIsNone(row["reward_percent"])

    def test_status_and_results_cli_work_without_torch_and_preserve_partial_exports(self):
        script = Path(__file__).resolve().parents[2] / "scripts/run_srgc_rebuttal.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = write_inputs(root, pending=True)
            for action in ("status", "results"):
                result = subprocess.run([sys.executable, str(script), action, "--plan", str(plan), "--json",
                    "--output", str(root / f"{action}.txt")], cwd="/tmp", text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(len(report["tasks"]), 30)
                self.assertFalse(report["complete"])
                self.assertTrue((root / f"{action}.txt").exists())


if __name__ == "__main__":
    unittest.main()
