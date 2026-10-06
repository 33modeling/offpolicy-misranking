"""Recovery and storage regressions for the Qwen shared worker."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import srgc_qwen35_worker as worker
from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs


class WorkerRecoveryTests(unittest.TestCase):
    def completed_queue(self, directory):
        directory.mkdir()
        queue = TaskQueue(write_inputs(directory))
        queue.bind()
        for seed in queue.plan["seeds"]:
            atomic_json(queue.root / f"seed-{seed}" / "run.json",
                        {**queue.identities[seed], "status": "running"})
        for task in queue.tasks:
            if task.arm != "cache":
                finish_fake(queue, task)
        return queue

    def test_completed_restart_recovers_reports_without_gpu_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = self.completed_queue(root / "complete")
            with patch.object(cluster, "gpu_identity") as gpu, \
                    patch("srgc_rebuttal.cost_report.compare", return_value={"costs": "recovered"}), \
                    patch("srgc_rebuttal.summarize.summarize", return_value={"results": "recovered"}), \
                    patch.object(worker, "runtime_signature") as signature:
                worker.worker([queue.plan_path], SimpleNamespace(), root, root / "common",
                              lambda *args, **kwargs: self.fail("unexpected admission"))
            gpu.assert_not_called()
            signature.assert_not_called()
            for seed in queue.plan["seeds"]:
                self.assertEqual(json.loads((queue.root / f"seed-{seed}" / "run.json").read_text())["status"],
                                 "complete")
            self.assertEqual(json.loads((queue.root / "results-summary.json").read_text()),
                             {"results": "recovered"})
            self.assertEqual(json.loads((queue.root / "cost-comparison.json").read_text()),
                             {"costs": "recovered"})

    def test_stopped_queue_does_not_prevent_completed_queue_report_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            done = self.completed_queue(root / "complete")
            (root / "stopped").mkdir()
            stopped = TaskQueue(write_inputs(root / "stopped"))
            stopped.bind()
            atomic_json(stopped.directory / "stop.json", {"immediate": False})
            with patch.object(cluster, "publish_reports") as publish, \
                    patch.object(cluster, "gpu_identity") as gpu:
                worker.worker([stopped.plan_path, done.plan_path], SimpleNamespace(), root, root / "common",
                              lambda *args, **kwargs: self.fail("unexpected admission"))
            gpu.assert_not_called()
            self.assertEqual([call.args[0].plan_path for call in publish.call_args_list], [done.plan_path])
            receipts = list((stopped.directory / "workers").glob("*.json"))
            self.assertEqual(json.loads(receipts[0].read_text())["status"], "stopped")
            self.assertFalse((stopped.root / "results-summary.json").exists())


class RuntimeBindingTests(unittest.TestCase):
    def test_rejected_runtime_does_not_bind_an_unstarted_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queues = [SimpleNamespace(directory=root / name) for name in ("new", "existing")]
            expected = {"packages": {"torch": "frozen"}}
            atomic_json(queues[1].directory / "runtime.json", expected)
            with self.assertRaisesRegex(ValueError, "node runtime differs"):
                worker.bind_runtime(queues, {"packages": {"torch": "different"}})
            self.assertFalse((queues[0].directory / "runtime.json").exists())
            self.assertEqual(json.loads((queues[1].directory / "runtime.json").read_text()), expected)
            worker.bind_runtime(queues, expected)
            self.assertEqual(json.loads((queues[0].directory / "runtime.json").read_text()), expected)


if __name__ == "__main__":
    unittest.main()
