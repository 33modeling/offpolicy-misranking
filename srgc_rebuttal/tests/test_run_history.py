import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.srgc_run_history import include_previous_runs, previous_plans
from scripts import srgc_shared_storage as storage
from srgc_rebuttal import reports
from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.plan import input_path, digest
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.tests import test_shared_storage


class RunHistoryTests(unittest.TestCase):
    def fixture(self, directory):
        source, env = test_shared_storage.SharedStorageTests().fixture(directory)
        old = storage.fresh_plan(source, env, "original")
        queue = TaskQueue(old)
        queue.bind()
        folder = queue.root / "seed-5"
        folder.mkdir(parents=True, exist_ok=True)
        expected = queue.identities[5]
        atomic_json(folder / "run.json", expected)
        checkpoint = folder / "prefix.pt"
        checkpoint.write_bytes(b"preserved checkpoint")
        atomic_json(folder / "prefix-ready.json", {**expected, "completed_updates": 25,
                                                    "checkpoint_sha256": digest(checkpoint)})
        bundle = json.loads(input_path(old, queue.plan, 5).read_text())
        atomic_json(folder / "sr-endpoint.json", {**expected, "arm": "sr", "total_updates": 275,
            "shared_prefix_updates": 25, "prefix_checkpoint_sha256": digest(checkpoint),
            "reward": .5, "per_question_reward": {p: .5 for p in bundle["evaluation_ids"]},
            "costs": {}, "switched_at": None})
        new = storage.fresh_plan(old, env, "legacy-auto-replacement")
        atomic_json(new.parent.parent / "automatic-restart.json", {
            "plan": str(new), "previous_plan": str(old), "reason": "implementation changed"})
        return env, old, new

    def test_old_completed_reward_is_visible_beside_new_waiting_queue_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            env, old, new = self.fixture(directory)
            files = {p: p.read_bytes() for p in Path(directory).rglob("*") if p.is_file()}
            original = reports.snapshot
            with include_previous_runs():
                report = reports.snapshot(new)
                current = next(r for r in report["tasks"] if r["task"] == "seed-5.sr")
                prior = next(r for r in report["previous_runs"][0]["tasks"] if r["task"] == "seed-5.sr")
                self.assertEqual(current["status"], "waiting_for_prefix")
                self.assertEqual(prior["status"], "complete")
                self.assertEqual(prior["reward_percent"], 50)
                self.assertIn("PREVIOUS RUN", reports.render(report))
                self.assertEqual(report["errors"], [])
            self.assertIs(reports.snapshot, original)
            self.assertEqual(files, {p: p.read_bytes() for p in files})

    def test_missing_previous_run_is_an_error_not_an_empty_success(self):
        with tempfile.TemporaryDirectory() as directory:
            env, old, new = self.fixture(directory)
            old.unlink()
            with include_previous_runs():
                report = reports.snapshot(new)
            self.assertTrue(report["errors"])
            self.assertFalse(report["complete"])

    def test_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            env, old, new = self.fixture(directory)
            atomic_json(old.parent.parent / "automatic-restart.json", {
                "plan": str(old), "previous_plan": str(new), "reason": "implementation changed"})
            with self.assertRaisesRegex(ValueError, "cycle"):
                list(previous_plans(new))

    def test_status_and_results_cli_show_previous_run_in_text_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            env, old, new = self.fixture(directory)
            root = Path(__file__).resolve().parents[2]
            environment = {**os.environ, **env, "PAIR_PYTHON": sys.executable, "SWITCH_PYTHON": sys.executable}
            for action in ("status", "results"):
                for output in ([], ["--json"]):
                    with self.subTest(action=action, output=output):
                        run = subprocess.run([sys.executable, "scripts/run_srgc_rebuttal.py", action,
                            "--plan", str(new), "--output", str(Path(directory) / f"{action}.txt"), *output],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=30)
                        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                        if output:
                            report = json.loads(run.stdout)
                            self.assertEqual(len(report["previous_runs"]), 1)
                        else:
                            self.assertIn("PREVIOUS RUN", run.stdout)
                            self.assertIn(str(old), run.stdout)


if __name__ == "__main__":
    unittest.main()
