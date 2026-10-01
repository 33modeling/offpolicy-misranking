import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import srgc_idle as idle
from scripts import srgc_shared_storage as storage
from srgc_rebuttal.plan import load_plan
from srgc_rebuttal.runtime import run_root
from srgc_rebuttal.tests import test_shared_storage, test_run_history


class IdleConflictTests(unittest.TestCase):
    def test_idle_leaves_only_when_plan_becomes_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            source, env, old = test_shared_storage.SharedStorageTests().old_code_run(directory)
            with patch.object(idle, "route_plan", side_effect=[idle.RunConflict(old, "code differs"), old]) as route:
                sleeps = []
                self.assertEqual(idle.wait_for_plan(source, sleep=sleeps.append), old)
            self.assertEqual(route.call_count, 2)
            self.assertEqual(sleeps, [10])

    def test_stop_marker_stops_idle_worker_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            source, env, old = test_shared_storage.SharedStorageTests().old_code_run(directory)
            (run_root(old, load_plan(old)) / ".queue/stop.json").write_text('{}')
            with patch.object(idle, "route_plan", side_effect=idle.RunConflict(old, "code differs")), \
                    self.assertRaises(SystemExit) as error:
                idle.wait_for_plan(source)
            self.assertEqual(error.exception.code, 0)

    def test_code_conflict_parks_without_claims_or_gpu_start_and_preserves_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            source, env, old = test_shared_storage.SharedStorageTests().old_code_run(directory)
            before = {p: p.read_bytes() for p in Path(directory).rglob("*") if p.is_file()}
            sleeps = []
            def sleep(seconds):
                sleeps.append(seconds)
                if len(sleeps) == 2:
                    raise KeyboardInterrupt
                records = list((run_root(old, load_plan(old)) / ".queue/workers").glob("*.json"))
                self.assertEqual(json.loads(records[0].read_text())["status"], "idle")
            with patch.dict(os.environ, env), patch.object(idle, "route_plan", storage.route_plan), \
                    patch.object(idle, "RunConflict", storage.RunConflict), \
                    patch("srgc_rebuttal.cluster.run_child") as child, \
                    patch("srgc_rebuttal.cluster.gpu_identity") as gpu:
                with self.assertRaises(KeyboardInterrupt):
                    idle.wait_for_plan(source, sleep=sleep, writing=True, start_or_continue=True)
            self.assertEqual(sleeps, [10, 10])
            child.assert_not_called()
            gpu.assert_not_called()
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertFalse(list(Path(directory).rglob("attempts/*.json")))

    def test_legacy_replacement_is_blocked_even_with_explicit_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            env, old, new = test_run_history.RunHistoryTests().fixture(directory)
            with patch.dict(os.environ, env), self.assertRaises(storage.RunConflict):
                storage.route_plan(new, writing=True)
            with patch.dict(os.environ, env):
                self.assertEqual(storage.route_plan(new, writing=False), new)

    def test_worker_process_really_stays_idle_and_sigterm_does_not_restart_training(self):
        with tempfile.TemporaryDirectory() as directory:
            env, old, new = test_run_history.RunHistoryTests().fixture(directory)
            root = Path(__file__).resolve().parents[2]
            environment = {**os.environ, **env, "PAIR_PYTHON": sys.executable, "SWITCH_PYTHON": sys.executable}
            worker = subprocess.Popen([sys.executable, "scripts/run_srgc_rebuttal.py", "worker", "--plan", str(new)],
                cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                folder = run_root(new, load_plan(new)) / ".queue"
                deadline = time.monotonic() + 10
                records = []
                while time.monotonic() < deadline and worker.poll() is None:
                    records = list((folder / "workers").glob("*.json"))
                    if records:
                        break
                    time.sleep(.05)
                self.assertTrue(records)
                record = json.loads(records[0].read_text())
                self.assertEqual(record["status"], "idle")
                self.assertIsNone(record["child_pid"])
                self.assertIsNone(worker.poll())
                self.assertFalse((folder / "tasks").exists())
                self.assertFalse((folder / "admission").exists())
                worker.send_signal(signal.SIGTERM)
                stdout, stderr = worker.communicate(timeout=10)
                self.assertEqual(worker.returncode, 130, stdout + stderr)
                self.assertIn("NODE idle", stdout)
                self.assertEqual(json.loads(records[0].read_text())["status"], "stopped")
            finally:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
