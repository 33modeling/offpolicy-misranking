import importlib.util
import io
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import Mock, patch
import zipfile

from srgc_rebuttal.plan import digest, input_path, load_plan
from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.runtime import atomic_json, code_digest, identity, lease, run_root
from srgc_rebuttal.tests.test_cluster import write_inputs
from srgc_rebuttal.tests.test_shared_storage import storage


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/srgc_checkpoint_backup.py"
spec = importlib.util.spec_from_file_location("srgc_checkpoint_backup", SCRIPT)
backup = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"srgc_shared_storage": storage}):
    spec.loader.exec_module(backup)


def checkpoint(path, content):
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w") as archive:
        archive.writestr("state", content)
    temporary.replace(path)


class CheckpointBackupTests(unittest.TestCase):
    def fixture(self, directory):
        group = Path(directory) / "group"
        group.mkdir()
        plan_path = write_inputs(group)
        plan = load_plan(plan_path)
        folder = run_root(plan_path, plan) / "seed-5"
        folder.mkdir(parents=True)
        atomic_json(folder / "run.json", identity(plan_path, plan, 5))
        source = folder / "sr-latest.pt"
        checkpoint(source, "step-25")
        env = {"GROUP_VOLUME": str(group), "OM_WORK": str(group / "work")}
        return plan_path, source, env

    def generations(self, source):
        return sorted((source.parent.parent / "checkpoint-backups/seed-5/sr-latest").glob("*/checkpoint.pt"))

    def capture(self, plan, env):
        with redirect_stdout(io.StringIO()):
            return backup.backup_once(plan, environment=env)

    def test_copy_checksum_identity_and_repeat_without_overwriting_live_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source, env = self.fixture(directory)
            original, runtime = source.read_bytes(), code_digest()
            report = self.capture(plan, env)
            self.assertEqual((report["saved"], report["errors"]), (1, []))
            copied, = self.generations(source)
            self.assertEqual(copied.read_bytes(), original)
            self.assertEqual(source.read_bytes(), original)
            receipt = json.loads(copied.with_name("receipt.json").read_text())
            self.assertEqual(receipt["sha256"], digest(source))
            self.assertEqual(receipt["identity"], identity(plan, load_plan(plan), 5))
            self.assertEqual(self.capture(plan, env)["unchanged"], 1)
            self.assertEqual(code_digest(), runtime)

    def test_keep_two_versions_and_do_not_replace_them_with_a_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source, env = self.fixture(directory)
            expected = []
            for step in (25, 50, 75):
                checkpoint(source, f"step-{step}")
                expected.append(digest(source))
                self.assertEqual(self.capture(plan, env)["saved"], 1)
            self.assertEqual({p.parent.name for p in self.generations(source)}, set(expected[-2:]))
            source.write_bytes(b"incomplete checkpoint")
            report = self.capture(plan, env)
            self.assertEqual(report["saved"], 0)
            self.assertEqual(len(report["errors"]), 1)
            self.assertEqual({p.parent.name for p in self.generations(source)}, set(expected[-2:]))

    def test_atomic_replacement_during_backup_keeps_old_snapshot_then_captures_new(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source, env = self.fixture(directory)
            original = source.read_bytes()
            def replace_while_verifying(path):
                if path.name == "checkpoint.pt":
                    checkpoint(source, "step-50")
                return digest(path)
            with patch.object(backup, "digest", side_effect=replace_while_verifying):
                self.assertEqual(self.capture(plan, env)["saved"], 1)
            self.assertEqual(self.generations(source)[0].read_bytes(), original)
            self.assertNotEqual(source.read_bytes(), original)
            self.assertEqual(self.capture(plan, env)["saved"], 1)
            self.assertEqual(len(self.generations(source)), 2)

    def test_temp_files_are_ignored_and_inputs_and_symlink_escapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source, env = self.fixture(directory)
            source.rename(source.with_suffix(".tmp"))
            self.assertEqual(self.capture(plan, env)["saved"], 0)
            source.symlink_to(source.with_suffix(".tmp"))
            self.assertTrue(self.capture(plan, env)["errors"])
            source.unlink()
            source.with_suffix(".tmp").rename(source)
            manifest = source.parent / "run.json"
            record = json.loads(manifest.read_text())
            atomic_json(manifest, {**record, "input_sha256": "wrong"})
            self.assertTrue(self.capture(plan, env)["errors"])
            atomic_json(manifest, record)
            destination = source.parent.parent / "checkpoint-backups"
            destination.rename(destination.with_name("old-backups"))
            outside = Path(directory) / "user"
            outside.mkdir()
            destination.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes group"):
                self.capture(plan, env)
            self.assertEqual(list(outside.iterdir()), [])

    def test_second_node_does_not_duplicate_copies_under_shared_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source, env = self.fixture(directory)
            lock = source.parent.parent / "checkpoint-backups/.backup.lock"
            with lease(lock):
                self.assertTrue(self.capture(plan, env)["busy"])
            self.assertEqual(self.capture(plan, env)["saved"], 1)
            self.assertEqual(self.capture(plan, env)["unchanged"], 1)

    def test_cpu_cli_does_not_need_torch_or_transformers_and_does_not_change_queue_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source, env = self.fixture(directory)
            wrapper = ("import runpy, sys; from pathlib import Path; "
                       "sys.modules.update(torch=None, transformers=None, peft=None); "
                       "sys.argv=sys.argv[1:]; sys.path.insert(0,str(Path(sys.argv[0]).parent)); "
                       "runpy.run_path(sys.argv[0],run_name='__main__')")
            command = [sys.executable, "-c", wrapper, str(SCRIPT.with_name("run_srgc_rebuttal.py")),
                       "backup", "--plan", str(plan)]
            result = subprocess.run(command, env={**os.environ, **env, "PAIR_PYTHON": sys.executable},
                                    text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"saved": 1', result.stdout)
            self.assertFalse((source.parent.parent / ".queue/protocol.json").exists())

    def test_backing_up_an_existing_queue_does_not_change_its_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _, env = self.fixture(directory)
            queue = TaskQueue(plan)
            queue.bind()
            marker = queue.directory / "protocol.json"
            before = marker.read_bytes()
            self.assertEqual(self.capture(plan, env)["saved"], 1)
            queue.verify()
            self.assertEqual(marker.read_bytes(), before)

    def test_worker_entry_point_automatically_backs_up_without_changing_worker_arguments(self):
        entry_spec = importlib.util.spec_from_file_location("rebuttal_entry", SCRIPT.with_name("run_srgc_rebuttal.py"))
        entry = importlib.util.module_from_spec(entry_spec)
        entry_spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as directory:
            plan, source, env = self.fixture(directory)
            def worker():
                self.assertIn("--plan", sys.argv)
                checkpoint(source, "step-50")
            with patch.dict(os.environ, env), patch.dict(sys.modules, {
                    "srgc_shared_storage": storage, "srgc_checkpoint_backup": backup,
                    "srgc_process_guard": Mock(process_guard=lambda plan: nullcontext()),
                    "srgc_step_checkpoints": Mock(worker_main=worker)}), \
                    patch("sys.argv", ["run_srgc_rebuttal.py", "worker", "--dataset", "math"]), \
                    patch("srgc_rebuttal.existing_runtime.select_python"), \
                    patch.object(storage, "route_plan", return_value=plan) as route, \
                    patch.object(entry.runpy, "run_module") as dispatch, redirect_stdout(io.StringIO()):
                entry.main()
                self.assertTrue(route.call_args.kwargs["start_or_continue"])
                dispatch.assert_not_called()
            self.assertTrue(any(p.read_bytes() == source.read_bytes() for p in self.generations(source)))

    def test_auto_watcher_takes_final_copy_and_backup_errors_do_not_interrupt_training(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _, env = self.fixture(directory)
            with patch.dict(os.environ, env), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                    patch.object(backup, "backup_once", side_effect=OSError("storage full")) as capture:
                with backup.automatic_backup(plan):
                    pass
                self.assertGreaterEqual(capture.call_count, 1)

    def test_watcher_reports_initial_state_and_final_scan(self):
        for state, changes in (
            ("waiting_for_checkpoint", {}),
            ("no_new_checkpoint", {"found": 1, "unchanged": 1}),
            ("copied", {"found": 1, "saved": 1}),
            ("another_watcher_copying", {"busy": True}),
            ("error", {"found": 1, "errors": ["copy failed"]}),
        ):
            with self.subTest(state=state):
                report = {"found": 0, "saved": 0, "unchanged": 0, "busy": False, "errors": [], **changes}
                stop = Mock()
                stop.is_set.side_effect = [False, True]
                output, errors = io.StringIO(), io.StringIO()
                with patch.object(backup, "backup_once", return_value=report), \
                        redirect_stdout(output), redirect_stderr(errors):
                    backup.watch(Path("fixture.json"), stop)
                stop.wait.assert_called_once_with(30)
                lines = output.getvalue().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertIn(f"state={state}", lines[0])
                self.assertIn(f"saved={report['saved']}", lines[0])
                self.assertIn("scan_interval=30s", lines[0])
                self.assertIn("final=true", lines[1])
                self.assertNotIn("next_check_in", lines[1])
                if report["errors"]:
                    self.assertIn("copy failed", errors.getvalue())

    def test_unchanged_waiting_scans_are_silent_but_polling_and_final_copy_continue(self):
        for changes in ({}, {"found": 1, "unchanged": 1}, {"busy": True}):
            with self.subTest(changes=changes):
                report = {"found": 0, "saved": 0, "unchanged": 0, "busy": False, "errors": [], **changes}
                stop, output = Mock(), io.StringIO()
                stop.is_set.side_effect = [False] * 5 + [True]
                with patch.object(backup, "backup_once", return_value=report) as scan, redirect_stdout(output):
                    backup.watch(Path("fixture.json"), stop)
                self.assertEqual(scan.call_count, 6)
                self.assertEqual(stop.wait.call_count, 5)
                lines = output.getvalue().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertIn("unchanged_scans_silent=true", lines[0])
                self.assertIn("final=true", lines[1])

    def test_state_changes_new_copies_and_repeated_errors_remain_visible(self):
        empty = {"found": 0, "saved": 0, "unchanged": 0, "busy": False, "errors": []}
        copied = {**empty, "found": 1, "saved": 1}
        unchanged = {**empty, "found": 1, "unchanged": 1}
        error = {**unchanged, "errors": ["copy failed"]}
        reports = [empty, empty, copied, copied, unchanged, unchanged, error, error, unchanged]
        stop, output, errors = Mock(), io.StringIO(), io.StringIO()
        stop.is_set.side_effect = [False] * (len(reports) - 1) + [True]
        with patch.object(backup, "backup_once", side_effect=reports), \
                redirect_stdout(output), redirect_stderr(errors):
            backup.watch(Path("fixture.json"), stop)
        text = output.getvalue()
        self.assertEqual(text.count("state=waiting_for_checkpoint"), 1)
        self.assertEqual(text.count("state=copied"), 2)
        self.assertEqual(text.count("state=no_new_checkpoint"), 2)
        self.assertEqual(text.count("state=error"), 2)
        self.assertEqual(errors.getvalue().count("copy failed"), 2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional PyTorch")
    def test_real_torch_checkpoint_restores_model_and_optimizer_values(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            plan, source, env = self.fixture(directory)
            state = {"step": 75, "model": torch.arange(4),
                     "optimizer": {"step": 75, "exp_avg": torch.ones(4)},
                     "used_training_ids": ["p1"], "switched_at": 50}
            torch.save(state, source)
            self.assertEqual(self.capture(plan, env)["saved"], 1)
            copied, = self.generations(source)
            restored = torch.load(copied, weights_only=False)
            self.assertEqual(restored["step"], 75)
            self.assertEqual(restored["switched_at"], 50)
            self.assertEqual(restored["used_training_ids"], ["p1"])
            torch.testing.assert_close(restored["model"], state["model"])
            torch.testing.assert_close(restored["optimizer"]["exp_avg"], state["optimizer"]["exp_avg"])


if __name__ == "__main__":
    unittest.main()
