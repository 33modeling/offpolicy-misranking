import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import srgc_qwen35_start as start
from srgc_rebuttal.runtime import lease


class QwenStartTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = {"MODELS_DIR": str(self.root / "models")}
        for name in ("setup_storage", "runtime_packages", "validate_saved"):
            mock = patch.object(start, name).start()
            self.addCleanup(patch.stopall)
            setattr(self, name, mock)

    def plan(self, dataset):
        return self.root / "experiments" / f"qwen35-9b-{dataset}.json"

    def saved(self, dataset):
        plan = self.plan(dataset)
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text("saved plan\n")
        return plan

    def prepare(self, command, **kwargs):
        self.assertTrue(kwargs["check"])
        self.assertEqual(command[3], "prepare")
        self.saved(command[2])

    def test_fresh_prepares_each_dataset_then_restart_reuses(self):
        with patch.object(start, "ensure_model"), patch.object(start.subprocess, "run", side_effect=self.prepare) as run:
            start.prepare_missing(("math", "mbpp"), self.root, self.env)
            self.assertEqual(run.call_count, 2)
            start.prepare_missing(("math", "mbpp"), self.root, self.env)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(self.plan("math").read_text(), "saved plan\n")

    def test_existing_plan_does_not_follow_changed_source_cohort(self):
        self.saved("math")
        with patch.object(start, "ensure_model"), patch.object(start.subprocess, "run", side_effect=self.prepare) as run:
            start.prepare_missing(("math", "mbpp"), self.root, self.env)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][2:4], ["mbpp", "prepare"])

    def test_invalid_existing_plan_stops_before_download_or_other_preparation(self):
        self.saved("mbpp")
        self.validate_saved.side_effect = ValueError("adapter mismatch")
        with patch.object(start, "ensure_model") as model, patch.object(start.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "adapter mismatch"):
                start.prepare_missing(("math", "mbpp"), self.root, self.env)
            model.assert_not_called()
            run.assert_not_called()

    def test_bad_packages_stop_before_model_download(self):
        self.runtime_packages.side_effect = ImportError("missing package")
        with patch.object(start, "ensure_model") as model:
            with self.assertRaises(ImportError):
                start.prepare_missing(("math",), self.root, self.env)
            model.assert_not_called()

    def test_download_failure_does_not_prepare_or_start_training(self):
        with patch.object(start, "specification", return_value={"local_directory": "qwen"}), \
             patch.object(start.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "download")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                start.prepare_missing(("math",), self.root, self.env)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][3], "download")

    def test_existing_invalid_model_is_not_downloaded_over(self):
        (self.root / "models/qwen").mkdir(parents=True)
        with patch.object(start, "specification", return_value={"local_directory": "qwen"}), \
             patch.object(start, "model_path", side_effect=ValueError("bad snapshot")), \
             patch.object(start.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "bad snapshot"):
                start.ensure_model(self.root, self.env)
            run.assert_not_called()

    def test_missing_explicit_model_is_not_replaced_by_default(self):
        self.env["SRGC_QWEN_MODEL_PATH"] = str(self.root / "missing-model")
        with patch.object(start.subprocess, "run") as run:
            with self.assertRaises(FileNotFoundError):
                start.ensure_model(self.root, self.env)
            run.assert_not_called()

    def test_download_is_skipped_after_model_appears(self):
        destination = self.root / "models/qwen"
        def downloaded(*args, **kwargs):
            destination.mkdir()
        with patch.object(start, "specification", return_value={"local_directory": "qwen"}), \
             patch.object(start, "model_path") as verify, \
             patch.object(start.subprocess, "run", side_effect=downloaded) as run:
            start.ensure_model(self.root, self.env)
            start.ensure_model(self.root, self.env)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(verify.call_count, 2)

    def test_two_processes_prepare_once_and_release_lock(self):
        ctx = multiprocessing.get_context("fork")
        entered, release = ctx.Event(), ctx.Event()
        calls = self.root / "prepare-calls.txt"
        def prepare(command, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test release timeout")
            with calls.open("a") as handle:
                handle.write(command[2] + "\n")
            self.saved(command[2])
        with patch.object(start, "ensure_model"), patch.object(start.subprocess, "run", side_effect=prepare):
            workers = [ctx.Process(target=start.prepare_missing, args=(("math", "mbpp"), self.root, self.env)) for _ in range(2)]
            try:
                workers[0].start()
                self.assertTrue(entered.wait(10))
                workers[1].start()
                release.set()
                for worker in workers:
                    worker.join(10)
                    self.assertEqual(worker.exitcode, 0)
            finally:
                release.set()
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(10)
        self.assertEqual(calls.read_text().splitlines(), ["math", "mbpp"])
        with lease(self.root / ".start-prepare.lock"):
            pass

    def test_main_execs_existing_worker_after_preparation(self):
        with patch.object(sys, "argv", ["start"]), patch.object(start, "default_root", return_value=self.root), \
             patch.object(start, "prepare_missing") as prepare, patch.object(os, "execv") as execute:
            start.main()
            prepare.assert_called_once_with(("math", "mbpp"), self.root, os.environ)
            command = start.controller("all", "run", self.root)
            execute.assert_called_once_with(sys.executable, command)


if __name__ == "__main__":
    unittest.main()
