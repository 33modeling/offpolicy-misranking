import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import srgc_qwen35_start as start
import run_srgc_qwen35 as launcher
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

    def prepare(self, command, root, environment, **kwargs):
        self.assertTrue(kwargs["pass_fds"])
        self.assertEqual(command[3], "prepare")
        self.saved(command[2])

    def test_fresh_prepares_each_dataset_then_restart_reuses(self):
        with patch.object(start, "ensure_model"), patch.object(start, "run_preparation", side_effect=self.prepare) as run:
            start.prepare_missing(("math", "mbpp"), self.root, self.env)
            self.assertEqual(run.call_count, 2)
            start.prepare_missing(("math", "mbpp"), self.root, self.env)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(self.plan("math").read_text(), "saved plan\n")

    def test_existing_plan_does_not_follow_changed_source_cohort(self):
        self.saved("math")
        with patch.object(start, "ensure_model"), patch.object(start, "run_preparation", side_effect=self.prepare) as run:
            start.prepare_missing(("math", "mbpp"), self.root, self.env)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][2:4], ["mbpp", "prepare"])

    def test_invalid_existing_plan_stops_before_download_or_other_preparation(self):
        self.saved("mbpp")
        self.validate_saved.side_effect = ValueError("adapter mismatch")
        with patch.object(start, "ensure_model") as model, patch.object(start, "run_preparation") as run:
            with self.assertRaisesRegex(ValueError, "adapter mismatch"):
                start.prepare_missing(("math", "mbpp"), self.root, self.env)
            model.assert_not_called()
            run.assert_not_called()

    def test_missing_plan_with_existing_run_is_not_recreated(self):
        run_root = self.root / 'runs/math/seed-5'
        run_root.mkdir(parents=True)
        checkpoint = run_root / 'prefix.pt'
        checkpoint.write_bytes(b'keep existing checkpoint')
        with patch.object(start, 'ensure_model') as model, patch.object(start, 'run_preparation') as run:
            with self.assertRaisesRegex(ValueError, 'missing plan.*existing'):
                start.prepare_missing(('math',), self.root, self.env)
            model.assert_not_called()
            run.assert_not_called()
        self.assertFalse(self.plan('math').exists())
        self.assertEqual(checkpoint.read_bytes(), b'keep existing checkpoint')

    def test_sigterm_during_preparation_stops_owned_child(self):
        ready, stopped = self.root / 'child-ready', self.root / 'child-stopped'
        code = ('import os, signal, time\nfrom pathlib import Path\n'
                f'def stop(*args):\n    Path({str(stopped)!r}).write_text("stopped")\n    raise SystemExit(143)\n'
                'signal.signal(signal.SIGTERM, stop)\n'
                f'Path({str(ready)!r}).write_text(str(os.getpid()))\n'
                'while True: time.sleep(.05)\n')
        ctx = multiprocessing.get_context('fork')
        def prepare_and_stop():
            try:
                start.prepare_missing(('math',), self.root, self.env)
            except KeyboardInterrupt:
                raise SystemExit(130)
        with patch.object(start, 'ensure_model'), \
             patch.object(start, 'controller', return_value=[sys.executable, '-c', code]):
            process = ctx.Process(target=prepare_and_stop)
            process.start()
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline and process.is_alive():
                    time.sleep(.05)
                self.assertTrue(ready.exists())
                process.terminate()
                process.join(10)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 130)
                self.assertTrue(stopped.exists(), 'preparation child survived parent SIGTERM')
                with lease(self.root / '.start-prepare.lock'):
                    pass
            finally:
                if ready.exists() and not stopped.exists():
                    try:
                        os.kill(int(ready.read_text()), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 5
                    while not stopped.exists() and time.monotonic() < deadline:
                        time.sleep(.05)
                if process.is_alive():
                    process.kill()
                process.join(10)

    def test_preparation_preserves_failure_code_environment_and_log(self):
        environment = {**os.environ, 'QWEN_TEST_VALUE': 'sentinel'}
        command = [sys.executable, '-c',
                   'import os; print(os.environ["QWEN_TEST_VALUE"], flush=True); raise SystemExit(7)']
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            start.run_preparation(command, self.root, environment)
        self.assertEqual(failure.exception.returncode, 7)
        logs = list((self.root / 'startup-logs').glob('*.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('sentinel', logs[0].read_text())

    def test_interrupted_main_does_not_enter_training(self):
        with patch.object(sys, 'argv', ['start', 'mbpp']), \
             patch.object(start, 'default_root', return_value=self.root), \
             patch.object(start, 'prepare_missing', side_effect=KeyboardInterrupt), \
             patch.object(os, 'execv') as execute:
            with self.assertRaises(SystemExit) as stopped:
                start.main()
            self.assertEqual(stopped.exception.code, 130)
            execute.assert_not_called()

    def test_start_preserves_admission_diagnostics_after_exec(self):
        with patch.object(sys, "argv", ["start", "math"]), \
             patch.object(start, "default_root", return_value=self.root), \
             patch.object(start, "prepare_missing"), patch.object(os, "execv") as execute:
            start.main()
        execute.assert_called_once_with(sys.executable, [sys.executable,
            str(ROOT / "scripts/srgc_qwen35_diagnostics.py"), "math", "run", "--root", str(self.root)])

    def test_bad_packages_stop_before_model_download(self):
        self.runtime_packages.side_effect = ImportError("missing package")
        with patch.object(start, "ensure_model") as model:
            with self.assertRaises(ImportError):
                start.prepare_missing(("math",), self.root, self.env)
            model.assert_not_called()

    def test_download_failure_does_not_prepare_or_start_training(self):
        with patch.object(start, "specification", return_value={"local_directory": "qwen"}), \
             patch.object(start, "run_preparation", side_effect=subprocess.CalledProcessError(1, "download")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                start.prepare_missing(("math",), self.root, self.env)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], [
                sys.executable, str(ROOT / "src/model_matrix.py"),
                "--config", str(ROOT / "configs/qwen35_9b_grpo.json"),
                "--models-dir", self.env["MODELS_DIR"], "download", "qwen3.5-9b-posttrained"])

    def test_existing_invalid_model_is_not_downloaded_over(self):
        (self.root / "models/qwen").mkdir(parents=True)
        with patch.object(start, "specification", return_value={"local_directory": "qwen"}), \
             patch.object(start, "model_path", side_effect=ValueError("bad snapshot")), \
             patch.object(start, "run_preparation") as run:
            with self.assertRaisesRegex(ValueError, "bad snapshot"):
                start.ensure_model(self.root, self.env)
            run.assert_not_called()

    def test_missing_explicit_model_is_not_replaced_by_default(self):
        self.env["SRGC_QWEN_MODEL_PATH"] = str(self.root / "missing-model")
        with patch.object(start, "run_preparation") as run:
            with self.assertRaises(FileNotFoundError):
                start.ensure_model(self.root, self.env)
            run.assert_not_called()

    def test_download_is_skipped_after_model_appears(self):
        destination = self.root / "models/qwen"
        def downloaded(*args, **kwargs):
            destination.mkdir()
        with patch.object(start, "specification", return_value={"local_directory": "qwen"}), \
             patch.object(start, "model_path") as verify, \
             patch.object(start, "run_preparation", side_effect=downloaded) as run:
            start.ensure_model(self.root, self.env)
            start.ensure_model(self.root, self.env)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(verify.call_count, 2)

    def test_manual_download_reuses_startup_validation_and_lock(self):
        model = self.root / "explicit-model"
        model.mkdir()
        environment = {**self.env, "SRGC_QWEN_MODEL_PATH": str(model)}
        with patch.dict(os.environ, environment), \
             patch.object(sys, "argv", ["run", "all", "download", "--root", str(self.root)]), \
             patch.object(launcher, "setup_storage", return_value=(self.root, self.root)), \
             patch.object(subprocess, "call", side_effect=AssertionError("unlocked download")), \
             patch.object(start, "model_path", side_effect=ValueError("bad snapshot")), \
             patch.object(start, "run_preparation") as run:
            with self.assertRaisesRegex(ValueError, "bad snapshot"):
                launcher.main()
        run.assert_not_called()
        self.assertTrue(model.is_dir())

    def test_manual_download_preserves_failure_and_interruption_codes(self):
        for failure, code in ((subprocess.CalledProcessError(7, "download"), 7),
                              (subprocess.CalledProcessError(-15, "download"), 143),
                              (KeyboardInterrupt(), 130)):
            with self.subTest(code=code), \
                 patch.object(sys, "argv", ["run", "all", "download", "--root", str(self.root)]), \
                 patch.object(launcher, "setup_storage", return_value=(self.root, self.root)), \
                 patch.object(start, "ensure_model", side_effect=failure):
                with self.assertRaises(SystemExit) as stopped:
                    launcher.main()
                self.assertEqual(stopped.exception.code, code)

    def test_manual_and_automatic_downloads_share_the_model_lease(self):
        ctx = multiprocessing.get_context("fork")
        entered, release = ctx.Event(), ctx.Event()
        calls = self.root / "download-calls.txt"
        destination = self.root / "models/qwen"
        def downloaded(command, root, environment, **kwargs):
            self.assertTrue(kwargs["pass_fds"])
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test release timeout")
            with calls.open("a") as handle:
                handle.write("download\n")
            destination.mkdir()
        with patch.dict(os.environ, self.env), \
             patch.object(sys, "argv", ["run", "all", "download", "--root", str(self.root / "other-run")]), \
             patch.object(launcher, "setup_storage", return_value=(self.root, self.root)), \
             patch.object(subprocess, "call", side_effect=AssertionError("unlocked download")), \
             patch.object(start, "specification", return_value={"local_directory": "qwen"}), \
             patch.object(start, "model_path", side_effect=lambda *args: str(destination) if destination.is_dir() else self.fail("model missing")), \
             patch.object(start, "run_preparation", side_effect=downloaded):
            automatic = ctx.Process(target=start.ensure_model, args=(self.root, self.env))
            manual = ctx.Process(target=launcher.main)
            workers = (automatic, manual)
            try:
                automatic.start()
                self.assertTrue(entered.wait(10))
                manual.start()
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
        self.assertEqual(calls.read_text().splitlines(), ["download"])

    def test_two_processes_prepare_once_and_release_lock(self):
        ctx = multiprocessing.get_context("fork")
        entered, release = ctx.Event(), ctx.Event()
        calls = self.root / "prepare-calls.txt"
        def prepare(command, root, environment, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test release timeout")
            with calls.open("a") as handle:
                handle.write(command[2] + "\n")
            self.saved(command[2])
        with patch.object(start, "ensure_model"), patch.object(start, "run_preparation", side_effect=prepare):
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
            command[1] = str(ROOT / "scripts/srgc_qwen35_diagnostics.py")
            execute.assert_called_once_with(sys.executable, command)


if __name__ == "__main__":
    unittest.main()
