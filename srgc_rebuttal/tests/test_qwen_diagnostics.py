"""Admission diagnostics only read bounded logs and preserve runner failures."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import srgc_qwen35_diagnostics as diagnostics


class QwenDiagnosticsTests(unittest.TestCase):
    def test_wrapper_preserves_failure_and_exposes_rank_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "qwen-smoke.log"
            original = ("[rank2]: RuntimeError: CUDA out of memory.\n"
                        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: \n")
            log.write_text(original)
            failure = RuntimeError(diagnostics.FAILURE_PREFIX + str(log))

            def run():
                raise failure

            with self.assertRaises(RuntimeError) as caught:
                diagnostics.run_with_diagnostics(run)
            self.assertIs(caught.exception, failure)
            self.assertEqual(str(caught.exception), diagnostics.FAILURE_PREFIX + str(log))
            note = "\n".join(caught.exception.__notes__)
            self.assertIn("[rank2]: RuntimeError: CUDA out of memory.", note)
            self.assertNotIn("ChildFailedError", note)
            self.assertEqual(log.read_text(), original)

    def test_error_mode_uses_latest_log_without_importing_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "runs/math/.queue/admission/old/qwen-smoke.log"
            second = root / "runs/math/.queue/admission/new/qwen-smoke.log"
            first.parent.mkdir(parents=True)
            first.write_text("[rank0]: ValueError: older failure\n")
            second.parent.mkdir(parents=True)
            second.write_text("[rank3]: TypeError: actual failure\n")
            import os
            os.utime(first, (1, 1))
            os.utime(second, (2, 2))
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = io.StringIO()
            runner = SimpleNamespace(main=lambda: self.fail("runner must not be invoked"))
            with patch.object(sys, "argv", ["diagnostics", "all", "error", "--root", str(root)]), \
                    patch.dict(sys.modules, {"run_srgc_qwen35": runner, "torch": None}), \
                    patch("scripts.srgc_qwen35_storage.setup_storage", side_effect=AssertionError("write attempted")), \
                    redirect_stdout(output):
                diagnostics.main()
            self.assertIn(str(second), output.getvalue())
            self.assertIn("actual failure", output.getvalue())
            self.assertNotIn("older failure", output.getvalue())
            self.assertIn("mbpp: no Qwen smoke log found", output.getvalue())
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_missing_log_does_not_mask_runner_exception(self):
        failure = RuntimeError(diagnostics.FAILURE_PREFIX + "/missing/qwen-smoke.log")
        with self.assertRaises(RuntimeError) as caught:
            diagnostics.run_with_diagnostics(lambda: (_ for _ in ()).throw(failure))
        self.assertIs(caught.exception, failure)
        self.assertIn("Unable to read log", "\n".join(caught.exception.__notes__))

    def test_normal_action_keeps_arguments_and_return_value(self):
        expected = ["diagnostics", "all", "status", "--root", "/example"]

        def run():
            self.assertEqual(sys.argv, expected)
            return 17

        with patch.object(sys, "argv", expected), \
                patch.dict(sys.modules, {"run_srgc_qwen35": SimpleNamespace(main=run)}):
            self.assertEqual(diagnostics.main(), 17)

    def test_run_wraps_admission_without_changing_arguments_or_leaking_patch(self):
        expected = ["diagnostics", "all", "run", "--root", "/example"]
        original = lambda *args, **kwargs: 17
        launcher = SimpleNamespace(admit_with_smoke=original)

        def run():
            self.assertEqual(sys.argv, expected)
            self.assertIsNot(launcher.admit_with_smoke, original)
            return launcher.admit_with_smoke(None, Path("/example"), {}, None)

        launcher.main = run
        with patch.object(sys, "argv", expected), patch.dict(sys.modules, {"run_srgc_qwen35": launcher}):
            self.assertEqual(diagnostics.main(), 17)
        self.assertIs(launcher.admit_with_smoke, original)

    def test_error_mode_follows_latest_recovery_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admission = root / "runs/math/.queue/admission/worker"
            latest = admission / "recovery-01/qwen-smoke.log"
            latest.parent.mkdir(parents=True)
            (admission / "qwen-smoke.log").write_text("RuntimeError: first failure\n")
            latest.write_text("RuntimeError: recovery failure\n")
            (admission / "qwen-admission.json").write_text(json.dumps({"qwen_smoke_log": str(latest)}))
            output = io.StringIO()
            with redirect_stdout(output):
                diagnostics.show_errors("math", root)
            self.assertIn("recovery failure", output.getvalue())
            self.assertNotIn("first failure", output.getvalue())

    def test_tail_read_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "qwen-smoke.log"
            log.write_text("RuntimeError: stale failure\n" + "x" * (300 * 1024) +
                           "\n[rank1]: ValueError: latest failure\n")
            details = diagnostics.failure_details(log)
            self.assertIn("latest failure", details)
            self.assertNotIn("stale failure", details)


if __name__ == "__main__":
    unittest.main()
