"""Shell regressions for run_h100.sh (Codex H100 review): CPU failures propagate; methods are forwarded."""
import os
import subprocess
import unittest
from pathlib import Path

from helpers import HERE, tmpdir

FAKE = """#!/bin/sh
echo "$*" >> "$FAKE_LOG"
case " $* " in *" $FAKE_FAIL "*) [ -n "$FAKE_FAIL" ] && exit 3 ;; esac
exit 0
"""


class RunH100(unittest.TestCase):
    def run_script(self, *args, fail=""):
        d = tmpdir()
        fake = d / "python"
        fake.write_text(FAKE)
        fake.chmod(0o755)
        log = d / "argv.log"
        env = dict(os.environ, V8_PYTHON=str(fake), V8_ROOT=str(d / "root"), V8_CODE_ROOT=str(d / "code"),
                   FAKE_LOG=str(log), FAKE_FAIL=fail, CUDA_VISIBLE_DEVICES="")
        r = subprocess.run(["sh", str(HERE / "run_h100.sh"), *args], env=env, capture_output=True, text=True, timeout=60)
        calls = log.read_text().splitlines() if log.exists() else []
        return r, calls

    def test_failing_cpu_step_fails_the_script(self):
        for stage, cmd in (("check", "inspect-repo"), ("check", "validate-config"), ("locate", "locate"),
                           ("profile", "profile"), ("collect", "collect"), ("freeze-budget", "freeze-budget")):
            with self.subTest(stage=stage, cmd=cmd):
                r, calls = self.run_script("olmo_math", stage, fail=cmd)
                self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("CPU step failed", r.stdout)

    def test_successful_cpu_stage_exits_zero(self):
        r, calls = self.run_script("olmo_math", "check")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(any("estimate olmo_math" in c for c in calls))

    def test_failing_pending_fails_gpu_stage(self):
        r, _ = self.run_script("olmo_math", "trajectory", fail="pending")
        self.assertNotEqual(r.returncode, 0)

    def test_failing_make_matrix_fails_gpu_stage(self):
        r, _ = self.run_script("olmo_math", "branches", fail="make-matrix")
        self.assertNotEqual(r.returncode, 0)

    def test_prospective_methods_forwarded_to_every_make_matrix(self):
        for stage, role in (("dev", "development"), ("test", "test")):
            with self.subTest(stage=stage):
                r, calls = self.run_script("olmo_math", stage)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                mm = [c for c in calls if "make-matrix" in c]
                self.assertTrue(mm)
                self.assertTrue(all("--methods" in c and f"--role {role}" in c for c in mm), mm)

    def test_seeds_forwarded(self):
        r, calls = self.run_script("olmo_math", "trajectory", "--seeds", "5,6")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(all("--seeds 5,6" in c for c in calls if "make-matrix" in c))

    def test_tune_freeze_is_explicitly_not_implemented(self):
        r, _ = self.run_script("olmo_math", "tune-freeze")
        self.assertEqual(r.returncode, 2)
        self.assertIn("NOT IMPLEMENTED", r.stdout)


if __name__ == "__main__":
    unittest.main()


class HistoryExport(unittest.TestCase):
    run_script = RunH100.run_script

    def test_skipped_without_export_root_and_failure_propagates(self):
        r, calls = self.run_script("olmo_math", "history-export")
        self.assertEqual(r.returncode, 0)
        self.assertIn("skipped", r.stdout)
        os.environ["EXPORT_CODE_ROOT"] = "/x/code"
        try:
            r, calls = self.run_script("olmo_math", "history-export", fail="--root")
            self.assertNotEqual(r.returncode, 0)
            self.assertTrue(any("srgc_v8_history.py --root" in c for c in calls))
        finally:
            del os.environ["EXPORT_CODE_ROOT"]
