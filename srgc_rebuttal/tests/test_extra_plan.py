import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from scripts.srgc_extra_plan import select_plan
from srgc_rebuttal.plan import digest, load_plan
from srgc_rebuttal.runtime import run_root
from srgc_rebuttal.tests import test_extra_arm_launch


class ExtraPlanTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_extra_arm_launch.ExtraArmLaunchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.old = self.fixture.plan("mbpp", pair=False)
        self.folder, _, _ = self.fixture.prefix(self.old)
        self.new = self.fixture.storage / "fresh/empty-replacement/experiments/mbpp_seeds.json"
        self.new.parent.mkdir(parents=True)
        self.new.write_bytes(self.old.read_bytes())
        (self.new.parent.parent / "automatic-restart.json").write_text(json.dumps({
            "plan": str(self.new), "previous_plan": str(self.old), "reason": "implementation changed"}))

    def test_mbpp_empty_replacement_uses_saved_prefix_without_changing_files(self):
        files = {p: p.read_bytes() for p in self.fixture.storage.rglob("*") if p.is_file()}
        self.assertEqual(select_plan(self.new, 5), self.old)
        self.assertEqual(select_plan(self.new), self.old)
        self.assertEqual(files, {p: p.read_bytes() for p in files})

    def test_present_or_incomplete_prefix_is_not_silently_bypassed(self):
        folder = run_root(self.new, load_plan(self.new)) / "seed-5"
        folder.mkdir(parents=True)
        for name in ("prefix-ready.json", "prefix.pt", "prefix-latest.pt"):
            path = folder / name
            path.write_text("corrupt fixture")
            self.assertEqual(select_plan(self.new, 5), self.new)
            path.unlink()

    def test_different_plan_and_cycles_fail_instead_of_selecting_another_experiment(self):
        plan = json.loads(self.old.read_text())
        plan["analysis"] = "different experiment"
        self.old.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "different settings"):
            select_plan(self.new, 5)
        self.old.write_bytes(self.new.read_bytes())
        (self.folder / "prefix.pt").unlink()
        (self.folder / "prefix-ready.json").unlink()
        (self.old.parent.parent / "automatic-restart.json").write_text(json.dumps({
            "plan": str(self.old), "previous_plan": str(self.new), "reason": "implementation changed"}))
        with self.assertRaisesRegex(ValueError, "cycle"):
            select_plan(self.new, 5)

    def test_actual_mbpp_shell_routes_fixed_controls_to_saved_cohort(self):
        source = test_extra_arm_launch.ROOT / "srgc_rebuttal/experiments/mbpp_seeds.json"
        (self.fixture.storage / ".mbpp_seeds-active.json").write_text(json.dumps({
            "plan": str(self.new), "source_plan_sha256": digest(source)}))
        fake = self.fixture.base / "python"
        fake.write_text(f"#!{sys.executable}\nimport json, subprocess, sys\n"
                        "if sys.argv[1] == '-':\n"
                        f"    raise SystemExit(subprocess.run([{sys.executable!r}, *sys.argv[1:]], input=sys.stdin.read(), text=True).returncode)\n"
                        "print(json.dumps(sys.argv[1:]))\n")
        fake.chmod(0o755)
        pgrep = self.fixture.base / "pgrep"
        pgrep.write_text("#!/bin/sh\nexit 1\n")
        pgrep.chmod(0o755)
        env = {**os.environ, **self.fixture.environment, "SWITCH_PYTHON": str(fake),
               "PATH": str(self.fixture.base) + os.pathsep + os.environ["PATH"]}
        env.pop("SRGC_STORAGE_ROOT", None)
        for arm in ("switch_fixed100", "switch_fixed200"):
            result = subprocess.run(["sh", str(test_extra_arm_launch.SCRIPT), "mbpp", "5", arm],
                                    cwd="/tmp", env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), ["scripts/srgc_extra_worker.py", "--plan", str(self.old),
                                                        "--seed", "5", "--arm", arm])


if __name__ == "__main__":
    unittest.main()
