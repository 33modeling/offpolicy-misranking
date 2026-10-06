"""Qwen runtime identity covers helpers that affect model execution and results."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import srgc_qwen35 as qwen
from srgc_rebuttal.tests.test_qwen_extension import ChatTokenizer


class QwenRuntimeIdentityTests(unittest.TestCase):
    def test_manual_prepare_cannot_replace_missing_plan_of_existing_run(self):
        for is_directory in (False, True):
            with self.subTest(is_directory=is_directory), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                run = destination / "runs/math"
                run.parent.mkdir(parents=True)
                if is_directory:
                    run.mkdir()
                    artifact = run / "checkpoint.pt"
                else:
                    artifact = run
                artifact.write_bytes(b"preserved experiment artifact")
                with patch.object(qwen, "make_bundle") as make_bundle:
                    with self.assertRaisesRegex(ValueError, "restore the original plan"):
                        qwen.prepare("math", ROOT / "srgc_rebuttal/experiments/additional_seeds.json",
                                     destination, ChatTokenizer())
                    make_bundle.assert_not_called()
                self.assertEqual(artifact.read_bytes(), b"preserved experiment artifact")
                self.assertFalse((destination / "experiments").exists())
                self.assertFalse((destination / "inputs").exists())

    def test_manual_prepare_allows_empty_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "runs/math").mkdir(parents=True)
            plan = qwen.prepare("math", ROOT / "srgc_rebuttal/experiments/additional_seeds.json",
                                destination, ChatTokenizer())
            self.assertEqual(qwen.validate_extension(plan)["model"], qwen.MODEL)

    def test_model_runtime_helpers_change_digest_and_cannot_resume_old_plan(self):
        helpers = ("scripts/srgc_resumable_rollouts.py", "scripts/srgc_child_tuning.py",
                   "scripts/srgc_direction_records.py")
        with tempfile.TemporaryDirectory() as directory:
            plan = qwen.prepare("math", ROOT / "srgc_rebuttal/experiments/additional_seeds.json",
                                Path(directory), ChatTokenizer())
            original = plan.read_bytes()
            recorded = json.loads(original)["adapter_sha256"]
            read_bytes = Path.read_bytes
            for helper in helpers:
                with self.subTest(helper=helper):
                    target = ROOT / helper

                    def changed_bytes(path):
                        value = read_bytes(path)
                        return value + b"\n# changed runtime helper\n" if path == target else value

                    with patch.object(Path, "read_bytes", changed_bytes):
                        self.assertNotEqual(qwen.adapter_digest(), recorded)
                        with self.assertRaisesRegex(ValueError, "adapter differs"):
                            qwen.validate_extension(plan)
                        historical = qwen.validate_extension(plan, read_only=True)
                        self.assertEqual(historical["adapter_sha256"], recorded)
                    self.assertEqual(plan.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
