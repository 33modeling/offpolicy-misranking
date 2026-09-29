import functools
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import srgc_child_tuning as tuning
from srgc_rebuttal import progress, run_experiment

try:
    from srgc_rebuttal import torch_backend
except ImportError:  # the backend needs torch; the recorder and attention tests still run
    torch_backend = None


class ProgressCountTest(unittest.TestCase):
    def test_counts_reset_at_each_update_and_reach_the_rank_file(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"SRGC_PROGRESS_DIR": folder, "RANK": "2"}):
            record = tuning.counting_record(progress.record)
            for _ in range(3):
                record("rollout", prompt="p")
                record("gradient_scoring", prompt="p")
            row = json.loads((Path(folder) / "rank-2.json").read_text())
            self.assertEqual((row["stage"], row["count"], row["prompt"]), ("gradient_scoring", 3, "p"))
            record("policy_update")
            record("gradient_scoring", prompt="q")
            self.assertEqual(json.loads((Path(folder) / "rank-2.json").read_text())["count"], 1)

    @unittest.skipIf(torch_backend is None, "torch backend unavailable")
    def test_count_progress_patches_backend_and_runner_bindings(self):
        original_backend, original_runner = torch_backend.progress, run_experiment.progress
        try:
            wrapper = tuning.count_progress()
            self.assertIs(torch_backend.progress, wrapper)
            self.assertIs(run_experiment.progress, wrapper)
        finally:
            torch_backend.progress, run_experiment.progress = original_backend, original_runner


class AttentionTest(unittest.TestCase):
    def test_default_keeps_eager_and_env_selects_kernel(self):
        self.assertIsNone(tuning.configured_attention({}))
        self.assertEqual(tuning.configured_attention({"SRGC_ATTENTION": "sdpa"}), "sdpa")
        with self.assertRaises(ValueError):
            tuning.configured_attention({"SRGC_ATTENTION": "xformers"})
        original = run_experiment.load_model
        try:
            self.assertEqual(tuning.apply_attention({}), "eager")
            self.assertIs(run_experiment.load_model, original)
            self.assertEqual(tuning.apply_attention({"SRGC_ATTENTION": "sdpa"}), "sdpa")
            self.assertIsInstance(run_experiment.load_model, functools.partial)
            self.assertEqual(run_experiment.load_model.keywords, {"attention": "sdpa"})
        finally:
            run_experiment.load_model = original


if __name__ == "__main__":
    unittest.main()
