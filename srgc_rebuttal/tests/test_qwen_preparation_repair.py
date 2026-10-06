"""Recover the known label-rejected preparation without touching experiment data."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import srgc_qwen35 as qwen
import srgc_qwen35_start as start
from srgc_rebuttal.plan import input_path
from srgc_rebuttal.tests.test_qwen_extension import ChatTokenizer


class RejectedPreparationTests(unittest.TestCase):
    def fixture(self, root):
        plan = qwen.prepare("math", ROOT / "srgc_rebuttal/experiments/additional_seeds.json", root, ChatTokenizer())
        spec = json.loads(plan.read_text())
        for seed in spec["seeds"]:
            path = input_path(plan, spec, seed)
            data = json.loads(path.read_text())
            data["dataset"] = "math500"
            data["provenance"]["source_provenance"].update(
                source_run="recorded-pair-run", reused_from_seed=3,
                prompt_format="olmo_rlzero_math")
            path.write_text(json.dumps(data))
        spec["adapter_sha256"] = start.REJECTED_PAIR_ADAPTER
        plan.write_text(json.dumps(spec))
        return plan, spec

    def test_start_recovers_only_preparation_and_preserves_inputs_and_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, spec = self.fixture(root)
            original = plan.read_bytes()
            inputs = {input_path(plan, spec, seed): input_path(plan, spec, seed).read_bytes()
                      for seed in spec["seeds"]}
            with patch.object(start, "setup_storage"), patch.object(start, "runtime_packages"), \
                    patch.object(start, "ensure_model"), patch.object(start, "run_preparation") as prepare:
                start.prepare_missing(("math",), root, {})
                start.prepare_missing(("math",), root, {})
            prepare.assert_not_called()
            start.validate_saved(plan)
            self.assertEqual(json.loads(plan.read_text()), {**spec, "adapter_sha256": qwen.adapter_digest()})
            backups = list(plan.parent.glob("*.before-math-label-fix-*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            for path, content in inputs.items():
                self.assertEqual(path.read_bytes(), content)

    def test_execution_and_cache_artifacts_block_repair(self):
        for kind in ("run", "cache", "rewards"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan, spec = self.fixture(root)
                original = plan.read_bytes()
                bundle = input_path(plan, spec, 5)
                if kind == "rewards":
                    data = json.loads(bundle.read_text())
                    data["cached_rewards"] = {data["candidate_ids"][0]: [0] * 8}
                    bundle.write_text(json.dumps(data))
                else:
                    artifact = ((root / "runs/math/seed-5/prefix.pt") if kind == "run"
                                else bundle.with_suffix(".cache") / "protocol.json")
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text("preserved")
                with self.assertRaisesRegex(ValueError, "already"):
                    start.repair_rejected_pair_plan(plan)
                self.assertEqual(plan.read_bytes(), original)
                self.assertFalse(list(plan.parent.glob("*.before-math-label-fix-*.json")))

    def test_unknown_adapter_stays_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, spec = self.fixture(Path(directory))
            spec["adapter_sha256"] = "a" * 64
            plan.write_text(json.dumps(spec))
            original = plan.read_bytes()
            self.assertFalse(start.repair_rejected_pair_plan(plan))
            self.assertEqual(plan.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
