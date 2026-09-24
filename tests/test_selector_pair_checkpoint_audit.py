"""The diagnostic explains failures without modifying any saved experiment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/selector_pair_checkpoint_audit.py"
spec = importlib.util.spec_from_file_location("checkpoint_audit", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CheckpointAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "view-50/selection_reduced"
        self.policy = self.directory / "policy/checkpoint-100"
        self.policy.mkdir(parents=True)
        self.expected = {"start_step": 50, "target_steps": 500,
                         "config": {"group_size": 8, "checkpoint_every": 5}}
        self.state = {**self.expected, "completed_steps": 100}
        for name in ["adapter_config.json", *audit.FILES]:
            (self.policy / name).write_bytes(b"original saved artifact")
            if name in audit.FILES:
                self.state[audit.FILES[name]] = audit.digest(self.policy / name)
        self.write_state()
        (self.directory.parent / "contract.json").write_text("{}")
        self.runner = self.root / "runner.py"
        self.runner.write_text("# frozen runner\n")

    def write_state(self):
        (self.policy / "checkpoint_state.json").write_text(json.dumps(self.state))

    def snapshot(self):
        return {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.root.rglob("*") if p.is_file()}

    def inspect(self):
        before = self.snapshot()
        report = audit.checkpoint_report(self.policy, self.expected)
        self.assertEqual(self.snapshot(), before)
        return report

    def test_valid_files_are_read_only(self):
        self.assertTrue(self.inspect()["contract_and_hashes_match"])

    def test_actual_artifact_corruption_is_identified(self):
        for name in audit.FILES:
            with self.subTest(name=name):
                path = self.policy / name
                original = path.read_bytes()
                path.write_bytes(b"changed saved bytes")
                report = self.inspect()
                self.assertFalse(report["contract_and_hashes_match"])
                self.assertIn({"file": name, "problem": "hash mismatch",
                               "recorded": self.state[audit.FILES[name]],
                               "actual": audit.digest(path)}, report["issues"])
                path.write_bytes(original)

    def test_contract_field_is_reported_separately_from_hashes(self):
        self.state["config"] = {"group_size": 16, "checkpoint_every": 5}
        self.write_state()
        report = self.inspect()
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["contract_differences"], [
            {"field": "config.group_size", "problem": "mismatch", "recorded": 16, "expected": 8}])

    def test_empty_or_missing_files_are_not_valid(self):
        (self.policy / "adapter_config.json").write_bytes(b"")
        (self.policy / "optimizer.pt").unlink()
        report = self.inspect()
        self.assertIn({"file": "adapter_config.json", "problem": "empty"}, report["issues"])
        self.assertIn({"file": "optimizer.pt", "problem": "FileNotFoundError"}, report["issues"])

    def test_missing_pinned_checkpoint_is_not_replaced(self):
        plan = self.directory / "budget-recovery/plan.json"
        plan.parent.mkdir()
        selected = self.policy.parent / "checkpoint-90"
        plan.write_text(json.dumps({"points": [{"adapter": str(selected)}],
                                    "runner_sha256": audit.digest(self.runner)}))
        before = self.snapshot()
        report = audit.branch_report(self.directory, lambda *_: self.expected, self.runner)
        self.assertEqual(report["planned_checkpoint"], str(selected))
        self.assertEqual(report["checkpoints"][0]["checkpoint"], str(selected))
        self.assertTrue(report["checkpoints"][0]["issues"])
        self.assertTrue(report["checkpoints"][1]["contract_and_hashes_match"])
        self.assertEqual(self.snapshot(), before)

    def test_runner_mismatch_does_not_reseal_plan(self):
        plan = self.directory / "budget-recovery/plan.json"
        plan.parent.mkdir()
        plan.write_text(json.dumps({"points": [{"adapter": str(self.policy)}],
                                    "runner_sha256": "0" * 64}))
        before = self.snapshot()
        report = audit.branch_report(self.directory, lambda *_: self.expected, self.runner)
        self.assertFalse(report["runner_hash_matches"])
        self.assertEqual(self.snapshot(), before)

    def test_malformed_metadata_stays_untouched(self):
        (self.policy / "checkpoint_state.json").write_text("unfinished{")
        self.assertTrue(self.inspect()["issues"])

    def test_live_file_change_is_inconclusive(self):
        original = audit.digest
        def concurrent_write(path):
            value = original(path)
            if path.name == "optimizer.pt":
                path.write_bytes(b"peer published new content")
            return value
        audit.digest = concurrent_write
        try:
            report = audit.checkpoint_report(self.policy, self.expected)
        finally:
            audit.digest = original
        self.assertFalse(report["contract_and_hashes_match"])
        self.assertIn({"file": "optimizer.pt", "problem": "changed during read; inconclusive"}, report["issues"])


if __name__ == "__main__":
    unittest.main()
