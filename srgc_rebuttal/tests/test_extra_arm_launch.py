import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.srgc_sr_refresh import prepare_run_storage, results
from srgc_rebuttal.plan import digest, input_path, load_plan
from srgc_rebuttal.runtime import identity, run_root


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_srgc_sr_refresh.sh"


class ExtraArmLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="extra arms ")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.group = self.base / "group"
        self.work = self.group / "u/offpolicy-misranking"
        self.storage = self.work / "srgc-rebuttal"
        self.environment = {"GROUP_VOLUME": str(self.group), "OM_WORK": str(self.work)}
        self.group.mkdir()

    def plan(self, dataset="math", pair=True):
        name = ("pair_seeds.json" if pair else "additional_seeds.json") if dataset == "math" else (
            "mbpp_pair_seeds.json" if pair else "mbpp_seeds.json")
        source = ROOT / "srgc_rebuttal/experiments" / name
        plan = self.storage / "fresh/candidate40-v2" / source.stem / "experiments" / name
        plan.parent.mkdir(parents=True)
        plan.write_bytes(source.read_bytes())
        pointer = self.storage / f".{source.stem}-active.json"
        pointer.write_text(json.dumps({"plan": str(plan), "source_plan_sha256": digest(source)}))
        return plan

    def prefix(self, plan):
        spec = load_plan(plan)
        bundle = input_path(plan, spec, 5)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_text('{"fixture": true}\n')
        folder = run_root(plan, spec) / "seed-5"
        folder.mkdir(parents=True)
        checkpoint = folder / "prefix.pt"
        checkpoint.write_bytes(b"synthetic checkpoint, not a GPU result")
        expected = identity(plan, spec, 5)
        receipt = {**expected, "completed_updates": 25, "checkpoint_sha256": digest(checkpoint)}
        (folder / "prefix-ready.json").write_text(json.dumps(receipt))
        return folder, expected, receipt

    def test_shell_reports_follow_active_math_and_mbpp_pair_cohorts(self):
        fake = self.base / "python"
        fake.write_text(f"#!{sys.executable}\nimport json, subprocess, sys\n"
                        f"real = {sys.executable!r}\n"
                        "if sys.argv[1] == '-':\n"
                        "    raise SystemExit(subprocess.run([real, *sys.argv[1:]], input=sys.stdin.read(), text=True).returncode)\n"
                        "print(json.dumps(sys.argv[1:]))\n")
        fake.chmod(0o755)
        for dataset in ("math", "mbpp"):
            with self.subTest(dataset=dataset):
                plan = self.plan(dataset)
                env = {**os.environ, **self.environment, "PAIR_PYTHON": str(fake), "SWITCH_PYTHON": str(fake)}
                env.pop("SRGC_STORAGE_ROOT", None)
                output = subprocess.run(["sh", str(SCRIPT), dataset, "results"], cwd="/tmp", env=env,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(output.returncode, 0, output.stderr)
                self.assertEqual(json.loads(output.stdout), ["scripts/srgc_sr_refresh.py", "results", "--plan", str(plan)])

    def test_explicit_missing_python_is_not_silently_replaced(self):
        result = subprocess.run(["sh", str(SCRIPT), "math", "results"], env={**os.environ,
                                "PAIR_PYTHON": str(self.base / "missing")}, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Python not found", result.stderr)

    def test_runtime_cache_environment_is_set_in_the_calling_process(self):
        plan = self.plan()
        self.prefix(plan)
        args = SimpleNamespace(plan=plan, seed=5)
        with patch.dict(os.environ, self.environment, clear=True), contextlib.redirect_stderr(io.StringIO()):
            prepare_run_storage(args)
            for key in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR", "TMPDIR"):
                self.assertTrue(Path(os.environ[key]).is_relative_to(self.group), key)
            self.assertEqual(args.plan, plan)

    def test_missing_or_changed_prefix_aborts_before_runtime_cache_creation(self):
        plan = self.plan()
        spec = load_plan(plan)
        bundle = input_path(plan, spec, 5)
        bundle.parent.mkdir(parents=True)
        bundle.write_text('{}\n')
        args = SimpleNamespace(plan=plan, seed=5)
        with patch.dict(os.environ, self.environment, clear=True):
            with self.assertRaisesRegex(ValueError, "prefix must finish"):
                prepare_run_storage(args)
        self.assertFalse((self.storage / "runtime-cache").exists())
        folder, _, _ = self.prefix(plan)
        (folder / "prefix.pt").write_bytes(b"changed")
        with patch.dict(os.environ, self.environment, clear=True):
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                prepare_run_storage(args)

    def test_user_volume_output_is_rejected(self):
        source = ROOT / "srgc_rebuttal/experiments/additional_seeds.json"
        with patch.dict(os.environ, self.environment, clear=True):
            with self.assertRaisesRegex(ValueError, "group-volume"):
                prepare_run_storage(SimpleNamespace(plan=source, seed=5))

    def endpoint(self, plan, cost=None):
        folder, expected, prefix = self.prefix(plan)
        value = {**expected, "arm": "sr_refresh", "total_updates": 275,
                 "prefix_checkpoint_sha256": prefix["checkpoint_sha256"], "reward": 0.5,
                 "costs": {"selection_gpu_seconds": cost, "training_gpu_seconds": 10},
                 "cost_measurement_complete": cost is not None}
        path = folder / "sr_refresh-endpoint.json"
        path.write_text(json.dumps(value))
        return path, value

    def test_results_do_not_display_missing_cost_as_zero(self):
        plan = self.plan()
        self.endpoint(plan)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=False))
        self.assertIn("unknown", output.getvalue())
        self.assertIn("cost measurement incomplete or unverified", output.getvalue())

    def test_results_keep_real_zero_cost_and_json_completeness(self):
        plan = self.plan()
        self.endpoint(plan, cost=0.0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=True))
        value = json.loads(output.getvalue())["rows"][0]["sr_refresh"]
        self.assertEqual(value["selection_gpu_seconds"], 0.0)
        self.assertTrue(value["cost_measurement_complete"])

    def test_results_reject_wrong_identity_prefix_arm_or_horizon(self):
        plan = self.plan()
        path, value = self.endpoint(plan)
        for key, wrong in (("input_sha256", "wrong"), ("prefix_checkpoint_sha256", "wrong"),
                           ("arm", "switch_repeat"), ("total_updates", 250)):
            with self.subTest(key=key):
                path.write_text(json.dumps({**value, key: wrong}))
                with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                    results(SimpleNamespace(plan=plan, json=True))


if __name__ == "__main__":
    unittest.main()
