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
        folder.mkdir(parents=True, exist_ok=True)
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

    def test_extra_run_prepares_verifier_before_distributed_startup(self):
        from scripts.srgc_sr_refresh import run
        options = SimpleNamespace(plan=self.plan(), seed=5)
        class Prepared(Exception):
            pass
        with patch("scripts.srgc_sr_refresh.prepare_run_storage") as storage, \
                patch("scripts.srgc_sr_refresh.prepare_verifier_runtime", side_effect=Prepared) as verifier:
            with self.assertRaises(Prepared):
                run(options)
        storage.assert_called_once_with(options)
        verifier.assert_called_once_with()

    def test_extra_runtime_bootstraps_verifier_without_installed_distribution(self):
        # Isolate site-packages to reproduce the missing distribution without
        # changing the user's Python environment or installing from the network.
        command = """
import importlib.metadata
import sys
import types
sys.path.insert(0, sys.argv[1])
sys.modules['numpy'] = types.ModuleType('numpy')
try:
    importlib.metadata.version('math-verify')
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError('fixture must start without installed math-verify')
from scripts.srgc_sr_refresh import prepare_verifier_runtime
prepare_verifier_runtime()
from math_verify import parse, verify
assert verify(parse('1/2'), parse('0.5'))
assert importlib.metadata.version('math-verify') == '0.9.0'
print('offline verifier ready')
"""
        result = subprocess.run([sys.executable, "-S", "-c", command, str(ROOT)],
            env={**os.environ, **self.environment, "PYTHONPATH": ""},
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("offline verifier ready", result.stdout)

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

    def test_results_discover_all_saved_fixed_controls_and_validate_them(self):
        plan = self.plan()
        path, value = self.endpoint(plan, cost=1.0)
        for step in (75, 200):
            target = path.with_name(f"switch_fixed{step}-endpoint.json")
            target.write_text(json.dumps({**value, "arm": f"switch_fixed{step}", "switched_at": step}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=True))
        row = json.loads(output.getvalue())["rows"][0]
        self.assertEqual(row["switch_fixed200"]["switched_at"], 200)
        self.assertEqual(row["switch_fixed75"]["reward_percent"], 50.0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=False))
        self.assertIn("switch_fixed200: step 200 -> sr", output.getvalue())
        target.write_text(json.dumps({**value, "arm": "switch_fixed200", "input_sha256": "wrong"}))
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            results(SimpleNamespace(plan=plan, json=True))

    def test_results_list_the_direction_and_cached_sr_controls(self):
        plan = self.plan()
        path, value = self.endpoint(plan, cost=1.0)
        for arm in ("direction_removed", "direction_magnitude", "direction_replaced", "sr_hold"):
            path.with_name(f"{arm}-endpoint.json").write_text(json.dumps({**value, "arm": arm}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=True))
        row = json.loads(output.getvalue())["rows"][0]
        self.assertEqual({row[arm]["reward_percent"] for arm in ("direction_removed", "direction_magnitude",
                                                                  "direction_replaced", "sr_hold")}, {50.0})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=False))
        self.assertIn("direction_magnitude", output.getvalue().splitlines()[1])

    def replicate(self, plan, replicate=1, rewards=None, manifest_override=None, endpoint_override=None):
        from scripts.srgc_replicate import REPLICATE_PROTOCOL, sampling_seed
        folder, expected, prefix = self.prefix(plan)
        out = folder / f"replicate-{replicate}"
        out.mkdir(exist_ok=True)
        record = {"protocol": REPLICATE_PROTOCOL, "id": replicate, "base_seed": 5, "sampling_seed": sampling_seed(5, replicate)}
        manifest = {**expected, "prefix_checkpoint_sha256": prefix["checkpoint_sha256"], "replicate": record,
                    **(manifest_override or {})}
        (out / "replicate.json").write_text(json.dumps(manifest))
        for arm, reward in (rewards or {"sr": 0.5, "switch": 0.52}).items():
            (out / f"{arm}-endpoint.json").write_text(json.dumps({**expected, "arm": arm, "total_updates": 275,
                "prefix_checkpoint_sha256": prefix["checkpoint_sha256"], "reward": reward, "replicate": record,
                "costs": {"selection_gpu_seconds": 1.0, "training_gpu_seconds": 10}, "cost_measurement_complete": True,
                **(endpoint_override or {})}))
        return out

    def test_results_pair_replicate_arms_and_validate_their_manifest(self):
        plan = self.plan()
        self.replicate(plan, 1)
        self.replicate(plan, 2, rewards={"sr": 0.4})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=True))
        replicates = json.loads(output.getvalue())["replicates"]
        self.assertEqual([(r["seed"], r["replicate"]) for r in replicates], [(5, 1), (5, 2)])
        self.assertAlmostEqual(replicates[0]["switch"]["reward_percent"] - replicates[0]["sr"]["reward_percent"], 2.0)
        self.assertNotIn("switch", replicates[1])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=False))
        self.assertIn("seed 5 replicate 1", output.getvalue())
        self.assertIn("switch - sr = +2.00 pp", output.getvalue())
        self.assertIn("seed 5 replicate 2", output.getvalue())
        import shutil
        from scripts.srgc_replicate import sampling_seed
        wrong_protocol = {"protocol": "x", "id": 3, "base_seed": 5, "sampling_seed": sampling_seed(5, 3)}
        for override, message in (({"replicate": wrong_protocol}, "sampling stream"),
                                  ({"prefix_checkpoint_sha256": "wrong"}, "different shared prefix")):
            with self.subTest(message=message):
                out = self.replicate(plan, 3, manifest_override=override)
                with self.assertRaisesRegex(ValueError, message), contextlib.redirect_stdout(io.StringIO()):
                    results(SimpleNamespace(plan=plan, json=True))
                shutil.rmtree(out)
        self.replicate(plan, 4, endpoint_override={"replicate": {"protocol": "x"}})
        with self.assertRaisesRegex(ValueError, "replicate record differs"), contextlib.redirect_stdout(io.StringIO()):
            results(SimpleNamespace(plan=plan, json=True))

    def test_replicate_manifest_is_written_once_and_must_match_later(self):
        from scripts.srgc_sr_refresh import replicate_manifest
        plan = self.plan()
        folder, expected, prefix = self.prefix(plan)
        out = folder / "replicate-1"
        out.mkdir()
        first = replicate_manifest(out, expected, 5, 1, prefix["checkpoint_sha256"])
        self.assertEqual(json.loads((out / "replicate.json").read_text()), first)
        self.assertEqual(replicate_manifest(out, expected, 5, 1, prefix["checkpoint_sha256"]), first)
        with self.assertRaisesRegex(ValueError, "different experiment, prefix or sampling stream"):
            replicate_manifest(out, expected, 5, 1, "other prefix")
        with self.assertRaisesRegex(ValueError, "different experiment, prefix or sampling stream"):
            replicate_manifest(out, {**expected, "input_sha256": "x"}, 5, 1, prefix["checkpoint_sha256"])

    def test_extra_arm_names_accepted_by_the_cli_and_the_shell(self):
        import argparse
        from scripts.srgc_sr_refresh import extra_arm
        for name in ("sr_hold", "direction_removed", "direction_magnitude", "direction_replaced",
                     "replicate1-sr", "replicate2-switch_fixed200", "switch_fixed75"):
            self.assertEqual(extra_arm(name), name)
        for name in ("direction_sideways", "replicate0-sr", "replicate1-sr_refresh", "sr_refresh-cached"):
            with self.subTest(name=name), self.assertRaises(argparse.ArgumentTypeError):
                extra_arm(name)
        fake = self.base / "python"
        fake.write_text(f"#!{sys.executable}\nimport subprocess, sys\n"
                        f"raise SystemExit(subprocess.run([{sys.executable!r}, *sys.argv[1:]], "
                        "input=sys.stdin.read(), text=True).returncode)\n")
        fake.chmod(0o755)
        self.plan()
        env = {**os.environ, **self.environment, "PAIR_PYTHON": str(fake)}
        env.pop("SRGC_STORAGE_ROOT", None)
        for token in ("sr_hold", "direction_replaced", "replicate1-switch", "replicate2-switch_fixed200"):
            with self.subTest(token=token):
                # A valid third argument passes the launcher's arm check and fails on the non-numeric seed.
                result = subprocess.run(["sh", str(SCRIPT), "math", "x", token], cwd="/tmp", env=env,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("seed must be an integer", result.stderr)
        for token in ("direction_sideways", "replicate-switch", "hold"):
            with self.subTest(token=token):
                result = subprocess.run(["sh", str(SCRIPT), "math", "5", token], cwd="/tmp", env=env,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 2)
                self.assertIn("third argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
