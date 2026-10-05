import contextlib
import csv
import io
import json
import os
import signal
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import time
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
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_bytes(source.read_bytes())
        pointer = self.storage / f".{source.stem}-active.json"
        pointer.write_text(json.dumps({"plan": str(plan), "source_plan_sha256": digest(source)}))
        return plan

    def prefix(self, plan):
        spec = load_plan(plan)
        bundle = input_path(plan, spec, 5)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_text(json.dumps({"fixture": True, "evaluation_ids": [f"e{i}" for i in range(300)]}))
        folder = run_root(plan, spec) / "seed-5"
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "prefix.pt"
        checkpoint.write_bytes(b"synthetic checkpoint, not a GPU result")
        expected = identity(plan, spec, 5)
        receipt = {**expected, "completed_updates": 25, "checkpoint_sha256": digest(checkpoint)}
        (folder / "prefix-ready.json").write_text(json.dumps(receipt))
        (folder / "run.json").write_text(json.dumps(expected))
        return folder, expected, receipt

    def test_shell_reports_follow_active_math_and_mbpp_pair_cohorts(self):
        fake = self.base / "python"
        fake.write_text(f"#!{sys.executable}\nimport json, subprocess, sys\n"
                        f"real = {sys.executable!r}\n"
                        "if sys.argv[1] == '-':\n"
                        "    raise SystemExit(subprocess.run([real, *sys.argv[1:]], input=sys.stdin.read(), text=True).returncode)\n"
                        "print(json.dumps(sys.argv[1:]))\n")
        fake.chmod(0o755)
        for dataset, flags in (("math", []), ("mbpp", []), ("math", ["--json"]), ("mbpp", ["--json"])):
            with self.subTest(dataset=dataset, flags=flags):
                plan = self.plan(dataset)
                env = {**os.environ, **self.environment, "PAIR_PYTHON": str(fake), "SWITCH_PYTHON": str(fake)}
                env.pop("SRGC_STORAGE_ROOT", None)
                output = subprocess.run(["sh", str(SCRIPT), dataset, "results", *flags], cwd="/tmp", env=env,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(output.returncode, 0, output.stderr)
                self.assertEqual(json.loads(output.stdout), ["scripts/srgc_sr_refresh.py", "results", "--plan", str(plan), *flags])

    def test_shell_sigterm_stops_owned_worker_and_does_not_retry(self):
        self.plan()
        fake = self.base / "python"
        ready, stopped = self.base / "ready", self.base / "stopped"
        fake.write_text(f"#!{sys.executable}\nimport os, pathlib, signal, subprocess, sys, time\n"
                        "if sys.argv[1] == '-':\n"
                        f"    raise SystemExit(subprocess.run([{sys.executable!r}, *sys.argv[1:]], input=sys.stdin.read(), text=True).returncode)\n"
                        "def stop(*args):\n"
                        f"    pathlib.Path({str(stopped)!r}).write_text('stopped')\n"
                        "    raise SystemExit(143)\n"
                        "signal.signal(signal.SIGTERM, stop)\n"
                        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\n"
                        "while True: time.sleep(.05)\n")
        fake.chmod(0o755)
        pgrep = self.base / "pgrep"
        pgrep.write_text("#!/bin/sh\nexit 1\n")
        pgrep.chmod(0o755)
        env = {**os.environ, **self.environment, "PAIR_PYTHON": str(fake),
               "PATH": str(self.base) + os.pathsep + os.environ["PATH"]}
        env.pop("SRGC_STORAGE_ROOT", None)
        process = subprocess.Popen(["sh", str(SCRIPT), "math", "5", "replicate1-sr"], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(.05)
            self.assertTrue(ready.exists())
            process.send_signal(signal.SIGTERM)
            _, error = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 143, error)
            self.assertTrue(stopped.exists())
            self.assertNotIn("retrying", error)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
            if ready.exists() and not stopped.exists():
                try:
                    os.kill(int(ready.read_text()), signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def test_explicit_missing_python_is_not_silently_replaced(self):
        result = subprocess.run(["sh", str(SCRIPT), "math", "results"], env={**os.environ,
                                "PAIR_PYTHON": str(self.base / "missing")}, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Python not found", result.stderr)

    def test_all_120_documented_task_commands_forward_the_right_tuple(self):
        plans = {dataset: self.plan(dataset) for dataset in ("math", "mbpp")}
        fake = self.base / "python"
        fake.write_text(f"#!{sys.executable}\nimport json, subprocess, sys\n"
                        "if sys.argv[1] == '-':\n"
                        f"    raise SystemExit(subprocess.run([{sys.executable!r}, *sys.argv[1:]], input=sys.stdin.read(), text=True).returncode)\n"
                        "print(json.dumps(sys.argv[1:]))\n")
        fake.chmod(0o755)
        pgrep = self.base / "pgrep"
        pgrep.write_text("#!/bin/sh\nexit 1\n")
        pgrep.chmod(0o755)
        env = {**os.environ, **self.environment, "PAIR_PYTHON": str(fake), "SWITCH_PYTHON": str(fake),
               "PATH": str(self.base) + os.pathsep + os.environ["PATH"]}
        env.pop("SRGC_STORAGE_ROOT", None)
        with (ROOT / "docs/REBUTTAL_EXTRA_TASKS.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({(r["dataset"], r["seed"], r["arm"]) for r in rows}), 120)
        for row in rows:
            with self.subTest(dataset=row["dataset"], seed=row["seed"], arm=row["arm"]):
                command = shlex.split(row["command"])
                command[1] = str(ROOT / command[1])
                result = subprocess.run(command, cwd="/tmp", env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                scope = {"sr_refresh": "candidates", "sr_refresh-pool": "pool"}.get(row["arm"])
                option = ["--scope", scope] if scope else ["--arm", row["arm"]]
                self.assertEqual(json.loads(result.stdout), ["scripts/srgc_extra_worker.py", "--plan",
                    str(plans[row["dataset"]]), "--seed", row["seed"], *option])

    def test_runtime_cache_environment_is_set_in_the_calling_process(self):
        plan = self.plan()
        self.prefix(plan)
        args = SimpleNamespace(plan=plan, seed=5)
        with patch.dict(os.environ, self.environment, clear=True), contextlib.redirect_stderr(io.StringIO()):
            prepare_run_storage(args)
            for key in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR", "TMPDIR"):
                self.assertTrue(Path(os.environ[key]).is_relative_to(self.group), key)
            self.assertEqual(args.plan, plan)

    def test_startup_streams_input_bundle_without_unbounded_read(self):
        plan = self.plan()
        self.prefix(plan)
        bundle = input_path(plan, load_plan(plan), 5)
        original = Path.read_bytes

        def read_bytes(path):
            if path == bundle:
                raise AssertionError("input bundle must be streamed")
            return original(path)

        output = io.StringIO()
        with patch.dict(os.environ, self.environment, clear=True), \
                patch.object(Path, "read_bytes", read_bytes), contextlib.redirect_stderr(output):
            prepare_run_storage(SimpleNamespace(plan=plan, seed=5))
        for stage in ("PLAN reading", "STORAGE resolving", "INPUT checking", "PREFIX verified"):
            self.assertIn(stage, output.getvalue())

    def test_startup_watchdog_is_cancelled_on_validation_error(self):
        from scripts import srgc_extra_worker as worker
        args = SimpleNamespace(plan=self.plan(), seed=5, arm="switch_fixed200", scope="candidates")
        with patch.object(worker, "prepare_run_storage", side_effect=ValueError("bad prefix")), \
                patch.object(worker.faulthandler, "dump_traceback_later") as start, \
                patch.object(worker.faulthandler, "cancel_dump_traceback_later") as stop:
            with self.assertRaisesRegex(ValueError, "bad prefix"):
                worker.launch(args)
        start.assert_called_once()
        stop.assert_called_once()

    def test_extra_worker_refuses_held_execution_lock_before_gpu_startup(self):
        from scripts import srgc_extra_worker as worker
        from srgc_rebuttal.runtime import Busy, lease
        plan = self.plan()
        folder, _, _ = self.prefix(plan)
        args = SimpleNamespace(plan=plan, seed=5, arm=None, scope="candidates")
        with patch.dict(os.environ, self.environment), \
                patch.object(worker, "process_guard", side_effect=lambda _: contextlib.nullcontext()), \
                patch.object(worker.cluster, "gpu_identity") as gpu, \
                lease(folder / ".sr_refresh.execution.lock"):
            with self.assertRaises(Busy):
                worker.launch(args)
        gpu.assert_not_called()

    def test_shell_uses_guarded_extra_worker_and_does_not_retry_busy_task(self):
        plan = self.plan()
        fake = self.base / "python"
        calls = self.base / "calls.jsonl"
        fake.write_text(f"#!{sys.executable}\nimport json, subprocess, sys\n"
                        f"real = {sys.executable!r}\n"
                        "if sys.argv[1] == '-':\n"
                        "    raise SystemExit(subprocess.run([real, *sys.argv[1:]], input=sys.stdin.read(), text=True).returncode)\n"
                        f"with open({str(calls)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                        "raise SystemExit(75)\n")
        fake.chmod(0o755)
        pgrep = self.base / "pgrep"
        pgrep.write_text("#!/bin/sh\nexit 1\n")
        pgrep.chmod(0o755)
        env = {**os.environ, **self.environment, "PAIR_PYTHON": str(fake),
               "PATH": str(self.base) + os.pathsep + os.environ["PATH"],
               "SRGC_WORKER_RESTART_DELAY": "0", "SRGC_WORKER_RESTARTS": "3"}
        env.pop("SRGC_STORAGE_ROOT", None)
        result = subprocess.run(["sh", str(SCRIPT), "math", "5", "replicate1-switch"],
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 75, result.stderr)
        commands = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual(commands, [["scripts/srgc_extra_worker.py", "--plan", str(plan),
                                    "--seed", "5", "--arm", "replicate1-switch"]])

    def test_extra_worker_refuses_duplicate_launch_before_gpu_startup(self):
        from scripts import srgc_extra_worker as worker
        from srgc_rebuttal.runtime import Busy, lease
        plan = self.plan()
        folder, _, _ = self.prefix(plan)
        args = SimpleNamespace(plan=plan, seed=5, arm="replicate1-switch", scope="candidates")
        with patch.dict(os.environ, self.environment), \
                patch.object(worker, "process_guard", side_effect=lambda _: contextlib.nullcontext()), \
                patch.object(worker.cluster, "gpu_identity") as gpu, \
                lease(folder / "replicate-1/.switch.launch.lock"):
            with self.assertRaises(Busy):
                worker.launch(args)
        gpu.assert_not_called()

    def test_extra_worker_probes_nccl_propagates_overrides_and_releases_locks(self):
        from scripts import srgc_extra_worker as worker
        from srgc_rebuttal.runtime import Busy, lease
        plan = self.plan()
        folder, _, _ = self.prefix(plan)
        args = SimpleNamespace(plan=plan, seed=5, arm="switch_fixed200", scope="candidates")
        calls = []

        def admit(root, environment, runner, **kwargs):
            calls.append("admit")
            self.assertTrue(kwargs["pass_fds"])
            environment["NCCL_NVLS_ENABLE"] = "0"
            return {"runtime_overrides": {"NCCL_NVLS_ENABLE": "0"}}

        def child(command, log, environment, **kwargs):
            calls.append("child")
            self.assertEqual(environment["NCCL_NVLS_ENABLE"], "0")
            self.assertIn("switch_fixed200", command)
            self.assertIn("SRGC_PROGRESS_DIR", environment)
            self.assertGreater(len(kwargs["pass_fds"]), 1)
            with self.assertRaises(Busy), lease(folder / ".switch_fixed200.launch.lock"):
                pass
            kwargs["heartbeat"](123)
            return 1

        with patch.dict(os.environ, self.environment), \
                patch.object(worker, "process_guard", side_effect=lambda _: contextlib.nullcontext()), \
                patch.object(worker.cluster, "gpu_identity", return_value=("0,1,2,3", ("a", "b", "c", "d"))), \
                patch.object(worker.cluster, "admit", side_effect=admit), \
                patch.object(worker.cluster, "run_child", side_effect=child), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.launch(args), 1)
        self.assertEqual(calls, ["admit", "child"])
        with lease(folder / ".switch_fixed200.launch.lock"):
            pass
        with worker.cluster.device_leases(folder.parent.parent / "gpu-node-locks", ("a", "b", "c", "d")):
            pass
        receipt = next((folder / "launches/switch_fixed200").glob("*/worker.json"))
        self.assertEqual(json.loads(receipt.read_text())["status"], "failed")

    def test_extra_run_prepares_verifier_before_distributed_startup(self):
        from scripts.srgc_sr_refresh import run
        options = SimpleNamespace(plan=self.plan(), seed=5)
        class Prepared(Exception):
            pass
        with patch("scripts.srgc_sr_refresh.prepare_run_storage") as storage, \
                patch("scripts.srgc_sr_refresh.prepare_verifier_runtime", side_effect=Prepared) as verifier:
            with self.assertRaises(Prepared):
                run(options)
        storage.assert_called_once_with(options, verify_checkpoint=False)
        verifier.assert_called_once_with()

    def test_extra_timeout_records_124_and_releases_launch_lock(self):
        from scripts import srgc_extra_worker as worker
        from srgc_rebuttal.runtime import lease
        plan = self.plan()
        folder, _, _ = self.prefix(plan)
        args = SimpleNamespace(plan=plan, seed=5, arm="switch_repeat", scope="candidates")
        with patch.dict(os.environ, self.environment), \
                patch.object(worker, "process_guard", side_effect=lambda _: contextlib.nullcontext()), \
                patch.object(worker.cluster, "gpu_identity", return_value=("0,1,2,3", ("a", "b", "c", "d"))), \
                patch.object(worker.cluster, "admit", return_value={}), \
                patch.object(worker.cluster, "run_child", side_effect=TimeoutError("no task progress")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(worker.launch(args), 124)
        receipt = json.loads(next((folder / "launches/switch_repeat").glob("*/worker.json")).read_text())
        self.assertEqual((receipt["status"], receipt["exit_code"]), ("failed", 124))
        self.assertIn("no task progress", receipt["error"])
        with lease(folder / ".switch_repeat.launch.lock"):
            pass

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
                 "shared_prefix_updates": 25, "per_question_reward": {f"e{i}": 0.5 for i in range(300)},
                 "costs": {"selection_gpu_seconds": cost, "training_gpu_seconds": 10},
                 "cost_measurement_complete": cost is not None}
        path = folder / "sr_refresh-endpoint.json"
        path.write_text(json.dumps(value))
        return path, value

    def test_completed_extra_refuses_wrong_prefix_before_gpu_startup(self):
        from scripts import srgc_extra_worker as worker
        plan = self.plan()
        path, value = self.endpoint(plan)
        path.write_text(json.dumps({**value, "prefix_checkpoint_sha256": "wrong"}))
        args = SimpleNamespace(plan=plan, seed=5, arm=None, scope="candidates")
        with patch.dict(os.environ, self.environment), patch.object(worker.cluster, "gpu_identity") as gpu:
            with self.assertRaisesRegex(ValueError, "different shared prefix"):
                worker.launch(args)
        gpu.assert_not_called()

    def test_results_reject_invalid_rewards_and_keep_other_endpoints(self):
        plan = self.plan()
        path, value = self.endpoint(plan, cost=1.0)
        path.with_name("sr_hold-endpoint.json").write_text(json.dumps({**value, "arm": "sr_hold"}))
        for wrong in ({"reward": float("nan")}, {"reward": 0.7}, {"per_question_reward": {"wrong": .5}},
                      {"per_question_reward": {**value["per_question_reward"], "e0": float("inf")}},
                      {"costs": {"selection_gpu_seconds": -1}}, {"shared_prefix_updates": 50}):
            with self.subTest(wrong=wrong):
                path.write_text(json.dumps({**value, **wrong}))
                output = io.StringIO()
                with self.assertRaises(ValueError), contextlib.redirect_stdout(output):
                    results(SimpleNamespace(plan=plan, json=True))
                report = json.loads(output.getvalue())
                self.assertTrue(report["errors"])
                self.assertNotIn("sr_refresh", report["rows"][0])
                self.assertEqual(report["rows"][0]["sr_hold"]["reward_percent"], 50)

    def test_results_remain_readable_after_code_change_but_resume_stays_blocked(self):
        from scripts import srgc_extra_worker as worker
        plan = self.plan()
        self.endpoint(plan, cost=1.0)
        output = io.StringIO()
        with patch("srgc_rebuttal.runtime.code_digest", return_value="new-code"), \
                contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=True))
        report = json.loads(output.getvalue())
        self.assertEqual(report["rows"][0]["sr_refresh"]["reward_percent"], 50)
        self.assertTrue(report["warnings"])
        with patch("srgc_rebuttal.runtime.code_digest", return_value="new-code"), \
                patch.dict(os.environ, self.environment):
            with self.assertRaisesRegex(ValueError, "different experiment"):
                worker.launch(SimpleNamespace(plan=plan, seed=5, arm=None, scope="candidates"))

    def test_corrupt_result_json_does_not_hide_valid_results_and_cli_fails(self):
        plan = self.plan()
        path, value = self.endpoint(plan, cost=1.0)
        path.with_name("sr_hold-endpoint.json").write_text(json.dumps({**value, "arm": "sr_hold"}))
        for corrupt in ("{broken", "[]", "null"):
            with self.subTest(corrupt=corrupt):
                path.write_text(corrupt)
                result = subprocess.run([sys.executable, str(ROOT / "scripts/srgc_sr_refresh.py"), "results",
                                         "--plan", str(plan), "--json"], text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 1, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["rows"][0]["sr_hold"]["reward_percent"], 50)
                self.assertTrue(report["errors"])
                self.assertNotIn("Traceback", result.stderr)

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
        self.assertTrue(any(line.startswith("seed  ") and "direction_magnitude" in line
                            for line in output.getvalue().splitlines()))

    def test_results_show_absolute_original_and_replicate_paths_for_both_datasets(self):
        for dataset in ("math", "mbpp"):
            with self.subTest(dataset=dataset):
                plan = self.plan(dataset)
                path, value = self.endpoint(plan)
                fixed = path.with_name("switch_fixed200-endpoint.json")
                fixed.write_text(json.dumps({**value, "arm": "switch_fixed200", "switched_at": 200}))
                replicate = self.replicate(plan)
                before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                          for p in self.storage.rglob("*") if p.is_file() and "results" not in p.parts}
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    results(SimpleNamespace(plan=plan, json=False))
                text = output.getvalue()
                self.assertTrue(text.startswith(f"RESULT DIRECTORY: {path.parent.parent}\n"))
                self.assertIn("RESULT FILES (validated originals):", text)
                for original in (path, fixed, replicate / "sr-endpoint.json", replicate / "switch-endpoint.json"):
                    self.assertIn(str(original), text)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    results(SimpleNamespace(plan=plan, json=True))
                report = json.loads(output.getvalue())
                self.assertEqual({item["path"] for item in report["result_files"]},
                                 {str(path), str(fixed), str(replicate / "sr-endpoint.json"),
                                  str(replicate / "switch-endpoint.json")})
                saved = Path(report["collection"]["report_path"])
                self.assertEqual(saved, path.parent.parent / "results/results.json")
                self.assertEqual(json.loads(saved.read_text()), report)
                self.assertIn("COLLECTED JSON:", text)
                self.assertIn("COLLECTED FILES:", text)
                self.assertEqual(len(report["source_results"]), 4)
                for item in report["result_files"]:
                    original = json.loads(Path(item["path"]).read_text())
                    self.assertEqual(json.loads(Path(item["collected_path"]).read_text()), original)
                    relative = Path(item["path"]).relative_to(path.parent.parent).as_posix()
                    self.assertEqual(report["source_results"][relative], original)
                after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                         for p in self.storage.rglob("*") if p.is_file() and "results" not in p.parts}
                self.assertEqual(before, after)

    def test_results_show_directory_without_inventing_missing_result_files(self):
        plan = self.plan()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=False))
        self.assertIn(f"RESULT DIRECTORY: {run_root(plan, load_plan(plan))}", output.getvalue())
        self.assertIn("No validated result files yet.", output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results(SimpleNamespace(plan=plan, json=True))
        self.assertEqual(json.loads(output.getvalue())["result_files"], [])

    def test_result_collection_excludes_invalid_json_and_keeps_prior_bundles(self):
        plan = self.plan()
        path, value = self.endpoint(plan)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            results(SimpleNamespace(plan=plan, json=True))
        first = json.loads(out.getvalue())
        original_bundle = Path(first["collection"]["bundle_report_path"])
        original_bytes = original_bundle.read_bytes()
        invalid = path.with_name("sr_hold-endpoint.json")
        invalid.write_text(json.dumps({**value, "arm": "sr_hold", "reward": .9}))
        out = io.StringIO()
        with self.assertRaises(ValueError), contextlib.redirect_stdout(out):
            results(SimpleNamespace(plan=plan, json=True))
        second = json.loads(out.getvalue())
        self.assertEqual(second["collection"]["validation_errors"], 1)
        self.assertEqual(len(second["source_results"]), 1)
        self.assertNotEqual(first["collection"]["directory"], second["collection"]["directory"])
        self.assertEqual(original_bundle.read_bytes(), original_bytes)
        self.assertFalse(any("sr_hold" in key for key in second["source_results"]))

    def test_collection_failure_does_not_replace_previous_combined_json(self):
        from scripts.srgc_sr_refresh import collect_result_json
        from srgc_rebuttal import runtime
        root = self.base / "collection"
        source = root / "seed-5/sr-endpoint.json"
        report = dict(result_files=[{"path": str(source)}], errors=[])
        first = collect_result_json(root, report, {str(source): {"fixture": 1}})
        saved = Path(first["collection"]["report_path"])
        before = saved.read_bytes()
        actual = runtime.atomic_json
        def fail_copy(path, value):
            if "raw" in path.parts:
                raise OSError("fixture disk full")
            return actual(path, value)
        with patch.object(runtime, "atomic_json", side_effect=fail_copy), self.assertRaises(OSError):
            collect_result_json(root, dict(result_files=[{"path": str(source)}], errors=[]),
                                {str(source): {"fixture": 2}})
        self.assertEqual(saved.read_bytes(), before)

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
                "shared_prefix_updates": 25, "per_question_reward": {f"e{i}": reward for i in range(300)},
                "costs": {"selection_gpu_seconds": 1.0, "training_gpu_seconds": 10}, "cost_measurement_complete": True,
                **(endpoint_override or {})}))
        return out

    def test_completed_replicate_checks_sampling_stream_before_gpu_startup(self):
        from scripts import srgc_extra_worker as worker
        plan = self.plan()
        self.replicate(plan, endpoint_override={"replicate": {"id": 999}})
        with patch.dict(os.environ, self.environment), patch.object(worker.cluster, "gpu_identity") as gpu:
            with self.assertRaisesRegex(ValueError, "replicate record differs"):
                worker.launch(SimpleNamespace(plan=plan, seed=5, arm="replicate1-switch", scope="candidates"))
        gpu.assert_not_called()

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
