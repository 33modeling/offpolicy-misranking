import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.build_cache import CacheStore
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, code_digest, lease, run_root, Busy
from srgc_rebuttal.tests.test_cluster import write_inputs


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/srgc_shared_storage.py"
spec = importlib.util.spec_from_file_location("srgc_shared_storage", SCRIPT)
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


class SharedStorageTests(unittest.TestCase):
    def test_all_runtime_caches_and_temporary_files_are_group_local(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, env = self.fixture(directory)
            keys = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME", "CUDA_CACHE_PATH",
                    "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR")
            overrides = {key: str(Path(directory) / "user-cache") for key in keys}
            with patch.dict(os.environ, {**env, **overrides}):
                target = storage.route_plan(plan, writing=True, fresh="cache-paths")
                for key in keys:
                    path = Path(os.environ[key])
                    self.assertTrue(path.is_relative_to(Path(env["GROUP_VOLUME"])), key)
                    self.assertTrue(path.is_dir(), key)
                self.assertTrue(target.is_relative_to(Path(env["GROUP_VOLUME"])))

    def test_existing_response_cache_cannot_symlink_to_user_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, env = self.fixture(directory)
            target = storage.fresh_plan(plan, env, "cache-paths")
            outside = Path(directory) / "user/cache"
            outside.mkdir(parents=True)
            input_path(target, load_plan(target), 5).with_suffix(".cache").symlink_to(outside, target_is_directory=True)
            with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, "outside group storage"):
                storage.route_plan(plan, writing=True, fresh="cache-paths")

    def test_fresh_ignores_old_work_and_partial_migration_and_never_resets_second_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, env = self.fixture(directory)
            old = TaskQueue(plan)
            old.bind()
            atomic_json(old.directory / "workers/stale.json", {"status": "running"})
            bundle = input_path(plan, old.plan, 5)
            data = json.loads(bundle.read_text())
            data["cached_rewards"] = {data["candidate_ids"][0]: [1] * 8}
            data["provenance"]["cache"] = {"old": True}
            atomic_json(bundle, data)
            _, root = storage.storage_root(env)
            atomic_json(root / "experiments" / plan.name, {"partial": True})
            target = storage.fresh_plan(plan, env, "restart1")
            first = input_path(target, old.plan, 5)
            clean = json.loads(first.read_text())
            self.assertEqual(clean["cached_rewards"], {})
            self.assertNotIn("cache", clean["provenance"])
            queue = TaskQueue(target)
            queue.bind()
            marker = first.with_suffix(".cache") / "completed.json"
            atomic_json(marker, {"new": True})
            self.assertEqual(storage.fresh_plan(plan, env, "restart1"), target)
            self.assertTrue(marker.exists())
            with patch.dict(os.environ, env):
                self.assertEqual(storage.route_plan(plan, writing=False), target)

    def test_two_fresh_cli_processes_share_one_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, env = self.fixture(directory)
            command = [sys.executable, str(SCRIPT.with_name("run_srgc_rebuttal.py")),
                       "storage", "--plan", str(plan), "--fresh", "node-pair"]
            children = [subprocess.Popen(command, env={**os.environ, **env},
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
            outputs = []
            for child in children:
                stdout, stderr = child.communicate(timeout=20)
                self.assertEqual(child.returncode, 0, stderr)
                outputs.append(stdout)
            self.assertEqual(outputs[0], outputs[1])

    def fixture(self, directory):
        root = Path(directory)
        source = root / "user/experiments"
        source.mkdir(parents=True)
        plan = write_inputs(source, pending=True)
        group = root / "group"
        group.mkdir()
        env = {"GROUP_VOLUME": str(group), "OM_WORK": str(group / "work")}
        return plan, env

    def test_rejects_user_volume_and_missing_group(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, env = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "not a user volume"):
                storage.stage(plan, environment={**env, "OM_WORK": str(Path(directory) / "user")})
            with self.assertRaisesRegex(ValueError, "unavailable"):
                storage.storage_root({"GROUP_VOLUME": str(Path(directory) / "missing")})

    def test_fresh_inputs_and_outputs_are_group_local_without_changing_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, env = self.fixture(directory)
            before = code_digest()
            target = storage.stage(plan, environment=env)
            self.assertEqual(plan.read_bytes(), target.read_bytes())
            spec = load_plan(target)
            for seed in spec["seeds"]:
                self.assertEqual(input_path(plan, spec, seed).read_bytes(), input_path(target, spec, seed).read_bytes())
                self.assertTrue(input_path(target, spec, seed).is_relative_to(Path(env["GROUP_VOLUME"])))
            queue = TaskQueue(target)
            queue.bind()
            self.assertTrue(queue.root.is_relative_to(Path(env["GROUP_VOLUME"])))
            self.assertEqual(storage.stage(plan, environment=env), target)
            self.assertEqual(code_digest(), before)

    def test_existing_work_requires_explicit_migration_and_preserves_partial_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, env = self.fixture(directory)
            queue = TaskQueue(plan)
            queue.bind()
            bundle = input_path(plan, queue.plan, 5)
            data = json.loads(bundle.read_text())
            store = CacheStore(bundle, data, {"cache_seed": 5})
            store.bind()
            candidate = data["candidate_ids"][0]
            store.write(candidate, [1] * 8, ["completed response"] * 8, 17)
            partial = store.path(candidate)
            atomic_json(queue.directory / "workers/stopped.json", {"status": "stopped"})
            with self.assertRaisesRegex(ValueError, "stop all"):
                storage.stage(plan, environment=env)
            target = storage.stage(plan, environment=env, migrate=True)
            moved = TaskQueue(target)
            self.assertEqual(queue.protocol, moved.protocol)
            moved_store = CacheStore(input_path(target, moved.plan, 5), data, {"cache_seed": 5})
            moved_store.bind()
            copied = moved_store.path(candidate)
            self.assertEqual(partial.read_bytes(), copied.read_bytes())
            self.assertEqual(moved_store.read(candidate), store.read(candidate))
            self.assertTrue(partial.exists())

    def test_live_worker_or_execution_lock_blocks_copy(self):
        for live in (True, False):
            with self.subTest(live=live), tempfile.TemporaryDirectory() as directory:
                plan, env = self.fixture(directory)
                queue = TaskQueue(plan)
                queue.bind()
                if live:
                    atomic_json(queue.directory / "workers/live.json", {"status": "running"})
                    with self.assertRaisesRegex(ValueError, "stop the source worker"):
                        storage.stage(plan, environment=env, migrate=True)
                else:
                    with lease(input_path(plan, queue.plan, 5).with_suffix(".cache") / "execution.lock"):
                        with self.assertRaises(Busy):
                            storage.stage(plan, environment=env, migrate=True)

    def test_group_symlink_escape_and_partial_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, env = self.fixture(directory)
            _, root = storage.storage_root(env)
            root.mkdir(parents=True)
            (root / "experiments").symlink_to(plan.parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside group"):
                storage.stage(plan, environment=env)

    def test_storage_cli_works_without_ml_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, env = self.fixture(directory)
            import os
            result = subprocess.run([sys.executable, str(SCRIPT.with_name("run_srgc_rebuttal.py")),
                                     "storage", "--plan", str(plan)], env={**os.environ, **env},
                                    text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(Path(env["OM_WORK"]) / "srgc-rebuttal"), result.stdout)


if __name__ == "__main__":
    unittest.main()
