import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from srgc_rebuttal.build_cache import CacheStore, GenerationProgress, candidate_seed, main
from srgc_rebuttal.build_inputs import build
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.tests.test_cluster import write_inputs
from srgc_rebuttal.timing import CostMeter


class CacheTests(unittest.TestCase):
    def test_unversioned_code_cache_cannot_resume_or_relabel_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = write_inputs(Path(directory), pending=True)
            path = input_path(plan_path, load_plan(plan_path), 5)
            bundle = json.loads(path.read_text())
            protocol = {"verifier": "srgc_rebuttal.verifiers:code_reward", "cache_seed": 5}
            store = CacheStore(path, bundle, protocol)
            store.bind()
            marker = store.root / "protocol.json"
            old = json.loads(marker.read_text())
            self.assertEqual(old.pop("code_verifier_version"), "parent-checked-values-v3")
            marker.write_text(json.dumps(old))
            before = marker.read_bytes()
            with self.assertRaisesRegex(ValueError, "settings"):
                CacheStore(path, bundle, protocol).bind()
            self.assertEqual(marker.read_bytes(), before)

    def test_generation_progress_is_completed_work_and_never_stops_decoding(self):
        now = [0.0]
        callback = GenerationProgress(10, "p1", clock=lambda: now[0])
        with patch("srgc_rebuttal.build_cache.progress") as record, redirect_stdout(io.StringIO()):
            self.assertFalse(callback(SimpleNamespace(shape=(8, 10)), None))
            record.assert_not_called()
            self.assertFalse(callback(SimpleNamespace(shape=(8, 11)), None))
            record.assert_called_once_with("cache_generation", prompt="p1", generated_steps=1)
            now[0] = 10.0
            self.assertFalse(callback(SimpleNamespace(shape=(8, 12)), None))
            self.assertEqual(record.call_count, 1)
            now[0] = 31.0
            self.assertFalse(callback(SimpleNamespace(shape=(8, 18)), None))
            record.assert_called_with("cache_generation", prompt="p1", generated_steps=8)

    def test_resume_after_last_receipt_exports_without_loading_model_or_regenerating(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = write_inputs(Path(directory), pending=True)
            plan = load_plan(plan_path)
            path = input_path(plan_path, plan, 5)
            bundle = json.loads(path.read_text())
            protocol = {key: plan[key] for key in
                        ("model", "model_revision", "verifier", "responses", "max_new_tokens")}
            protocol.update(cache_seed=5, attention="sdpa")
            store = CacheStore(path, bundle, protocol)
            store.bind()
            for candidate in bundle["candidate_ids"]:
                store.write(candidate, [0, 1] * 4, ["saved response"] * 8, 1.0)
            command = ["cache", "--plan", str(plan_path), "--bundle", str(path), "--cache-seed", "5"]
            with patch("sys.argv", command), patch("srgc_rebuttal.build_cache.initialize", return_value=(0, 0)), \
                    patch("torch.distributed.destroy_process_group"), \
                    patch("srgc_rebuttal.build_cache.torch_meter", side_effect=lambda record: CostMeter(
                        local_gpu_count=4, record=record)), \
                    patch("srgc_rebuttal.build_cache.load_model") as model, \
                    patch("srgc_rebuttal.build_cache.generate_rewards") as generate, redirect_stdout(io.StringIO()):
                main()
                model.assert_not_called()
                generate.assert_not_called()
            exported = json.loads(path.read_text())
            self.assertEqual(len(exported["cached_rewards"]), 400)
            self.assertEqual(exported["cached_rewards"][bundle["candidate_ids"][0]], [0, 1] * 4)
            self.assertTrue(path.with_suffix(".cache-responses.jsonl").is_file())
            costs = json.loads((store.root / "cost-summary.json").read_text())
            self.assertFalse(costs["complete"])  # Missing prior timing must not be invented.
            self.assertIsNone(costs["total_gpu_seconds"])

    def test_partial_generation_resume_preserves_responses_and_seeds(self):
        rows = [{"question": f"question-{i}", "answer": str(i)} for i in range(800)]
        bundle = build("fixture", rows, split_seed=0, kind="math", ranking_validation=50,
                       cache=None, provenance={})
        protocol = {"model": "fixture", "model_revision": "a" * 40, "cache_seed": 5,
                    "responses": 8, "max_new_tokens": 10, "verifier": "test"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seed-5.json"
            store = CacheStore(path, bundle, protocol)
            store.bind()
            ids = bundle["candidate_ids"]
            original_seeds = {i: candidate_seed(5, i) for i in ids}
            for candidate in ids[:3]:
                store.write(candidate, [0, 1] * 4, ["response"] * 8, 1.25)
            resumed = CacheStore(path, bundle, protocol)
            resumed.bind()
            todo = [i for i in ids if resumed.read(i) is None]
            self.assertEqual(todo, ids[3:])
            self.assertTrue(all(candidate_seed(5, i) == original_seeds[i] for i in todo))
            self.assertEqual(resumed.read(ids[0])["responses"], ["response"] * 8)
            with self.assertRaisesRegex(ValueError, "settings"):
                CacheStore(path, bundle, {**protocol, "model_revision": "b" * 40}).bind()
            changed = copy.deepcopy(bundle)
            changed["records"][ids[0]]["prompt"] += "changed"
            with self.assertRaisesRegex(ValueError, "settings"):
                CacheStore(path, changed, protocol).bind()

    def test_nonbinary_receipts_rejected(self):
        rows = [{"question": f"question-{i}", "answer": str(i)} for i in range(800)]
        bundle = build("fixture", rows, split_seed=0, kind="math", ranking_validation=50,
                       cache=None, provenance={})
        with tempfile.TemporaryDirectory() as folder:
            store = CacheStore(Path(folder) / "input.json", bundle, {"cache_seed": 5})
            store.bind()
            with self.assertRaisesRegex(ValueError, "binary"):
                store.write(bundle["candidate_ids"][0], [0.5] * 8, ["x"] * 8, 1.0)
