import copy
from pathlib import Path
import tempfile
import unittest

from srgc_rebuttal.build_cache import CacheStore, candidate_seed
from srgc_rebuttal.build_inputs import build


class CacheTests(unittest.TestCase):
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
