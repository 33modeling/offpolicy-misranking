import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from srgc_rebuttal import import_pair_inputs
from srgc_rebuttal.plan import validate_inputs

ROOT = Path(__file__).resolve().parents[2]


def make_run(folder: Path, seed: int, *, drop_rollout: bool = False) -> Path:
    run = folder / f"family-math500-s{seed}" / f"run-s{seed}-math500-d0"
    run.mkdir(parents=True)
    offset = seed * 10_000
    prompts = {"train": [{"question": f"Q{offset + i}: what is {i}+1?", "answer": f"${i + 1}$"} for i in range(400)],
               "val": [{"question": f"V{offset + i}: what is {i}+2?", "answer": f"${i + 2}$"} for i in range(100)]}
    (run / "prompts.json").write_text(json.dumps(prompts))
    with (run / "rollouts_behavior_train.jsonl").open("w") as handle:
        for i in range(400):
            for j in range(8):
                if drop_rollout and i == 7 and j == 3:
                    continue
                handle.write(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": float((i + j) % 2),
                                         "input_ids": [1, 2], "resp_start": 1, "resp_end": 2}) + "\n")
    return run


def make_evaluation() -> dict:
    return {"test": [{"question": f"E{i}: what is {i}+3?", "answer": f"${i + 3}$"} for i in range(300)],
            "provenance": {"dataset": "EleutherAI/hendrycks_math", "revision": "abc", "split": "train"}}


class ImportPairInputsTest(unittest.TestCase):
    def test_existing_different_inputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seed-5.json"
            import_pair_inputs.write_bundle(path, {"original": True})
            with self.assertRaisesRegex(ValueError, "overwrite"):
                import_pair_inputs.write_bundle(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_string_indices_do_not_hide_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            run = make_run(Path(folder), 3)
            with (run / "rollouts_behavior_train.jsonl").open("a") as handle:
                handle.write(json.dumps({"prompt_idx": "0", "rollout_idx": "0", "reward": 1}) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                import_pair_inputs.cached_rewards(run, 400)

    def test_bundle_reuses_split_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            run = make_run(Path(folder), 3)
            bundle = import_pair_inputs.build_bundle(5, 3, run, make_evaluation())
            validate_inputs(bundle)
            self.assertEqual(len(bundle["ranking_validation_ids"]), 50)
            self.assertEqual(bundle["ranking_validation_ids"], bundle["validation_pool_ids"][:50])
            first = bundle["cached_rewards"][bundle["candidate_ids"][0]]
            self.assertEqual(first, [0, 1, 0, 1, 0, 1, 0, 1])
            self.assertTrue(bundle["records"][bundle["candidate_ids"][0]]["prompt"].startswith("Solve the following problem"))
            self.assertEqual(bundle["provenance"]["reused_from_seed"], 3)

    def test_missing_rollout_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = make_run(Path(folder), 3, drop_rollout=True)
            with self.assertRaises(ValueError):
                import_pair_inputs.build_bundle(5, 3, run, make_evaluation())

    def test_pair_root_cli_writes_five_seeds_alternating(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            runs = {s: make_run(base / "matrix", s) for s in (3, 4)}
            branch = base / "pair" / "branches" / "on_policy"
            branch.mkdir(parents=True)
            (branch / "switch.json").write_text(json.dumps({
                "sources": {str(s): {"path": str(r)} for s, r in runs.items()}, "evaluation": make_evaluation()}))
            out = base / "inputs"
            result = subprocess.run([sys.executable, "-m", "srgc_rebuttal.import_pair_inputs",
                                     "--pair-root", str(base / "pair"), "--output-dir", str(out)],
                                    text=True, capture_output=True, cwd=str(ROOT))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(p.name for p in out.iterdir()), [f"seed-{s}.json" for s in range(5, 10)])
            for seed, source in import_pair_inputs.DEFAULT_ASSIGNMENT.items():
                bundle = json.loads((out / f"seed-{seed}.json").read_text())
                validate_inputs(bundle)
                self.assertEqual(bundle["provenance"]["reused_from_seed"], source)


if __name__ == "__main__":
    unittest.main()
