import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from srgc_rebuttal import build_inputs, plan_dataset, verifiers
from srgc_rebuttal.plan import load_plan, validate_inputs


def rows(kind: str, count: int = 900) -> list[dict]:
    if kind == "math":
        return [{"question": f"What is {i} + {i}?", "answer": f"${2 * i}$"} for i in range(count)]
    return [{"question": f"Write add{i}(a, b) returning a + b + {i}.",
             "answer": f"assert add{i}(1, 2) == {3 + i}"} for i in range(count)]


class BuildInputsTest(unittest.TestCase):
    def test_math_gold_and_evaluation_only_dataset_rejection(self):
        self.assertEqual(build_inputs.math_answer(r"old \boxed{2}, final \boxed{\frac{1}{3}}."),
                         r"$\boxed{\frac{1}{3}}$")
        self.assertEqual(build_inputs.math_answer(r"answer is $\boxed 2$."), r"$\boxed{2}$")
        with self.assertRaisesRegex(ValueError, "math_train"):
            build_inputs.load_rows("math500", None)
        self.assertFalse(build_inputs.has_gold({"answer": r"$\boxed{}$"}))
        self.assertTrue(build_inputs.has_gold({"answer": r"$\boxed{0}$"}))
        self.assertEqual(build_inputs.dedupe([{"question": "missing", "answer": r"$\boxed{}$"},
                                             {"question": "valid", "answer": r"$\boxed{0}$"}]),
                         [{"question": "valid", "answer": r"$\boxed{0}$"}])

    def test_invalid_validation_and_cache_are_not_silently_coerced(self):
        for count in [0, -1, 101]:
            with self.assertRaises(ValueError):
                build_inputs.build("toy", rows("math"), split_seed=0, kind="math", ranking_validation=count,
                                   cache=None, provenance={})
        bundle = build_inputs.build("toy", rows("math"), split_seed=0, kind="math", ranking_validation=50,
                                    cache=None, provenance={})
        cache = {i: [0.4] * 8 for i in bundle["candidate_ids"]}
        with self.assertRaisesRegex(ValueError, "binary"):
            build_inputs.build("toy", rows("math"), split_seed=0, kind="math", ranking_validation=50,
                               cache=cache, provenance={})

    def test_bundle_without_cache_then_with_cache_validates(self):
        bundle = build_inputs.build("toy", rows("math"), split_seed=5, kind="math", ranking_validation=50,
                                    cache=None, provenance={"source": "test"})
        self.assertEqual(len(bundle["candidate_ids"]), 400)
        self.assertEqual(len(bundle["validation_pool_ids"]), 100)
        self.assertEqual(len(bundle["evaluation_ids"]), 300)
        self.assertEqual(len(bundle["ranking_validation_ids"]), 50)
        self.assertTrue(bundle["records"][bundle["candidate_ids"][0]]["prompt"].startswith("Solve the following problem"))
        with self.assertRaises((ValueError, KeyError)):
            validate_inputs(bundle)  # no cached rewards yet
        cache = {i: [0, 1] * 4 for i in bundle["candidate_ids"]}
        complete = build_inputs.build("toy", rows("math"), split_seed=5, kind="math", ranking_validation=50,
                                      cache=cache, provenance={"source": "test", "cache": "test"})
        validate_inputs(complete)
        self.assertEqual(complete["candidate_ids"], bundle["candidate_ids"])  # same split seed, same split

    def test_splits_are_disjoint_and_deduplicated(self):
        duplicated = rows("code") + rows("code")[:50]
        bundle = build_inputs.build("toy", duplicated, split_seed=1, kind="code", ranking_validation=50,
                                    cache=None, provenance={})
        ids = bundle["candidate_ids"] + bundle["validation_pool_ids"] + bundle["evaluation_ids"]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("```", bundle["records"][ids[0]]["prompt"])

    def test_too_few_rows_is_rejected(self):
        with self.assertRaises(ValueError):
            build_inputs.build("toy", rows("math", 700), split_seed=1, kind="math", ranking_validation=50,
                               cache=None, provenance={})

    def test_code_reward_executes_tests(self):
        record = {"answer": "assert add(1, 2) == 3\nassert add(0, 0) == 0"}
        self.assertEqual(verifiers.code_reward(record, "```python\ndef add(a, b):\n    return a + b\n```"), 1.0)
        self.assertEqual(verifiers.code_reward(record, "```python\ndef add(a, b):\n    return a - b\n```"), 0.0)
        self.assertEqual(verifiers.code_reward(record, "no code here"), 0.0)
        self.assertEqual(verifiers.code_reward(record, "```python\nwhile True: pass\n```"), 0.0)
        with self.assertRaises(ValueError):
            verifiers.code_reward({"answer": "$3$"}, "x")

    def test_dataset_plan_keeps_frozen_settings(self):
        base = load_plan(plan_dataset.DEFAULT_PLAN)
        plan = plan_dataset.make_plan(base, dataset="mbpp", seeds=[5, 6], verifier="srgc_rebuttal.verifiers:code_reward")
        self.assertEqual(plan["seeds"], [5, 6])
        self.assertEqual(plan["input_pattern"], "../inputs/mbpp-seed-{seed}.json")
        for key in ("shared_prefix_updates", "total_updates", "check_interval", "scoring_prompts_per_set"):
            self.assertEqual(plan[key], base[key])

    def test_cli_builds_a_jsonl_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "rows.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in rows("math")))
            output = Path(folder) / "toy-seed-5.json"
            result = subprocess.run([sys.executable, "-m", "srgc_rebuttal.build_inputs", "--dataset", "jsonl",
                                     "--rows", str(source), "--kind", "math", "--seed", "5", "--split-seed", "0",
                                     "--output", str(output)],
                                    text=True, capture_output=True, cwd=str(Path(__file__).resolve().parents[2]))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("build_cache", result.stdout)
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text())["provenance"]["split_seed"], 0)


if __name__ == "__main__":
    unittest.main()
