import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from srgc_rebuttal.plan import DEFAULT_PLAN, load_plan
from srgc_rebuttal.prepare_inputs import prepare
from srgc_rebuttal.cluster_queue import TaskQueue


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_srgc_rebuttal.py"
MBPP_PLAN = DEFAULT_PLAN.with_name("mbpp_seeds.json")


class DatasetCLITests(unittest.TestCase):
    def test_math_and_mbpp_options_dispatch_to_separate_plans(self):
        for dataset, output in (("math", "additional-seeds"), ("mbpp", "mbpp-seeds")):
            result = subprocess.run([sys.executable, str(SCRIPT), "plan", "--dataset", dataset],
                                    cwd="/tmp", text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual([j["seed"] for j in report["jobs"]], [5, 6, 7, 8, 9])
            self.assertEqual(load_plan(MBPP_PLAN if dataset == "mbpp" else DEFAULT_PLAN)["output_root"], f"../runs/{output}")
            self.assertIn("mbpp-seed-5" if dataset == "mbpp" else "inputs/seed-5", report["jobs"][0]["input"])

    def test_conflicting_dataset_and_plan_are_rejected(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "plan", "--dataset", "mbpp", "--plan", str(DEFAULT_PLAN)],
                                text=True, capture_output=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different datasets", result.stderr)

    def test_prepare_mbpp_all_seeds_preserves_existing_data_and_queue_can_start_without_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = load_plan(MBPP_PLAN)
            plan.update(input_pattern="inputs/mbpp-seed-{seed}.json", output_root="runs/mbpp")
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            rows = root / "rows.jsonl"
            rows.write_text("\n".join(json.dumps({"question": f"Write function task{i}.", "answer": f"assert task{i}() == {i}"}) for i in range(800)))
            report = prepare(path, rows)
            self.assertEqual(len(report["jobs"]), 5)
            bundles = [json.loads(Path(j["input"]).read_text()) for j in report["jobs"]]
            self.assertTrue(all(b["candidate_ids"] == bundles[0]["candidate_ids"] for b in bundles))
            self.assertTrue(all(b["cached_rewards"] == {} for b in bundles))
            self.assertIn("code", bundles[0]["provenance"]["prompt_format"])
            self.assertEqual(prepare(path, rows)["jobs"], report["jobs"])
            queue = TaskQueue(path)
            queue.bind()
            self.assertEqual([r["status"] for r in queue.status()].count("ready"), 5)
            result = subprocess.run([sys.executable, str(SCRIPT), "status", "--dataset", "mbpp", "--plan", str(path)],
                                    cwd="/tmp", text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["dataset"], "mbpp")
            rows.write_text(rows.read_text().replace("task0", "changed0"))
            with self.assertRaisesRegex(ValueError, "different existing inputs"):
                prepare(path, rows)


if __name__ == "__main__":
    unittest.main()
