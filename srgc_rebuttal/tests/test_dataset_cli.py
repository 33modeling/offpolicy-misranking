import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from srgc_rebuttal.plan import DEFAULT_PLAN, digest, input_path, load_plan, validate_inputs
from srgc_rebuttal.prepare_inputs import prepare
from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.tests.test_reference import ScriptedBackend


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_srgc_rebuttal.py"
MBPP_PLAN = DEFAULT_PLAN.with_name("mbpp_seeds.json")


class DatasetCLITests(unittest.TestCase):
    def test_both_dataset_plans_use_400_to_distinct_40_to_4_for_every_arm(self):
        for plan_path in (DEFAULT_PLAN, MBPP_PLAN):
            plan = load_plan(plan_path)
            for seed in plan["seeds"]:
                with self.subTest(dataset=plan["dataset"], seed=seed):
                    bundle = json.loads(input_path(plan_path, plan, seed).read_text())
                    ids = bundle["candidate_ids"]
                    self.assertEqual(len(ids), 400)
                    # Synthetic rewards exercise selection only, never written to real inputs.
                    cache = {i: [0, 1] * 4 for i in ids}
                    config = Config(seed=seed, projection_dim=2,
                                    scoring_prompts=plan["scoring_prompts_per_set"],
                                    training_prompts=plan["training_prompts"],
                                    selection_interval=plan["selection_interval"])
                    draws = []
                    for arm in plan["arms"]:
                        engine = Engine(ScriptedBackend(), ids, bundle["ranking_validation_ids"],
                                        cache, config=config, arm=arm)
                        record = engine.update()
                        candidates = record.get("training_candidate_ids", record.get("on_ids"))
                        self.assertEqual(len(candidates), 40)
                        self.assertEqual(len(set(candidates)), 40)
                        self.assertEqual(len(set(record["train_ids"])), 4)
                        self.assertTrue(set(record["train_ids"]) <= set(candidates))
                        draws.append(candidates)
                        if arm == "sr":
                            self.assertEqual(record["train_ids"], [i for i in engine.sr_ranked_ids if i in candidates][:4])
                        if arm in {"on_policy", "switch"}:
                            # At step zero neither arm performs a Switch check.
                            self.assertEqual(record["sr_ids"], [])
                            self.assertEqual(record["scored_distinct_prompts"], 40)
                    self.assertTrue(all(draw == draws[0] for draw in draws))

    def test_shipped_mbpp_inputs_are_real_valid_and_queue_ready_without_preparation(self):
        repo = SCRIPT.parents[1]
        plan = load_plan(MBPP_PLAN)
        manifest = json.loads(MBPP_PLAN.with_name("prepared_mbpp_inputs.json").read_text())
        self.assertEqual(manifest["plan_sha256"], digest(MBPP_PLAN))
        self.assertEqual(manifest["source_rows"], 974)
        self.assertEqual([j["seed"] for j in manifest["jobs"]], plan["seeds"])
        for job in manifest["jobs"]:
            path = repo / job["input"]
            self.assertEqual(path.resolve(), input_path(MBPP_PLAN, plan, job["seed"]))
            self.assertEqual(digest(path), job["input_sha256"])
            data = json.loads(path.read_text())
            validate_inputs(data, require_cache=False)
            self.assertEqual(data["cached_rewards"], {})
            self.assertEqual(data["provenance"]["dataset_revision"], manifest["source_revision"])
            self.assertEqual(data["provenance"]["source_rows_sha256"], manifest["source_rows_sha256"])
            self.assertEqual(data["provenance"]["experiment_seed"], job["seed"])
        with tempfile.TemporaryDirectory() as directory:
            plan.update(input_pattern=str(input_path(MBPP_PLAN, plan, "{seed}")), output_root="runs")
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            queue = TaskQueue(path)
            queue.bind()
            self.assertEqual([r["status"] for r in queue.status()].count("ready"), 5)

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
            result = subprocess.run([sys.executable, str(SCRIPT), "status", "--json", "--output", str(root / "status.txt"), "--dataset", "mbpp", "--plan", str(path)],
                                    cwd="/tmp", text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["dataset"], "mbpp")
            rows.write_text(rows.read_text().replace("task0", "changed0"))
            with self.assertRaisesRegex(ValueError, "different existing inputs"):
                prepare(path, rows)


if __name__ == "__main__":
    unittest.main()
