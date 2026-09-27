import json
from pathlib import Path
import tempfile
import unittest

from srgc_rebuttal.plan import DEFAULT_PLAN, digest
from srgc_rebuttal.summarize import summarize


class SummaryTests(unittest.TestCase):
    def test_preserves_every_seed_and_uses_paired_seed_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            plan = json.loads(DEFAULT_PLAN.read_text())
            plan["output_root"] = "runs"
            plan_path = folder / "plan.json"
            plan_path.write_text(json.dumps(plan))
            for seed in plan["seeds"]:
                root = folder / "runs" / f"seed-{seed}"
                root.mkdir(parents=True)
                identity = {"seed": seed, "plan_sha256": digest(plan_path), "input_sha256": str(seed),
                            "implementation_sha256": "synthetic-code-hash"}
                (root / "run.json").write_text(json.dumps({**identity, "status": "complete"}))
                (root / "prefix-ready.json").write_text(json.dumps({**identity, "completed_updates": 25,
                    "checkpoint_sha256": f"synthetic-prefix-{seed}"}))
                for arm in plan["arms"]:
                    # Includes an unfavorable seed; no outcome-based exclusions.
                    reward = 0.5 + (seed - 6) * 0.01 if arm == "switch" else 0.5
                    endpoint = {**identity, "arm": arm, "total_updates": 275, "shared_prefix_updates": 25,
                        "prefix_checkpoint_sha256": f"synthetic-prefix-{seed}",
                        "reward": reward, "per_question_reward": {str(i): reward for i in range(300)},
                        "switched_at": 100 if arm == "switch" else None,
                        "cost_measurement_complete": True,
                        "costs": {"selection_gpu_seconds": 3600, "training_gpu_seconds": 7200,
                                  "sr_preparation_gpu_seconds": 0}}
                    (root / f"{arm}-endpoint.json").write_text(json.dumps(endpoint))
            result = summarize(plan_path)
            self.assertEqual([r["seed"] for r in result["per_seed"]], [5, 6, 7, 8, 9])
            contrast = result["paired_seed_statistics"]["switch_minus_sr_pp"]
            self.assertAlmostEqual(contrast["mean"], 1)
            self.assertEqual(contrast["positive_seeds"], 3)
            self.assertEqual(result["per_seed"][0]["selection_and_training_gpu_hours"]["switch"], 3)
            endpoint_path = folder / "runs/seed-9/switch-endpoint.json"
            original = endpoint_path.read_text()
            for field in ("implementation_sha256", "prefix_checkpoint_sha256"):
                broken = json.loads(original)
                broken[field] = "different"
                endpoint_path.write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError, "incomparable"):
                    summarize(plan_path)
            endpoint_path.write_text(original)
            incomplete = json.loads(original)
            incomplete["cost_measurement_complete"] = False
            endpoint_path.write_text(json.dumps(incomplete))
            result = summarize(plan_path)
            self.assertIsNone(result["per_seed"][-1]["selection_and_training_gpu_hours"]["switch"])
            self.assertEqual(result["per_seed"][-1]["incomplete_cost_arms"], ["switch"])
            (folder / "runs/seed-9/switch-endpoint.json").unlink()
            with self.assertRaises(FileNotFoundError):
                summarize(plan_path)


if __name__ == "__main__":
    unittest.main()
