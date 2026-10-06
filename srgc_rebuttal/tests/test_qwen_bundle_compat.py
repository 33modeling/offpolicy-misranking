"""Qwen preparation preserves and accepts the historical Pair MATH cohort."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import srgc_qwen35 as qwen
from srgc_rebuttal.import_pair_inputs import build_bundle
from srgc_rebuttal.plan import digest, input_path, load_plan
from srgc_rebuttal.tests.test_import_pair_inputs import make_evaluation, make_run


class ChatTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"


class QwenBundleCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        run = make_run(self.root / "sources", 3)
        self.bundle = build_bundle(5, 3, run, make_evaluation())
        self.source = self.root / "source/experiments/pair_seeds.json"
        self.source.parent.mkdir(parents=True)
        self.plan = load_plan(ROOT / "srgc_rebuttal/experiments/pair_seeds.json")
        self.source.write_text(json.dumps(self.plan))
        for seed in self.plan["seeds"]:
            path = input_path(self.source, self.plan, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            source_bundle = copy.deepcopy(self.bundle)
            source_bundle["provenance"]["experiment_seed"] = seed
            path.write_text(json.dumps(source_bundle))
        self.destination = self.root / "qwen"

    def test_pair_import_prepares_and_enters_qwen_queue_without_relabeling(self):
        from srgc_rebuttal.cluster_queue import input_info

        target = qwen.prepare("math", self.source, self.destination, ChatTokenizer())
        plan = qwen.validate_extension(target)
        self.assertEqual(plan["dataset"], "math_train")
        for seed in self.plan["seeds"]:
            source_path = input_path(self.source, self.plan, seed)
            source = json.loads(source_path.read_text())
            data = json.loads(input_path(target, plan, seed).read_text())
            qwen.validate_bundle_model(data, plan, seed)
            self.assertEqual(data["dataset"], "math500")
            self.assertEqual(data["provenance"]["source_provenance"], source["provenance"])
            self.assertEqual(data["provenance"]["source_bundle_sha256"], digest(source_path))
            self.assertEqual(data["provenance"]["source_data_sha256"],
                             input_info(source_path, recorded_rewards=True)[0]["source_sha256"])
            self.assertFalse(data["cached_rewards"])
            for group in ("candidate_ids", "validation_pool_ids", "ranking_validation_ids", "evaluation_ids"):
                self.assertEqual(data[group], source[group])
            for rid, record in source["records"].items():
                self.assertEqual(data["records"][rid]["question"], record["question"])
                self.assertEqual(data["records"][rid]["answer"], record["answer"])
                self.assertIn(record["prompt"], data["records"][rid]["prompt"])
        with qwen.runtime_adapter():
            from srgc_rebuttal.cluster_queue import TaskQueue
            queue = TaskQueue(target)
            self.assertEqual(queue.cache_ready, {seed: False for seed in self.plan["seeds"]})

    def test_dataset_alias_does_not_accept_other_datasets_or_missing_pair_provenance(self):
        original = qwen.make_bundle(self.bundle, ChatTokenizer(), source_plan=self.source, source_sha256="hash")
        for dataset in ("mbpp", "gsm8k", "math", "unknown"):
            with self.subTest(dataset=dataset):
                data = copy.deepcopy(original)
                data["dataset"] = dataset
                with self.assertRaisesRegex(ValueError, "bundle dataset"):
                    qwen.validate_bundle_model(data, self.plan, 5)
        with self.assertRaisesRegex(ValueError, "bundle dataset"):
            qwen.validate_bundle_model(original, {**self.plan, "dataset": "mbpp"}, 5)
        for field in ("prompt_format", "source_run", "reused_from_seed"):
            with self.subTest(missing=field):
                data = copy.deepcopy(original)
                del data["provenance"]["source_provenance"][field]
                with self.assertRaisesRegex(ValueError, "bundle dataset"):
                    qwen.validate_bundle_model(data, self.plan, 5)

    def test_pair_alias_keeps_reference_size_and_model_identity_checks(self):
        original = qwen.make_bundle(self.bundle, ChatTokenizer(), source_plan=self.source, source_sha256="hash")
        data = copy.deepcopy(original)
        data["ranking_validation_ids"].pop()
        with self.assertRaisesRegex(ValueError, "reference size 49.*expected 50"):
            qwen.validate_bundle_model(data, self.plan, 5)
        data = copy.deepcopy(original)
        data["provenance"]["model"] = self.plan["model"]
        with self.assertRaisesRegex(ValueError, "pinned Qwen model"):
            qwen.validate_bundle_model(data, self.plan, 5)

    def test_invalid_last_seed_is_rejected_before_any_prepared_inputs_are_written(self):
        for change, expected in (("dataset", "bundle dataset"), ("reference", "reference size")):
            with self.subTest(change=change):
                last = input_path(self.source, self.plan, self.plan["seeds"][-1])
                original = json.loads(last.read_text())
                data = copy.deepcopy(original)
                if change == "dataset":
                    data["dataset"] = "mbpp"
                else:
                    data["ranking_validation_ids"].pop()
                last.write_text(json.dumps(data))
                try:
                    with self.assertRaisesRegex(ValueError, expected):
                        qwen.prepare("math", self.source, self.destination, ChatTokenizer())
                    self.assertFalse((self.destination / "inputs").exists())
                    self.assertFalse((self.destination / "experiments").exists())
                finally:
                    last.write_text(json.dumps(original))


if __name__ == "__main__":
    unittest.main()
