"""Model-port tests: real hybrid decoder, cache isolation, queue and resume routing."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import srgc_qwen35 as qwen
if HAS_QWEN:
    from srgc_qwen35_memory import QwenBackend as TorchBackend, bounded_generate
    from srgc_rebuttal.tests.test_torch_backend import TinyTokenizer


def tiny_model():
    torch.manual_seed(7)
    config = Qwen3_5TextConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1,
        linear_num_value_heads=2, layer_types=["linear_attention"] * 3 + ["full_attention"],
        pad_token_id=0, eos_token_id=7,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.0})
    config._attn_implementation = "eager"
    return Qwen3_5ForCausalLM(config)


@unittest.skipUnless(HAS_QWEN, "Qwen model tests require the separate Transformers 5 environment")
class QwenModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        model = qwen.attach_adapter(tiny_model(), LoraConfig(r=2, lora_alpha=4, task_type="CAUSAL_LM"), get_peft_model)
        base = model.get_base_model()
        base.generate = bounded_generate(base.generate, torch.device("cpu"))
        self.backend = TorchBackend(model, TinyTokenizer(), {"p0": {"prompt": "problem", "answer": "3"}},
            lambda record, text: float("3" in text.split()), projection_dim=16, max_new_tokens=3)

    def mixed_rollout(self, prompt_id, responses, seed):
        return ([torch.tensor([1, 2, 3]) if i % 2 else torch.tensor([1, 2, 4, 5]) for i in range(responses)],
                np.array([float(i % 2) for i in range(responses)]), 2)

    def test_hybrid_layers_all_adapted_and_dense_scoring_excludes_lora(self):
        names = [n for n, _ in self.backend.train_parameters]
        self.assertTrue(any("linear_attn.in_proj_qkv" in n for n in names))
        self.assertTrue(any("self_attn.q_proj" in n for n in names))
        self.assertFalse(any("lora_" in n for n, _ in self.backend.score_parameters))
        self.assertTrue(any(n.startswith("final_norm.") for n, _ in self.backend.score_parameters))

    def test_real_generation_reproducible_eight_responses(self):
        first = self.backend._rollout("p0", 8, 19)
        second = self.backend._rollout("p0", 8, 19)
        self.assertEqual(len(first[0]), 8)
        for left, right in zip(first[0], second[0]):
            torch.testing.assert_close(left, right)

    def test_dense_gradient_training_and_resume(self):
        self.backend._rollout = self.mixed_rollout
        before = copy.deepcopy(self.backend.state_dict())
        vector = self.backend.score_gradients(["p0"], responses=8, group_size=4, seed=11)["p0"]
        self.assertTrue(np.isfinite(vector).all())
        self.assertGreater(np.linalg.norm(vector), 0)
        for name, parameter in self.backend.train_parameters:
            torch.testing.assert_close(parameter, before["trainable"][name])
        self.backend.train(["p0"], responses=8, objective="grpo", seed=11)
        after = self.backend.state_dict()
        self.assertTrue(any(not torch.equal(after["trainable"][n], p) for n, p in before["trainable"].items()))
        self.backend.load_state_dict(before)
        self.backend.train(["p0"], responses=8, objective="grpo", seed=11)
        for name, value in self.backend.state_dict()["trainable"].items():
            torch.testing.assert_close(value, after["trainable"][name], rtol=0, atol=0)

    def test_official_multimodal_key_layout_loads_text_exactly(self):
        from safetensors.torch import save_file
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        original = tiny_model()
        state = {name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name: value
                 for name, value in original.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": original.config.to_dict()}))
            save_file(state, path / "model.safetensors")
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(8)}, unk_token="0")),
                                               pad_token="0", eos_token="7", unk_token="0")
            tokenizer.save_pretrained(path)
            loaded, _ = qwen.load_text_model(path, "cpu")
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
            del state["model.language_model.layers.3.self_attn.q_proj.weight"]
            save_file(state, path / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "incomplete Qwen"):
                qwen.load_text_model(path, "cpu")


class ChatTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class QwenProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name)
        self.source = ROOT / "srgc_rebuttal/experiments/additional_seeds.json"

    def prepared(self):
        return qwen.prepare("math", self.source, self.destination, ChatTokenizer())

    def test_old_adapter_results_are_read_only_and_training_remains_blocked(self):
        path = self.prepared()
        before = path.read_bytes()
        with patch.object(qwen, "adapter_digest", return_value="a" * 64), \
                patch.object(qwen, "engine_digest", return_value="b" * 64):
            with self.assertRaisesRegex(ValueError, "adapter differs"):
                qwen.validate_extension(path)
            self.assertEqual(qwen.validate_extension(path, read_only=True)["model"], qwen.MODEL)
        self.assertEqual(path.read_bytes(), before)

    def test_rank_keeps_plan_attention_despite_olmo_environment_override(self):
        import os
        from contextlib import nullcontext
        import srgc_qwen35_rank as rank
        import srgc_step_checkpoints as checkpoints
        path = self.prepared()
        def child():
            self.assertEqual(os.environ["SRGC_ATTENTION"], "eager")
        with patch.dict(os.environ, {"SRGC_ATTENTION": "sdpa"}), \
                patch.object(sys, "argv", ["rank", "--stage", "train", "--plan", str(path),
                                          "--seed", "5", "--task", "random", "--resume"]), \
                patch("srgc_qwen35_storage.setup_storage"), patch("srgc_verifier_fallback.install"), \
                patch.object(qwen, "runtime_adapter", side_effect=nullcontext), \
                patch.object(qwen, "training_adapter", side_effect=nullcontext), \
                patch.object(checkpoints, "main", side_effect=child) as main:
            rank.main()
            main.assert_called_once_with()

    def test_all_results_exports_both_datasets_even_when_one_has_errors(self):
        from contextlib import nullcontext, redirect_stdout
        import io
        import run_srgc_qwen35 as launcher
        self.prepared()
        qwen.prepare("mbpp", ROOT / "srgc_rebuttal/experiments/mbpp_seeds.json", self.destination, ChatTokenizer())
        with patch.object(sys, "argv", ["qwen", "all", "results", "--root", str(self.destination)]), \
                patch.object(launcher, "setup_storage", return_value=(self.destination, self.destination)), \
                patch.object(launcher, "runtime_adapter", side_effect=nullcontext), \
                patch("srgc_rebuttal.reports.snapshot", side_effect=[{"errors": ["invalid math result"]}, {"errors": []}]) as read, \
                patch("srgc_rebuttal.reports.render", return_value="fixture"), \
                patch("srgc_rebuttal.reports.export") as export, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                launcher.main()
            self.assertEqual(error.exception.code, 1)
        self.assertEqual(read.call_count, 2)
        self.assertEqual(export.call_count, 2)

    def test_preparation_isolates_cache_and_preserves_question_splits(self):
        from srgc_rebuttal.plan import input_path, load_plan
        target = self.prepared()
        plan = qwen.validate_extension(target)
        source = load_plan(self.source)
        self.assertEqual(plan["seeds"], [5, 6, 7, 8, 9])
        self.assertEqual(plan["selection_interval"], 25)
        self.assertEqual(plan["total_updates"], 275)
        for seed in plan["seeds"]:
            data = json.loads(input_path(target, plan, seed).read_text())
            old = json.loads(input_path(self.source, source, seed).read_text())
            self.assertEqual(data["cached_rewards"], {})
            for group in ("candidate_ids", "validation_pool_ids", "ranking_validation_ids", "evaluation_ids"):
                self.assertEqual(data[group], old[group])
            first = data["candidate_ids"][0]
            self.assertIn(old["records"][first]["prompt"], data["records"][first]["prompt"])
            self.assertEqual(data["records"][first]["answer"], old["records"][first]["answer"])
        self.assertEqual(self.prepared(), target)

    def test_existing_olmo_rewards_not_carried_into_qwen(self):
        old = json.loads((ROOT / "srgc_rebuttal/inputs/seed-5.json").read_text())
        old["cached_rewards"] = {i: [1] * 8 for i in old["candidate_ids"]}
        data = qwen.make_bundle(old, ChatTokenizer(), source_plan=self.source, source_sha256="hash")
        self.assertFalse(data["cached_rewards"])
        self.assertTrue(old["cached_rewards"])

    def test_resume_preserves_completed_qwen_cache(self):
        from srgc_rebuttal.plan import input_path
        target = self.prepared()
        plan = qwen.validate_extension(target)
        path = input_path(target, plan, 5)
        data = json.loads(path.read_text())
        data["cached_rewards"] = {i: [0, 1] * 4 for i in data["candidate_ids"]}
        path.write_text(json.dumps(data))
        self.prepared()
        self.assertEqual(json.loads(path.read_text())["cached_rewards"], data["cached_rewards"])

    def test_changed_adapter_or_base_model_cannot_resume(self):
        target = self.prepared()
        data = json.loads(target.read_text())
        with patch.object(qwen, "adapter_digest", return_value="different"):
            with self.assertRaises(ValueError):
                qwen.validate_extension(target)
            with self.assertRaises(ValueError):
                self.prepared()
        data["model"] += "-Base"
        target.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            qwen.validate_extension(target)

    def test_each_stage_uses_qwen_rank_entry_and_cache_depends_on_qwen(self):
        from srgc_rebuttal.cluster_queue import Task, TaskQueue
        from srgc_rebuttal import cluster_queue, existing_runtime
        from srgc_rebuttal.runtime import code_digest
        before = code_digest()
        original_loader = existing_runtime.load_model
        target = self.prepared()
        with qwen.runtime_adapter():
            queue = TaskQueue(target)
            self.assertNotEqual(queue.protocol["implementation_sha256"], before)
            self.assertEqual(len(queue.tasks), 30)
            states = queue.status()
            self.assertEqual(sum(row["status"] == "ready" and row["task"].endswith(".cache") for row in states), 5)
            for arm in ("cache", "prefix", "random", "sr", "on_policy", "switch"):
                command = qwen.task_command(queue, Task(5, arm))
                self.assertIn(str(ROOT / "scripts/srgc_qwen35_rank.py"), command)
                self.assertIn("cache" if arm == "cache" else "train", command)
                self.assertNotIn("srgc_rebuttal.run_experiment", command)
                if arm == "cache":
                    self.assertEqual(command[-2:], ["--attention", "eager"])
        self.assertEqual(code_digest(), before)
        self.assertEqual(cluster_queue.code_digest(), before)
        self.assertIs(existing_runtime.load_model, original_loader)

    def test_qwen_live_owner_protects_rank_orphans_are_detected(self):
        import os
        from scripts.srgc_process_guard import orphan_pids
        uid = os.getuid()
        table = {900001: (1, uid, "python scripts/run_srgc_qwen35.py all run"),
                 900002: (900001, uid, "python -m torch.distributed.run scripts/srgc_qwen35_rank.py"),
                 900003: (900002, uid, "python scripts/srgc_qwen35_rank.py --stage train")}
        self.assertEqual(orphan_pids(table=table), [])
        table[900003] = (1, uid, table[900003][2])
        self.assertEqual(orphan_pids(table=table), [900003])


if __name__ == "__main__":
    unittest.main()
