import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from srgc_rebuttal.existing_runtime import load_model, model_path, python_path, runtime_packages
from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.tests.test_cluster import write_inputs


class ExistingRuntimeTests(unittest.TestCase):
    def test_interpreter_uses_pair_and_mbpp_overrides_without_installation(self):
        environment = {"PAIR_PYTHON": "/pair/python", "SWITCH_PYTHON": "/mbpp/python"}
        with patch("srgc_rebuttal.existing_runtime.shutil.which", side_effect=lambda path: path):
            self.assertEqual(python_path("math", environment), "/pair/python")
            self.assertEqual(python_path("mbpp", environment), "/mbpp/python")
            self.assertEqual(python_path("math", {"VENV_DIR": "/existing/venv"}), "/existing/venv/bin/python")
            self.assertEqual(python_path("math", {"OM_WORK": "/work"}), "/work/.venv-cu126/bin/python")
        with patch("srgc_rebuttal.existing_runtime.shutil.which", return_value=None):
            self.assertEqual(python_path("math", {}), sys.executable)
            with self.assertRaises(ValueError):
                python_path("math", environment)

    def test_version_check_reuses_existing_olmo_gate_not_exact_new_pins(self):
        calls = []
        modules = {"model_matrix": SimpleNamespace(_require_runtime=calls.append),
                   "transformers": SimpleNamespace(AutoModelForCausalLM=object(), AutoTokenizer=object()),
                   "peft": SimpleNamespace(LoraConfig=object(), get_peft_model=object())}
        with patch.dict(sys.modules, modules), patch("importlib.metadata.version", return_value="4.57.1"):
            versions = runtime_packages()
        self.assertEqual(calls, [{"key": "olmo3-7b-base", "model_type": "olmo3"}])
        self.assertEqual(versions["transformers"], "4.57.1")

    def test_loading_delegates_to_existing_rollout_and_restores_attention_environment(self):
        tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=7)
        calls = []
        def existing(source, **kwargs):
            self.assertEqual(os.environ["OM_ATTN"], "eager")
            calls.append((source, kwargs))
            return "model", tokenizer
        with patch.dict(sys.modules, {"rollout": SimpleNamespace(load_model=existing)}), \
                patch.dict(os.environ, {"SRGC_LOCAL_MODEL": "/existing/pinned-model", "OM_ATTN": "sdpa"}):
            self.assertEqual(load_model("model", "revision", "cuda:2"), ("model", tokenizer))
            self.assertEqual(os.environ["OM_ATTN"], "sdpa")
        self.assertEqual(calls, [("/existing/pinned-model", {"device": "cuda:2", "dtype": "bfloat16"})])
        self.assertEqual(tokenizer.pad_token_id, 7)

    def test_existing_snapshot_is_verified_and_reused_without_download(self):
        calls = []
        config = json.loads((Path(__file__).parents[2] / "configs/olmo3_rlzero.json").read_text())["models"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            modules = {"model_matrix": SimpleNamespace(validate_snapshot_provenance=lambda spec, p: calls.append((spec, p)))}
            with patch.dict(sys.modules, modules):
                source = model_path(config["repository"], config["revision"], {"OM_OLMO3_MODEL_PATH": str(path)})
            self.assertEqual(source, str(path.resolve()))
            self.assertEqual(calls, [(config, path)])

    def test_failed_startup_can_update_empty_queue_without_losing_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = write_inputs(Path(directory), pending=True)
            with patch("srgc_rebuttal.cluster_queue.code_digest", return_value="old-runtime"):
                queue = TaskQueue(plan)
                queue.bind()
                atomic_json(queue.directory / "workers/failed.json", {"status": "failed", "error": "transformers pin mismatch"})
            repaired = TaskQueue(plan)
            repaired.bind()
            self.assertTrue((queue.directory / "startup-history/old-runtime.json").is_file())
            self.assertTrue((queue.directory / "workers/failed.json").is_file())

    def test_started_or_live_queue_cannot_bypass_code_identity(self):
        for kind in ("task", "worker", "cache", "checkpoint"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                plan = write_inputs(Path(directory), pending=True)
                with patch("srgc_rebuttal.cluster_queue.code_digest", return_value="old-runtime"):
                    queue = TaskQueue(plan)
                    queue.bind()
                    if kind == "task":
                        atomic_json(queue.directory / "tasks/seed-5.cache.json", {"status": "failed"})
                    elif kind == "worker":
                        atomic_json(queue.directory / "workers/live.json", {"status": "preflight"})
                    elif kind == "cache":
                        atomic_json(Path(directory) / "inputs-5.cache/protocol.json", {})
                    else:
                        (queue.root / "seed-5").mkdir()
                with self.assertRaisesRegex(ValueError, "code or plan changed"):
                    TaskQueue(plan)


if __name__ == "__main__":
    unittest.main()
