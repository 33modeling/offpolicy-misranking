"""Failure regressions for shared storage, concurrent workers and Qwen memory."""

import json
import os
from pathlib import Path
import sys
import tempfile
import copy
import multiprocessing
import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_srgc_qwen35 as launcher
from srgc_rebuttal.tests import test_qwen_extension as model_tests
HAS_QWEN = model_tests.HAS_QWEN


class StorageRegressionTests(unittest.TestCase):
    def test_prepare_rejects_home_destination_before_tokenizer_download(self):
        with tempfile.TemporaryDirectory() as temp:
            group = Path(temp) / "group"
            group.mkdir()
            env = {"GROUP_VOLUME": str(group), "OM_WORK": str(group / "work")}
            argv = ["run_srgc_qwen35.py", "math", "prepare", "--root", str(Path(temp) / "home"),
                    "--allow-tokenizer-download"]
            with patch.dict(os.environ, env), patch.object(sys, "argv", argv), \
                    patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("download reached before storage validation")):
                with self.assertRaises((ValueError, SystemExit)):
                    launcher.main()
            self.assertFalse((Path(temp) / "home").exists())

    def test_missing_mount_and_symlink_escape_fail_without_creating_artifacts(self):
        from srgc_qwen35_storage import setup_storage
        with tempfile.TemporaryDirectory() as temp:
            group = Path(temp) / "group"
            with self.assertRaisesRegex(ValueError, "unavailable"):
                setup_storage(group / "run", {"GROUP_VOLUME": str(group)})
            self.assertFalse(group.exists())
            group.mkdir()
            (group / "run").mkdir()
            (group / "run" / "runs").symlink_to(Path(temp))
            with self.assertRaises(ValueError):
                setup_storage(group / "run", {"GROUP_VOLUME": str(group)})

    def test_all_library_caches_are_under_group_even_with_home_environment(self):
        from srgc_qwen35_storage import setup_storage, default_root
        with tempfile.TemporaryDirectory() as temp:
            group = Path(temp)
            env = {"GROUP_VOLUME": str(group), "OM_WORK": "/home/wrong", "HF_HOME": "/home/old-cache",
                   "HF_HUB_CACHE": "/home/old-hub", "TRITON_CACHE_DIR": "/home/old-triton"}
            root = default_root(env)
            setup_storage(root, env)
            self.assertTrue(root.is_relative_to(group))
            for name in ("OM_WORK", "MODELS_DIR", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
                         "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR"):
                self.assertTrue(Path(env[name]).is_relative_to(group), name)

    def test_other_training_torchrun_is_not_an_orphan_target(self):
        from scripts.srgc_process_guard import is_target
        self.assertFalse(is_target("python -m torch.distributed.run unrelated_training.py --plan /elsewhere.json"))
        self.assertTrue(is_target("python -m torch.distributed.run scripts/srgc_qwen35_rank.py --stage train"))

    def test_cached_model_integrity_rechecks_changed_files(self):
        import srgc_qwen35 as qwen
        sys.path.insert(0, str(ROOT / "src"))
        import model_matrix
        with tempfile.TemporaryDirectory() as temp:
            group = Path(temp)
            model = group / "models/snapshot"
            model.mkdir(parents=True)
            for name in ("config.json", "tokenizer_config.json", ".om_snapshot.json"):
                (model / name).write_text("{}")
            (model / "model.safetensors").write_bytes(b"tiny integrity fixture")
            env = {"GROUP_VOLUME": str(group), "OM_WORK": str(group / "work"), "SRGC_QWEN_MODEL_PATH": str(model)}
            with patch.object(model_matrix, "validate_snapshot_provenance") as verify:
                for _ in range(5):
                    self.assertEqual(qwen.model_path(qwen.MODEL, qwen.REVISION, env), str(model))
                self.assertEqual(verify.call_count, 1)
                (model / "config.json").write_text('{"changed": true}')
                verify.side_effect = ValueError("hash mismatch")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    qwen.model_path(qwen.MODEL, qwen.REVISION, env)
                self.assertEqual(verify.call_count, 2)


@unittest.skipUnless(HAS_QWEN, "requires Qwen test environment")
class MemoryRegressionTests(unittest.TestCase):
    def test_snapshot_never_deepcopies_optimizer_tensors_on_device(self):
        import torch
        test = model_tests.QwenModelTests()
        test.setUp()
        backend = test.backend
        backend._rollout = test.mixed_rollout
        backend.train(["p0"], responses=8, objective="grpo", seed=11)
        with patch.object(torch.Tensor, "__deepcopy__", side_effect=AssertionError("device deepcopy")):
            state = backend.state_dict()
        self.assertTrue(state["optimizer"]["state"])

    def test_generation_oom_restores_rng_keeps_eight_and_bounds_batch(self):
        import torch
        from srgc_qwen35_memory import bounded_generate
        calls, fail = [], [True]
        def original(**kwargs):
            n = kwargs["num_return_sequences"]
            calls.append(n)
            samples = torch.randint(1, 7, (n, 4))
            if n == 2 and fail[0]:
                fail[0] = False
                raise torch.cuda.OutOfMemoryError("injected allocation failure")
            return samples
        torch.manual_seed(101)
        expected = bounded_generate(original, torch.device("cpu"), batch_size=1)(num_return_sequences=8, pad_token_id=0)
        calls.clear()
        torch.manual_seed(101)
        actual = bounded_generate(original, torch.device("cpu"), batch_size=2)(num_return_sequences=8, pad_token_id=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(calls, [2] + [1] * 8)

    def test_generation_does_not_retry_nonmemory_failures(self):
        import torch
        from srgc_qwen35_memory import bounded_generate
        original = MagicMock(side_effect=RuntimeError("kernel error"))
        with self.assertRaisesRegex(RuntimeError, "kernel error"):
            bounded_generate(original, torch.device("cpu"))(num_return_sequences=8, pad_token_id=0)
        self.assertEqual(original.call_count, 1)

    def test_checkpointed_gradients_match_reference_and_retain_less_activations(self):
        import torch
        import numpy as np
        from srgc_rebuttal.torch_backend import TorchBackend
        test = model_tests.QwenModelTests()
        test.setUp()
        compact = test.backend
        reference_model = copy.deepcopy(compact.model)
        reference_model.get_base_model().gradient_checkpointing_disable()
        reference = TorchBackend(reference_model, compact.tokenizer, compact.records, compact.verifier,
                                 projection_dim=16, max_new_tokens=3, logprob_micro_batch=1, logit_chunk_tokens=64)
        sequence = torch.tensor([1, 2] + [3, 4, 5] * 20)
        footprints = []
        for backend in (reference, compact):
            saved = []
            def pack(tensor):
                saved.append(tensor.numel() * tensor.element_size())
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                backend._logps_batch([sequence], 2)[0].sum().backward()
            footprints.append(sum(saved))
            backend.optimizer.zero_grad(set_to_none=True)
            backend._rollout = test.mixed_rollout
        self.assertLess(footprints[1], footprints[0] / 2)
        a = reference.score_gradients(["p0"], responses=8, group_size=4, seed=11)["p0"]
        b = compact.score_gradients(["p0"], responses=8, group_size=4, seed=11)["p0"]
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
        reference.train(["p0"], responses=8, objective="grpo", seed=11)
        compact.train(["p0"], responses=8, objective="grpo", seed=11)
        for (_, a), (_, b) in zip(reference.train_parameters, compact.train_parameters):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)

    def test_snapshot_is_independent_cpu_state_and_fsyncs_boundary_save(self):
        import torch
        from srgc_qwen35_memory import durable_checkpoints
        test = model_tests.QwenModelTests()
        test.setUp()
        backend = test.backend
        backend._rollout = test.mixed_rollout
        backend.train(["p0"], responses=8, objective="grpo", seed=11)
        state = backend.state_dict()
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "checkpoint.tmp"
            with patch("os.fsync", wraps=os.fsync) as sync, durable_checkpoints():
                torch.save(state, target)
            self.assertEqual(sync.call_count, 1)
            restored = torch.load(target, map_location="cpu", weights_only=False)
            backend.load_state_dict(restored)
            for value in restored["optimizer"]["state"].values():
                self.assertTrue(all(not isinstance(v, torch.Tensor) or v.device.type == "cpu" for v in value.values()))


def simulate_node(plans, group, node, barrier):
    import srgc_qwen35 as qwen
    import srgc_qwen35_worker as worker
    from srgc_rebuttal import cluster
    from srgc_rebuttal.plan import input_path, digest
    from srgc_rebuttal.runtime import atomic_json
    work = []
    def child(command, log_path, environment, **kwargs):
        path = Path(command[command.index("--plan") + 1])
        queue = cluster.TaskQueue(path)
        seed_flag = "--cache-seed" if "--cache-seed" in command else "--seed"
        seed = int(command[command.index(seed_flag) + 1])
        arm = "cache" if seed_flag == "--cache-seed" else command[command.index("--task") + 1]
        task = next(t for t in queue.tasks if t.seed == seed and t.arm == arm)
        if arm != "cache":
            assert queue.complete(queue.dependency(task))
        kwargs["heartbeat"](os.getpid())
        receipts = list((queue.directory / "workers").glob("*.json"))
        assert any(json.loads(p.read_text()).get("active_plan") == str(path) for p in receipts)
        start = time.monotonic()
        # Hold each node's first claimed task until both are actually executing.
        # A 25 ms sleep alone can serialize on a loaded CPU despite a correct queue.
        if not work:
            barrier.wait(timeout=30)
        time.sleep(0.025)
        folder = queue.root / f"seed-{seed}"
        if arm == "cache":
            bundle = input_path(path, queue.plan, seed)
            data = json.loads(bundle.read_text())
            data["cached_rewards"] = {i: [0, 1] * 4 for i in data["candidate_ids"]}
            data["provenance"]["cache"] = {k: queue.plan[k] for k in ("model", "model_revision", "responses", "max_new_tokens", "verifier", "attention")}
            data["provenance"]["cache"]["cache_seed"] = seed
            from srgc_rebuttal.verifiers import verifier_protocol
            data["provenance"]["cache"].update(verifier_protocol(queue.plan["verifier"]))
            atomic_json(bundle, data)
            atomic_json(bundle.with_suffix(".cache") / "cost-summary.json", {"bundle_sha256": digest(bundle)})
        elif arm == "prefix":
            folder.mkdir(parents=True, exist_ok=True)
            checkpoint = folder / "prefix.pt"
            checkpoint.write_bytes(b"CPU simulation, not a model checkpoint")
            atomic_json(folder / "prefix-ready.json", {**queue.identities[seed], "completed_updates": 25,
                                                       "checkpoint_sha256": digest(checkpoint)})
        else:
            atomic_json(folder / f"{arm}-endpoint.json", {**queue.identities[seed], "arm": arm, "total_updates": 275})
        work.append((queue.plan["dataset"], task.key, start, time.monotonic()))
        return 0
    args = SimpleNamespace(retry_failed=False, max_attempts=3, retry_delay=0, poll_seconds=0.01,
                           heartbeat_seconds=0.01, stall_seconds=30)
    with qwen.runtime_adapter(), patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", tuple(f"node{node}-gpu{i}" for i in range(4)))), \
            patch.object(cluster, "run_child", child), patch.object(cluster, "publish_reports"), \
            patch.object(worker, "runtime_signature", return_value={"packages": "same CPU simulation"}):
        barrier.wait(timeout=30)
        worker.worker([Path(p) for p in plans], args, Path(group), Path(group) / "common", lambda *a, **kw: {"simulation": True})
    return work


class MultiNodeRegressionTests(unittest.TestCase):
    def test_two_nodes_drain_sixty_tasks_once_with_dataset_correct_status_and_dependencies(self):
        import srgc_qwen35 as qwen
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plans = [qwen.prepare(dataset, ROOT / "srgc_rebuttal/experiments" / source, root / "experiment", model_tests.ChatTokenizer())
                     for dataset, source in (("math", "additional_seeds.json"), ("mbpp", "mbpp_seeds.json"))]
            context = multiprocessing.get_context("spawn")
            with context.Manager() as manager:
                barrier = manager.Barrier(2)
                with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
                    futures = [pool.submit(simulate_node, [str(p) for p in plans], str(root), node, barrier) for node in range(2)]
                    jobs = [job for future in futures for job in future.result(timeout=90)]
            self.assertEqual(len(jobs), 60)
            self.assertEqual(len({(dataset, key) for dataset, key, *_ in jobs}), 60)
            times = {(d, k): (s, e) for d, k, s, e in jobs}
            for dataset in ("math_train", "mbpp"):
                for seed in range(5, 10):
                    self.assertGreaterEqual(times[dataset, f"seed-{seed}.prefix"][0], times[dataset, f"seed-{seed}.cache"][1])
                    for arm in ("sr", "random", "on_policy", "switch"):
                        self.assertGreaterEqual(times[dataset, f"seed-{seed}.{arm}"][0], times[dataset, f"seed-{seed}.prefix"][1])
            active = peak = 0
            for _, delta in sorted([(s, 1) for _, _, s, _ in jobs] + [(e, -1) for _, _, _, e in jobs]):
                active += delta
                peak = max(active, peak)
            self.assertEqual(peak, 2)
            import srgc_qwen35_worker as worker
            from srgc_rebuttal import cluster
            args = SimpleNamespace()
            with qwen.runtime_adapter(), patch.object(cluster, "gpu_identity", side_effect=AssertionError("unnecessary GPU admission")):
                worker.worker(plans, args, root, root / "common", lambda *a, **k: None)

    def test_runtime_mismatch_is_rejected_before_work(self):
        from srgc_qwen35_worker import bind_runtime
        with tempfile.TemporaryDirectory() as temp:
            queues = [SimpleNamespace(directory=Path(temp) / "queue")]
            bind_runtime(queues, {"torch": "same"})
            bind_runtime(queues, {"torch": "same"})
            with self.assertRaisesRegex(ValueError, "runtime differs"):
                bind_runtime(queues, {"torch": "different"})


class QueueHandoffRegressionTests(unittest.TestCase):
    def setUp(self):
        import srgc_qwen35 as qwen
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.plan_path = qwen.prepare("math", ROOT / "srgc_rebuttal/experiments/additional_seeds.json",
                                      Path(self.temp.name) / "experiment", model_tests.ChatTokenizer())

    def export(self, queue, *, cost=True):
        from srgc_rebuttal.plan import input_path, digest
        from srgc_rebuttal.runtime import atomic_json
        path = input_path(queue.plan_path, queue.plan, 5)
        data = json.loads(path.read_text())
        data["cached_rewards"] = {i: [0, 1] * 4 for i in data["candidate_ids"]}
        data["provenance"]["cache"] = {k: queue.plan[k] for k in ("model", "model_revision", "responses", "max_new_tokens", "verifier", "attention")}
        data["provenance"]["cache"]["cache_seed"] = 5
        atomic_json(path, data)
        if cost:
            atomic_json(path.with_suffix(".cache") / "cost-summary.json", {"bundle_sha256": digest(path)})
        return path

    def test_complete_cache_without_cost_receipt_still_requires_export_recovery(self):
        import srgc_qwen35 as qwen
        from srgc_rebuttal import cluster
        with qwen.runtime_adapter():
            initial = cluster.TaskQueue(self.plan_path)
            self.export(initial, cost=False)
            queue = cluster.TaskQueue(self.plan_path)
            self.assertFalse(queue.cache_ready[5])
            with queue.claim() as task:
                self.assertEqual(task.key, "seed-5.cache")

    def test_wrong_model_cache_refused_even_before_first_queue_bind(self):
        import srgc_qwen35 as qwen
        from srgc_rebuttal import cluster
        with qwen.runtime_adapter():
            initial = cluster.TaskQueue(self.plan_path)
            path = self.export(initial)
            data = json.loads(path.read_text())
            data["provenance"]["cache"]["model"] = "allenai/Olmo-3-1025-7B"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "cached rewards"):
                cluster.TaskQueue(self.plan_path)

    def test_cache_finished_between_scan_and_claim_is_not_reexecuted(self):
        import srgc_qwen35 as qwen
        from srgc_rebuttal import cluster, cluster_queue
        with qwen.runtime_adapter():
            queue = cluster.TaskQueue(self.plan_path)
            queue.bind()
            original = cluster_queue.lease
            fired = []
            @contextmanager
            def lease(path, **kwargs):
                with original(path, **kwargs) as handle:
                    if path.name == "seed-5.cache.lock" and not fired:
                        fired.append(True)
                        self.export(queue)
                    yield handle
            with patch.object(cluster_queue, "lease", lease):
                with queue.claim() as task:
                    self.assertIsNone(task)
            self.assertTrue(fired)
            self.assertTrue(queue.cache_ready[5])

    def test_plan_cannot_redirect_output_to_home_or_other_dataset(self):
        import srgc_qwen35 as qwen
        plan = json.loads(self.plan_path.read_text())
        for target in ("/home/somewhere", "../runs/mbpp"):
            self.plan_path.write_text(json.dumps({**plan, "output_root": target}))
            with self.assertRaises(ValueError):
                qwen.validate_extension(self.plan_path)

    def test_preimported_reports_use_qwen_identity(self):
        from srgc_rebuttal import reports, cluster
        import srgc_qwen35 as qwen
        with qwen.runtime_adapter():
            cluster.TaskQueue(self.plan_path).bind()
            report = reports.snapshot(self.plan_path)
            self.assertFalse(any("code changed" in w for w in report["warnings"]))

    def test_raw_cache_receipts_pin_microbatch_and_adapter(self):
        import srgc_qwen35 as qwen
        from srgc_rebuttal import build_cache
        from srgc_rebuttal.plan import input_path
        plan = qwen.validate_extension(self.plan_path)
        path = input_path(self.plan_path, plan, 5)
        data = json.loads(path.read_text())
        protocol = {"model": qwen.MODEL, "model_revision": qwen.REVISION, "cache_seed": 5}
        with qwen.runtime_adapter():
            store = build_cache.CacheStore(path, data, protocol)
            store.bind()
            self.assertEqual(store.protocol["generation_micro_batch"], 2)
            self.assertEqual(store.protocol["adapter_sha256"], qwen.adapter_digest())
            receipt = store.root / "protocol.json"
            saved = json.loads(receipt.read_text())
            saved["generation_micro_batch"] = 8
            receipt.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "settings"):
                store.bind()


if __name__ == "__main__":
    unittest.main()
