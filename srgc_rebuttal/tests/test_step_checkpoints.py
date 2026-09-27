import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_rebuttal.runtime import code_digest
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.tests.test_cluster import write_inputs
from srgc_rebuttal.tests.test_reference import ScriptedBackend
from srgc_rebuttal.timing import CostMeter


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/srgc_step_checkpoints.py"
spec = importlib.util.spec_from_file_location("srgc_step_checkpoints", SCRIPT)
checkpoints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoints)
POLICY = {"interval_updates": 1, "storage_adapter_sha256": "test-adapter",
          "legacy_boundary_saves_retained": True}


def make_engine(folder, task="prefix", step=0, *, adapted=True):
    backend = ScriptedBackend()
    backend.cost_meter = CostMeter(record=PhaseLedger(folder / "costs").record)
    ids = [f"s{i}" for i in range(40)] + [f"o{i}" for i in range(360)]
    cache = {i: ([0, 1] * 4 if i.startswith("s") else [1] * 8) for i in ids}
    cls = checkpoints.checkpoint_engine(Engine, folder, task, POLICY) if adapted else Engine
    engine = cls(backend, ids, ["sv"], cache, arm="on_policy" if task == "prefix" else task,
                 config=Config(projection_dim=2), step=step)
    backend.strengths = {i: float(len(ids) - n) for n, i in enumerate(engine.sr_ranked_ids)}
    return engine


class StepCheckpointTests(unittest.TestCase):
    def test_every_completed_update_saved_with_original_boundaries_retained(self):
        for task in ("prefix", "random", "sr", "on_policy", "switch"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory, \
                    redirect_stdout(io.StringIO()):
                folder = Path(directory)
                start, stop, interval = (0, 7, 5) if task == "prefix" else (25, 52, 25)
                engine = make_engine(folder, task, start)
                path = folder / f"{task}-latest.pt"
                with patch.object(checkpoints, "save_checkpoint", wraps=checkpoints.save_checkpoint) as save:
                    for step in range(start + 1, stop + 1):
                        engine.update()
                        # Equivalent to the unchanged runner's original boundary save.
                        if step % interval == 0:
                            checkpoints.save_checkpoint(engine, folder, task)
                        state = torch.load(path, weights_only=False)
                        self.assertEqual(state["step"], step)
                        self.assertEqual(state["backend"]["optimizer_steps"], step - start)
                        self.assertEqual(state["checkpoint_policy"], POLICY)
                    self.assertEqual(save.call_count, stop - start)
                ledger = PhaseLedger(folder / "costs").totals()
                self.assertTrue(ledger["complete"])
                self.assertGreater(ledger["phase_wall_seconds"]["checkpoint_save"], 0)
                self.assertEqual(ledger["exclusive_stages"]["checkpoint_save.write"]["calls"], stop - start)
                self.assertEqual(ledger["exclusive_stages"]["checkpoint_save.state_snapshot"]["calls"], stop - start)

    def test_intermediate_resume_preserves_sampling_model_optimizer_and_switch_rule(self):
        for task in ("prefix", "random", "sr", "on_policy", "switch"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory, \
                    redirect_stdout(io.StringIO()):
                folder = Path(directory)
                start = 0 if task == "prefix" else 25
                original = make_engine(folder, task, start, adapted=False)
                persisted = make_engine(folder, task, start)
                for _ in range(2):
                    original.update()
                    persisted.update()
                state = torch.load(folder / f"{task}-latest.pt", weights_only=False)
                restored = make_engine(folder, task, start)
                restored.load_state_dict(state)
                self.assertEqual(restored.active_selection, original.active_selection)
                self.assertEqual(restored.costs, persisted.costs)
                for _ in range(27):
                    expected, actual = original.update(), restored.update()
                    for row in (expected, actual):
                        row.pop("selection_gpu_seconds")
                        row.pop("training_gpu_seconds")
                    self.assertEqual(actual, expected)
                self.assertEqual(restored.backend.state_dict(), original.backend.state_dict())
                self.assertEqual(restored.used_training_ids, original.used_training_ids)
                self.assertEqual(restored.sampling_cycle, original.sampling_cycle)
                self.assertEqual(restored.rule, original.rule)
                self.assertEqual(restored.switched_at, original.switched_at)

    def test_failed_save_keeps_previous_checkpoint_and_records_unfinished_cost(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            engine = make_engine(folder)
            engine.update()
            path = folder / "prefix-latest.pt"
            before = path.read_bytes()
            def fail(state, handle):
                handle.write(b"interrupted write")
                raise OSError("storage full")
            with patch.object(torch, "save", side_effect=fail):
                with self.assertRaisesRegex((OSError, RuntimeError), "storage full"):
                    engine.update()
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_suffix(".tmp").exists())
            self.assertEqual(torch.load(path, weights_only=False)["step"], 1)
            ledger = PhaseLedger(folder / "costs").totals()
            self.assertFalse(ledger["complete"])
            self.assertEqual(ledger["unfinished_phases"][0]["phase"], "checkpoint_save")

    def test_failed_update_does_not_publish_partially_updated_state(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            engine = make_engine(folder)
            engine.update()
            path = folder / "prefix-latest.pt"
            before = path.read_bytes()
            with patch.object(engine.backend, "train", side_effect=RuntimeError("interrupted update")):
                with self.assertRaisesRegex(RuntimeError, "interrupted update"):
                    engine.update()
            self.assertEqual(path.read_bytes(), before)

    def test_worker_commands_preserve_cache_plan_seed_resume_and_torchrun(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            original = cluster.task_command
            for arm in ("cache", "prefix", "random", "sr", "on_policy", "switch"):
                task = Task(5, arm)
                before = original(queue, task)
                after = checkpoints.checkpoint_command(original, queue, task)
                if arm == "cache":
                    self.assertEqual(after, before)
                else:
                    index = before.index("srgc_rebuttal.run_experiment")
                    self.assertEqual(after[:index - 1], before[:index - 1])
                    self.assertEqual(after[index - 1], str(SCRIPT))
                    self.assertEqual(after[index:], before[index + 1:])
                    self.assertIn("--resume", after)

    def test_worker_hook_restored_after_success_and_failure(self):
        original = cluster.task_command
        for failure in (False, True):
            def worker():
                self.assertIsNot(cluster.task_command, original)
                if failure:
                    raise RuntimeError("worker failed")
            with patch.object(cluster, "main", side_effect=worker):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "worker failed"):
                        checkpoints.worker_main()
                else:
                    checkpoints.worker_main()
            self.assertIs(cluster.task_command, original)

    def test_frozen_experiment_digest_unchanged(self):
        self.assertEqual(code_digest(), "1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a")


if __name__ == "__main__":
    unittest.main()
