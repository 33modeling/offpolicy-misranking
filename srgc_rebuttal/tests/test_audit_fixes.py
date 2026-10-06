"""Regression cases independently reproduced in the October 4 runtime audit."""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import srgc_process_guard as guard
from scripts import srgc_step_checkpoints as checkpoints
from scripts.srgc_step_checkpoints import checkpoint_engine
from srgc_rebuttal import verifiers
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.code_assertions import check_returned_values, prepare_assertions
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


class AuditRegressionTests(unittest.TestCase):
    def test_shared_memory_cleanup_preserves_live_and_unknown_foreign_files(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            for name in ("torch_other_job", "nccl-unknown", "ordinary-file"):
                (folder / name).write_bytes(b"preserve")
            (folder / "torch_directory").mkdir()
            (folder / "torch_directory" / "data").write_bytes(b"preserve")
            with (folder / "torch_other_job").open("rb") as handle:
                table = {os.getpid(): (os.getppid(), os.getuid(), "python unrelated_training.py")}
                self.assertEqual(guard.clean_shm(shm_dir=folder, table=table), [])
                self.assertEqual(handle.read(), b"preserve")
            self.assertEqual(guard.clean_shm(shm_dir=folder, table={}), [])
            self.assertEqual(len(list(folder.iterdir())), 4)

    def test_checkpoint_wrapper_rejects_attention_mismatch_before_restore(self):
        features, answers, candidates, validation, _, cache = make_problem(5)
        config = Config(seed=5, projection_dim=64)

        def make(cls):
            backend = ToyBackend(features, answers, projection_dim=64, seed=5)
            return cls(backend, candidates, validation, cache, arm="on_policy", config=config)

        original = make(Engine)
        original.run_until(1)
        state = {**original.state_dict(), "checkpoint_policy": {"attention": "sdpa"}}
        with tempfile.TemporaryDirectory() as directory:
            resumed = make(checkpoint_engine(Engine, Path(directory), "prefix",
                           {"attention": "eager"}, total_updates=25))
            with self.assertRaisesRegex(ValueError, "attention"):
                resumed.load_state_dict(state)
            self.assertEqual(resumed.step, 0)

    def test_main_restores_saved_kernel_before_loading_model_and_restores_hooks(self):
        import torch
        import srgc_child_tuning as tuning
        import srgc_resumable_rollouts as rollouts
        import srgc_verifier_fallback as fallback
        from srgc_rebuttal import run_experiment
        from srgc_rebuttal.plan import load_plan
        from srgc_rebuttal.runtime import run_root
        cases = [
            ("prefix", {}, "sdpa", "new-run"),
            ("prefix", {"prefix-latest.pt": {"checkpoint_policy": {"attention": "eager"}}},
             "eager", "prefix-latest.pt"),
            ("switch", {"prefix.pt": {"checkpoint_policy": {"attention": "flash_attention_2"}}},
             "flash_attention_2", "prefix.pt"),
            ("switch", {"prefix.pt": {"checkpoint_policy": {"attention": "sdpa"}},
                        "switch-latest.pt": {}}, "eager", "switch-latest.pt"),
        ]
        for task, states, expected, source in cases:
            with self.subTest(task=task, source=source), tempfile.TemporaryDirectory() as directory:
                plan_path = write_inputs(Path(directory))
                folder = run_root(plan_path, load_plan(plan_path)) / "seed-5"
                folder.mkdir(parents=True)
                for filename, state in states.items():
                    torch.save(state, folder / filename)
                original_engine = run_experiment.Engine
                def run():
                    self.assertIsNot(run_experiment.Engine, original_engine)
                    run_experiment.load_model("model", "revision", "device")
                    raise RuntimeError("stop after model load")
                with patch("sys.argv", ["checkpoint", "--plan", str(plan_path), "--seed", "5", "--task", task]), \
                        patch.object(tuning, "configured_attention", return_value="sdpa"), \
                        patch.object(tuning, "count_progress"), patch.object(fallback, "install"), \
                        patch.object(rollouts, "install"), patch.object(run_experiment, "main", side_effect=run), \
                        patch.object(run_experiment, "load_model") as loader, \
                        redirect_stdout(io.StringIO()) as output:
                    with self.assertRaisesRegex(RuntimeError, "stop after model load"):
                        checkpoints.main()
                    loader.assert_called_once_with("model", "revision", "device", attention=expected)
                    self.assertIs(run_experiment.load_model, loader)
                self.assertIs(run_experiment.Engine, original_engine)
                self.assertIn(f"ATTENTION {expected} source={source}", output.getvalue())

    def test_legacy_attention_is_eager_and_invalid_saved_settings_fail_closed(self):
        self.assertEqual(checkpoints.saved_attention({}), "eager")
        for state in ({"checkpoint_policy": None}, {"checkpoint_policy": {"attention": "unknown"}}):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "attention"):
                checkpoints.saved_attention(state)

    def test_interrupts_do_not_exhaust_retries_or_erase_attempt_history(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            queue.tasks = [Task(5, "prefix")]
            for attempt in range(1, 7):
                with queue.claim(max_attempts=3) as task:
                    self.assertIsNotNone(task)
                    queue.finish(task, 130, interrupted=True)
                row = queue.status(max_attempts=3)[0]
                self.assertEqual(row["status"], "interrupted")
                self.assertEqual(row["attempt"], attempt)
                self.assertEqual(row["retry_attempt"], 0)
            self.assertEqual(len(list((queue.directory / "attempts").glob("*.json"))), 6)
            with queue.claim(max_attempts=3) as task:
                finish_fake(queue, task)
                queue.finish(task, 0)
            self.assertTrue(queue.ready(Task(5, "switch")))

    def test_failures_still_exhaust_the_budget_after_interruptions(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            queue.tasks = [Task(5, "prefix")]
            for _ in range(3):
                with queue.claim(max_attempts=3, retry_failed=True, retry_delay=0) as task:
                    self.assertIsNotNone(task)
                    queue.finish(task, 130, interrupted=True)
                with queue.claim(max_attempts=3, retry_failed=True, retry_delay=0) as task:
                    self.assertIsNotNone(task)
                    queue.finish(task, 1)
            with queue.claim(max_attempts=3, retry_failed=True, retry_delay=0) as task:
                self.assertIsNone(task)
            row = queue.status(max_attempts=3)[0]
            self.assertEqual((row["attempt"], row["retry_attempt"]), (6, 3))
            self.assertEqual(row["status"], "attempts_exhausted")

    def test_all_workers_wait_for_retry_cooldown_using_failure_budget_not_launch_count(self):
        from scripts import srgc_multi_queue as multi
        from scripts import srgc_qwen35_worker as qwen
        from srgc_rebuttal import cluster
        from srgc_rebuttal.tests.test_multi_queue import args
        for name, runner in (("base", cluster.run_worker),
                             ("multi", lambda q, *a: multi.run_worker_multi([q], *a)),
                             ("qwen", lambda q, *a: qwen.drain([q], *a))):
            with self.subTest(worker=name), tempfile.TemporaryDirectory() as directory:
                queue = TaskQueue(write_inputs(Path(directory)))
                queue.bind()
                queue.tasks = [Task(5, "prefix")]
                for _ in range(6):
                    with queue.claim(max_attempts=3) as task:
                        queue.finish(task, 130, interrupted=True)
                with queue.claim(max_attempts=3) as task:
                    queue.finish(task, 1)
                options = args()
                options.retry_delay = 600
                def stop_after_wait(_):
                    cluster.atomic_json(queue.directory / "stop.json", {"immediate": False})
                with patch.object(cluster.time, "sleep", side_effect=stop_after_wait) as sleep, \
                        patch.object(cluster, "run_child") as child:
                    runner(queue, options, {}, (), "test", lambda *a, **k: None)
                sleep.assert_called_once()
                child.assert_not_called()

    def test_mbpp_cannot_forge_a_successful_assertion(self):
        candidates = [
            'import os, sys\nos.write(int(sys.argv[2]), sys.argv[3].encode("ascii"))\nos._exit(0)',
            'import os, sys\nos.write(int(sys.argv[2]), b\'{"values": []}\')\nos._exit(0)',
            'import os, sys\nos.write(int(sys.argv[2]), b\'{"passed": true}\')\nos._exit(0)',
        ]
        for code in candidates:
            with self.subTest(code=code):
                self.assertEqual(verifiers.code_reward({"answer": "assert False"}, code), 0.0)

    def test_parent_rejects_wrong_exported_values_and_candidate_equality_hooks(self):
        code = ('class AlwaysEqual:\n    def __eq__(self, other): return True\n'
                'def wrong(): return AlwaysEqual()')
        self.assertEqual(verifiers.code_reward({"answer": "assert wrong() == 42"}, code), 0.0)
        spoof = ('import os, sys\n'
                 'os.write(int(sys.argv[2]), b\'{"values": ["41"]}\')\n'
                 'os._exit(0)')
        self.assertEqual(verifiers.code_reward({"answer": "assert wrong() == 42"}, spoof), 0.0)

    def test_actual_mbpp_checks_use_literal_expected_values(self):
        path = Path(__file__).resolve().parents[1] / "inputs/mbpp-seed-5.json"
        records = json.loads(path.read_text())["records"]
        count = 0
        for record in records.values():
            expressions, expected, constants_pass = prepare_assertions(record["answer"])
            self.assertEqual(len(expressions), 3)
            self.assertEqual(len(expected), 3)
            self.assertTrue(constants_pass)
            for assertion in ast.parse(record["answer"]).body:
                self.assertIsInstance(assertion, ast.Assert)
                self.assertEqual(len(assertion.test.ops), 1)
                self.assertIsInstance(assertion.test.ops[0], ast.Eq)
                ast.literal_eval(assertion.test.comparators[0])
                count += 1
        self.assertEqual(count, 2400)

    def test_mbpp_literal_results_preserve_real_value_comparisons(self):
        values = [None, True, 42, 1.25, 2 + 3j, "text", b"bytes", [1, (2, 3)],
                  {"key": [1, 2]}, {1, 2}, set(), (), {}]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(verifiers.code_reward({"answer": f"assert result() == {value!r}"},
                                                      f"def result(): return {value!r}"), 1.0)
        code = "class Pair:\n    def __init__(self, a, b): self.a, self.b = a, b\ndef add(p): return p.a + p.b"
        self.assertEqual(verifiers.code_reward({"answer": "assert add(Pair(2, 3)) == 5"}, code), 1.0)

    def test_mbpp_output_is_bounded_data_and_never_executed_in_parent(self):
        for report in ({"values": ["__import__('os')._exit(91)"]}, {"values": [42]},
                       {"values": ["42", "42"]}, {"values": ["42"], "passed": True}, None):
            with self.subTest(report=report):
                self.assertFalse(check_returned_values(report, (42,), True))
        for source in ("assert f() != 42", "assert f() == unknown", "assert True\nimport os"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "fix input"):
                verifiers.code_reward({"answer": source}, "pass")
        self.assertEqual(verifiers.code_reward({"answer": "assert result() == 'x'"},
            f"def result(): return 'x' * {verifiers.CODE_RESULT_BYTES + 1}"), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent-must-not-execute"
            expression = f"__import__('pathlib').Path({str(path)!r}).touch()"
            self.assertFalse(check_returned_values({"values": [expression]}, (None,), True))
            self.assertFalse(path.exists())

    def test_old_mbpp_results_remain_readable_but_cannot_enter_current_training(self):
        from srgc_rebuttal import reports
        from srgc_rebuttal.plan import validate_inputs
        from srgc_rebuttal.tests.test_hardening import ReportTests
        with tempfile.TemporaryDirectory() as directory:
            plan = write_inputs(Path(directory))
            path = plan.parent / "inputs-5.json"
            bundle = json.loads(path.read_text())
            bundle["provenance"] = {"cache": {"verifier": "srgc_rebuttal.verifiers:code_reward",
                "code_verifier_version": "assertion-completion-v2"}}
            path.write_text(json.dumps(bundle))
            with patch.object(verifiers, "CODE_REWARD_VERSION", "assertion-completion-v2"):
                queue = TaskQueue(plan)
                queue.bind()
                ReportTests().endpoint(queue, 5, "switch")
            before = {p: p.read_bytes() for p in Path(directory).rglob("*") if p.is_file()}
            report = reports.snapshot(plan, include_costs=True)
            self.assertEqual(report["errors"], [])
            row = next(r for r in report["tasks"] if r["task"] == "seed-5.switch")
            self.assertEqual((row["status"], row["reward_percent"]), ("complete", 50))
            self.assertTrue(any("assertion-completion-v2" in w and "display only" in w for w in report["warnings"]))
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaisesRegex(ValueError, "unverified verifier version"):
                validate_inputs(bundle)
            with self.assertRaisesRegex(ValueError, "unverified verifier version"):
                TaskQueue(plan)
            bundle["cached_rewards"][bundle["candidate_ids"][0]] = [0.5] * 8
            with self.assertRaisesRegex(ValueError, "binary"):
                validate_inputs(bundle, recorded_rewards=True)

    def test_qwen_can_copy_old_mbpp_questions_without_reusing_old_rewards(self):
        from scripts import srgc_qwen35 as qwen
        from srgc_rebuttal.plan import input_path, load_plan
        from srgc_rebuttal.tests.test_qwen_extension import ChatTokenizer
        with tempfile.TemporaryDirectory() as directory:
            plan = write_inputs(Path(directory))
            spec = json.loads(plan.read_text())
            spec.update(dataset="mbpp", verifier="srgc_rebuttal.verifiers:code_reward")
            plan.write_text(json.dumps(spec))
            before = {}
            for seed in spec["seeds"]:
                path = input_path(plan, spec, seed)
                bundle = json.loads(path.read_text())
                bundle["dataset"] = "mbpp"
                bundle["provenance"] = {"cache": {"verifier": spec["verifier"],
                    "code_verifier_version": "assertion-completion-v2"}}
                path.write_text(json.dumps(bundle))
                before[path] = path.read_bytes()
            destination = Path(directory) / "qwen"
            prepared = qwen.prepare("mbpp", plan, destination, ChatTokenizer())
            self.assertEqual(qwen.prepare("mbpp", plan, destination, ChatTokenizer()), prepared)
            for seed in spec["seeds"]:
                data = json.loads(input_path(prepared, load_plan(prepared), seed).read_text())
                self.assertEqual(data["cached_rewards"], {})
                self.assertEqual(data["provenance"]["source_provenance"]["cache"]["code_verifier_version"],
                                 "assertion-completion-v2")
            self.assertEqual(before, {p: p.read_bytes() for p in before})


if __name__ == "__main__":
    unittest.main()
