import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts.srgc_worker_status import run_with_status
from scripts.srgc_live_status import this_node_lines
from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.runtime import atomic_json, code_digest
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs


class WorkerStatusTests(unittest.TestCase):
    def test_node_serving_other_dataset_does_not_show_this_datasets_progress(self):
        report = {"dataset": "math_train", "workers": [{"host": "node", "status": "idle",
            "heartbeat_age_seconds": 0, "task": None, "active_plan": "/mbpp/plan.json",
            "active_dataset": "mbpp", "active_task": "seed-5.prefix"}],
            "tasks": [{"task": "seed-5.prefix", "completed_steps": 20, "total_steps": 25}]}
        text = "\n".join(this_node_lines(report, host="node", gpu_summary=lambda: None))
        self.assertIn("serving mbpp:seed-5.prefix", text)
        self.assertNotIn("20/25", text)

    def args(self):
        return SimpleNamespace(retry_failed=True, max_attempts=3, retry_delay=0,
                               heartbeat_seconds=.01, poll_seconds=.01)

    def test_cache_to_prefix_and_all_arms_log_work_without_waiting_for_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory), pending=True))
            queue.bind()
            before = (queue.directory / "protocol.json").read_bytes()
            output, updates = io.StringIO(), []

            def child(command, log, env, **kwargs):
                seed, arm = log.stem.split(".")
                task = Task(int(seed.removeprefix("seed-")), arm)
                kwargs["heartbeat"](123)
                finish_fake(queue, task)
                return 0

            with patch.object(cluster, "gpu_identity"), patch.object(cluster, "run_child", side_effect=child), \
                    patch.object(cluster, "publish_reports"), redirect_stdout(output):
                run_with_status(cluster.run_worker, queue, self.args(), {}, (), "test",
                                lambda *args: updates.append(args))
            self.assertEqual(sum(state == "running" for state, *_ in updates), 30)
            self.assertEqual(updates[-1][0], "complete")
            self.assertIn("NODE running seed-5.prefix · attempt 1 · pid 123 · log ", output.getvalue())
            self.assertIn("NODE seed-5.prefix update 0/25 · starting · gpus 0/4 reporting",
                          output.getvalue())
            self.assertEqual(output.getvalue().count("RUN seed-5.cache\n"), 1)
            self.assertEqual(before, (queue.directory / "protocol.json").read_bytes())

    def test_waiting_reports_actual_locked_parent_and_does_not_release_its_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory), pending=True))
            queue.bind()
            output = io.StringIO()
            with queue.claim() as task:
                finish_fake(queue, task)
                queue.verify()

                def waiting(queue, args, env, fds, worker, report):
                    report("idle")
                    self.assertTrue(queue.locked(task))

                with redirect_stdout(output):
                    run_with_status(waiting, queue, self.args(), {}, (), "test", lambda *args: None)
                self.assertIn("NODE idle (nothing claimable) · ", output.getvalue())
                self.assertIn("running 1", output.getvalue())
                self.assertIn("waiting ", output.getvalue())

    def test_phase_and_rank_progress_reported_periodically_without_writing_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            with queue.claim() as task:
                receipt = json.loads(queue.receipt(task).read_text())
                phase = queue.root / "seed-5/cost-receipts/shared-prefix/a.json"
                atomic_json(phase, {"phase": "selection", "state": "started", "checkpoint": 0})
                progress = queue.directory / "progress" / receipt["attempt_id"] / "rank-0.json"
                atomic_json(progress, {"stage": "gradient_scoring", "updated": receipt["started"]})
                before = (phase.read_bytes(), progress.read_bytes(), queue.receipt(task).read_bytes())
                output, calls = io.StringIO(), []

                def running(queue, args, env, fds, worker, report):
                    for _ in range(3):
                        report("running", task, 123)

                with redirect_stdout(output):
                    run_with_status(running, queue, self.args(), {}, (), "test", lambda *args: calls.append(args),
                                    interval=30, clock=iter([0, 5, 30]).__next__)
                self.assertEqual(len(calls), 3)
                self.assertEqual(output.getvalue().count("NODE seed-5.prefix update"), 2)
                self.assertIn("update 0/25 · selection · gpus 1/4 busy (last ", output.getvalue())
                self.assertEqual(before, (phase.read_bytes(), progress.read_bytes(), queue.receipt(task).read_bytes()))

    def test_blocked_queue_prints_failed_task_receipts_and_log_tails(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            with queue.claim() as task:
                (queue.directory / "logs").mkdir(parents=True, exist_ok=True)
                (queue.directory / "logs" / f"{task.key}.log").write_text("line1\nTraceback\nValueError: boom\n")
                queue.finish(task, 1)
            output = io.StringIO()

            def blocked(queue, args, env, fds, worker, report):
                raise RuntimeError("failed task blocks remaining work; inspect logs before retry")

            with redirect_stdout(output), self.assertRaises(RuntimeError):
                run_with_status(blocked, queue, self.args(), {}, (), "test", lambda *args: None)
            text = output.getvalue()
            self.assertIn("BLOCKED seed-5.prefix status=failed attempts=1 exit=1", text)
            self.assertIn("BLOCKED seed-5.prefix | ValueError: boom", text)

    def test_bad_progress_is_reported_without_interrupting_the_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            with queue.claim() as task:
                receipt = json.loads(queue.receipt(task).read_text())
                progress = queue.directory / "progress" / receipt["attempt_id"] / "rank-0.json"
                atomic_json(progress, {})
                output = io.StringIO()

                def running(queue, args, env, fds, worker, report):
                    report("running", task, 123)
                    return "still running"

                with redirect_stdout(output):
                    result = run_with_status(running, queue, self.args(), {}, (), "test", lambda *args: None)
                self.assertEqual(result, "still running")
            self.assertIn("NODE status read error · KeyError", output.getvalue())

    def test_blocked_secondary_queue_prints_its_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "math").mkdir(); (root / "mbpp").mkdir()
            primary, secondary = [TaskQueue(write_inputs(root / name)) for name in ("math", "mbpp")]
            primary._worker_queues = [primary, secondary]
            with secondary.claim() as task:
                log = secondary.directory / "logs" / f"{task.key}.log"
                log.parent.mkdir(parents=True)
                log.write_text("RuntimeError: MBPP verifier failed\n")
                secondary.finish(task, 1)

            def blocked(*args):
                raise RuntimeError("failed task blocks remaining work")

            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(RuntimeError):
                run_with_status(blocked, primary, self.args(), {}, (), "test", lambda *args: None)
            self.assertIn(str(log), output.getvalue())
            self.assertIn("MBPP verifier failed", output.getvalue())

    def test_corrected_experiment_code_identity_is_preserved(self):
        self.assertEqual(code_digest(), "12cf5ef830ebfd92fa8a87ea62dc7df734cd9ceab57fbce18fc4b2548385f960")

    def test_multiple_queues_report_both_totals_and_active_queue_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("math", "mbpp")]
            for root in roots:
                root.mkdir()
            primary, secondary = [TaskQueue(write_inputs(root)) for root in roots]
            # Status labels only; both queues use the synthetic MATH test inputs.
            secondary.plan["dataset"] = "mbpp"
            primary._worker_queues = [primary, secondary]
            output = io.StringIO()
            with primary.claim() as first, secondary.claim() as second:
                self.assertEqual(first, second)

                def running(queue, args, env, fds, worker, report):
                    for active in (primary, secondary):
                        queue._active_queue = active
                        report("running", first, 123)
                    queue._active_queue = None
                    report("idle")

                with redirect_stdout(output):
                    run_with_status(running, primary, self.args(), {}, (), "test", lambda *args: None)
            text = output.getvalue()
            self.assertNotIn("status read error", text)
            self.assertEqual(text.count("NODE running seed-5.prefix"), 2)
            for queue in (primary, secondary):
                self.assertIn(str(queue.directory / "logs" / "seed-5.prefix.log"), text)
            self.assertIn("math_train: done ", text)
            self.assertIn(" | mbpp: done ", text)


if __name__ == "__main__":
    unittest.main()
