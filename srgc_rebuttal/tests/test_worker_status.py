import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts.srgc_worker_status import run_with_status
from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.runtime import atomic_json, code_digest
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs


class WorkerStatusTests(unittest.TestCase):
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
            self.assertIn("WORKER seed-5.prefix started · attempt 1 · pid 123 · log ", output.getvalue())
            self.assertIn("WORKER seed-5.prefix update 0/25 · starting · gpus 0/4 reporting",
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
                self.assertIn("WORKER idle · ", output.getvalue())
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
                self.assertEqual(output.getvalue().count("WORKER seed-5.prefix update"), 2)
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
                self.assertIn("status read error · KeyError", output.getvalue())

    def test_experiment_code_identity_is_preserved(self):
        self.assertEqual(code_digest(), "1869fe1cf898d4ff3a6d5e9054790836442b5e0b81b485fb04bc27de4ebab20a")


if __name__ == "__main__":
    unittest.main()
