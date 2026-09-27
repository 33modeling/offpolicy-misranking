from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest

from srgc_rebuttal.cluster import Task, TaskQueue, child_environment, ssh_command
from srgc_rebuttal.plan import DEFAULT_PLAN, digest
from srgc_rebuttal.runtime import Busy, atomic_json, finalize_seed, lease


def write_inputs(root):
    plan = json.loads(DEFAULT_PLAN.read_text())
    plan.update(output_root="runs", input_pattern="inputs-{seed}.json")
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    ids = [f"p{i}" for i in range(800)]
    data = {"schema": "srgc-inputs-v1", "records": {i: {"question": i, "prompt": i, "answer": "$1$"} for i in ids},
        "candidate_ids": ids[:400], "validation_pool_ids": ids[400:500],
        "ranking_validation_ids": ids[400:450], "evaluation_ids": ids[500:],
        "cached_rewards": {i: [0, 1] * 4 for i in ids[:400]}, "provenance": "synthetic queue test"}
    for seed in plan["seeds"]:
        (root / f"inputs-{seed}.json").write_text(json.dumps(data))
    return plan_path


def finish_fake(queue, task):
    folder = queue.root / f"seed-{task.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    expected = queue.identities[task.seed]
    if task.arm == "prefix":
        checkpoint = folder / "prefix.pt"
        checkpoint.write_bytes(b"synthetic checkpoint for queue test")
        atomic_json(folder / "prefix-ready.json", {**expected, "completed_updates": 25,
                                                   "checkpoint_sha256": digest(checkpoint)})
    else:
        atomic_json(folder / f"{task.arm}-endpoint.json", {**expected, "arm": task.arm, "total_updates": 275})


def simulate_worker(plan, barrier):
    queue = TaskQueue(Path(plan))
    queue.bind()
    barrier.wait(timeout=15)
    work = []
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with queue.claim() as task:
            if task is not None:
                started = time.monotonic()
                if task.arm != "prefix":
                    assert queue.complete(Task(task.seed, "prefix"))
                time.sleep(0.03)
                finish_fake(queue, task)
                finished = time.monotonic()
                assert queue.finish(task, 0) == 0
                work.append((task.key, started, finished))
            elif all(queue.complete(t) for t in queue.tasks):
                return work
        if task is None:
            time.sleep(0.005)
    raise RuntimeError("queue simulation did not finish")


class QueueTests(unittest.TestCase):
    def test_completion_receipt_recovers_if_process_dies_after_last_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            seed = 5
            folder = queue.root / f"seed-{seed}"
            expected = queue.identities[seed]
            atomic_json(folder / "run.json", {**expected, "status": "running"})
            for arm in queue.plan["arms"]:
                finish_fake(queue, Task(seed, arm))
            self.assertTrue(finalize_seed(folder, expected, queue.plan["arms"], 275))
            self.assertEqual(json.loads((folder / "run.json").read_text())["status"], "complete")
            self.assertTrue(finalize_seed(folder, expected, queue.plan["arms"], 275))

    def test_four_processes_execute_25_tasks_once_and_respect_prefix_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = write_inputs(Path(directory))
            context = multiprocessing.get_context("spawn")
            with context.Manager() as manager:
                barrier = manager.Barrier(4)
                with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
                    futures = [pool.submit(simulate_worker, str(plan), barrier) for _ in range(4)]
                    jobs = [job for future in futures for job in future.result(timeout=30)]
            self.assertEqual(len(jobs), 25)
            self.assertEqual(len({key for key, _, _ in jobs}), 25)
            times = {key: (start, end) for key, start, end in jobs}
            for seed in range(5, 10):
                for arm in ("random", "sr", "on_policy", "switch"):
                    self.assertGreaterEqual(times[f"seed-{seed}.{arm}"][0], times[f"seed-{seed}.prefix"][1])
            events = sorted([(s, 1) for _, s, _ in jobs] + [(e, -1) for _, _, e in jobs])
            active = peak = 0
            for _, change in events:
                active += change
                peak = max(peak, active)
            self.assertGreaterEqual(peak, 2)
            queue = TaskQueue(plan)
            with queue.claim() as task:
                self.assertIsNone(task)  # Completed work is not repeated on restart.

    def test_task_and_node_leases_exclude_duplicates_and_release_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease"
            with lease(path):
                with self.assertRaises(Busy):
                    with lease(path):
                        pass
            with lease(path):
                pass
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            with self.assertRaises(Busy):
                with queue.claim() as task:
                    self.assertIsNotNone(task)
                    raise Busy("GPU occupied")
            with queue.claim() as task:
                self.assertEqual(task, Task(5, "prefix"))

    def test_failed_task_requires_retry_and_input_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = TaskQueue(write_inputs(root))
            queue.bind()
            with queue.claim() as task:
                self.assertEqual(task, Task(5, "prefix"))
                queue.finish(task, 1)
            with queue.claim() as task:
                self.assertEqual(task, Task(6, "prefix"))
            with queue.claim(retry_failed=True) as task:
                self.assertEqual(task, Task(5, "prefix"))
            path = root / "inputs-5.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                with queue.claim():
                    pass

    def test_commands_are_parallel_workers_and_thread_pools_are_bounded(self):
        command = ssh_command("node-a", "/shared/repo with space", "/venv/bin/python", "v7/plan.json")
        self.assertEqual(command[0], "ssh")
        self.assertIn("srgc_rebuttal.cluster", command[-1])
        self.assertIn("nohup", command[-1])
        self.assertEqual(child_environment()["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(child_environment()["TOKENIZERS_PARALLELISM"], "false")
        with self.assertRaises(ValueError):
            ssh_command("-oProxyCommand=bad", "/repo", "python", "plan")


if __name__ == "__main__":
    unittest.main()
