from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import signal
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from srgc_rebuttal.cluster import (Task, TaskQueue, child_environment, ssh_command, device_leases,
                                  cluster_status, task_command, run_child, run_worker, probe, launch, remote_plan,
                                  publish_reports)
from srgc_rebuttal.build_cache import write_cost_summary
from srgc_rebuttal.plan import DEFAULT_PLAN, digest
from srgc_rebuttal.runtime import Busy, atomic_json, finalize_seed, lease


def write_inputs(root, *, pending=False):
    plan = json.loads(DEFAULT_PLAN.read_text())
    plan.update(output_root="runs", input_pattern="inputs-{seed}.json")
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    ids = [f"p{i}" for i in range(800)]
    data = {"schema": "srgc-inputs-v1", "records": {i: {"question": i, "prompt": i, "answer": "$1$"} for i in ids},
        "candidate_ids": ids[:400], "validation_pool_ids": ids[400:500],
        "ranking_validation_ids": ids[400:450], "evaluation_ids": ids[500:],
        "cached_rewards": {i: [0, 1] * 4 for i in ids[:400]}, "provenance": "synthetic queue test"}
    if pending:
        data.update(cached_rewards={}, provenance={"source": "synthetic queue test", "cache": "pending"})
    for seed in plan["seeds"]:
        (root / f"inputs-{seed}.json").write_text(json.dumps(data))
    return plan_path


def finish_fake(queue, task):
    folder = queue.root / f"seed-{task.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    expected = queue.identities[task.seed]
    if task.arm == "cache":
        path = queue.plan_path.parent / f"inputs-{task.seed}.json"
        bundle = json.loads(path.read_text())
        bundle["cached_rewards"] = {i: [0, 1] * 4 for i in bundle["candidate_ids"]}
        bundle["provenance"]["cache"] = {k: queue.plan[k] for k in ("model", "model_revision", "verifier", "responses", "max_new_tokens")}
        bundle["provenance"]["cache"]["cache_seed"] = task.seed
        atomic_json(path, bundle)
        write_cost_summary(path, bundle)
    elif task.arm == "prefix":
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
                if task.arm != "cache":
                    assert queue.complete(queue.dependency(task))
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
            with queue.claim(retry_failed=True, retry_delay=0) as task:
                self.assertEqual(task, Task(5, "prefix"))
            path = root / "inputs-5.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                with queue.claim():
                    pass

    def test_two_nodes_execute_all_30_tasks_from_pending_caches_once(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = write_inputs(Path(directory), pending=True)
            context = multiprocessing.get_context("spawn")
            with context.Manager() as manager:
                barrier = manager.Barrier(2)
                with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
                    futures = [pool.submit(simulate_worker, str(plan), barrier) for _ in range(2)]
                    jobs = [j for f in futures for j in f.result(timeout=30)]
            self.assertEqual(len(jobs), 30)
            self.assertEqual(len({key for key, _, _ in jobs}), 30)
            times = {key: (start, end) for key, start, end in jobs}
            for seed in range(5, 10):
                self.assertGreaterEqual(times[f"seed-{seed}.prefix"][0], times[f"seed-{seed}.cache"][1])
                for arm in ("random", "sr", "on_policy", "switch"):
                    self.assertGreaterEqual(times[f"seed-{seed}.{arm}"][0], times[f"seed-{seed}.prefix"][1])
            restarted = TaskQueue(plan)
            restarted.bind()
            self.assertTrue(all(restarted.complete(t) for t in restarted.tasks))
            self.assertEqual(cluster_status(restarted)["counts"], {"complete": 30})

    def test_cache_handoff_freezes_rewards_and_requires_receipt_before_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = TaskQueue(write_inputs(root, pending=True))
            queue.bind()
            with queue.claim() as task:
                self.assertEqual(task, Task(5, "cache"))
                finish_fake(queue, task)
                summary = root / "inputs-5.cache/cost-summary.json"
                saved = summary.read_text()
                summary.unlink()
                queue.verify()
                self.assertFalse(queue.ready(Task(5, "prefix")))
                summary.write_text(saved)
                queue.finish(task, 0)
            self.assertTrue(queue.ready(Task(5, "prefix")))
            bundle = root / "inputs-5.json"
            bundle.write_text(bundle.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "frozen inputs"):
                queue.verify()

    def test_failed_retries_are_bounded_and_attempt_history_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            for _ in range(2):
                with queue.claim(retry_failed=True, max_attempts=2, retry_delay=0) as task:
                    self.assertEqual(task, Task(5, "prefix"))
                    queue.finish(task, 1)
            with queue.claim(retry_failed=True, max_attempts=2, retry_delay=0) as task:
                self.assertEqual(task, Task(6, "prefix"))
            self.assertEqual(len(list((queue.directory / "attempts").glob("*.json"))), 2)

    def test_live_lease_wins_over_stale_heartbeat_and_unlocked_work_is_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            with queue.claim() as task:
                self.assertEqual(next(r for r in queue.status() if r["task"] == task.key)["status"], "running")
            self.assertEqual(next(r for r in queue.status() if r["task"] == task.key)["status"], "recoverable")

    def test_partially_overlapping_gpu_allocations_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with device_leases(root, ("a", "b", "c", "d")):
                with self.assertRaises(Busy):
                    with device_leases(root, ("c", "d", "e", "f")):
                        pass
                with device_leases(root, ("e", "f", "g", "h")):
                    pass
            with device_leases(root, ("c", "d", "e", "f")):
                pass

    def test_child_retains_passed_task_lease_after_parent_descriptor_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.lock"
            handle = path.open("a+")
            fcntl.flock(handle, fcntl.LOCK_EX)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"], pass_fds=(handle.fileno(),))
            try:
                handle.close()  # SIGKILL likewise closes without an explicit LOCK_UN.
                with self.assertRaises(Busy):
                    with lease(path):
                        pass
            finally:
                child.terminate()
                child.wait(timeout=5)
            with lease(path):
                pass

    def test_run_child_stop_terminates_owned_process_and_updates_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            pids = []
            with self.assertRaises(KeyboardInterrupt):
                run_child([sys.executable, "-c", "import time; time.sleep(15)"], Path(directory) / "task.log",
                          child_environment(), heartbeat=pids.append, should_stop=lambda: True, interval=0.01)
            self.assertEqual(len(pids), 1)
            with self.assertRaises(ProcessLookupError):
                os.kill(pids[0], 0)

    def test_cache_and_continuation_commands_are_separate_and_seed_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory), pending=True))
            cache = task_command(queue, Task(5, "cache"))
            arm = task_command(queue, Task(5, "switch"))
            self.assertIn("srgc_rebuttal.build_cache", cache)
            self.assertEqual(cache[cache.index("--cache-seed") + 1], "5")
            self.assertIn("srgc_rebuttal.run_experiment", arm)
            self.assertIn("--resume", arm)

    def test_worker_drives_cache_and_all_arms_then_publishes_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory), pending=True))
            queue.bind()
            states = []
            args = SimpleNamespace(retry_failed=False, max_attempts=3, retry_delay=0, heartbeat_seconds=.01, poll_seconds=.01)
            def child(command, log, env, **kwargs):
                seed, arm = log.stem.split(".")
                task = Task(int(seed.removeprefix("seed-")), arm)
                self.assertTrue(kwargs["pass_fds"])
                kwargs["heartbeat"](123)
                finish_fake(queue, task)
                return 0
            with patch("srgc_rebuttal.cluster.gpu_identity"), patch("srgc_rebuttal.cluster.run_child", side_effect=child), \
                    patch("srgc_rebuttal.cluster.publish_reports") as publish:
                run_worker(queue, args, {}, (), "test", lambda state, *args: states.append(state))
            self.assertEqual(states.count("running"), 30)
            self.assertEqual(states[-1], "complete")
            publish.assert_called_once_with(queue)

    def test_probe_rejects_nonshared_or_unlocked_coordinator_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            with self.assertRaisesRegex(ValueError, "cannot see"):
                probe(queue, "token")
            atomic_json(queue.directory / "preflight/token.json", {"token": "token"})
            with self.assertRaisesRegex(ValueError, "does not enforce"):
                probe(queue, "token")

    def test_launch_checks_all_nodes_and_waits_for_worker_acknowledgements(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            args = SimpleNamespace(plan=queue.plan_path, repo=str(Path(directory)), python="python3", hosts=["n1", "n2"],
                                   retry_failed=True, max_attempts=3, retry_delay=1, startup_timeout=1)
            calls = []
            def remote(commands):
                import shlex
                calls.append(commands)
                if len(calls) == 1:
                    return [json.dumps({"host": host, "gpu_uuids": [f"{host}-{i}" for i in range(4)],
                        "gpu_models": ["fixture"] * 4, "python": [3, 12], "packages": {"torch": "test"},
                        "code_verifier_environment": {}}) for host in args.hosts]
                for command in commands:
                    shell = shlex.split(command[-1])[-1]
                    tokens = shlex.split(shell)
                    worker_id = tokens[tokens.index("--worker-id") + 1]
                    atomic_json(queue.directory / "workers" / f"{worker_id}.json", {"status": "idle"})
                return ["123\n"] * len(commands)
            with patch("srgc_rebuttal.cluster.remote_calls", side_effect=remote):
                launch(args)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(list((queue.directory / "launches").glob("*.json"))), 1)
            self.assertEqual(remote_plan(args), str(args.plan))

    def test_ssh_worker_command_does_not_acknowledge_a_failed_cd(self):
        import shlex
        command = ssh_command("node", "/nonexistent/srgc-test-repo", sys.executable, "plan.json")
        local_shell = shlex.split(command[-1])
        result = subprocess.run(local_shell, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_final_summary_publication_keeps_every_seed_and_marks_unmeasured_costs_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            for seed in queue.plan["seeds"]:
                folder = queue.root / f"seed-{seed}"
                atomic_json(folder / "run.json", {**queue.identities[seed], "status": "running"})
                finish_fake(queue, Task(seed, "prefix"))
                prefix_hash = json.loads((folder / "prefix-ready.json").read_text())["checkpoint_sha256"]
                for arm in queue.plan["arms"]:
                    atomic_json(folder / f"{arm}-endpoint.json", {**queue.identities[seed], "arm": arm,
                        "total_updates": 275, "shared_prefix_updates": 25, "prefix_checkpoint_sha256": prefix_hash,
                        "reward": .5, "per_question_reward": {str(i): .5 for i in range(300)},
                        "switched_at": None, "cost_measurement_complete": False,
                        "costs": {"selection_gpu_seconds": 0, "training_gpu_seconds": 0, "sr_preparation_gpu_seconds": 0}})
            publish_reports(queue)
            report = json.loads((queue.root / "results-summary.json").read_text())
            self.assertEqual([r["seed"] for r in report["per_seed"]], [5, 6, 7, 8, 9])
            self.assertIsNone(report["per_seed"][0]["selection_and_training_gpu_hours"]["switch"])
            self.assertTrue((queue.root / "cost-comparison.json").exists())

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
