import io
import json
import tempfile
import sys
import unittest
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import srgc_multi_queue as multi
from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs
from scripts.srgc_seed_order import seed_first_queue


def args():
    return SimpleNamespace(retry_failed=True, max_attempts=3, retry_delay=0, poll_seconds=0.01,
                           heartbeat_seconds=1, stall_seconds=1800)


def simulate_multi_worker(primary, secondary, worker_id, barrier):
    primary, secondary = Path(primary), Path(secondary)
    queues = {p: TaskQueue(p) for p in (primary, secondary)}
    options = args()
    options.plan, options.worker_id = primary, worker_id
    options.node_lock_root = primary.parent.parent / "devices"
    jobs = []

    def child(command, log, environment, **kwargs):
        plan = Path(command[command.index("--plan") + 1])
        queue = queues[plan]
        queue.verify()
        task = next(t for t in queue.tasks if t.key == log.stem)
        parent = queue.dependency(task)
        if parent is not None:
            assert queue.complete(parent)
        started = time.monotonic()
        kwargs["heartbeat"](123)
        for each in queues.values():
            record = json.loads((each.directory / "workers" / f"{worker_id}.json").read_text())
            assert record["active_plan"] == str(plan)
            assert record["task"] == (task.key if each.plan_path == plan else None)
        if not jobs:
            barrier.wait(timeout=20)
        time.sleep(.01)
        finish_fake(queue, task)
        jobs.append((plan.parent.name, task.key, started, time.monotonic()))
        return 0

    with seed_first_queue(), multi.multi_queue([secondary]), \
            patch.object(cluster, "run_child", child), \
            patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", tuple(f"{worker_id}-{i}" for i in range(4)))), \
            patch.object(cluster, "admit", return_value={}), patch.object(cluster, "publish_reports"), \
            redirect_stdout(io.StringIO()):
        cluster.worker(options)
    return jobs


class MultiQueueTest(unittest.TestCase):
    def test_timeout_retries_without_losing_worker_and_records_real_error(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            self.queues = {queue.plan_path: queue}
            order, calls = [], []
            finish = self.fake_child(order)
            output = io.StringIO()

            def child(command, log, environment, **kwargs):
                calls.append(log.stem)
                self.assertIn("--resume", command)
                if len(calls) == 1:
                    raise TimeoutError("no task progress for 1800s")
                return finish(command, log, environment, **kwargs)

            with patch.object(cluster, "run_child", child), \
                    patch.object(cluster, "gpu_identity"), patch.object(cluster, "publish_reports"), \
                    redirect_stdout(output):
                multi.run_worker_multi([queue], args(), {}, (), "timeoutworker", lambda *a, **kw: None)
            self.assertTrue(all(queue.complete(t) for t in queue.tasks))
            attempts = [json.loads(p.read_text()) for p in (queue.directory / "attempts").glob("*.json")]
            failed = [r for r in attempts if r["status"] == "failed"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["exit_code"], 124)
            self.assertIn("TimeoutError: no task progress for 1800s", output.getvalue())
            self.assertEqual(calls.count(calls[0]), 2)

    def test_repeated_timeouts_are_bounded_and_other_tasks_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            self.queues = {queue.plan_path: queue}
            order = []
            finish = self.fake_child(order)

            def child(command, log, environment, **kwargs):
                if log.stem == "seed-5.prefix":
                    raise TimeoutError("stalled seed 5")
                return finish(command, log, environment, **kwargs)

            with patch.object(cluster, "run_child", child), \
                    patch.object(cluster, "gpu_identity"), patch.object(cluster, "publish_reports"), \
                    redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "failed task blocks"):
                multi.run_worker_multi([queue], args(), {}, (), "w1", lambda *a, **kw: None)
            task = next(t for t in queue.tasks if t.key == "seed-5.prefix")
            receipt = json.loads(queue.receipt(task).read_text())
            self.assertEqual(receipt["attempt"], args().max_attempts)
            self.assertEqual(receipt["exit_code"], 124)
            self.assertFalse(queue.locked(task))
            self.assertTrue(all(queue.complete(t) for t in queue.tasks if t.seed != 5))

    def test_nonzero_exit_prints_error_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = TaskQueue(write_inputs(Path(directory)))
            queue.bind()
            self.queues = {queue.plan_path: queue}
            order = []
            finish = self.fake_child(order)
            output = io.StringIO()
            failed = False

            def child(command, log, environment, **kwargs):
                nonlocal failed
                if not failed:
                    failed = True
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text("RuntimeError: test child failure\n")
                    return 1
                self.assertIn("RuntimeError: test child failure", output.getvalue())
                return finish(command, log, environment, **kwargs)

            with patch.object(cluster, "run_child", child), \
                    patch.object(cluster, "gpu_identity"), patch.object(cluster, "publish_reports"), \
                    redirect_stdout(output):
                multi.run_worker_multi([queue], args(), {}, (), "w1", lambda *a, **kw: None)
            self.assertTrue(all(queue.complete(t) for t in queue.tasks))

    def test_two_processes_drain_sixty_tasks_once_and_publish_both_heartbeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("math", "mbpp"):
                (root / name).mkdir()
            plans = [write_inputs(root / name, pending=True) for name in ("math", "mbpp")]
            context = multiprocessing.get_context("spawn")
            with context.Manager() as manager:
                barrier = manager.Barrier(2)
                with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
                    futures = [pool.submit(simulate_multi_worker, *map(str, plans), f"node{i}", barrier) for i in range(2)]
                    jobs = [job for future in futures for job in future.result(timeout=40)]
            self.assertEqual(len(jobs), 60)
            self.assertEqual(len({(dataset, task) for dataset, task, *_ in jobs}), 60)
            active = peak = 0
            for _, delta in sorted([(start, 1) for _, _, start, _ in jobs] + [(end, -1) for _, _, _, end in jobs]):
                active += delta
                peak = max(peak, active)
            self.assertEqual(peak, 2)
            for plan in plans:
                queue = TaskQueue(plan)
                self.assertTrue(all(queue.complete(task) for task in queue.tasks))
                for worker in (queue.directory / "workers").glob("*.json"):
                    self.assertEqual(json.loads(worker.read_text())["status"], "complete")

    def test_completed_queues_need_no_gpu_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir(); (root / "b").mkdir()
            queues = [TaskQueue(write_inputs(root / name)) for name in ("a", "b")]
            for queue in queues:
                for task in queue.tasks:
                    if task.arm != "cache":
                        finish_fake(queue, task)
            options = args()
            options.plan, options.worker_id, options.node_lock_root = queues[0].plan_path, "doneworker", root / "devices"
            with multi.multi_queue([queues[1].plan_path]), patch.object(cluster, "gpu_identity") as gpu, \
                    patch.object(cluster, "admit") as admit, patch.object(cluster, "publish_reports") as publish:
                cluster.worker(options)
            gpu.assert_not_called(); admit.assert_not_called()
            self.assertEqual(publish.call_count, 2)

    def test_child_error_records_failure_and_releases_task_and_gpu_leases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir(); (root / "b").mkdir()
            a = TaskQueue(write_inputs(root / "a")); b = TaskQueue(write_inputs(root / "b"))
            options = args()
            options.plan, options.worker_id, options.node_lock_root = a.plan_path, "errorworker", root / "devices"
            uuids = ("u0", "u1", "u2", "u3")

            def fail(command, log, environment, **kwargs):
                kwargs["heartbeat"](123)
                raise RuntimeError("simulated child timeout")

            with multi.multi_queue([b.plan_path]), patch.object(cluster, "run_child", fail), \
                    patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", uuids)), \
                    patch.object(cluster, "admit", return_value={}), redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, "simulated child timeout"):
                cluster.worker(options)
            task = next(t for t in a.tasks if t.key == "seed-5.prefix")
            self.assertFalse(a.locked(task))
            self.assertEqual(json.loads(a.receipt(task).read_text())["status"], "failed")
            with cluster.device_leases(options.node_lock_root, uuids):
                pass
            for queue in (a, b):
                record = json.loads((queue.directory / "workers/errorworker.json").read_text())
                self.assertEqual(record["status"], "failed")
                self.assertIn("simulated child timeout", record["error"])

    def test_worker_receipts_follow_the_dataset_that_is_actually_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir(); (root / "b").mkdir()
            a = TaskQueue(write_inputs(root / "a")); b = TaskQueue(write_inputs(root / "b"))
            self.queues = {a.plan_path: a, b.plan_path: b}
            order = []
            finish = self.fake_child(order)

            def child(command, log, environment, **kwargs):
                plan = Path(command[command.index("--plan") + 1])
                kwargs["heartbeat"](123)
                for queue in (a, b):
                    path = queue.directory / "workers/debugworker.json"
                    self.assertTrue(path.exists(), f"missing worker heartbeat for {queue.plan_path}")
                    record = json.loads(path.read_text())
                    self.assertEqual(record["active_plan"], str(plan))
                    self.assertEqual(record["status"], "running" if queue.plan_path == plan else "idle")
                    self.assertEqual(record["task"], log.stem if queue.plan_path == plan else None)
                return finish(command, log, environment, **kwargs)

            options = args()
            options.plan, options.worker_id, options.node_lock_root = a.plan_path, "debugworker", root / "devices"
            with multi.multi_queue([b.plan_path]), patch.object(cluster, "run_child", child), \
                    patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", ("u0", "u1", "u2", "u3"))), \
                    patch.object(cluster, "admit", return_value={}), patch.object(cluster, "publish_reports"), \
                    redirect_stdout(io.StringIO()):
                cluster.worker(options)
            self.assertEqual(len(order), 50)
            for queue in (a, b):
                self.assertEqual(json.loads((queue.directory / "workers/debugworker.json").read_text())["status"], "complete")

    def test_stopping_one_dataset_does_not_prevent_the_other_from_starting(self):
        from srgc_rebuttal.runtime import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir(); (root / "b").mkdir()
            a = TaskQueue(write_inputs(root / "a")); b = TaskQueue(write_inputs(root / "b"))
            a.bind(); b.bind()
            atomic_json(a.directory / "stop.json", {"immediate": False})
            self.queues = {a.plan_path: a, b.plan_path: b}
            order = []
            options = args()
            options.plan, options.worker_id, options.node_lock_root = a.plan_path, "stopworker", root / "devices"
            with multi.multi_queue([b.plan_path]), patch.object(cluster, "run_child", self.fake_child(order)), \
                    patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", ("u0", "u1", "u2", "u3"))), \
                    patch.object(cluster, "admit", return_value={}), patch.object(cluster, "publish_reports"), \
                    redirect_stdout(io.StringIO()):
                cluster.worker(options)
            self.assertEqual(len(order), 25)
            self.assertTrue(all(name == "b" for name, _ in order))
            self.assertFalse(any(a.complete(t) for t in a.tasks if t.arm != "cache"))

    def test_launcher_resolves_both_dataset_cohorts(self):
        from scripts import run_srgc_rebuttal
        root = Path(__file__).resolve().parents[2]
        with patch.object(sys, "path", [str(root / "scripts"), *sys.path]):
            for primary, secondary in (("math", "mbpp"), ("mbpp", "math")):
                for pair, fresh in ((True, None), (False, None), (True, "code-fix"), (False, "code-fix")):
                    with self.subTest(primary=primary, pair=pair, fresh=fresh), ExitStack() as stack:
                        paths = {"math": root / ("pair_seeds.json" if pair else "additional_seeds.json"),
                                 "mbpp": root / ("mbpp_pair_seeds.json" if pair else "mbpp_seeds.json")}
                        resolver = stack.enter_context(patch("srgc_pair_inputs.default_plan",
                            side_effect=lambda root, dataset, env, **kw: paths[dataset]))
                        stack.enter_context(patch("srgc_rebuttal.existing_runtime.select_python"))
                        route = stack.enter_context(patch("srgc_shared_storage.route_plan", side_effect=lambda p, **kw: p))
                        stack.enter_context(patch("srgc_shared_storage.storage_root", return_value=(root, root)))
                        wrappers = {}
                        for module, method in (("srgc_checkpoint_backup", "automatic_backup"),
                                ("srgc_log_format", "uniform_log"), ("srgc_process_guard", "process_guard"),
                                ("srgc_seed_order", "seed_first_queue")):
                            wrappers[module] = stack.enter_context(patch(f"{module}.{method}", side_effect=lambda *a, **kw: nullcontext()))
                        queues = stack.enter_context(patch("srgc_multi_queue.multi_queue",
                                                          side_effect=lambda *a: nullcontext()))
                        stack.enter_context(patch("srgc_step_checkpoints.worker_main"))
                        stack.enter_context(patch.object(sys, "argv", ["run", "worker", "--dataset", primary,
                                                                      "--with-dataset", secondary,
                                                                      *(["--fresh", fresh] if fresh else [])]))
                        run_srgc_rebuttal.main()
                        wrappers["srgc_process_guard"].assert_called_once_with(paths[primary])
                        self.assertEqual([c.args[1] for c in resolver.call_args_list], [primary, secondary])
                        self.assertTrue(all(c.kwargs["writing"] for c in resolver.call_args_list))
                        self.assertEqual([c.args[0] for c in route.call_args_list], [paths[primary], paths[secondary]])
                        self.assertTrue(all(c.kwargs["fresh"] == fresh for c in route.call_args_list))
                        self.assertTrue(all(c.kwargs["start_or_continue"] == (fresh is None)
                                            for c in route.call_args_list))
                        queues.assert_called_once_with([paths[secondary]])

    def fake_child(self, order):
        def run_child(command, log_path, environment, **kwargs):
            plan = Path(command[command.index("--plan") + 1])
            seed = int(command[command.index("--seed") + 1]) if "--seed" in command else int(command[command.index("--cache-seed") + 1])
            arm = command[command.index("--task") + 1] if "--task" in command else "cache"
            queue = self.queues[plan]
            task = next(t for t in queue.tasks if t.seed == seed and t.arm == arm)
            finish_fake(queue, task)
            order.append((plan.parent.name, task.key))
            return 0
        return run_child

    def test_worker_drains_both_queues_preferring_the_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir(); (root / "b").mkdir()
            a = TaskQueue(write_inputs(root / "a")); a.bind()
            b = TaskQueue(write_inputs(root / "b", pending=True)); b.bind()
            self.queues = {a.plan_path: a, b.plan_path: b}
            todo = sum(not q.complete(t) for q in (a, b) for t in q.tasks)  # a's caches are already complete
            order, states = [], []
            with patch.object(cluster, "run_child", self.fake_child(order)), \
                    patch.object(cluster, "gpu_identity", lambda: ("0,1,2,3", ("u0", "u1", "u2", "u3"))), \
                    patch.object(cluster, "publish_reports", lambda queue: states.append(("published", queue.plan_path.parent.name))), \
                    redirect_stdout(io.StringIO()):
                multi.run_worker_multi([a, b], args(), {}, (), "w1", lambda state, *rest: states.append(state))
            self.assertEqual(states[-1], "complete")
            self.assertTrue(all(a.complete(t) for t in a.tasks))
            self.assertTrue(all(b.complete(t) for t in b.tasks))
            first_b = next(i for i, (name, _) in enumerate(order) if name == "b")
            self.assertTrue(all(name == "a" for name, _ in order[:first_b]))  # queue a was drained first
            self.assertEqual(len(order), todo)
            self.assertEqual([s[1] for s in states if isinstance(s, tuple)], ["a", "b"])

    def test_multi_queue_patches_and_restores_run_worker(self):
        original = cluster.run_worker
        original_entry = cluster.worker
        with multi.multi_queue([]):
            self.assertIsNot(cluster.run_worker, original)
        self.assertIs(cluster.run_worker, original)
        self.assertIs(cluster.worker, original_entry)


if __name__ == "__main__":
    unittest.main()
