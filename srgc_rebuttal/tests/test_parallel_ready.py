from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest

from scripts.srgc_seed_order import seed_first_queue
from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import Task
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs


class ParallelReadyTests(unittest.TestCase):
    def test_all_twenty_arms_can_be_leased_without_a_ten_worker_cap(self):
        with tempfile.TemporaryDirectory() as directory, seed_first_queue(), ExitStack() as leases:
            plan = write_inputs(Path(directory))
            producer = cluster.TaskQueue(plan)
            prefixes = [leases.enter_context(cluster.TaskQueue(plan).claim()) for _ in range(5)]
            self.assertEqual(set(prefixes), {Task(seed, "prefix") for seed in range(5, 10)})
            waiting = cluster.TaskQueue(plan)
            with waiting.claim() as task:
                self.assertIsNone(task)
            for prefix in prefixes:
                finish_fake(producer, prefix)
            tasks = [leases.enter_context(cluster.TaskQueue(plan).claim()) for _ in range(20)]
            self.assertEqual(set(tasks), {Task(seed, arm) for seed in range(5, 10)
                                         for arm in producer.plan["arms"]})
            with waiting.claim() as task:
                self.assertIsNone(task)

    def test_published_prefix_releases_all_arms_while_producer_is_locked(self):
        with tempfile.TemporaryDirectory() as directory, seed_first_queue(), ExitStack() as leases:
            plan = write_inputs(Path(directory))
            producer = cluster.TaskQueue(plan)
            prefix = leases.enter_context(producer.claim())
            self.assertEqual(prefix, Task(5, "prefix"))
            finish_fake(producer, prefix)
            claimed = []
            for _ in range(4):
                consumer = cluster.TaskQueue(plan)
                task = leases.enter_context(consumer.claim())
                claimed.append(task)
                self.assertTrue(consumer.locked(task))
                self.assertTrue(consumer.locked(prefix))
            self.assertEqual(set(claimed), {Task(5, arm) for arm in producer.plan["arms"]})
            next_worker = cluster.TaskQueue(plan)
            self.assertEqual(leases.enter_context(next_worker.claim()), Task(6, "prefix"))

    def test_unpublished_latest_does_not_release_arms_or_block_other_seeds(self):
        with tempfile.TemporaryDirectory() as directory, seed_first_queue(), ExitStack() as leases:
            plan = write_inputs(Path(directory))
            producer = cluster.TaskQueue(plan)
            prefix = leases.enter_context(producer.claim())
            folder = producer.root / "seed-5"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "prefix-latest.pt").write_bytes(b"partial checkpoint")
            (folder / "prefix.pt").write_bytes(b"not yet published")
            consumer = cluster.TaskQueue(plan)
            for arm in producer.plan["arms"]:
                self.assertFalse(consumer.ready(Task(5, arm)))
            self.assertEqual(leases.enter_context(consumer.claim()), Task(6, "prefix"))
            self.assertTrue(producer.locked(prefix))

    def test_published_prefix_still_requires_valid_digest(self):
        with tempfile.TemporaryDirectory() as directory, seed_first_queue():
            queue = cluster.TaskQueue(write_inputs(Path(directory)))
            with queue.claim() as prefix:
                finish_fake(queue, prefix)
                (queue.root / "seed-5/prefix.pt").write_bytes(b"corrupted checkpoint")
                with self.assertRaises(ValueError):
                    queue.ready(Task(5, "random"))

    def test_cache_handoff_still_waits_for_producer_lease(self):
        with tempfile.TemporaryDirectory() as directory, seed_first_queue():
            plan = write_inputs(Path(directory), pending=True)
            queue = cluster.TaskQueue(plan)
            with queue.claim() as cache:
                self.assertEqual(cache, Task(5, "cache"))
                finish_fake(queue, cache)
                consumer = cluster.TaskQueue(plan)
                consumer.verify()
                self.assertTrue(consumer.complete(cache))
                self.assertFalse(consumer.ready(Task(5, "prefix")))
            self.assertTrue(consumer.ready(Task(5, "prefix")))


if __name__ == "__main__":
    unittest.main()
