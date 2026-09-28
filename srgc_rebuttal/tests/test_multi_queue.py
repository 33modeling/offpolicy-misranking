import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import srgc_multi_queue as multi
from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs


def args():
    return SimpleNamespace(retry_failed=True, max_attempts=3, retry_delay=0, poll_seconds=0.01,
                           heartbeat_seconds=1, stall_seconds=1800)


class MultiQueueTest(unittest.TestCase):
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
        with multi.multi_queue([]):
            self.assertIsNot(cluster.run_worker, original)
        self.assertIs(cluster.run_worker, original)


if __name__ == "__main__":
    unittest.main()
