"""Prefer earlier seeds without imposing dependencies between seeds or arms.

The queue itself lists tasks phase by phase, so a free worker picks another
seed's cache as soon as one cache finishes. This wrapper only changes the claim
order. A verified immutable prefix releases all four arms even while its
producer is still shutting down. Task/device leases and cache handoff remain.
"""

from contextlib import contextmanager

ORDER = ("cache", "prefix", "on_policy", "switch", "sr", "random")


def seed_first(tasks, seeds):
    position = {seed: index for index, seed in enumerate(seeds)}
    return sorted(tasks, key=lambda task: (position[task.seed], ORDER.index(task.arm)))


@contextmanager
def seed_first_queue():
    from srgc_rebuttal import cluster
    original = cluster.TaskQueue

    class SeedFirstQueue(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tasks = seed_first(self.tasks, self.plan["seeds"])

        def ready(self, task):
            parent = self.dependency(task)
            if parent is not None and parent.arm == "prefix":
                # prefix.pt + its verified receipt are immutable after publication.
                # GPU teardown/cost finalization need not serialize the consumers.
                return self.complete(parent)
            return super().ready(task)

    cluster.TaskQueue = SeedFirstQueue
    try:
        yield
    finally:
        cluster.TaskQueue = original
