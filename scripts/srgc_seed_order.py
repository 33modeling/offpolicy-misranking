"""Workers finish each seed's cache, prefix and arms before starting the next seed's cache.

The queue itself lists tasks phase by phase, so a free worker picks another
seed's cache as soon as one cache finishes. This wrapper only changes the claim
order: dependencies, leases, attempts and the experiment package are untouched.
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

    cluster.TaskQueue = SeedFirstQueue
    try:
        yield
    finally:
        cluster.TaskQueue = original
