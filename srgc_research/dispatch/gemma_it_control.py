"""Stop Gemma startup and prepared queues without rebinding experiment data."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from srgc_rebuttal import cluster
from srgc_rebuttal.runtime import atomic_json


class RunControl:
    def __init__(self, options):
        self.root = Path(options.root).resolve()
        self.datasets = ("math", "mbpp") if options.dataset == "all" else (options.dataset,)
        phase = os.environ.get("SRGC_DATASET_PHASE")
        self.active = tuple(name for name in self.datasets if phase is None or name == phase)
        self.paths = {
            dataset: (self.root / f"runs/{dataset}/.queue/stop.json",
                      self.root / f".startup-control/{dataset}/stop.json")
            for dataset in self.datasets
        }

    def validate_storage(self):
        from .gemma4_it.storage import group_work, inside, validate_tree

        group, _ = group_work(os.environ)
        inside(self.root, group)
        validate_tree(self.root)
        for paths in self.paths.values():
            for path in paths:
                inside(path, self.root)

    def requested(self, *, immediate=False):
        active = self.datasets if os.environ.get("SRGC_CACHE_ONLY") == "1" else self.active
        return any(cluster.stop_requested(SimpleNamespace(directory=path.parent), immediate=immediate)
                   for dataset in active for path in self.paths[dataset])

    def check(self):
        if self.requested():
            print("GEMMA stopped; no further stage launched. Use the normal resume action to continue.", flush=True)
            raise KeyboardInterrupt

    def stop(self, *, immediate):
        self.validate_storage()
        for dataset, (queue, startup) in self.paths.items():
            plan = self.root / "experiments" / f"gemma4-12b-it-{dataset}.json"
            # A queue created before its plan would make the immutable preparer
            # refuse that root. Keep pre-plan cancellation outside runs instead.
            target = queue if plan.is_file() or queue.parent.exists() else startup
            now = immediate or any(self.was_immediate(path) for path in (queue, startup))
            atomic_json(target, {"immediate": now, "requested": time.time()})
            print(f"GEMMA STOP {dataset}: {'immediate' if now else 'after current stage'}; {target}", flush=True)
        return 0

    @staticmethod
    def was_immediate(path):
        try:
            record = json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            return False
        return isinstance(record, dict) and record.get("immediate") is True

    def resume(self):
        from .gemma4_it import adapter

        self.validate_storage()
        # Do not unpause a historical/incompatible experiment merely because a
        # newer launcher was invoked against its root.
        for dataset in self.datasets:
            path = self.root / "experiments" / f"gemma4-12b-it-{dataset}.json"
            if path.exists():
                adapter.validate_extension(path)
        for paths in self.paths.values():
            for path in paths:
                path.unlink(missing_ok=True)

    def preparing(self, original):
        def prepare(*args, **kwargs):
            self.check()
            previous_child = cluster.run_child

            def child(command, log, environment, **options):
                self.check()
                previous_stop = options.get("should_stop", lambda: False)
                options["should_stop"] = lambda: previous_stop() or self.requested(immediate=True)
                result = previous_child(command, log, environment, **options)
                # Graceful cancellation lets the current atomic preparation
                # finish, then prevents the next dataset or GPU admission.
                self.check()
                return result

            with patch.object(cluster, "run_child", child):
                result = original(*args, **kwargs)
            self.check()
            return result

        return prepare
