"""Start a new IT comparison without reusing caches or deleting original records."""

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from srgc_rebuttal.runtime import atomic_json

from .storage import group_work, inside, validate_tree


def request_stop(root):
    from ..gemma_it_control import RunControl
    from types import SimpleNamespace

    RunControl(SimpleNamespace(root=root, dataset="all")).stop(immediate=True)


def stop_previous_pt():
    """An explicit reset cancels old PT queues, retaining every original file."""
    from ..gemma_text_control import RunControl
    from types import SimpleNamespace

    group, work = group_work(os.environ)
    parent = work / "srgc-rebuttal"
    roots = {parent / "gemma4-12b-pt-v1"}
    roots.update(parent.glob("gemma4-12b-text-v[0-9]"))
    for key in ("SRGC_GEMMA_TEXT_ROOT", "SRGC_GEMMA_ROOT"):
        if os.environ.get(key):
            roots.add(Path(os.environ[key]))
    for root in sorted(roots):
        root = inside(root, group)
        plans = list((root / "experiments").glob("gemma4-12b-*.json"))
        if plans and all(json.loads(path.read_text()).get("model") == "google/gemma-4-12B" for path in plans):
            print(f"GEMMA stopping previous PT comparison: {root}", flush=True)
            RunControl(SimpleNamespace(root=root, dataset="all")).stop(immediate=True)


def validate_root(root):
    group, _ = group_work(os.environ)
    root = inside(root, group)
    if root.exists():
        validate_tree(root)
        plans = list((root / "experiments").glob("*.json"))
        if any(json.loads(plan.read_text()).get("model") != "google/gemma-4-12B-it"
               for plan in plans):
            raise ValueError("reset root contains another experiment; use the default IT root")
        if any(root.iterdir()) and not plans:
            allowed = {".startup-control", "startup-logs", ".start-prepare.lock", "initialization.json"}
            if {path.name for path in root.iterdir()} - allowed:
                raise ValueError("reset root is not an identified Gemma IT experiment")
    return root


@contextmanager
def session(root, *, reset=False):
    """External lifetime lease prevents moving a root while a worker writes it."""
    root = validate_root(root) if reset else Path(root).resolve()
    group, _ = group_work(os.environ)
    inside(root, group)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.session.lock").open("a+") as handle:
        if reset:
            stop_previous_pt()
            if root.exists():
                request_stop(root)
            deadline = time.monotonic() + 120
            announced = False
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not announced:
                        print("GEMMA stopping the previous IT worker before resetting", flush=True)
                        announced = True
                    if time.monotonic() >= deadline:
                        raise ValueError("previous Gemma worker is still stopping; reset did not move its files")
                    time.sleep(1)
            validate_root(root)
            if root.exists():
                archive = root.parent / f"{root.name}.archive-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
                root.rename(archive)
                print(f"GEMMA previous IT results archived: {archive}", flush=True)
            root.mkdir()
            atomic_json(root / "initialization.json", {
                "model": "google/gemma-4-12B-it", "created": time.time(),
                "cached_rewards_reused": False, "checkpoints_reused": False,
                "datasets": ["math", "mbpp"],
            })
            # Downgrade atomically: other workers may join, another reset cannot.
            fcntl.flock(handle, fcntl.LOCK_SH)
            print("GEMMA initialized: fresh MATH and MBPP caches, then training", flush=True)
        else:
            fcntl.flock(handle, fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
