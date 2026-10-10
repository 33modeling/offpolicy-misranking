"""One atomic, consolidated JSON artifact per experiment."""

import os
from contextlib import contextmanager
from pathlib import Path

from srgc_rebuttal.runtime import atomic_json, lease


def work_root(environment=None):
    environment = os.environ if environment is None else environment
    group = Path(environment.get("GROUP_VOLUME", "/group-volume"))
    return Path(environment.get("OM_WORK", str(group / environment.get("OM_USER", "minsoo3.kim") /
                                              "offpolicy-misranking"))).resolve()


def save_result(name, result, *, work=None):
    work = work_root() if work is None else Path(work).resolve()
    destination = work / "results" / f"{name}-results.json"
    if not destination.resolve().is_relative_to(work):
        raise ValueError("result destination escapes the experiment workspace")
    atomic_json(destination, result)
    print(f"RESULT: {destination}", flush=True)
    print(f"SIZE: {destination.stat().st_size / (1024 * 1024):.2f} MiB", flush=True)
    return destination


@contextmanager
def export_lock(name, work):
    work = Path(work).resolve()
    path = work / ".result-export-locks" / f"{name}.lock"
    if not path.resolve().is_relative_to(work):
        raise ValueError("result lock escapes the experiment workspace")
    with lease(path, wait=True):
        yield
