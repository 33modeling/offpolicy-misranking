"""One atomic, consolidated JSON artifact per experiment."""

import os
from pathlib import Path

from srgc_rebuttal.runtime import atomic_json


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
    return destination
