"""Replace one current GPU-time/result snapshot without reading older snapshots."""

import importlib
from pathlib import Path
import sys
import time

from srgc_rebuttal.runtime import atomic_json


def save_current(model, dataset, root, snapshots, environment):
    from .model_status import CONFIGS

    storage = importlib.import_module(CONFIGS[model][1])
    adapter = importlib.import_module(CONFIGS[model][0])
    _, work = storage.group_work(environment)
    destination = work / "status" / f"{model}-latest.json"
    if not destination.resolve().is_relative_to(work):
        raise ValueError("status destination escapes the experiment workspace")
    datasets = {}
    for report in snapshots:
        tasks = []
        for row in report["tasks"]:
            shared = row["arm"] in {"cache", "prefix"}
            timing = row.get("timing" if shared else "submission_timing", {"record": "MISSING"})
            measured = {key: timing[key] for key in (
                "record", "gpu_seconds", "gpu_hours", "recorded_updates", "expected_updates",
                "known_recorded_subtotal_gpu_seconds", "source", "preparation_source", "error") if key in timing}
            tasks.append({"seed": row["seed"], "arm": row["arm"], "task": row["task"],
                          "status": row["status"],
                          "reward_percent": row.get("reward_percent") if row["status"] == "complete" else None,
                          "switched_at": row.get("switched_at"), "gpu_time": measured,
                          "time_basis": "shared cache/prefix receipts" if shared else "retained-update timers"})
        name = "math" if report["label"] == "MATH" else "mbpp"
        datasets[name] = {"complete": report["complete"], "tasks": tasks,
                          "plan_path": str(Path(root).resolve() / "experiments" / f"{CONFIGS[model][2]}-{name}.json"),
                          "initial_candidate_accuracy": report.get("initial_candidate_accuracy", []),
                          "errors": report["errors"]}
    atomic_json(destination, {"schema": "model-status-results-v1", "experiment": model,
                             "model": adapter.MODEL, "model_revision": adapter.REVISION,
                             "source_root": str(Path(root).resolve()), "dataset_scope": dataset,
                             "generated": time.time(), "datasets": datasets,
                             "units": "reward percent; allocated GPU-seconds and GPU-hours; missing values are null"})
    return destination


def print_current(text):
    if sys.stdout.isatty():
        print("\x1b[H\x1b[2J", end="", flush=True)
    print(text, flush=True)
