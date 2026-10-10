"""Consolidate SRGC model results without training, downloads or extra exports."""

import argparse
import hashlib
import importlib
import io
import json
import os
import time
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from srgc_rebuttal import reports

from .result_export import export_lock, save_result

CONFIGS = {
    "qwen35": ("scripts.srgc_qwen35", "scripts.srgc_qwen35_storage", "qwen35-9b"),
    "gemma4": ("srgc_research.dispatch.gemma4.adapter", "srgc_research.dispatch.gemma4.storage", "gemma4-12b-pt"),
    "llama31": ("srgc_research.dispatch.llama31.adapter", "srgc_research.dispatch.llama31.storage", "llama31-8b"),
}


def compact(result):
    """Retain final outcomes, switching decisions, identities and measured costs."""
    def fields(value, names):
        return {key: value[key] for key in names if key in value}

    def ledger(value):
        if value is None:
            return None
        row = fields(value, ("complete", "total_gpu_seconds", "known_gpu_seconds", "recorded_phases"))
        row["unfinished_phase_count"] = value.get("unfinished_phase_count", len(value.get("unfinished_phases", [])))
        return row

    datasets = {}
    for name, snapshot in result["datasets"].items():
        row = fields(snapshot, ("dataset", "requested_dataset", "label", "prepared", "complete", "counts",
                                "arm_statistics", "initial_candidate_accuracy", "plan_path", "errors", "warnings"))
        row["tasks"] = [fields(task, ("seed", "arm", "task", "status", "step", "reward_percent", "switched_at",
                                     "cost_complete", *reports.COST_FIELDS)) for task in snapshot["tasks"]]
        row["plan"] = fields(snapshot.get("plan", {}), (
            "schema", "protocol_revision", "dataset", "model", "model_revision", "seeds", "arms", "split_seed",
            "shared_prefix_updates", "total_updates", "world_size", "objective", "responses", "training_prompts",
            "ranking_validation_prompts", "scoring_prompts_per_set", "selection_protocol", "selection_interval",
            "check_interval", "first_check", "projection_dim", "projection_seed", "max_new_tokens", "verifier",
            "primary_outcome", "primary_contrasts"))
        row["endpoints"] = []
        for item in snapshot.get("endpoints", []):
            entry = fields(item, ("seed", "arm", "source", "sha256"))
            entry["endpoint"] = fields(item["endpoint"], (
                "seed", "arm", "plan_sha256", "input_sha256", "implementation_sha256", "prefix_checkpoint_sha256",
                "total_updates", "shared_prefix_updates", "switched_at", "sampling_protocol",
                "sr_gc_comparison_prompts_per_set", "reward", "per_question_reward", "cost_measurement_complete",
                "selection_interval", "selection_steps", "checks"))
            row["endpoints"].append(entry)
        row["costs"] = []
        for cost in snapshot.get("costs", []):
            entry = fields(cost, ("seed", "experiment_accounting"))
            entry["arms"] = {arm: fields(value, ("complete", *reports.COST_FIELDS, "known_completed_phase_gpu_seconds"))
                             for arm, value in cost.get("arms", {}).items()}
            entry["shared_prefix"] = ledger(cost.get("shared_prefix"))
            entry["shared_all_task_startup_and_load"] = ledger(cost.get("shared_all_task_startup_and_load"))
            row["costs"].append(entry)
        datasets[name] = row
    return {**result, "format": "compact", "datasets": datasets}


def file_version(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def export(model, dataset="all", *, root=None, environment=None):
    environment = os.environ if environment is None else environment
    storage = importlib.import_module(CONFIGS[model][1])
    _, work = storage.group_work(environment)
    with export_lock(model, work):
        return _export(model, dataset, root=root, environment=environment)


def _export(model, dataset, *, root, environment):
    environment = os.environ if environment is None else environment
    adapter_name, storage_name, prefix = CONFIGS[model]
    adapter, storage = importlib.import_module(adapter_name), importlib.import_module(storage_name)
    group, work = storage.group_work(environment)
    root = storage.inside(storage.default_root(environment) if root is None else root, group)
    storage.validate_tree(root)
    names = ("math", "mbpp") if dataset == "all" else (dataset,)
    versions = {path: file_version(path) for name in names
                for path in (root / "runs" / name).glob("seed-*/*-endpoint.json")}
    original = reports.snapshot

    def with_costs(path, **kwargs):
        return original(path, **{**kwargs, "include_costs": True})

    with patch.object(reports, "snapshot", with_costs):
        if model == "qwen35":
            from . import qwen_status

            with patch.object(qwen_status, "code_digest", adapter.engine_digest):
                snapshots = qwen_status.load_reports(dataset, root, environment)
        else:
            status = importlib.import_module(adapter_name.removesuffix("adapter") + "status")
            output = io.StringIO()
            with redirect_stdout(output), patch.object(status, "code_digest", adapter.engine_digest):
                status.show(dataset, root, environment, as_json=True)
            snapshots = json.loads(output.getvalue())["datasets"]
    result = {"export_schema": "srgc-model-results-v1", "experiment": model,
              "model": adapter.MODEL, "model_revision": adapter.REVISION,
              "generated": time.time(), "source_root": str(root), "complete": False,
              "datasets": {}, "coverage": {}, "errors": [], "warnings": [],
              "units": "reward percent; allocated GPU-seconds; missing values are null"}
    completed, planned = 0, 0
    for name, snapshot in zip(names, snapshots, strict=True):
        snapshot["requested_dataset"] = name
        plan_path = root / "experiments" / f"{prefix}-{name}.json"
        endpoints = []
        try:
            plan = adapter.validate_extension(plan_path, read_only=True)
            snapshot["plan"] = plan
            snapshot["plan_path"] = str(plan_path)
            planned += len(plan["seeds"]) * len(plan["arms"])
        except (OSError, ValueError, KeyError, TypeError):
            planned += 5 * 4
        for row in snapshot["tasks"]:
            if row["arm"] in {"cache", "prefix"} or row["status"] != "complete":
                continue
            path = root / "runs" / name / f"seed-{row['seed']}" / f"{row['arm']}-endpoint.json"
            try:
                before = versions.get(path)
                if before is None or file_version(path) != before:
                    raise ValueError("endpoint changed during result inspection; rerun results")
                payload = path.read_bytes()
                endpoint = json.loads(payload)
                if file_version(path) != before:
                    raise ValueError("endpoint changed during result inspection; rerun results")
                endpoints.append({"seed": row["seed"], "arm": row["arm"], "source": str(path),
                                  "sha256": hashlib.sha256(payload).hexdigest(), "endpoint": endpoint})
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["status"] = "invalid"
                row["reward_percent"] = None
                row["cost_complete"] = False
                for field in reports.COST_FIELDS:
                    row[field] = None
                snapshot["errors"].append(f"{row['task']}: {error}")
        snapshot["endpoints"] = endpoints
        snapshot["counts"] = dict(Counter(row["status"] for row in snapshot["tasks"]))
        for arm, statistics in snapshot.get("arm_statistics", {}).items():
            rows = [row for row in snapshot["tasks"] if row["arm"] == arm]
            statistics["completed_seeds"] = sum(row["status"] == "complete" for row in rows)
            if statistics["completed_seeds"] != statistics["planned_seeds"]:
                statistics["mean_reward_percent"] = None
                statistics["mean_selection_training_preparation_gpu_seconds"] = None
        snapshot["complete"] = snapshot["complete"] and not snapshot["errors"]
        completed += len(endpoints)
        result["datasets"][name] = snapshot
        result["errors"].extend(f"{name}: {error}" for error in snapshot["errors"])
        result["warnings"].extend(f"{name}: {warning}" for warning in snapshot["warnings"])
    result["complete"] = not result["errors"] and all(snapshot["complete"] for snapshot in snapshots)
    result["coverage"] = {"completed_continuations": completed, "planned_continuations": planned}
    save_result(model, compact(result), work=work)
    print(f"{model.upper()} {completed}/{planned} continuations; "
          f"{'COMPLETE' if result['complete'] else 'INCOMPLETE'}", flush=True)
    for error in result["errors"]:
        print(f"ERROR: {error}", flush=True)
    return int(bool(result["errors"]))


def publish(model, queue):
    """Exporting results cannot turn a successful GPU task into a failed task."""
    try:
        return export(model, root=queue.plan_path.parent.parent)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"RESULT export failed for {model}: {error}; rerun results", flush=True)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(CONFIGS))
    parser.add_argument("dataset", choices=("math", "mbpp", "all"), nargs="?", default="all")
    parser.add_argument("action", choices=("results",), nargs="?", default="results")
    parser.add_argument("--root", type=Path)
    # Existing launch commands may retain status/runner switches. They do not
    # authorize training when the selected action is results.
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        return export(args.model, args.dataset, root=args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
