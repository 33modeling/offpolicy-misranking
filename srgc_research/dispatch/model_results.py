"""Consolidate SRGC model results without training, downloads or extra exports."""

import argparse
import gzip
import hashlib
import importlib
import json
import os
import time
import zlib
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from srgc_rebuttal import reports

from .model_status import CONFIGS, load_reports
from .result_export import export_lock, save_result
from .reward_curve_results import collect as collect_reward_curve


def compact(result):
    """Retain outcomes, scalar reward curves, identities and measured costs."""
    def fields(value, names):
        return {key: value[key] for key in names if key in value}

    def task(value):
        row = fields(value, ("seed", "arm", "task", "status", "step", "reward_percent", "switched_at",
                             "cost_complete", *reports.COST_FIELDS))
        for name in ("timing", "submission_timing"):
            if name not in value:
                continue
            measured = value[name]
            # MISSING already identifies all time values as unknown. Avoid
            # repeating empty timers and the same 250-update missing range per task.
            row[name] = ({"record": "MISSING"} if measured.get("record") == "MISSING" else measured)
        return row

    def ledger(value):
        if value is None:
            return None
        row = fields(value, ("complete", "total_gpu_seconds", "known_gpu_seconds", "recorded_phases",
                             "exclusive_stages", "counts"))
        row["unfinished_phase_count"] = value.get("unfinished_phase_count", len(value.get("unfinished_phases", [])))
        return row

    datasets = {}
    for name, snapshot in result["datasets"].items():
        row = fields(snapshot, ("dataset", "requested_dataset", "label", "prepared", "complete", "counts",
                                "arm_statistics", "initial_candidate_accuracy", "plan_path", "errors", "warnings",
                                "submission_time_basis", "submission_time_statistics", "recorded_execution_settings",
                                "timing_units", "timing_accounting"))
        row["tasks"] = [task(item) for item in snapshot["tasks"]]
        row["plan"] = fields(snapshot.get("plan", {}), (
            "schema", "protocol_revision", "dataset", "model", "model_revision", "seeds", "arms", "split_seed",
            "shared_prefix_updates", "total_updates", "world_size", "objective", "responses", "training_prompts",
            "ranking_validation_prompts", "scoring_prompts_per_set", "selection_protocol", "selection_interval",
            "check_interval", "first_check", "projection_dim", "projection_seed", "max_new_tokens", "verifier",
            "primary_outcome", "primary_contrasts", "generation_micro_batch", "logprob_micro_batch",
            "logit_chunk_tokens", "gradient_checkpointing", "attention", "checkpoint_snapshot",
            "lora_targets", "thinking", "extension", "sampling", "model_snapshot_sha256",
            "adapter_sha256", "engine_sha256"))
        row["endpoints"] = []
        for item in snapshot.get("endpoints", []):
            entry = fields(item, ("seed", "arm", "source", "sha256"))
            entry["endpoint"] = fields(item["endpoint"], (
                "seed", "arm", "plan_sha256", "input_sha256", "implementation_sha256", "prefix_checkpoint_sha256",
                "total_updates", "shared_prefix_updates", "switched_at", "sampling_protocol",
                "sr_gc_comparison_prompts_per_set", "reward", "per_question_reward", "cost_measurement_complete",
                "selection_interval", "selection_steps", "checks", "checkpoint_policy"))
            row["endpoints"].append(entry)
        row["costs"] = []
        row["reward_curves"] = snapshot.get("reward_curves", [])
        for cost in snapshot.get("costs", []):
            entry = fields(cost, ("seed", "experiment_accounting"))
            entry["arms"] = {arm: fields(value, ("complete", *reports.COST_FIELDS, "known_completed_phase_gpu_seconds",
                                                "exclusive_stages", "counts"))
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


def export(model, dataset="all", *, root=None, environment=None, text=False, full=False):
    environment = os.environ if environment is None else environment
    storage = importlib.import_module(CONFIGS[model][1])
    _, work = storage.group_work(environment)
    with export_lock(model, work):
        return _export(model, dataset, root=root, environment=environment, text=text, full=full)


def _export(model, dataset, *, root, environment, text=False, full=False):
    environment = os.environ if environment is None else environment
    adapter_name, storage_name, prefix = CONFIGS[model]
    adapter, storage = importlib.import_module(adapter_name), importlib.import_module(storage_name)
    group, work = storage.group_work(environment)
    root = storage.inside(storage.default_root(environment) if root is None else root, group)
    storage.validate_tree(root)
    names = ("math", "mbpp") if dataset == "all" else (dataset,)
    patterns = ("seed-*/*-endpoint.json", "seed-*/*-progress.json", "seed-*/run.json",
                "seed-*/cost-receipts/*/*.json",
                "seed-*/reward-curves/*/protocol.json", "seed-*/reward-curves/*/training.json",
                "seed-*/reward-curves/*/recording-extensions/*.json",
                "seed-*/reward-curves/*/step-*.json", "seed-*/reward-curves/*/rollouts/step-*.json.gz")
    versions = {path: file_version(path) for name in names for pattern in patterns
                for path in (root / "runs" / name).glob(pattern)}
    finished, continuations = set(), {}

    def settled(path):
        """Only a complete continuation's records must stay fixed during export."""
        try:
            name, folder, *parts = path.relative_to(root / "runs").parts
            seed = int(folder.removeprefix("seed-"))
        except ValueError:
            return True
        if not parts:
            return True
        if parts == ["run.json"]:
            return False  # Its status changes when the seed's last arm completes.
        if parts[0] in {"cost-receipts", "reward-curves"}:
            owner = parts[1]
        else:
            owner = parts[0].removesuffix("-endpoint.json").removesuffix("-progress.json")
        if owner == "prefix":
            # Each arm can publish the shared-prefix evaluation when it starts.
            arms = continuations.get((name, seed), set())
            return bool(arms) and all((name, seed, arm) in finished for arm in arms)
        return (name, seed, owner) in finished

    def parse(path, payload):
        try:
            value = json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)
            if not isinstance(value, dict):
                raise TypeError("expected a JSON object")
            if path.suffix == ".gz":
                groups = value.get("groups")
                if not isinstance(groups, list) or any(not isinstance(group, dict) for group in groups):
                    raise TypeError("rollout groups must be an array of JSON objects")
                if not isinstance(value.get("history_record"), dict):
                    raise TypeError("rollout history must be a JSON object")
        except (OSError, EOFError, ValueError, TypeError, zlib.error) as error:
            # A corrupt archive is local to this arm's reward records. Keep
            # successful endpoints and other arms in the partial export, and
            # never let automatic publication turn this into a learner crash.
            raise ValueError(f"invalid reward record {path}: {error}") from error
        return value

    def read_stable(path):
        if not settled(path):
            # A running arm keeps writing while results are read. Its files are
            # replaced atomically, so read one whole current version instead.
            for _ in range(3):
                current = file_version(path)
                if current is None:
                    return None
                payload = path.read_bytes()
                if file_version(path) == current:
                    return parse(path, payload)
            raise ValueError("reward records changed during result inspection; rerun results")
        before = versions.get(path)
        if before is None and not path.exists():
            return None
        if before is None or file_version(path) != before:
            raise ValueError("reward records changed during result inspection; rerun results")
        value = parse(path, path.read_bytes())
        if file_version(path) != before:
            raise ValueError("reward records changed during result inspection; rerun results")
        return value
    original = reports.snapshot

    def with_costs(path, **kwargs):
        return original(path, **{**kwargs, "include_costs": True})

    with patch.object(reports, "snapshot", with_costs):
        snapshots = load_reports(model, dataset, root, environment)
    for name, snapshot in zip(names, snapshots, strict=True):
        for row in snapshot["tasks"]:
            if row["arm"] in {"cache", "prefix"}:
                continue
            continuations.setdefault((name, row["seed"]), set()).add(row["arm"])
            if row["status"] == "complete":
                finished.add((name, row["seed"], row["arm"]))
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
        plan = None
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
        snapshot["reward_curves"] = []
        if plan is not None:
            from srgc_rebuttal.plan import digest, input_path

            from .submission_time import attach as attach_submission_time

            attach_submission_time(snapshot, plan_path, plan, read=read_stable)

            identity = hashlib.sha256((adapter.engine_digest() + adapter.adapter_digest()).encode()).hexdigest()
            for row in snapshot["tasks"]:
                if row["arm"] not in plan["arms"] or row["status"] == "invalid":
                    continue
                folder = root / "runs" / name / f"seed-{row['seed']}"
                try:
                    bundle_path = input_path(plan_path, plan, row["seed"])
                    data = json.loads(bundle_path.read_text())
                    expected = {"seed": row["seed"], "plan_sha256": digest(plan_path),
                                "input_sha256": digest(bundle_path), "implementation_sha256": identity}
                    curve = collect_reward_curve(folder, row["arm"], plan, expected, data["evaluation_ids"], read_stable,
                                                 live=row["status"] != "complete")
                    endpoint = next((e for e in endpoints if e["seed"] == row["seed"] and e["arm"] == row["arm"]), None)
                    if endpoint is not None:
                        value = endpoint["endpoint"]["reward"]
                        curve["held_out_rewards"].append({"step": plan["total_updates"], "reward": value,
                            "reward_percent": 100 * value, "kind": "fixed_held_out_evaluation",
                            "source": endpoint["source"], "responses_per_prompt": 8,
                            "question_count": len(data["evaluation_ids"]), "shared_prefix": False})
                    curve["held_out_rewards"].sort(key=lambda point: point["step"])
                    measured = {p["step"] for p in curve["held_out_rewards"]}
                    curve["missing_held_out_steps"] = [step for step in range(25, plan["total_updates"] + 1, 25) if step not in measured]
                    curve["training_reward_is_held_out"] = False
                    snapshot["reward_curves"].append(curve)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    snapshot["errors"].append(f"{row['task']} reward curve: {error}")
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
    artifact = compact(result)
    if not full:
        from .model_result_transfer import compact as transfer

        artifact = transfer(artifact)
    destination = save_result(f"{model}-full" if full else model, artifact, work=work, quiet=text, dense=not full)
    if text:
        from .model_result_text import render

        print(render(artifact), flush=True)
        print(f"\nJSON: {destination}", flush=True)
    else:
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
    parser.add_argument("--json", action="store_true", help="retained for compatibility; results always save JSON and print tables")
    parser.add_argument("--full", action="store_true", help="export response-level details to a separate large JSON; default is a small transfer JSON")
    args = parser.parse_args(argv)
    try:
        return export(args.model, args.dataset, root=args.root, text=True, full=args.full)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
