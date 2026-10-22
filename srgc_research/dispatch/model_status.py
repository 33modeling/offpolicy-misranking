"""One read-only status loader for Qwen, Llama and Gemma experiment queues."""

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts import srgc_live_status as live
from srgc_rebuttal import reports
from srgc_rebuttal.plan import input_path
from srgc_rebuttal.srgc import stream_seed

CONFIGS = {
    "gemma4_it": ("srgc_research.dispatch.gemma4_it.adapter", "srgc_research.dispatch.gemma4_it.storage", "gemma4-12b-it"),
    "qwen35": ("scripts.srgc_qwen35", "scripts.srgc_qwen35_storage", "qwen35-9b"),
    "gemma4": ("srgc_research.dispatch.gemma4.adapter", "srgc_research.dispatch.gemma4.storage", "gemma4-12b-pt"),
    "gemma4_text": ("srgc_research.dispatch.gemma4_text.adapter", "srgc_research.dispatch.gemma4_text.storage", "gemma4-12b-text"),
    "llama31": ("srgc_research.dispatch.llama31.adapter", "srgc_research.dispatch.llama31.storage", "llama31-8b"),
}
LABELS = {"qwen35": "Qwen3.5-9B", "llama31": "Llama-3.1-8B-Instruct", "gemma4": "Gemma-4-12B-PT"}
LABELS["gemma4_text"] = "Gemma-4-12B-PT (text-v3)"
LABELS["gemma4_it"] = "Gemma-4-12B-IT (thinking off)"


def interrupted_receipts(report):
    """Interpret old signal exits without rewriting receipts or retry history."""
    for row in report["tasks"]:
        if row["status"] != "failed":
            continue
        path = Path(report["output_root"]) / ".queue/tasks" / f"{row['task']}.json"
        try:
            receipt = live.read_object(path)
        except (OSError, ValueError, TypeError) as error:
            report["errors"].append(f"{row['task']} interruption status: {error}")
            continue
        if receipt.get("status") != "failed":
            continue
        code = receipt.get("exit_code")
        row["validation_error"] = receipt.get("validation_error")
        if type(code) is not int or row["validation_error"]:
            continue
        normalized = 128 - code if code < 0 else code
        # HUP/INT/TERM identify a terminated session. SIGKILL/137 may be OOM,
        # so an old ambiguous failure must retain its ERROR and original log.
        if normalized in (129, 130, 143):
            row.update(status="interrupted", receipt_status="failed", exit_code=code)


def initial_accuracy(data, responses):
    cache = data.get("cached_rewards", {})
    candidates = data["candidate_ids"]
    if not isinstance(cache, dict):
        raise TypeError("initial candidate rewards must be an object")
    if not cache:
        return None
    if set(cache) - set(candidates) or not candidates:
        raise ValueError("initial candidate rewards include unknown candidate prompts")
    if any(not isinstance(values, list) or len(values) != responses
           or any(reward not in (0, 1) for reward in values)
           for values in cache.values()):
        raise ValueError("invalid initial candidate rewards")
    if set(cache) != set(candidates):
        return None
    rewards = [reward for values in cache.values() for reward in values]
    return {"responses": len(rewards), "success_percent": 100 * sum(rewards) / len(rewards)}


def final_evaluation_progress(report, plan, seed, data):
    """Observe saved rollouts, without treating their presence as a valid endpoint."""
    # run_experiment.evaluate uses eight responses and this stream, independently
    # of the training/scoring caches. Read names only: token arrays can be large.
    responses = 8
    expected = {f"{hashlib.sha256(rid.encode()).hexdigest()[:24]}-{responses}.npz"
                for rid in data["evaluation_ids"]}
    evaluation_seed = stream_seed(seed, plan["total_updates"], "reporting-evaluation")
    for row in report["tasks"]:
        if (row["seed"] != seed or row["status"] != "running"
                or row.get("phase_stage") != "evaluation"):
            continue
        progress = {"saved_prompts": None, "total_prompts": len(data["evaluation_ids"]),
                    "responses_per_prompt": responses}
        directory = (Path(report["output_root"]) / f"seed-{seed}" / "rollout-cache"
                     / row["arm"] / f"eval-{evaluation_seed}")
        try:
            with os.scandir(directory) as entries:
                count = sum(entry.name in expected and entry.is_file(follow_symlinks=False)
                            for entry in entries)
            # Evaluation removes this directory before writing its endpoint.
            # Absence during that transition means unknown, not zero completed.
            if directory.is_dir():
                progress["saved_prompts"] = count
        except FileNotFoundError:
            pass
        except OSError as error:
            progress["read_error"] = str(error)
        row["final_evaluation"] = progress


def load_reports(model, dataset, root, environment):
    """Read recorded identities and isolate input failures to the affected seed."""
    if model not in CONFIGS or dataset not in {"all", "math", "mbpp"}:
        raise ValueError("unknown model or dataset")
    scripts_path = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    adapter_name, storage_name, prefix = CONFIGS[model]
    adapter, storage = importlib.import_module(adapter_name), importlib.import_module(storage_name)
    group, _ = storage.group_work(environment)
    root = storage.inside(root, group)
    storage.validate_tree(root)
    # engine_digest remains the original engine even inside runtime_adapter().
    identity = hashlib.sha256((adapter.engine_digest() + adapter.adapter_digest()).encode()).hexdigest()
    snapshots = []
    for name in ("math", "mbpp") if dataset == "all" else (dataset,):
        path = root / "experiments" / f"{prefix}-{name}.json"
        try:
            plan = adapter.validate_extension(path, read_only=True)
            with patch.object(reports, "code_digest", return_value=identity):
                report = live.snapshot(reports.snapshot, path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report = {"dataset": name, "output_root": str(root / "runs" / name), "tasks": [],
                      "workers": [], "errors": [str(error)], "warnings": [], "complete": False,
                      "prepared": False, "stop_requested": False, "generated": time.time()}
        else:
            report["prepared"] = True
            interrupted_receipts(report)
            report["initial_candidate_accuracy"] = []
            for seed in plan["seeds"]:
                try:
                    data = live.read_object(input_path(path, plan, seed))
                    adapter.validate_bundle_model(data, plan, seed)
                    accuracy = initial_accuracy(data, plan["responses"])
                    if accuracy is not None:
                        report["initial_candidate_accuracy"].append({"seed": seed, **accuracy})
                    final_evaluation_progress(report, plan, seed, data)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    report["errors"].append(f"seed {seed} {LABELS[model]} inputs: {error}")
                    for row in report["tasks"]:
                        if row["seed"] == seed:
                            row.update(status="invalid", reward_percent=None, cost_complete=False)
                            for field in reports.COST_FIELDS:
                                row[field] = None
                            from scripts.srgc_time_status import invalid
                            invalid(row, "model-specific input validation failed")
            report["complete"] = report["complete"] and not report["errors"]
            report["counts"] = dict(Counter(row["status"] for row in report["tasks"]))
            for arm, statistics in report.get("arm_statistics", {}).items():
                statistics["completed_seeds"] = sum(row["status"] == "complete" for row in report["tasks"]
                                                    if row["arm"] == arm)
                if statistics["completed_seeds"] != statistics["planned_seeds"]:
                    statistics["mean_reward_percent"] = None
                    statistics["mean_selection_training_preparation_gpu_seconds"] = None
        if model in {"gemma4_text", "gemma4_it"}:
            # Initial preparation can be cancelled before a plan/queue exists.
            # Report that stop even when the plan is still missing or invalid.
            report["stop_requested"] |= any(path.is_file() for path in (
                root / f".startup-control/{name}/stop.json",
                root / f"runs/{name}/.queue/stop.json",
            ))
        report["label"] = "MATH" if name == "math" else "MBPP"
        if report["prepared"]:
            from .submission_time import attach as attach_submission_time

            attach_submission_time(report, path, plan)
        snapshots.append(report)
    from .worker_history import prune_workers

    prune_workers(snapshots, gpu_lock_roots=(Path(group) / ".srgc-gpu-node-locks", root / "gpu-node-locks"))
    return snapshots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(CONFIGS))
    parser.add_argument("dataset", choices=("math", "mbpp", "all"), nargs="?", default="all")
    parser.add_argument("action", choices=("status",), nargs="?", default="status")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        storage = importlib.import_module(CONFIGS[args.model][1])
        root = storage.default_root(os.environ) if args.root is None else args.root
        snapshots = load_reports(args.model, args.dataset, root, os.environ)
        if args.json:
            from .model_chain import snapshot as chain_snapshot

            adapter = importlib.import_module(CONFIGS[args.model][0])
            print(json.dumps({"model": adapter.MODEL, "root": str(root.resolve()), "datasets": snapshots, "model_chain": chain_snapshot()}, indent=2))
        else:
            from .qwen_status import render

            print(render(snapshots, root.resolve(), model=LABELS[args.model], include_initial=True))
        return int(any(report["errors"] for report in snapshots))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"STATUS ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
