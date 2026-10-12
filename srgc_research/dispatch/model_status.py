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
    "qwen35": ("scripts.srgc_qwen35", "scripts.srgc_qwen35_storage", "qwen35-9b"),
    "gemma4": ("srgc_research.dispatch.gemma4.adapter", "srgc_research.dispatch.gemma4.storage", "gemma4-12b-pt"),
    "llama31": ("srgc_research.dispatch.llama31.adapter", "srgc_research.dispatch.llama31.storage", "llama31-8b"),
}
LABELS = {"qwen35": "Qwen3.5-9B", "llama31": "Llama-3.1-8B-Instruct", "gemma4": "Gemma-4-12B-PT"}


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
            report["complete"] = report["complete"] and not report["errors"]
            report["counts"] = dict(Counter(row["status"] for row in report["tasks"]))
            for arm, statistics in report.get("arm_statistics", {}).items():
                statistics["completed_seeds"] = sum(row["status"] == "complete" for row in report["tasks"]
                                                    if row["arm"] == arm)
                if statistics["completed_seeds"] != statistics["planned_seeds"]:
                    statistics["mean_reward_percent"] = None
                    statistics["mean_selection_training_preparation_gpu_seconds"] = None
        report["label"] = "MATH" if name == "math" else "MBPP"
        snapshots.append(report)
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
            adapter = importlib.import_module(CONFIGS[args.model][0])
            print(json.dumps({"model": adapter.MODEL, "root": str(root.resolve()), "datasets": snapshots}, indent=2))
        else:
            from .qwen_status import render

            print(render(snapshots, root.resolve(), model=LABELS[args.model], include_initial=True))
        return int(any(report["errors"] for report in snapshots))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"STATUS ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
