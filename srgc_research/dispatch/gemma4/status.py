"""Read the Gemma queues without GPU imports, downloads, or storage writes."""

import hashlib
import json
from collections import Counter
from unittest.mock import patch

from scripts import srgc_live_status as live
from srgc_rebuttal import reports
from srgc_rebuttal.plan import input_path
from srgc_rebuttal.runtime import code_digest
from srgc_research.dispatch.qwen_status import render

from . import adapter
from .storage import group_work, inside, validate_tree


def show(dataset, root, environment, *, as_json=False):
    group, _ = group_work(environment)
    root = inside(root, group)
    validate_tree(root)
    identity = hashlib.sha256(
        (code_digest() + adapter.adapter_digest()).encode()
    ).hexdigest()
    snapshots = []
    for name in ("math", "mbpp") if dataset == "all" else (dataset,):
        path = root / "experiments" / f"gemma4-12b-pt-{name}.json"
        try:
            plan = adapter.validate_extension(path, read_only=True)
            with patch.object(reports, "code_digest", return_value=identity):
                report = live.snapshot(reports.snapshot, path)
            for seed in plan["seeds"]:
                try:
                    adapter.validate_bundle_model(
                        live.read_object(input_path(path, plan, seed)), plan, seed
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report["errors"].append(f"seed {seed} Gemma inputs: {exc}")
                    for row in report["tasks"]:
                        if row["seed"] == seed:
                            row["status"] = "invalid"
            report["complete"] = report["complete"] and not report["errors"]
            report["counts"] = dict(Counter(row["status"] for row in report["tasks"]))
            report["prepared"] = True
            report["initial_candidate_accuracy"] = []
            for seed in plan["seeds"]:
                data = live.read_object(input_path(path, plan, seed))
                cache = data.get("cached_rewards", {})
                if set(cache) == set(data["candidate_ids"]):
                    adapter.validate_bundle_model(data, plan, seed)
                    rewards = [reward for values in cache.values() for reward in values]
                    if len(rewards) != 8 * len(data["candidate_ids"]) or any(reward not in (0, 1) for reward in rewards):
                        raise ValueError("invalid Gemma initial candidate rewards")
                    report["initial_candidate_accuracy"].append({"seed": seed, "responses": len(rewards), "success_percent": 100 * sum(rewards) / len(rewards)})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report = {
                "dataset": name,
                "tasks": [],
                "workers": [],
                "errors": [str(exc)],
                "warnings": [],
                "prepared": False,
                "complete": False,
                "stop_requested": False,
            }
        report["label"] = "MATH" if name == "math" else "MBPP"
        snapshots.append(report)
    if as_json:
        print(
            json.dumps(
                {"model": adapter.MODEL, "root": str(root), "datasets": snapshots},
                indent=2,
            )
        )
    else:
        print(
            render(snapshots, root).replace(
                "Qwen3.5-9B | SRGC", "Gemma-4-12B-PT | SRGC", 1
            )
        )
        for report in snapshots:
            values = report.get("initial_candidate_accuracy", [])
            if values:
                print(f"{report['label']} initial candidate success (8 responses/prompt): " + " | ".join(f"seed {row['seed']} {row['success_percent']:.1f}%" for row in values))
    return int(any(report["errors"] for report in snapshots))
