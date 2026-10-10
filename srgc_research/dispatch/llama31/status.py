"""Read the Llama queues without GPU imports, downloads, or storage writes."""

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
        path = root / "experiments" / f"llama31-8b-{name}.json"
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
                    report["errors"].append(f"seed {seed} Llama inputs: {exc}")
                    for row in report["tasks"]:
                        if row["seed"] == seed:
                            row["status"] = "invalid"
            report["complete"] = report["complete"] and not report["errors"]
            report["counts"] = dict(Counter(row["status"] for row in report["tasks"]))
            report["prepared"] = True
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
                "Qwen3.5-9B | SRGC", "Llama-3.1-8B-Instruct | SRGC", 1
            )
        )
    return int(any(report["errors"] for report in snapshots))
