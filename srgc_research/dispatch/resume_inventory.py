"""Read small queue receipts to prioritize abandoned work across model runtimes."""

import importlib
import json
import re
from pathlib import Path

from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.reports import locked
from srgc_rebuttal.runtime import run_root


def signature(model, dataset, environment, *, root=None):
    return _signature(model, dataset, environment, root=root, failed_only=False)


def failed_signature(model, dataset, environment, *, root=None):
    return _signature(model, dataset, environment, root=root, failed_only=True)


def _signature(model, dataset, environment, *, root, failed_only):
    # No CUDA/model imports, checkpoint deserialization or queue mutation.
    from .model_status import CONFIGS

    _, storage_name, prefix = CONFIGS[model]
    storage = importlib.import_module(storage_name)
    root = Path(root) if root is not None else storage.default_root(environment)
    path = root / "experiments" / f"{prefix}-{dataset}.json"
    try:
        plan = load_plan(path)
    except (OSError, ValueError, KeyError, TypeError):
        return None  # The normal launcher prepares/validates absent or invalid plans.
    output = run_root(path, plan)
    found = {}

    def owned(seed, arm):
        task = f"seed-{seed}.{arm}"
        execution = (input_path(path, plan, seed).with_suffix(".cache") / "execution.lock"
                     if arm == "cache" else output / f"seed-{seed}" / f".{arm}.execution.lock")
        return locked(output / ".queue" / "leases" / f"{task}.lock") or locked(execution)

    def completed(seed, arm):
        folder = output / f"seed-{seed}"
        if arm == "cache":
            from srgc_rebuttal.cluster_queue import input_info

            bundle = input_path(path, plan, seed)
            try:
                info, _ = input_info(bundle)
                if info["pending_cache"]:
                    return False
                protocol = output / ".queue" / "protocol.json"
                if protocol.is_file() and json.loads(protocol.read_text())["inputs"][str(seed)]["pending_cache"]:
                    cost = bundle.with_suffix(".cache") / "cost-summary.json"
                    receipt = json.loads(cost.read_text())
                    return isinstance(receipt, dict) and receipt.get("bundle_sha256") == info["input_sha256"]
                return True
            except (OSError, ValueError, KeyError, TypeError):
                return False  # Dispatch validates incomplete or malformed evidence.
        if arm == "prefix":
            return (folder / "prefix-ready.json").is_file() and (folder / "prefix.pt").is_file()
        return arm != "cache" and (folder / f"{arm}-endpoint.json").is_file()

    def remember(item):
        try:
            info = item.stat()
            found[str(item)] = (info.st_mtime_ns, info.st_size)
        except FileNotFoundError:
            pass

    for item in (output / ".queue" / "tasks").glob("*.json"):
        match = re.fullmatch(r"seed-(\d+)\.(cache|prefix|random|sr|on_policy|switch)", item.stem)
        if not match:
            continue
        seed, arm = int(match[1]), match[2]
        if seed not in plan["seeds"] or owned(seed, arm) or completed(seed, arm):
            continue
        try:
            row = json.loads(item.read_text())
        except (OSError, ValueError):
            remember(item)  # Actual dispatch will report invalid evidence.
            continue
        statuses = {"failed"} if failed_only else {"running", "failed", "interrupted"}
        if not isinstance(row, dict) or row.get("status") in statuses:
            remember(item)
    if failed_only:
        return tuple(sorted(found.items())) or None
    for item in output.glob("seed-*/*-latest.pt"):
        match = re.fullmatch(r"seed-(\d+)", item.parent.name)
        arm = item.stem.removesuffix("-latest")
        if (not match or arm not in {"prefix", "random", "sr", "on_policy", "switch"}
                or int(match[1]) not in plan["seeds"]):
            continue
        seed = int(match[1])
        if not owned(seed, arm) and not completed(seed, arm):
            remember(item)
    # Cache generation commits individual prompt receipts before the queue may
    # have written its task receipt. Those prompts must also resume first.
    for seed in plan["seeds"]:
        items = list(input_path(path, plan, seed).with_suffix(".cache").glob("*.json"))
        if items and not owned(seed, "cache") and not completed(seed, "cache"):
            for item in items:
                remember(item)
    return tuple(sorted(found.items())) or None
