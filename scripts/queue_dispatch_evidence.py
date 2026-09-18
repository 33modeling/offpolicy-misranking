"""Bounded, CPU-only metadata evidence for one about-to-dispatch queue root."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from collections import Counter
from pathlib import Path

STATUSES = ("DONE", "EVAL", "RESUME", "REVIEW", "READY", "RUNNING", "FAILED", "STALE", "INVALID", "BUDGET", "WAIT", "SAVING")
PRIORITY = {name: i for i, name in enumerate(("REVIEW", "INVALID", "BUDGET", "FAILED", "STALE", "RESUME", "EVAL", "SAVING", "RUNNING", "WAIT"))}


def clean(value, limit=200):
    text = " ".join(str(value).split())
    text = re.sub(r"(?i)\b(token|secret|password|api[_-]?key|authorization)\s*[=:]\s*(?:Bearer\s+)?[^\s,;]+", r"\1=[redacted]", text)
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", text)
    return text[:limit]


def load_snapshot(kind, root):
    module = importlib.import_module("selection_switch_status" if kind == "switch" else "mopps_comparison_status")
    # Status also inventories every node and calls nvidia-smi for interactive
    # views. This helper is a separate short-lived process: use only this root's
    # saved task metadata, without global node scans or GPU subprocesses.
    old_nodes, old_gpus = module.node_view.launcher_nodes, module.node_view.local_gpus
    module.node_view.launcher_nodes = lambda *args, **kwargs: []
    module.node_view.local_gpus = lambda: {"available": False, "gpus": [], "processes": []}
    try:
        return module.snapshot(root)
    finally:
        module.node_view.launcher_nodes, module.node_view.local_gpus = old_nodes, old_gpus


def describe(root, revision, *, snapshot_loader=None):
    root = Path(root).resolve()
    yield f"[dispatch] root={clean(root, 600)} checkout={clean(revision, 64)}"
    manifest_path = next((root / name for name in ("switch.json", "mopps.json") if (root / name).is_file()), None)
    if manifest_path is None:
        yield "[dispatch] manifest=absent; preparation/prerequisites remain for the worker to check"
        return
    try:
        with manifest_path.open("rb") as handle:
            raw = handle.read(1048577)
        if len(raw) > 1048576:
            raise ValueError("manifest exceeds metadata read limit")
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise TypeError("manifest must be an object")
        kind = "switch" if manifest_path.name == "switch.json" else "mopps"
        fields = " ".join(f"{key}={clean(manifest.get(key, '?'), 80)}" for key in ("dataset", "selector", "accounting", "gate"))
        yield f"[dispatch] frozen {fields} (stored protocol values; no relabel/reset)"
        data = (snapshot_loader or load_snapshot)(kind, root)
        tasks = data.get("tasks", [])
        branches = [task for task in tasks if task.get("kind", "branch") == "branch"]
        counts = Counter(task.get("status", "UNKNOWN") for task in branches)
        labels = {"RUNNING": "RUN", "FAILED": "FAIL"}
        counts_text = " ".join(f"{labels.get(name, name)}={counts[name]}" for name in STATUSES)
        yield f"[dispatch] branches={len(branches)} {counts_text} trained={data.get('training_published', '?')} gate={'ready' if data.get('gate_ready') else 'waiting'}"
        notable = sorted((task for task in tasks if task.get("status") in PRIORITY),
                         key=lambda task: (PRIORITY[task["status"]], str(task.get("directory", ""))))
        for task in notable[:3]:
            identity = task.get("directory") or f"s{task.get('seed', '?')}/t{task.get('step', '?')}/{task.get('arm', '?')}"
            checkpoint = re.search(r"\bcheckpoint step (\d+)\b", str(task.get("reason", "")))
            saved_step = checkpoint.group(1) if checkpoint else "?"
            yield (f"[dispatch-task] {clean(identity, 180)} status={clean(task.get('status'))} "
                   f"phase={clean(task.get('phase') or '-', 60)} checkpoint_step={saved_step} "
                   f"logged_step={clean(task.get('training_step') if task.get('training_step') is not None else '?', 30)} "
                   f"host={clean(task.get('host') or '-', 100)} reason={clean(task.get('reason') or '-', 200)}")
        if data.get("notices"):
            notice = data["notices"][0]
            yield f"[dispatch-note] {clean(notice.get('path', '?'), 180)}: {clean(notice.get('error', '?'))}"
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
        yield f"[dispatch] metadata snapshot unavailable: {clean(exc)}; worker validation remains authoritative"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkout", default="unknown")
    args = parser.parse_args()
    for line in describe(args.root, args.checkout):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
