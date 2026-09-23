#!/usr/bin/env python3
"""One status screen for both experiments: selection switch, then MoPPS.

Read-only. Renders the switch view and the MoPPS view back to back and shows
this node's GPUs once, at the end. --json emits one object with both
snapshots. --watch [N] refreshes every N seconds (default 15).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import textwrap
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mopps_comparison_status as mopps_status
import selection_switch_status as switch_status
from _status_summary import random_counts, random_text, suite_label
from _status_watch import StatusWatch
from _status_execution import execution_tasks

SEPARATOR = "=" * 24


def planned_branches(data, kind):
    """Keep registered slots even when their files cannot be observed."""
    if "registered_tasks" in data:
        registered = {tuple(key) for key in data["registered_tasks"]}
    elif kind == "switch":
        rule = switch_status.rule
        registered = {(seed, step, arm) for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS)
                      for step in rule.STEPS
                      for arm in (rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS)}
    else:
        registered = {(seed, step, arm)
                      for seed in data.get("seeds", mopps_status.rule.TEST_SEEDS)
                      for step in data.get("steps", mopps_status.rule.STEPS)
                      for arm in data.get("arms", mopps_status.ARM_LABELS)}
    branches, conflicts = {}, set()
    for task in execution_tasks(data.get("tasks", [])):
        key = (task.get("seed"), task.get("step"), task.get("arm"))
        if task.get("kind", "branch") != "branch" or key not in registered or task.get("unverified"):
            continue
        if key in branches and branches[key] != task:
            conflicts.add(key)
        branches[key] = task
    for key in conflicts:
        branches.pop(key)
    return [branches.get(key, {"kind": "branch", "seed": key[0], "step": key[1], "arm": key[2],
                               "status": "WAIT", "unverified": True,
                               "reason": "registered branch record missing or conflicting"})
            for key in sorted(registered)]


def sibling_status(switch_root, mopps_root, *, now):
    """Show sibling results as well as workers; another suite is not a reset."""
    runs = Path(switch_root).resolve().parent
    known = {Path(switch_root).resolve(), Path(mopps_root).resolve()}
    tasks, summaries = [], []
    if not runs.is_dir():
        return tasks, summaries
    for marker, module in (("switch.json", switch_status), ("mopps.json", mopps_status)):
        for path in sorted(runs.glob(f"*/{marker}")):
            root = path.parent.resolve()
            if root in known:
                continue
            try:
                data = module.snapshot(root, now=now, local_gpus=False)  # GPUs are shown once, not per root
            except Exception as exc:
                summaries.append({"root": str(root), "error": str(exc)})
                continue
            observed = execution_tasks(data.get("tasks", []))
            kind = "switch" if marker == "switch.json" else "mopps"
            branches = planned_branches(data, kind)
            summaries.append({"root": str(root), "kind": kind,
                              "prepared": data.get("prepared", False), "branches": len(branches),
                              "branch_counts": dict(Counter(task["status"] for task in branches)),
                              "random_counts": random_counts(branches),
                              "unverified_branches": sum(bool(task.get("unverified")) for task in branches),
                              "archived_tasks": sum(bool(task.get("archived_work")) for task in branches),
                              "training_published": data.get("training_published", 0)})
            for task in observed:
                if task.get("status") in {"RUNNING", "STALE"} and task.get("host"):
                    label = root.name.replace("selection-switch-", "").replace("mopps-comparison", "mopps")
                    label = label[:-3] if label.endswith("-v1") else label
                    tasks.append({**task, "arm": f"{label}: {task.get('arm', '')}"})
    return tasks, summaries


def snapshot(switch_root, mopps_root, *, now=None):
    now = time.time() if now is None else now
    # nvidia-smi waits out a 20 s timeout per call on a GPU-faulted node, exactly when status
    # is needed: query this node's GPUs once per frame and give both views the same result.
    switch = switch_status.snapshot(switch_root, now=now, local_gpus=False)
    mopps = mopps_status.snapshot(mopps_root, now=now, local_gpus=False)
    if switch.get("prepared") or mopps.get("prepared"):
        gpus = switch_status.node_view.local_gpus()
        for item in (switch, mopps):
            if item.get("prepared"):
                item["local_gpus"] = gpus
    # Every host once: the node launcher's logs plus every experiment's launcher logs and tasks.
    sibling_tasks, summaries = sibling_status(switch_root, mopps_root, now=now)
    tasks = switch.get("tasks", []) + mopps.get("tasks", []) + sibling_tasks
    nodes = switch_status.node_view.launcher_nodes(switch_root, tasks, now=now)
    return {"updated": now, "nodes": nodes, "node_summary": switch_status.node_view.summarize(nodes),
            "selection_switch": switch, "mopps_comparison": mopps, "other_experiments": summaries}


def render(data, *, all_tasks=False, width=120):
    stamp = datetime.fromtimestamp(data["updated"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"EXPERIMENTS  {stamp}",
             switch_status.node_view.render_summary(data["nodes"], all_nodes=all_tasks),
             "RUN training/claiming  ADMIT NCCL probe  WAIT no claimable task  HOLD between passes  COOL GPU-fault cooldown",
             switch_status.node_view.render_idle(data["nodes"]),
             "", "NODES (state then node number; inactive history: --all)"]
    lines += switch_status.node_view.render_nodes(data["nodes"], switch_status.table, width, all_nodes=all_tasks)
    lines += ['', 'RANDOM CONTROLS (RF=full, RR=reduced, RO=online; current saved state)']
    random_roots = [
        {'root': item.get('root', name), 'random_counts': random_counts(planned_branches(item, kind)),
         'archived_tasks': sum(bool(task.get('archived_work')) for task in item.get('tasks', [])
                               if task.get('kind', 'branch') == 'branch')}
        for name, kind, item in (('on-policy', 'switch', data['selection_switch']),
                                 ('MoPPS', 'mopps', data['mopps_comparison']))
        if item.get('prepared')
    ] + data.get('other_experiments', [])
    for item in random_roots:
        counts = item.get('random_counts', {})
        if not counts:
            continue
        history = f" | HISTORY {item['archived_tasks']}" if item.get('archived_tasks') else ''
        lines += textwrap.wrap(f"{suite_label(item['root'], item.get('protocol'))}: {random_text(counts)}{history}",
                               width=width, subsequent_indent='  ')
    if data.get("other_experiments"):
        lines += ["", "OTHER EXPERIMENT RESULTS (separate roots; not a restart of the primary suite)"]
        for item in data["other_experiments"]:
            counts = item.get("branch_counts", {})
            if item.get("error"):
                summary = f"{Path(item['root']).name}: UNREADABLE: {item['error']}"
            else:
                parts = [f"DONE {counts.get('DONE', 0)}/{item['branches']}"]
                if item["kind"] == "switch":
                    parts.append(f"TRAINED {item['training_published']}")
                parts += [f"{name} {counts[name]}" for name in switch_status.CELLS if name != "DONE" and counts.get(name)]
                if item.get("unverified_branches"):
                    parts.append(f"UNVERIFIED {item['unverified_branches']} (included in WAIT)")
                summary = f"{Path(item['root']).name}: " + "  ".join(parts)
            lines += textwrap.wrap(summary, width=width, subsequent_indent="  ")
            lines += textwrap.wrap(f"  ROOT {item['root']}", width=width, subsequent_indent="    ")
    lines += ["", f"{SEPARATOR} SELECTION SWITCH {SEPARATOR}",
              switch_status.render(data["selection_switch"], all_tasks=all_tasks, width=width, local_gpus=False, nodes=False),
              "", f"{SEPARATOR} MOPPS COMPARISON {SEPARATOR}",
              mopps_status.render(data["mopps_comparison"], all_tasks=all_tasks, width=width, local_gpus=False, nodes=False)]
    view = (data["selection_switch"] if data["selection_switch"].get("prepared") else data["mopps_comparison"]).get("local_gpus")
    if not view:  # neither root prepared (or a snapshot taken with local_gpus=False): query here, once
        view = switch_status.node_view.local_gpus()
    lines += ["", "THIS NODE GPUS"]
    lines += switch_status.node_view.render_local_gpus(view, switch_status.table, width)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switch-root", type=Path, required=True)
    parser.add_argument("--mopps-root", type=Path, required=True)
    parser.add_argument("--all", action="store_true", dest="all_tasks", help="include inactive node history and task details")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--watch", nargs="?", const=15., type=float)
    args = parser.parse_args()
    if args.watch is not None and (not math.isfinite(args.watch) or args.watch < 1):
        parser.error("watch interval must be at least one second")
    watcher = StatusWatch() if args.watch is not None else None
    try:
        while True:
            if watcher:
                watcher.refresh()
            data = snapshot(args.switch_root, args.mopps_root)
            if args.watch is not None and sys.stdout.isatty() and not args.as_json:
                print("\033[2J\033[H", end="")
            print(json.dumps(data, indent=2) if args.as_json else render(data, all_tasks=args.all_tasks,
                  width=max(80, shutil.get_terminal_size((120, 40)).columns)), flush=True)
            if args.watch is None:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"[status unavailable] {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
