"""Read-only terminal dashboard for Qwen SRGC queues and their recorded progress."""

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts import srgc_live_status as live
from scripts.srgc_qwen35_storage import default_root, group_work, inside, validate_tree
from srgc_rebuttal import reports
from srgc_rebuttal.plan import input_path
from srgc_rebuttal.runtime import code_digest

REPO = Path(__file__).resolve().parents[2]
ARMS = ("cache", "prefix", "on_policy", "switch", "sr", "random")
HEADERS = ("Cache", "Prefix", "On-policy", "Switch", "SR", "Random")
LIVE_WORKERS = {"running", "preflight", "idle", "serving_other_dataset"}


def load_reports(dataset, root, environment):
    """Use recorded identities without importing model packages or binding queues."""
    group, _ = group_work(environment)
    root = inside(root, group)
    validate_tree(root)
    # Existing Qwen validation helpers use the scripts directory for imports.
    scripts_path = str(REPO / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from scripts import srgc_qwen35 as qwen

    expected_digest = hashlib.sha256((code_digest() + qwen.adapter_digest()).encode()).hexdigest()
    result = []
    for name in ("math", "mbpp") if dataset == "all" else (dataset,):
        plan_path = root / "experiments" / f"qwen35-9b-{name}.json"
        try:
            plan = qwen.validate_extension(plan_path, read_only=True)
            with patch.object(reports, "code_digest", return_value=expected_digest):
                report = live.snapshot(reports.snapshot, plan_path)
            for seed in plan["seeds"]:
                try:
                    qwen.validate_bundle_model(live.read_object(input_path(plan_path, plan, seed)), plan, seed)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    report["errors"].append(f"seed {seed} Qwen inputs: {exc}")
                    for task in report["tasks"]:
                        if task["seed"] == seed:
                            task["status"] = "invalid"
            report["complete"] = report["complete"] and not report["errors"]
            report["counts"] = dict(Counter(task["status"] for task in report["tasks"]))
            report["prepared"] = True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report = {"dataset": name, "output_root": str(root / "runs" / name), "tasks": [],
                      "workers": [], "errors": [str(exc)], "warnings": [], "complete": False,
                      "prepared": False, "stop_requested": False, "generated": time.time()}
        report["label"] = "MATH" if name == "math" else "MBPP"
        result.append(report)
    return result


def age(seconds):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return "-"
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def bar(done, total, width=16):
    filled = int(width * done / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def cell(task):
    if task is None:
        return "-"
    state = task["status"]
    if state == "complete":
        reward = task.get("reward_percent")
        return "DONE" if reward is None else f"DONE {reward:.1f}%"
    if state == "running":
        if task["arm"] == "cache" and "cache_saved_prompts" in task:
            count = max(task["cache_saved_prompts"], task["cache_exported_prompts"])
            return f"RUN {count}/{task['cache_total_prompts']}"
        done, total = task.get("completed_steps"), task.get("total_steps")
        return f"RUN {done}/{total}" if done is not None and total else "RUN"
    if state.startswith("waiting_for_"):
        return "WAIT " + state.removeprefix("waiting_for_")
    if state == "failed":
        return f"ERROR x{task.get('attempt', 0)}"
    return {"ready": "READY", "recoverable": "RESUME", "interrupted": "STOPPED",
            "invalid": "INVALID", "attempts_exhausted": "RETRY LIMIT"}.get(state, state.upper())


def table(headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
              for i, header in enumerate(headers)]
    line = lambda values: "  ".join(f"{value:<{width}}" for value, width in zip(values, widths)).rstrip()
    return [line(headers), line(["-" * width for width in widths]), *(line(row) for row in rows)]


def node_rows(snapshots):
    """Collapse the two dataset receipts and show the newest session per host."""
    sessions = {}
    for report in snapshots:
        for worker in report["workers"]:
            key = (worker.get("host", "-"), worker.get("worker_id", worker.get("pid")))
            row = {**worker, "label": report["label"], "task_row": next(
                (task for task in report["tasks"] if task["task"] == worker.get("task")), None)}
            # A shared worker writes an idle receipt to the inactive dataset.
            # In a dataset-only view this still describes a busy allocation.
            if row["status"] == "idle" and row.get("active_dataset"):
                row["status"] = "serving_other_dataset"
            previous = sessions.get(key)
            if previous is not None:
                current_age = row.get("heartbeat_age_seconds", float("inf"))
                previous_age = previous.get("heartbeat_age_seconds", float("inf"))
                if abs(current_age - previous_age) <= 5:
                    priority = {"running": 4, "serving_other_dataset": 3, "preflight": 2, "idle": 1}
                    if priority.get(previous["status"], 0) > priority.get(row["status"], 0):
                        continue
                elif current_age > previous_age:
                    continue
            sessions[key] = row
    nodes = {}

    def activity(row):
        # A newer idle launcher must not hide a different live session whose
        # task lease proves it is actually running on this host.
        if row["status"] == "running" and row["task_row"] and row["task_row"]["status"] == "running":
            return 3
        if row["status"] == "serving_other_dataset":
            return 2
        return int(row["status"] in LIVE_WORKERS)

    for row in sessions.values():
        host = row.get("host", "-")
        previous = nodes.get(host)
        if previous is None or (activity(row), -row.get("heartbeat_age_seconds", float("inf"))) > (
                activity(previous), -previous.get("heartbeat_age_seconds", float("inf"))):
            nodes[host] = row
    return sorted(nodes.values(), key=lambda row: (row["status"] not in LIVE_WORKERS, row.get("host", "-")))


def progress(task):
    parts = []
    if task["arm"] == "cache":
        if "cache_saved_prompts" in task:
            count = max(task["cache_saved_prompts"], task["cache_exported_prompts"])
            parts.append(f"{count}/{task['cache_total_prompts']} prompts")
            write_age = task.get("cache_last_write_age_seconds")
            if write_age is not None:
                parts.append(f"last write {age(write_age)} ago")
    else:
        if task.get("current_step") is not None:
            parts.append(f"update {task['current_step']}/{task['total_steps']} in progress")
        if task.get("completed_steps") is not None:
            parts.append(f"{task['completed_steps']}/{task['total_steps']} completed")
        if task.get("phase_stage"):
            parts.append(task["phase_stage"].replace("_", " "))
            parts.append(f"last progress {age(task.get('progress_age_seconds'))} ago")
    if not parts:
        parts.append("starting; progress not recorded yet")
    ranks = []
    for row in task.get("progress", []):
        updated = row.get("updated")
        if isinstance(updated, (int, float)) and math.isfinite(updated) and row.get("stage"):
            rank = f"r{row['rank']} " if type(row.get("rank")) is int else ""
            ranks.append(f"{rank}{row['stage']} ({age(time.time() - updated)} ago)")
    if ranks:
        parts.append("rank progress: " + "; ".join(ranks))
    return " | ".join(parts)


def resume_backlog(snapshots, root):
    """Inspect the dispatch marker without claiming jobs or changing receipts."""
    path = Path(root) / ".dispatch/resume-first.json"
    if not path.exists():
        return [], None
    try:
        saved = live.read_object(path)
        entries = saved.get("tasks")
        if saved.get("schema") != "qwen-resume-first-v1" or not isinstance(entries, list):
            raise ValueError("invalid resume-first marker")
        lookup = {(report["label"], task["task"]): task for report in snapshots for task in report["tasks"]}
        visible = {report["label"] for report in snapshots}
        pending = {}
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2 or any(not isinstance(value, str) for value in entry):
                raise ValueError("invalid resume-first task")
            plan, key = entry
            if Path(plan).parent.resolve() != (Path(root) / "experiments").resolve():
                raise ValueError("resume-first plan is outside this experiment")
            label = "MATH" if Path(plan).name.endswith("-math.json") else "MBPP" if Path(plan).name.endswith("-mbpp.json") else None
            if label is None:
                raise ValueError("unrecognized resume-first dataset")
            task = lookup.get((label, key))
            if task is None and label in visible:
                raise ValueError("resume-first task is absent from this dataset report")
            if task is None or task["status"] != "complete":
                pending[(label, key)] = task or {"status": "outside_view"}
        return list(pending.values()), None
    except (OSError, ValueError, TypeError) as error:
        return [], f"resume-first status unavailable: {error}"


def idle_reason(node, snapshots, backlog):
    if node.get("idle_reason"):
        return " ".join(str(node["idle_reason"]).split())
    if backlog:
        counts = Counter(row["status"] for row in backlog)
        return (f"resume-first: {len(backlog)} unfinished, {counts['running']} running; "
                "fresh tasks blocked")
    tasks = [row for report in snapshots for row in report["tasks"]]
    if any(report.get("stop_requested") for report in snapshots):
        return "stop requested"
    if tasks and all(row["status"] == "complete" for row in tasks):
        return "tasks complete; worker finishing"
    if any(row["status"] in {"ready", "recoverable", "interrupted"} for row in tasks):
        return "ready/resume work recorded; waiting for claim"
    if any(row["status"] in live.ATTENTION for row in tasks):
        return "failed/unavailable tasks; see ATTENTION"
    return "waiting for cache/prefix or another node"


def render(snapshots, root, *, host=None, now=None):
    host = socket.gethostname() if host is None else host
    now = time.time() if now is None else now
    tasks = [task for report in snapshots for task in report["tasks"]]
    counts = Counter(task["status"] for task in tasks)
    done, total = counts["complete"], len(tasks)
    waiting = sum(count for state, count in counts.items() if state.startswith("waiting_for_"))
    issues = sum(counts[state] for state in live.ATTENTION)
    lines = [f"Qwen3.5-9B | SRGC | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
        (f"Tasks {bar(done, total)} {done}/{total} completed"
         f" | RUN {counts['running']} | READY {counts['ready']} | WAIT {waiting} | ISSUE {issues}"),
        f"Local node: {host}"]
    error_count = sum(len(report["errors"]) for report in snapshots)
    if error_count:
        lines.insert(2, f"READ ERRORS: {error_count}; inspect ATTENTION below")
    if any(report.get("stop_requested") for report in snapshots):
        labels = ", ".join(report["label"] for report in snapshots if report.get("stop_requested"))
        lines.append(f"STOP REQUESTED: {labels}")
    nodes = node_rows(snapshots)
    backlog, backlog_error = resume_backlog(snapshots, root)
    if backlog_error:
        lines.append(backlog_error)
    if nodes:
        live_nodes = sum(row["status"] in LIVE_WORKERS for row in nodes)
        stale_nodes = sum(row["status"] == "heartbeat_stale" for row in nodes)
        node_counts = Counter(row["status"] for row in nodes)
        lines += ["", (f"NODES  {live_nodes} active | {stale_nodes} stale | {len(nodes) - live_nodes - stale_nodes} stopped"
                       f" | RUN {node_counts['running'] + node_counts['serving_other_dataset']} | IDLE {node_counts['idle']}"
                       f" | ADMISSION {node_counts['preflight']}")]
        rows = []
        for node in nodes:
            state = {"running": "RUN", "serving_other_dataset": "RUN OTHER", "preflight": "ADMISSION", "heartbeat_stale": "STALE"}.get(
                node["status"], node["status"].upper())
            current = f"{node['label']} {node['task']}" if node.get("task") else "-"
            reason = idle_reason(node, snapshots, backlog) if node["status"] == "idle" else "-"
            if node["status"] == "serving_other_dataset":
                current = str(node["active_dataset"])
                reason = "shared worker is running the other dataset"
            rows.append([("* " if node.get("host") == host else "") + node.get("host", "-"), state,
                         current, cell(node["task_row"]).removeprefix("RUN ") if node["task_row"] else "-",
                         age(node.get("heartbeat_age_seconds")), reason])
        lines += table(("Node", "State", "Task", "Progress", "Heartbeat", "Wait reason"), rows)
    else:
        lines += ["", "NODES  no worker records"]
    for report in snapshots:
        rows = report["tasks"]
        if not report.get("prepared", True):
            lines += ["", f"{report['label']}  NOT PREPARED / UNREADABLE"]
            continue
        finished = Counter(row["arm"] for row in rows if row["status"] == "complete")
        train_total = sum(row["arm"] not in {"cache", "prefix"} for row in rows)
        train_done = sum(finished[arm] for arm in ARMS[2:])
        seeds = sorted({row["seed"] for row in rows})
        lines += ["", (f"{report['label']}  {bar(train_done, train_total)} train {train_done}/{train_total}"
                       f" | cache {finished['cache']}/{len(seeds)} | prefix {finished['prefix']}/{len(seeds)}")]
        lookup = {(row["seed"], row["arm"]): row for row in rows}
        lines += table(("Seed", *HEADERS), [[seed, *(cell(lookup.get((seed, arm))) for arm in ARMS)]
                                            for seed in seeds])
    running = [(report["label"], row) for report in snapshots for row in report["tasks"]
               if row["status"] == "running"]
    if running:
        lines += ["", "RUNNING"]
        for label, row in running:
            lines.append(f"  {label} {row['task']} | node {row.get('host') or '-'} | {progress(row)}")
    attention = [(report["label"], row) for report in snapshots for row in report["tasks"]
                 if row["status"] in live.ATTENTION]
    errors = [(report["label"], error) for report in snapshots for error in dict.fromkeys(report["errors"])]
    worker_errors = [node for node in nodes if node.get("error")]
    if attention or errors or worker_errors:
        lines += ["", "ATTENTION"]
        for label, row in attention:
            reason = live.last_error_line(row.get("log", ""))
            lines.append(f"  {label} {row['task']} | {cell(row)}" + (f" | {reason}" if reason else ""))
            if row.get("log"):
                lines.append(f"    log: {row['log']}")
        for node in worker_errors:
            message = " ".join(str(node["error"]).split())
            lines.append(f"  node {node.get('host', '-')} | {message}")
        lines += [f"  {label} | {error}" for label, error in errors]
    warnings = [(report["label"], warning) for report in snapshots for warning in dict.fromkeys(report["warnings"])]
    if warnings:
        lines += ["", "WARNINGS", *(f"  {label} | {warning}" for label, warning in warnings)]
    lines += ["", "Seed table: RUN = active lease; STALE = node heartbeat older than 60s.",
              "Update counts show completed work; cache counts show saved prompts.",
              "Node IDLE = dispatcher waiting; rank progress and heartbeat are not GPU utilization.",
              f"Root: {root}"]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    parser.add_argument("action", choices=("status",))
    parser.add_argument("--root", type=Path, default=default_root(os.environ))
    parser.add_argument("--json", action="store_true", help="print the same recorded status as JSON")
    args = parser.parse_args(argv)
    try:
        snapshots = load_reports(args.dataset, args.root, os.environ)
        if args.json:
            print(json.dumps({"model": "Qwen/Qwen3.5-9B", "root": str(args.root.resolve()),
                              "datasets": snapshots}, indent=2))
        else:
            print(render(snapshots, args.root.resolve()))
        return int(any(report["errors"] for report in snapshots))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"STATUS ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
