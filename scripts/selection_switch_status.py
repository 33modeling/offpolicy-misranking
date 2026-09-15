#!/usr/bin/env python3
"""Read-only operational status for running selected-prefix experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import textwrap
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import selection_gate as core
import selection_switch as rule

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _node_view as node_view

ARM_LABELS = {"selection_reduced": "SEL", "random_reduced": "RND",
              "selection_full": "FULL-S", "random_full": "FULL-R", "gated": "GATE"}
CELLS = {"DONE": "DONE", "RUNNING": "RUN", "READY": "READY", "WAIT": "WAIT",
         "FAILED": "FAIL", "STALE": "STALE", "INVALID": "INVALID", "SAVING": "SAVING"}


def number(value, default=0.):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else default


def short_error(value):
    lines = str(value).splitlines()
    causes = [line.strip() for line in lines if re.search(r"\b[\w.]+(?:Error|Exception):", line)
              and "worker failed" not in line and "ChildFailedError" not in line]
    return (causes[0] if causes else next((line.strip() for line in lines if line.strip()), "unknown failure"))


def last_training_step(path):
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell()-65536))
            rows = handle.read().splitlines()
        for line in reversed(rows):
            try:
                value = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            step = value.get("step") if isinstance(value, dict) else None
            if type(step) is int and step >= 0:
                return step
    except OSError:
        pass
    return None


def snapshot(root, *, now=None):
    root = root.resolve()
    now = time.time() if now is None else now
    notices = []

    def read(path):
        if not path.is_file():
            return {}
        try:
            value = core.read(path)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value
        except (OSError, ValueError) as exc:
            notices.append({"path": str(path.relative_to(root)), "error": str(exc)})
            return {"_invalid": str(exc)}

    manifest = read(root / "switch.json")
    if not manifest:
        return {"prepared": False, "root": str(root), "updated": now}
    if manifest.get("schema") != rule.SCHEMA:
        raise ValueError("switch.json has an invalid or unsupported schema")
    model = read(root / "model.json")
    gate_ready = bool(model) and "_invalid" not in model
    tasks, cost_pending = [], []

    def observe(directory, *, seed, step, kind, arm, done_path=None, dependency=None):
        progress = read(directory / "progress.json")
        failure = read(directory / "failure.json")
        age = now-number(progress.get("updated"), -1e30)
        fresh = progress.get("state") == "running" and -5 <= age < 60
        task = {"seed": seed, "step": step, "kind": kind, "arm": arm,
                "role": "DEV" if seed in rule.DEV_SEEDS else "TEST",
                "directory": str(directory.relative_to(root)), "status": "WAIT" if dependency else "READY",
                "reason": dependency or "", "host": progress.get("host", ""), "pid": progress.get("pid"),
                "phase": progress.get("phase", ""), "seconds": number(progress.get("seconds")),
                "timeout": number(progress.get("timeout")), "heartbeat_age": max(0., age) if progress else None}
        policy_dir = directory / ("fresh_r/policy" if kind == "prefix" else "policy")
        task["training_step"] = last_training_step(policy_dir / "grpo_stats.jsonl") if kind != "diagnostic" else None
        if done_path is not None and done_path.is_file():
            result = read(done_path)
            task.update(status="DONE", reason="published")
            if "_invalid" in result:
                task.update(status="INVALID", reason="unreadable completion record")
            elif kind == "branch":
                receipt_path = directory / "result.sha256.json"
                receipt = read(receipt_path)
                if not receipt_path.exists():
                    task.update(status="SAVING", reason="result receipt pending")
                elif (result.get("complete") is not True or result.get("schema") != rule.SCHEMA
                      or receipt.get("sha256") != hashlib.sha256(done_path.read_bytes()).hexdigest()):
                    task.update(status="INVALID", reason="result receipt or completion flag invalid")
            elif kind == "diagnostic" and result.get("status") != "complete":
                task.update(status="FAILED", reason="diagnostic failed; no retry")
        elif fresh:
            task.update(status="RUNNING", reason="")
        elif failure or progress.get("state") == "failed":
            task.update(status="FAILED", reason=short_error(failure.get("error", "worker failed; inspect errors")))
        elif progress.get("state") == "running":
            task.update(status="STALE", reason="heartbeat older than 60s; owner not confirmed alive")
        elif "_invalid" in progress:
            task.update(status="INVALID", reason="unreadable progress record")

        cost_path = directory / "cost.jsonl"
        if cost_path.is_file():
            try:
                events = [json.loads(line) for line in cost_path.read_text().splitlines() if line.strip()]
                summary = core.cost_summary(events)
                for event_id in summary["incomplete_events"]:
                    if fresh and event_id == progress.get("event_id"):
                        continue
                    cost_pending.append({"directory": task["directory"], "event_id": event_id,
                                         "scope": "prefix research" if kind == "prefix" else "branch budget"})
                if summary["missing_starts"]:
                    notices.append({"path": str(cost_path.relative_to(root)), "error": "missing cost start records"})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                notices.append({"path": str(cost_path.relative_to(root)), "error": f"cost snapshot unreadable: {exc}"})
        tasks.append(task)
        return task

    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        prefix = root / "prefixes" / f"seed-{seed}"
        reached = {step for step in rule.STEPS if (prefix / f"prefix-{step}.json").is_file()
                   and "_invalid" not in read(prefix / f"prefix-{step}.json")}
        previous = None
        for step in rule.STEPS:
            observe(prefix / f"segment-{step}", seed=seed, step=step, kind="prefix", arm="prefix",
                    done_path=prefix / f"prefix-{step}.json",
                    dependency=f"prefix {previous}" if previous is not None and previous not in reached else None)
            previous = step
            child = root / "states" / f"s{seed}-t{step}"
            points = sorted(path for path in (child / "points").glob("*") if path.is_dir())
            out = points[0] if len(points) == 1 else child / "points" / f"view-{step}"
            if len(points) > 1:
                notices.append({"path": str(child.relative_to(root)), "error": "multiple state points; cannot select an owner"})
            diagnostic_dir = out / ("measurement" if seed in rule.DEV_SEEDS else "gate_measurement")
            diagnostic = None
            if diagnostic_dir.is_dir():
                diagnostic = observe(diagnostic_dir, seed=seed, step=step, kind="diagnostic", arm="diagnostic",
                                     done_path=diagnostic_dir / "initial.json")
            unfinished = next((task for task in tasks if task["kind"] == "prefix" and task["seed"] == seed
                               and task["step"] <= step and task["status"] != "DONE"), None)
            dependency = f"prefix {unfinished['step']} ({CELLS[unfinished['status']].lower()})" if unfinished else None
            if dependency is None and seed in rule.TEST_SEEDS and not gate_ready:
                dependency = "development gate"
            if dependency is None and not (out / "decisions-frozen.json").exists() and diagnostic:
                if (diagnostic["status"] in {"RUNNING", "STALE", "INVALID"}
                        or diagnostic["status"] == "FAILED" and seed in rule.DEV_SEEDS):
                    dependency = f"diagnostic {CELLS[diagnostic['status']].lower()}"
            for arm in rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS:
                observe(out / arm, seed=seed, step=step, kind="branch", arm=arm,
                        done_path=out / arm / "result.json", dependency=dependency)

    active = [task for task in tasks if task["status"] == "RUNNING"]
    active_hosts = {task["host"] for task in active if task["host"]}
    stale_hosts = {task["host"] for task in tasks if task["status"] == "STALE" and task["host"]} - active_hosts
    waiting = []
    for path in sorted((root / "logs").glob("launcher.*.log")):
        if not -5 <= now-path.stat().st_mtime < 60:
            continue
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell()-8192))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        last = next((line for line in reversed(lines) if line.strip()), "")
        host = path.name[len("launcher."):-len(".log")].rstrip("_")
        if last.startswith("[waiting]") and host not in active_hosts:
            waiting.append({"host": host, "reason": "no claimable task (fresh launcher log)"})
        elif last.startswith("[holding]") and host not in active_hosts:
            waiting.append({"host": host, "state": "HOLD", "reason": "node retained between queue passes"})
    branches = [task for task in tasks if task["kind"] == "branch"]
    nodes = node_view.launcher_nodes(root, tasks, now=now)
    return {"prepared": True, "root": str(root), "updated": now, "gate_ready": gate_ready,
            "nodes": nodes, "local_gpus": node_view.local_gpus(),
            "active_nodes": len(active_hosts), "stale_nodes": len(stale_hosts), "waiting_nodes": waiting,
            "branch_counts": dict(Counter(task["status"] for task in branches)),
            "prefix_done": sum(task["status"] == "DONE" for task in tasks if task["kind"] == "prefix"),
            "development_done": sum(task["status"] == "DONE" and task["role"] == "DEV" for task in branches),
            "test_done": sum(task["status"] == "DONE" and task["role"] == "TEST" for task in branches),
            "tasks": tasks, "cost_pending": cost_pending, "notices": notices,
            "verification": "published result receipts only; full scientific validation is separate"}


def duration(seconds):
    seconds = max(0, int(number(seconds)))
    if seconds >= 3600:
        return f"{seconds//3600}h{seconds%3600//60:02}m"
    return f"{seconds//60}m{seconds%60:02}s" if seconds >= 60 else f"{seconds}s"


def clip(value, width):
    value = " ".join(str(value).split())
    return value if len(value) <= width else value[:width-3] + "..."


def table(headers, rows, widths):
    return ["  ".join(clip(cell, width).ljust(width) for cell, width in zip(row, widths)).rstrip()
            for row in [headers, *rows]]


def render(data, *, all_tasks=False, width=120):
    if not data["prepared"]:
        return f"NOT PREPARED  {data['root']}"
    stamp = datetime.fromtimestamp(data["updated"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"SELECTION SWITCH  {stamp}",
             f"NODES  {data['active_nodes']} active  |  {len(data['waiting_nodes'])} waiting  |  {data['stale_nodes']} stale",
             f"PROGRESS  Prefix {data['prefix_done']}/15 segments  |  Dev {data['development_done']}/18  |  Test {data['test_done']}/30",
             "BRANCHES  " + "  ".join(f"{name} {data['branch_counts'].get(name, 0)}" for name in CELLS)]
    lines.append("GATE  " + ("READY" if data["gate_ready"] else
                 f"WAIT: {18-data['development_done']} development branches unpublished" if data["development_done"] < 18
                 else "FIT PENDING: 18/18 development results published"))
    tasks = data["tasks"]
    alerts = Counter(task["status"] for task in tasks if task["status"] in {"FAILED", "STALE", "INVALID"})
    if alerts:
        lines.append("ALERTS  " + "  ".join(f"{key} {value}" for key, value in alerts.items()) + "  (all phases)")
    observed = [task for task in tasks if task["status"] in {"RUNNING", "STALE"}]
    lines += ["", "CURRENT WORK"]
    rows = [[task["host"] or "unknown", task["pid"] or "-", CELLS[task["status"]], f"s{task['seed']}/t{task['step']} {ARM_LABELS.get(task['arm'], task['arm'])}",
             task["phase"] or "-", duration(task["seconds"]), duration(task["timeout"]) if task["timeout"] else "-",
             duration(task["heartbeat_age"])] for task in sorted(observed, key=lambda item: (item["host"], item["seed"], item["step"]))]
    rows += [[item["host"], "-", item.get("state", "WAIT"), "-",
              "between passes" if item.get("state") == "HOLD" else "no claimable task", "-", "-", "<60s"]
             for item in data["waiting_nodes"]]
    if rows:
        headers = ["NODE", "PID", "STATE", "TASK", "PHASE", "ELAPSED", "LIMIT", "BEAT"]
        if width < 100:
            lines += table(headers[:6] + headers[7:], [row[:6] + row[7:] for row in rows], [12, 6, 6, 16, 14, 6, 5])
        else:
            lines += table(headers, rows, [max(12, min(20, width-87)), 7, 6, 20, 18, 8, 8, 6])
    else:
        lines.append("No fresh worker heartbeat or waiting launcher observed.")
    lines += ["", "NODES (every host with launcher evidence; ALIVE is known only on that host)"]
    lines += node_view.render_nodes(data.get("nodes", []), table, width)
    lines += ["", "THIS NODE GPUS"]
    lines += node_view.render_local_gpus(data.get("local_gpus", {"host": "?", "available": False, "gpus": [], "processes": []}), table, width)
    lines += ["", "PREFIXES"]
    rows = []
    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        items = [task for task in tasks if task["kind"] == "prefix" and task["seed"] == seed]
        reached = max((task["step"] for task in items if task["status"] == "DONE"), default=0)
        logged = max((task["training_step"] for task in items if task["training_step"] is not None), default=None)
        rows.append([f"s{seed}", items[0]["role"], f"{reached}/100", logged if logged is not None else "-",
                     *[CELLS[task["status"]] for task in items]])
    lines += table(["SEED", "ROLE", "PUBLISHED", "LAST STEP", "TO 25", "TO 50", "TO 100"], rows, [5, 5, 10, 10, 8, 8, 8])
    lines += ["", "CONTINUATIONS"]
    branches = [task for task in tasks if task["kind"] == "branch"]
    rows = []
    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        for step in rule.STEPS:
            items = {task["arm"]: task for task in branches if task["seed"] == seed and task["step"] == step}
            reasons = sorted({task["reason"] for task in items.values() if task["status"] == "WAIT"})
            rows.append([f"s{seed}/t{step}", "DEV" if seed in rule.DEV_SEEDS else "TEST",
                         *[CELLS[items[arm]["status"]] if arm in items else "-" for arm in ARM_LABELS], ", ".join(reasons)])
    lines += table(["STATE", "ROLE", *ARM_LABELS.values(), "WAIT FOR"], rows, [8, 5, 7, 7, 7, 7, 7, max(15, width-70)])
    attention = [task for task in tasks if task["status"] in {"FAILED", "STALE", "INVALID", "SAVING"}]
    if attention or data["cost_pending"] or data["notices"]:
        lines += ["", "ATTENTION"]
        for task in attention[:8]:
            lines.append(clip(f"{CELLS[task['status']]} s{task['seed']}/t{task['step']} {task['arm']}: {task['reason']}", width))
        if len(attention) > 8:
            lines.append(f"... {len(attention)-8} more; use --all or --json")
        if data["cost_pending"]:
            scopes = Counter(item["scope"] for item in data["cost_pending"])
            lines.append("COST REVIEW  " + ", ".join(f"{count} {scope} event(s)" for scope, count in scopes.items()))
        for notice in data["notices"][:3]:
            lines.append(clip(f"READ WARNING  {notice['path']}: {notice['error']}", width))
        lines.append("Details: errors | recover-cost")
    if all_tasks:
        lines += ["", "TASK DETAILS"]
        for task in tasks:
            lines.append(f"{task['status']:8} {task['directory']}")
            if task["reason"]:
                lines.append("  " + task["reason"])
    lines += ["", "SEL/RND: diagnostic-paid selection/random; FULL-S/FULL-R: full-budget controls.",
              "DONE: published receipt checked. RUN: heartbeat <60s. STALE: not confirmed running.",
              f"ROOT  {data['root']}"]
    return "\n".join(part for line in lines for part in
                     (textwrap.wrap(line, width=width, subsequent_indent="  ", break_long_words=True, break_on_hyphens=False) if len(line) > width else [line]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--all", action="store_true", dest="all_tasks")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--watch", nargs="?", const=15., type=float)
    args = parser.parse_args()
    if args.watch is not None and (not math.isfinite(args.watch) or args.watch < 1):
        parser.error("watch interval must be at least one second")
    try:
        while True:
            data = snapshot(args.root)
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
    raise SystemExit(main())
