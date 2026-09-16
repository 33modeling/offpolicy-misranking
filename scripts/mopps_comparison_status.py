#!/usr/bin/env python3
"""Read-only operational status for the MoPPS comparison, in the switch status layout."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import textwrap
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import selection_gate as core
import selection_switch as rule

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _node_view as node_view

ARM_LABELS = {"mopps": "MOPPS", "random_online": "RANDOM"}
CELLS = {"DONE": "DONE", "RUNNING": "RUN", "READY": "READY", "WAIT": "WAIT", "BLOCKED": "BLOCKED",
         "FAILED": "FAIL", "STALE": "STALE", "INVALID": "INVALID", "SAVING": "SAVING"}
PARENT_CELLS = {"DONE": "DONE", "RUNNING": "RUN", "FAILED": "FAIL", "STALE": "STALE", "QUEUED": "QUEUED"}


def number(value, default=0.):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else default


def short_error(value):
    import re
    lines = str(value).splitlines()
    causes = [line.strip() for line in lines if re.search(r"\b[\w.]+(?:Error|Exception):", line)
              and "worker failed" not in line and "ChildFailedError" not in line]
    return causes[0] if causes else next((line.strip() for line in lines if line.strip()), "unknown failure")


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
            notices.append({"path": str(path), "error": str(exc)})
            return {"_invalid": str(exc)}

    manifest = read(root / "mopps.json")
    if not manifest:
        return {"prepared": False, "root": str(root), "updated": now}
    seeds = [int(s) for s in manifest.get("seeds", rule.TEST_SEEDS)]
    steps = [int(t) for t in manifest.get("steps", rule.STEPS)]
    arms = list(manifest.get("arms", ARM_LABELS))
    parent = Path(str(manifest.get("parent", "")))
    tasks, cost_pending = [], []

    def heartbeat(directory):
        progress = read(directory / "progress.json")
        age = now-number(progress.get("updated"), -1e30)
        fresh = progress.get("state") == "running" and -5 <= age < 60
        return progress, age, fresh

    def costs(directory, task, scope):
        path = directory / "cost.jsonl"
        if not path.is_file():
            return
        try:
            events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            summary = core.cost_summary(events)
            for event_id in summary["incomplete_events"]:
                if task.get("status") == "RUNNING" and event_id == task.get("event_id"):
                    continue
                cost_pending.append({"directory": task["directory"], "event_id": event_id, "scope": scope})
            if summary["missing_starts"]:
                notices.append({"path": str(path), "error": "missing cost start records"})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            notices.append({"path": str(path), "error": f"cost snapshot unreadable: {exc}"})

    # Parent prefixes: the only prerequisite for a new branch.
    prefixes = {}
    for seed in seeds:
        directory = parent / "prefixes" / f"seed-{seed}"
        segments = []
        for step in rule.STEPS:
            segment = directory / f"segment-{step}"
            cert = directory / f"prefix-{step}.json"
            progress, age, fresh = heartbeat(segment)
            failure = read(segment / "failure.json")
            if cert.is_file() and "_invalid" not in read(cert):
                state, reason = "DONE", ""
            elif fresh:
                state, reason = "RUNNING", f"{progress.get('host', '?')} {progress.get('seconds', 0):.0f}s"
            elif progress.get("state") == "running":
                state, reason = "STALE", "heartbeat older than 60s"
            elif failure:
                state, reason = "FAILED", short_error(failure.get("error", "prefix failure"))
            else:
                state, reason = "QUEUED", ""
            segments.append({"step": step, "status": state, "reason": reason, "host": progress.get("host", ""),
                             "training_step": last_training_step(segment / "fresh_r/policy/grpo_stats.jsonl")})
        prefixes[seed] = segments

    def prerequisite(seed, step):
        """Earliest unmet parent prefix up to `step`: (blocking, reason) or None when satisfied."""
        for segment in prefixes[seed]:
            if segment["step"] > step:
                break
            if segment["status"] == "DONE":
                continue
            label = PARENT_CELLS[segment["status"]].lower()
            if segment["status"] == "FAILED":
                return True, f"prefix {segment['step']} failed: {segment['reason']}"
            return False, f"prefix {segment['step']} ({label})"
        return None

    def observe(directory, *, seed, step, kind, arm, done_path=None, dependency=None, blocking=False):
        progress, age, fresh = heartbeat(directory)
        failure = read(directory / "failure.json")
        task = {"seed": seed, "step": step, "kind": kind, "arm": arm, "directory": str(directory.relative_to(root)),
                "status": "BLOCKED" if blocking else "WAIT" if dependency else "READY", "reason": dependency or "",
                "host": progress.get("host", ""), "pid": progress.get("pid"), "phase": progress.get("phase", ""),
                "seconds": number(progress.get("seconds")), "timeout": number(progress.get("timeout")),
                "heartbeat_age": max(0., age) if progress else None, "event_id": progress.get("event_id")}
        if kind == "branch":
            task["training_step"] = last_training_step(directory / "policy/grpo_stats.jsonl")
            task["trained"] = (directory / "policy/budget_stop.json").is_file()
            task["evaluated_shards"] = sum((directory / "evaluation" / f"shard-{i}.done.json").is_file() for i in range(4))
        if done_path is not None and done_path.is_file():
            task.update(status="DONE", reason="published")
            if kind == "branch":
                receipt_path = directory / "result.sha256.json"
                receipt = read(receipt_path)
                result = read(done_path)
                if not receipt_path.exists():
                    task.update(status="SAVING", reason="result receipt pending")
                elif (result.get("complete") is not True
                      or receipt.get("sha256") != hashlib.sha256(done_path.read_bytes()).hexdigest()):
                    task.update(status="INVALID", reason="result receipt or completion flag invalid")
        elif fresh:
            task.update(status="RUNNING", reason="")
        elif failure or progress.get("state") == "failed":
            task.update(status="FAILED", reason=short_error(failure.get("error", "worker failed; inspect errors")))
        elif progress.get("state") == "running":
            task.update(status="STALE", reason="heartbeat older than 60s; owner not confirmed alive")
        elif "_invalid" in progress:
            task.update(status="INVALID", reason="unreadable progress record")
        costs(directory, task, "shared import" if kind == "import" else "branch budget")
        tasks.append(task)
        return task

    gate_results = {}
    for seed in seeds:
        for step in steps:
            state_dir = root / "states" / f"s{seed}-t{step}"
            origin = parent / "states" / f"s{seed}-t{step}" / "points" / f"view-{step}"
            gate_results[(seed, step)] = (origin / "gated" / "result.json").is_file()
            unmet = prerequisite(seed, step)
            blocking = bool(unmet and unmet[0])
            dependency = unmet[1] if unmet else None
            observe(state_dir / "import-cost", seed=seed, step=step, kind="import", arm="import",
                    done_path=state_dir / "import.done.json", dependency=dependency, blocking=blocking)
            for arm in arms:
                observe(state_dir / arm, seed=seed, step=step, kind="branch", arm=arm,
                        done_path=state_dir / arm / "result.json", dependency=dependency, blocking=blocking)

    admissions = {}
    for path in sorted((root / "node-preflight").glob("*/admission.json")):
        value = read(path)
        host = str(value.get("host", path.parent.name))
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if host not in admissions or age < admissions[host]["age"]:
            attempts = value.get("attempts", []) if isinstance(value.get("attempts"), list) else []
            admissions[host] = {"host": host, "state": str(value.get("state", "?")), "age": age,
                                "overrides": value.get("overrides", {}), "attempts": len(attempts),
                                "failure_kind": value.get("failure_kind", ""), "directory": str(path.parent)}

    active = [task for task in tasks if task["status"] == "RUNNING"]
    active_hosts = {task["host"] for task in active if task["host"]}
    stale_hosts = {task["host"] for task in tasks if task["status"] == "STALE" and task["host"]} - active_hosts
    waiting = []
    for path in sorted((root / "logs").glob("launcher.*.log")):
        try:
            if not -5 <= now-path.stat().st_mtime < 60:
                continue
            with path.open("rb") as handle:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell()-8192))
                lines = handle.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            continue
        last = next((line for line in reversed(lines) if line.strip()), "")
        host = path.name[len("launcher."):-len(".log")].rstrip("_")
        if host in active_hosts:
            continue
        if last.startswith("[waiting]"):
            waiting.append({"host": host, "state": "WAIT", "reason": "no claimable task (fresh launcher log)"})
        elif last.startswith("[holding]") or last.startswith("[hold]"):
            waiting.append({"host": host, "state": "HOLD", "reason": "node retained between queue passes"})
        elif last.startswith("[blocked]"):
            waiting.append({"host": host, "state": "BLOCKED", "reason": "node admission failed; launcher left"})
    branches = [task for task in tasks if task["kind"] == "branch"]
    nodes = node_view.launcher_nodes(root, tasks, now=now)
    return {"prepared": True, "root": str(root), "parent": str(parent), "updated": now,
            "nodes": nodes, "local_gpus": node_view.local_gpus(),
            "seeds": seeds, "steps": steps, "arms": arms,
            "active_nodes": len(active_hosts), "stale_nodes": len(stale_hosts), "waiting_nodes": waiting,
            "branch_counts": dict(Counter(task["status"] for task in branches)),
            "imports_done": sum(task["status"] == "DONE" for task in tasks if task["kind"] == "import"),
            "branches_done": sum(task["status"] == "DONE" for task in branches),
            "gate_results": {f"s{seed}-t{step}": value for (seed, step), value in gate_results.items()},
            "prefixes": {str(seed): segments for seed, segments in prefixes.items()},
            "admissions": sorted(admissions.values(), key=lambda item: item["host"]),
            "tasks": tasks, "cost_pending": cost_pending, "notices": notices,
            "verification": "published result receipts only; the Gate-versus-MoPPS comparison is computed by summarize"}


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


def render(data, *, all_tasks=False, width=120, local_gpus=True):
    if not data["prepared"]:
        return f"NOT PREPARED  {data['root']}"
    stamp = datetime.fromtimestamp(data["updated"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    seeds, steps, arms = data["seeds"], data["steps"], data["arms"]
    states = len(seeds)*len(steps)
    gate_available = sum(data["gate_results"].values())
    lines = [f"MOPPS COMPARISON  {stamp}",
             f"NODES  {data['active_nodes']} active  |  {len(data['waiting_nodes'])} waiting  |  {data['stale_nodes']} stale",
             f"PROGRESS  Imports {data['imports_done']}/{states} states  |  Branches {data['branches_done']}/{states*len(arms)}"
             f"  |  Parent gate results {gate_available}/{states}",
             "BRANCHES  " + "  ".join(f"{name} {data['branch_counts'].get(name, 0)}" for name in CELLS)]
    tasks = data["tasks"]
    alerts = Counter(task["status"] for task in tasks if task["status"] in {"FAILED", "BLOCKED", "STALE", "INVALID"})
    if alerts:
        lines.append("ALERTS  " + "  ".join(f"{key} {value}" for key, value in alerts.items()) + "  (all phases)")
    observed = [task for task in tasks if task["status"] in {"RUNNING", "STALE"}]
    lines += ["", "CURRENT WORK"]
    rows = [[task["host"] or "unknown", task["pid"] or "-", CELLS[task["status"]],
             f"s{task['seed']}/t{task['step']} {ARM_LABELS.get(task['arm'], task['arm'])}",
             task["phase"] or "-", duration(task["seconds"]), duration(task["timeout"]) if task["timeout"] else "-",
             duration(task["heartbeat_age"])] for task in sorted(observed, key=lambda item: (item["host"], item["seed"], item["step"]))]
    rows += [[item["host"], "-", item["state"], "-",
              "between passes" if item["state"] == "HOLD" else "admission failed" if item["state"] == "BLOCKED" else "no claimable task",
              "-", "-", "<60s"] for item in data["waiting_nodes"]]
    if rows:
        headers = ["NODE", "PID", "STATE", "TASK", "PHASE", "ELAPSED", "LIMIT", "BEAT"]
        if width < 100:
            lines += table(headers[:6] + headers[7:], [row[:6] + row[7:] for row in rows], [12, 6, 7, 16, 14, 6, 5])
        else:
            lines += table(headers, rows, [max(12, min(20, width-88)), 7, 7, 20, 18, 8, 8, 6])
    else:
        lines.append("No fresh worker heartbeat or waiting launcher observed.")
    lines += ["", "NODES (every host with launcher evidence; ALIVE is known only on that host)"]
    lines += node_view.render_nodes(data.get("nodes", []), table, width)
    if local_gpus:
        lines += ["", "THIS NODE GPUS"]
        lines += node_view.render_local_gpus(data.get("local_gpus", {"host": "?", "available": False, "gpus": [], "processes": []}), table, width)
    if data["admissions"]:
        lines += ["", "NODE ADMISSION (latest NCCL probe per host)"]
        rows = [[item["host"], item["state"], str(item["attempts"]),
                 ",".join(f"{k}={v}" for k, v in item["overrides"].items()) or "-",
                 item["failure_kind"] or "-", duration(item["age"]) + " ago"] for item in data["admissions"]]
        # 18+9+6+22+9 fixed columns plus five two-space separators leave width-74 for overrides.
        lines += table(["HOST", "STATE", "PROBES", "OVERRIDES", "FAILURE", "AGE"], rows,
                       [18, 9, 6, max(6, width-74), 22, 9])
    lines += ["", "PARENT PREFIXES (selection-switch)"]
    rows = []
    for seed in seeds:
        segments = data["prefixes"][str(seed)]
        reached = max((s["step"] for s in segments if s["status"] == "DONE"), default=0)
        logged = max((s["training_step"] for s in segments if s["training_step"] is not None), default=None)
        rows.append([f"s{seed}", f"{reached}/100", logged if logged is not None else "-",
                     *[PARENT_CELLS[s["status"]] for s in segments]])
    lines += table(["SEED", "PUBLISHED", "LAST STEP", "TO 25", "TO 50", "TO 100"], rows, [5, 10, 10, 8, 8, 8])
    lines += ["", "STATES"]
    rows = []
    for seed in seeds:
        for step in steps:
            items = {task["arm"]: task for task in tasks if task["seed"] == seed and task["step"] == step}
            reasons = sorted({task["reason"] for task in items.values() if task["status"] in {"WAIT", "BLOCKED"} and task["reason"]})
            gate = "DONE" if data["gate_results"].get(f"s{seed}-t{step}") else "WAIT"
            progress = []
            for arm in arms:
                task = items.get(arm)
                if task and task["status"] == "RUNNING":
                    progress.append(f"{ARM_LABELS.get(arm, arm)} {task['phase'] or '?'}")
            rows.append([f"s{seed}/t{step}", CELLS[items["import"]["status"]] if "import" in items else "-",
                         *[CELLS[items[arm]["status"]] if arm in items else "-" for arm in arms], gate,
                         "; ".join(reasons) if reasons else ", ".join(progress)])
    lines += table(["STATE", "IMPORT", *[ARM_LABELS.get(arm, arm) for arm in arms], "GATE", "WAIT FOR / PHASE"],
                   rows, [8, 7, 8, 8, 6, max(15, width-52)])
    attention = [task for task in tasks if task["status"] in {"FAILED", "BLOCKED", "STALE", "INVALID", "SAVING"}]
    if attention or data["cost_pending"] or data["notices"]:
        lines += ["", "ATTENTION"]
        shown = 0
        seen = set()
        for task in attention:
            key = (task["status"], task["reason"]) if task["status"] == "BLOCKED" else id(task)
            if key in seen:
                continue
            seen.add(key)
            lines.append(clip(f"{CELLS[task['status']]} s{task['seed']}/t{task['step']} {task['arm']}: {task['reason']}", width))
            shown += 1
            if shown == 8:
                break
        if len(attention) > shown:
            lines.append(f"... {len(attention)-shown} more; use --all or --json")
        if data["cost_pending"]:
            scopes = Counter(item["scope"] for item in data["cost_pending"])
            lines.append("COST REVIEW  " + ", ".join(f"{count} {scope} event(s)" for scope, count in scopes.items()))
        for notice in data["notices"][:3]:
            lines.append(clip(f"READ WARNING  {notice['path']}: {notice['error']}", width))
        lines.append("Details: errors | recover-cost | retry")
    if all_tasks:
        lines += ["", "TASK DETAILS"]
        for task in tasks:
            lines.append(f"{task['status']:8} {task['directory']}")
            if task["reason"]:
                lines.append("  " + task["reason"])
    lines += ["", "MOPPS/RANDOM: online arms of this comparison; GATE: the parent's executed gate result the primary contrast needs.",
              "DONE: published receipt checked. RUN: heartbeat <60s. STALE: not confirmed running. BLOCKED: parent prefix failed.",
              f"ROOT  {data['root']}", f"PARENT  {data['parent']}"]
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
