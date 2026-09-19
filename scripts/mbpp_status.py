"""One read-only dashboard for the MBPP suites, without unrelated experiments."""
from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selection_switch_status as switch_status


def label(root):
    name = Path(root).name
    if name == "selection-switch-mbpp-v1":
        return "on-policy"
    return name.removeprefix("selection-switch-mbpp-").removesuffix("-v1")


def snapshot(roots, *, now=None):
    now = time.time() if now is None else now
    suites = []
    for root in roots:
        try:
            suites.append(switch_status.snapshot(Path(root), now=now, local_gpus=False,
                                                node_namespace="mbpp"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            suite = {"prepared": False, "root": str(root), "updated": now, "error": str(exc)}
            try:
                suite["nodes"] = switch_status.node_view.launcher_nodes(
                    Path(root), [], now=now, node_namespace="mbpp")
            except OSError:
                suite["nodes"] = []
            suites.append(suite)
    return {"updated": now, "suites": suites}


def active(task):
    # Publication status and process activity are independent: an EVAL branch
    # can still have a worker. Never downgrade its saved result to RUN/READY.
    return task.get("heartbeat_fresh", task.get("status") == "RUNNING")


def node_assignments(data):
    """Merge shared launcher evidence once, retaining *every* live assignment."""
    hosts = {}
    for suite in data["suites"]:
        for node in suite.get("nodes", []):
            host = str(node["host"]).rstrip("_")
            previous = hosts.get(host)
            log_age, pid_age = node.get("last_age"), node.get("pid_age")
            ages = [age for age in (log_age, pid_age) if age is not None]
            age = min(ages) if ages else None
            if (pid_age is not None and -5 <= pid_age < switch_status.node_view.HEARTBEAT_GRACE
                    and (log_age is None or log_age >= switch_status.node_view.HEARTBEAT_GRACE)):
                node = {**node, "state": "UNKNOWN", "source_root": None, "reason": "",
                        "detail": "Recent launcher PID record; waiting for current task or controller log."}
            prior_age = previous.get("evidence_age") if previous else None
            if previous is None or (age is not None and (prior_age is None or age < prior_age)):
                hosts[host] = {**node, "host": host, "evidence_age": age, "assignments": []}
    for suite in data["suites"]:
        for task in suite.get("tasks", []):
            if task.get("status") == "STALE" and task.get("host"):
                host = str(task["host"]).rstrip("_")
                node = hosts.setdefault(host, {"host": host, "state": "STALE", "evidence_age": None,
                                               "assignments": []})
                age = task.get("heartbeat_age")
                prior_age = node.get("evidence_age")
                if age is not None and (prior_age is None or age < prior_age):
                    node.update(evidence_age=age, state="STALE", source_root=str(Path(suite["root"]).resolve()),
                                detail=f"Last task s{task['seed']}/t{task['step']} {task['arm']}; heartbeat stale, ownership unconfirmed.")
            if not active(task):
                continue
            host = str(task.get("host") or "unknown-owner").rstrip("_")
            node = hosts.setdefault(host, {"host": host, "state": "RUN", "evidence_age": None,
                                           "assignments": []})
            node["state"] = "RUN"
            node["assignments"].append((suite["root"], task))
    for node in hosts.values():
        if node["assignments"]:
            node["state"] = "RUN"
        node["assignments"].sort(key=lambda pair: (pair[0], pair[1]["seed"], pair[1]["step"], pair[1]["arm"]))
        age = node.get("evidence_age")
        node["current"] = bool(node["assignments"] or node.get("launcher_alive") is True
                               or age is not None and -5 <= age < switch_status.node_view.HEARTBEAT_GRACE)
        if node["state"] == "-":
            node["state"] = "UNKNOWN"
    return sorted(hosts.values(), key=lambda node: (not node["current"], not bool(node["assignments"]),
                                                   switch_status.node_view.host_sort_key(node["host"])))


def wrapped_table(headers, rows, widths):
    """Wrap inside columns instead of truncating distinguishing node suffixes."""
    lines = []
    for row in [headers, *rows]:
        cells = [textwrap.wrap(str(value), width=width, break_on_hyphens=False) or [""]
                 for value, width in zip(row, widths)]
        for index in range(max(map(len, cells))):
            lines.append("  ".join((cell[index] if index < len(cell) else "").ljust(width)
                                   for cell, width in zip(cells, widths)).rstrip())
    return lines


def render_nodes(data, *, width, all_nodes=False):
    nodes = node_assignments(data)
    current = [node for node in nodes if node["current"]]
    running = sum(bool(node["assignments"]) for node in current)
    lines = ["NODE ASSIGNMENTS", f"NODES {len(current)} current | RUN {running} | OTHER {len(current) - running}",
             "RUN = fresh task heartbeat; other rows show launcher evidence, not confirmed GPU activity."]
    rows, details = [], []
    labels = {str(Path(suite["root"]).resolve()): label(suite["root"]) for suite in data["suites"]}
    for node in nodes:
        if not all_nodes and not node["current"]:
            continue
        if node["assignments"]:
            for root, task in node["assignments"]:
                arm = switch_status.ARM_LABELS.get(task["arm"], task["arm"])
                step = task.get("training_step")
                age = task.get("heartbeat_age")
                rows.append([node["host"], "RUN", label(root), f"s{task['seed']}/t{task['step']} {arm}",
                             task.get("phase") or "?", step if step is not None else "-",
                             task.get("pid") or "?", switch_status.duration(age) if age is not None else "?"])
            if len(node["assignments"]) > 1:
                details.append(f"{node['host']}: {len(node['assignments'])} fresh task records; all shown above.")
        else:
            age = node.get("evidence_age")
            suite = labels.get(node.get("source_root"), "MBPP queue")
            detail = node.get("reason") or node.get("detail") or "No fresh task heartbeat; assignment unconfirmed."
            if not all_nodes:
                detail = switch_status.clip(detail, 200)
            phase = {"ADMIT": "GPU admission", "HOLD": "between passes", "WAIT": "waiting",
                     "COOL": "GPU cooldown"}.get(node["state"], "unconfirmed")
            if node.get("detail", "").startswith("[recover-cost]"):
                phase = "cost recovery"
            rows.append([node["host"], node["state"], suite, "-", phase, "-",
                         node.get("launcher_pid") or "?", switch_status.duration(age) if age is not None else "?"])
            details.append(f"{node['host']}: {detail}")
    # Even at 80 columns all identity text is kept, with continuation lines.
    widths = [width - 81, 7, 12, 17, 14, 5, 7, 5] if width >= 100 else [15, 7, 10, 12, 9, 4, 5, 4]
    lines += wrapped_table(["NODE", "STATUS", "SUITE", "TASK", "PHASE", "STEP", "PID", "AGE"], rows, widths)
    if not rows:
        lines.append("No current MBPP node evidence; saved experiment results are retained.")
    lines += details
    hidden = len(nodes) - len(current)
    if hidden and not all_nodes:
        lines.append(f"{hidden} old node(s) grouped in history; status --all shows them. Nothing deleted.")
    lines.append("AGE: last task heartbeat (RUN) or launcher evidence. '-' TASK: no confirmed task assignment.")
    return lines


def render(data, *, width=120, all_tasks=False):
    width = max(80, width)
    stamp = datetime.fromtimestamp(data["updated"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"MBPP EXPERIMENTS  {stamp}", "DONE = completed experiment; LEFT = unfinished (including EVAL)."]
    rows, running, notices = [], [], []
    for suite in data["suites"]:
        name = label(suite["root"])
        if not suite.get("prepared"):
            rows.append([name, "ERROR" if suite.get("error") else "--", "--", "--", "--", "--", "--", "--", "--"])
            notices.append(f"{name}: ERROR {suite['error']}" if suite.get("error") else f"{name}: NOT PREPARED")
            continue
        tasks = suite.get("tasks", [])
        branches = [task for task in tasks if task.get("kind") == "branch"]
        counts = Counter(task["status"] for task in branches)
        block = sum(counts[state] for state in ("FAILED", "STALE", "INVALID", "REVIEW", "BUDGET", "BLOCKED"))
        active_tasks = [task for task in tasks if active(task)]
        running += [(name, task) for task in active_tasks]
        prefixes = [task for task in tasks if task.get("kind") == "prefix"]
        prefix_done = sum(task["status"] == "DONE" for task in prefixes)
        rows.append([name, f"{counts['DONE']}/{len(branches)}", len(branches) - counts['DONE'], len(active_tasks),
                     counts['EVAL'], counts['RESUME'], block, f"{counts['READY']}/{counts['WAIT']}",
                     f"{prefix_done}/{len(prefixes)}"])
        trained = suite.get("training_published", 0)
        if trained > counts['DONE']:
            notices.append(f"{name}: {trained} training results saved; {counts['EVAL']} task(s) await evaluation/publication.")
    lines += switch_status.table(["SUITE", "DONE/TOTAL", "LEFT", "RUN", "EVAL", "RESUME", "BLOCK", "READY/WAIT", "PREFIX"],
                                 rows, [12, 10, 5, 4, 4, 6, 5, 10, 7])
    for suite in data["suites"]:
        lines += ["", f"FULL STATUS — {label(suite['root'])}"]
        if not suite.get("prepared"):
            lines.append("ERROR: suite could not be read." if suite.get("error") else "NOT PREPARED (no saved suite manifest)")
            continue
        tasks = suite.get("tasks", [])
        prefixes = {(task["seed"], task["step"]): task for task in tasks if task.get("kind") == "prefix"}
        branches = {(task["seed"], task["step"], task["arm"]): task for task in tasks if task.get("kind") == "branch"}

        def cell(task):
            return switch_status.CELLS.get(task["status"], task["status"]) if task else "-"

        matrix = []
        for seed in (*switch_status.rule.DEV_SEEDS, *switch_status.rule.TEST_SEEDS):
            for step in switch_status.rule.STEPS:
                matrix.append([f"s{seed}/t{step}", "DEV" if seed in switch_status.rule.DEV_SEEDS else "TEST",
                               cell(prefixes.get((seed, step))),
                               *[cell(branches.get((seed, step, arm))) for arm in switch_status.ARM_LABELS]])
        lines += switch_status.table(["STATE", "ROLE", "PREFIX", *switch_status.ARM_LABELS.values()],
                                     matrix, [8, 5, 8, 7, 7, 7, 7, 7])
    lines += ["SEL/RND: measured selection/random. FULL-S/FULL-R: full budget. GATE: gated policy.",
              "DONE: receipt checked. RUN: fresh heartbeat. READY: no saved work at that task path. -: not scheduled.",
              "", f"CURRENT RUN {len(running)}"]
    if not running:
        lines.append("No fresh RUN heartbeat in these MBPP suites; saved completions above are retained.")
    lines += render_nodes(data, width=width, all_nodes=all_tasks)
    lines += ["", "EVAL: evaluate saved work. RESUME: validate checkpoint. BLOCK: failed/stale/review/budget."]
    lines += notices
    if all_tasks:
        for suite in data["suites"]:
            lines += ["", f"ROOT {suite['root']}"]
            for task in suite.get("tasks", []):
                lines.append(f"{task['status']} {task['directory']}" + (f" — {task['reason']}" if task.get('reason') else ""))
    return "\n".join(part for line in lines for part in
                     (textwrap.wrap(line, width=width, subsequent_indent="  ", break_long_words=True,
                                    break_on_hyphens=False) if len(line) > width else [line]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--all", action="store_true", dest="all_tasks")
    args = parser.parse_args()
    data = snapshot(args.root)
    print(render(data, width=max(80, shutil.get_terminal_size((120, 40)).columns),
                 all_tasks=args.all_tasks))
    return int(any(suite.get("error") for suite in data["suites"]))


if __name__ == "__main__":
    raise SystemExit(main())
