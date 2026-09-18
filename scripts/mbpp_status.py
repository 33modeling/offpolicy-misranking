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
            suites.append(switch_status.snapshot(Path(root), now=now, local_gpus=False))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            suites.append({"prepared": False, "root": str(root), "updated": now, "error": str(exc)})
    return {"updated": now, "suites": suites}


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
        active = [task for task in tasks if task.get("status") == "RUNNING"]
        running += [(name, task) for task in active]
        prefixes = [task for task in tasks if task.get("kind") == "prefix"]
        prefix_done = sum(task["status"] == "DONE" for task in prefixes)
        rows.append([name, f"{counts['DONE']}/{len(branches)}", len(branches) - counts['DONE'], len(active),
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
    for name, task in sorted(running, key=lambda pair: (
            pair[0], switch_status.node_view.host_sort_key(pair[1].get("host", "")),
            pair[1]["seed"], pair[1]["step"], pair[1]["arm"])):
        step = task.get("training_step")
        lines.append(f"{name} s{task['seed']}/t{task['step']} {switch_status.ARM_LABELS.get(task['arm'], task['arm'])} "
                     f"| {task.get('phase') or '?'} | step {step if step is not None else '?'} "
                     f"| {task.get('host') or '?'} | {switch_status.duration(task.get('seconds'))}")
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
