#!/usr/bin/env python3
"""One phone-width screen: how every experiment is training right now.

Per prepared root (v1, difficulty, hard, quality, long, then MoPPS): the branch
counts, every running branch with its updates so far, elapsed time and node, and
every failed or stale branch with its reason. Read-only. Node names are the
identities the launchers recorded, so two containers with one hostname show
separately once they run the node-identity launcher.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import textwrap
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))
import selection_switch_status as switch_status  # noqa: E402
import mopps_comparison_status as mopps_status  # noqa: E402
from _status_summary import random_counts, random_text, suite_label
from _status_watch import StatusWatch

ORDER = ("selection-switch-v1", "difficulty", "hard", "quality", "long")


def rank(root):
    name = root.name
    for i, key in enumerate(ORDER):
        if name == key or (i and key in name):
            return i
    return len(ORDER)


def label(root, protocol=None):
    return suite_label(root, protocol)


def prepared_roots(work):
    runs = Path(work) / "runs"
    switch = sorted((p.parent for p in runs.glob("*/switch.json")), key=lambda r: (rank(r), r.name))
    mopps = sorted(p.parent for p in runs.glob("*/mopps.json"))
    return switch, mopps


def scoped_roots(roots):
    """Explicit switch roots in the order given, prepared or not.

    The MBPP launcher hands over its three suite roots; a sibling that is still
    waiting for the on-policy prefixes has no switch.json yet and must be shown
    as not prepared rather than silently dropped. MoPPS is never in scope here.
    """
    seen, ordered = set(), []
    for root in roots:
        root = Path(root).resolve()
        if root not in seen:
            seen.add(root)
            ordered.append(root)
    return ordered, []


def updates(task):
    step = task.get("training_step")
    return f"{int(step) - int(task['step'])}u" if isinstance(step, (int, float)) and task.get("step") is not None and step >= task["step"] else ""


def clip(value, width):
    value = str(value)
    return value if len(value) <= width else value[:width - 1] + "~"


def render_root(root, data, *, width, kind):
    name = label(root, data.get("protocol"))
    if not data.get("prepared", True) or "tasks" not in data:
        return [f"{name}: not prepared"]
    tasks = data["tasks"]
    branches = [t for t in tasks if t.get("kind", "branch") == "branch"]
    counts = Counter(t["status"] for t in branches)
    running = [t for t in tasks if t["status"] == "RUNNING"]
    parts = [f"DONE {counts.get('DONE', 0)}/{len(branches)}", f"RUN {len(running)}"]
    for key in ("EVAL", "RESUME", "SAVING", "REVIEW", "READY", "WAIT", "FAILED", "STALE", "INVALID", "BUDGET"):
        if counts.get(key):
            parts.append(f"{'FAIL' if key == 'FAILED' else key} {counts[key]}")
    if kind == "switch":
        published = data.get("training_published", 0)
        if published:
            parts.append(f"TRAINED {published}")
        gate = "READY" if data.get("gate_ready") else f"WAIT (dev {data.get('development_done', 0)}/18)"
        parts.append(f"gate {gate}")
    lines = textwrap.wrap(f"{name}: " + "  ".join(parts), width=width, subsequent_indent="  ")
    random = random_text(random_counts(branches))
    if random:
        lines += textwrap.wrap('  RANDOM ' + random, width=width, subsequent_indent='    ')
    histories = sum(bool(task.get('archived_work')) for task in branches)
    if histories:
        lines += textwrap.wrap(f'  HISTORY {histories}: archived attempts retained; current status shown separately.',
                               width=width, subsequent_indent='    ')
    host_width = max(12, width - 52)
    for t in sorted(running, key=lambda t: (t["seed"], t["step"], t["arm"])):
        lines.append(clip(f"  RUN  s{t['seed']}/t{t['step']} {t['arm']:<17} {t.get('phase') or '-':<9} {updates(t):>5} "
                          f"{switch_status.duration(t.get('seconds')):>6} {clip(t.get('host') or '?', host_width)}", width))
    for t in sorted((t for t in tasks if t["status"] in {"FAILED", "STALE", "INVALID", "BUDGET", "REVIEW"}), key=lambda t: (t["status"], t["seed"], t["step"])):
        tag = {"FAILED": "FAIL", "STALE": "STALE", "INVALID": "INVAL", "BUDGET": "BUDGET", "REVIEW": "REVIEW"}[t["status"]]
        prefix = (tag + " ").ljust(5)
        lines.append(clip(f"  {prefix}s{t['seed']}/t{t['step']} {t['arm']:<17} {t.get('reason') or ''}", width))
    return lines


def render(work, *, width=80, now=None, roots=None):
    now = time.time() if now is None else now
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    switch, mopps = scoped_roots(roots) if roots else prepared_roots(work)
    scope = "  (MBPP roots only)" if roots and all("mbpp" in Path(r).name for r in roots) else ("  (selected roots only)" if roots else "")
    lines = [f"PROGRESS  {stamp}{scope}", clip("RUN: updates (u), elapsed, node. BUDGET: allocation exhausted; needs review.", width)]
    lines.append(clip("EVAL: evaluation only. RESUME: checkpoint validation. REVIEW: saved work blocked.", width))
    hosts = set()
    for root in switch:
        if not (root / "switch.json").is_file():
            lines += ["", f"{label(root)}: not prepared"]
            continue
        try:
            data = switch_status.snapshot(root, now=now)
        except Exception as exc:  # noqa: BLE001 - one unreadable root must not hide the others
            lines += ["", f"{label(root)}: unreadable ({exc})"]
            continue
        hosts |= {t.get("host") for t in data.get("tasks", []) if t["status"] == "RUNNING" and t.get("host")}
        lines += ["", *render_root(root, data, width=width, kind="switch")]
    for root in mopps:
        try:
            data = mopps_status.snapshot(root, now=now)
        except Exception as exc:  # noqa: BLE001
            lines += ["", f"{label(root)}: unreadable ({exc})"]
            continue
        hosts |= {t.get("host") for t in data.get("tasks", []) if t.get("status") == "RUNNING" and t.get("host")}
        lines += ["", *render_root(root, data, width=width, kind="mopps")]
    if not switch and not mopps:
        lines += ["", "no prepared experiment root under " + str(Path(work) / "runs")]
    elif roots and not any((root / "switch.json").is_file() for root in switch):
        lines += ["", "none of the selected roots is prepared yet"]
    lines.insert(2, f"NODES TRAINING NOW  {len(hosts)}  (distinct node identities with a running task)")
    lines += ["", *render_nodes(switch, mopps, width=width, now=now)]
    return "\n".join(lines)


def render_nodes(switch, mopps, *, width, now):
    """Every node with launcher evidence or a running task: state, task, phase, silence."""
    tasks = []
    for root, module in [(r, switch_status) for r in switch] + [(r, mopps_status) for r in mopps]:
        try:
            data = module.snapshot(root, now=now)
        except Exception:  # noqa: BLE001
            continue
        for task in data.get("tasks", []):
            if task.get("status") in {"RUNNING", "STALE"} and task.get("host"):
                tasks.append({**task, "arm": f"{label(root)}: {task.get('arm', '')}"})
    anchor = switch[0] if switch else (mopps[0] if mopps else None)
    if anchor is None:
        return ["NODES  none"]
    view = switch_status.node_view
    nodes = view.listed(view.launcher_nodes(anchor, tasks, now=now))
    lines = [f"NODES  {view.render_summary(nodes)[7:]}"]
    for item in nodes:
        age = "" if item["last_age"] is None else f"{int(item['last_age'])//60}m"
        what = item["task"] or ("between passes" if item["state"] == "HOLD" else item["reason"] or "")
        lines.append(clip(f"  {item['state']:<6} {clip(item['host'], 24):<24} {clip(item['phase'] or '', 12):<12} {age:>4} {what}", width))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(os.environ.get("OM_WORK", "")))
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--watch", type=float, nargs="?", const=60.)
    parser.add_argument("--root", type=Path, action="append", default=None,
                        help="show only this switch root (repeatable); unprepared roots are listed as such")
    args = parser.parse_args()
    if not args.work or not str(args.work):
        parser.error("--work or OM_WORK is required")
    watcher = StatusWatch() if args.watch is not None else None
    try:
        while True:
            if watcher:
                watcher.refresh()
            text = render(args.work, width=args.width, roots=args.root)
            if args.watch:
                print("\033[2J\033[H", end="")
            print(text, flush=True)
            if not args.watch:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
