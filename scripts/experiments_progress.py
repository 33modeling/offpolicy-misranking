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
import shutil
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
from _status_execution import current_tasks, execution_tasks

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


def fit(text, width, indent="  "):
    """Wrap rather than clip, so a phone-width screen keeps every word."""
    return [text] if len(text) <= width else textwrap.wrap(text, width=width, subsequent_indent=indent,
                                                           break_on_hyphens=False)


def render_root(root, data, *, width, kind):
    name = label(root, data.get("protocol"))
    if not data.get("prepared", True) or "tasks" not in data:
        return [f"{name}: not prepared"]
    tasks = execution_tasks(data["tasks"])
    branches = [t for t in tasks if t.get("kind", "branch") == "branch"]
    counts = Counter(t["status"] for t in branches)
    running = current_tasks(tasks)
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
    random = random_text(random_counts(data["tasks"]))
    if random:
        lines += textwrap.wrap('  RANDOM (saved results) ' + random, width=width, subsequent_indent='    ')
    histories = sum(bool(task.get('archived_work')) for task in branches)
    if histories:
        lines += textwrap.wrap(f'  HISTORY {histories}: archived attempts retained; current status shown separately.',
                               width=width, subsequent_indent='    ')
    host_width = max(12, width - 52)
    for t in sorted(running, key=lambda t: (str(t["seed"]), str(t["step"]), t["arm"])):
        head = f"  RUN  s{t['seed']}/t{t['step']} {t['arm']:<17} {t.get('phase') or '-':<9}"
        tail = f"{updates(t):>5} {switch_status.duration(t.get('seconds')):>6}"
        row = f"{head} {tail} {clip(t.get('host') or '?', host_width)}"
        if width >= 100 or len(row) <= width:
            lines.append(clip(row, width))
        else:
            # Narrow terminal: updates, elapsed and node move to their own line instead of being clipped away.
            lines += [clip(head.rstrip(), width), clip(f"{'':7}{tail} {t.get('host') or '?'}", width)]
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
    lines = [f"PROGRESS  {stamp}{scope}", *fit("RUN: updates (u), elapsed, node. BUDGET: allocation exhausted; needs review.", width)]
    at = len(lines)
    lines += fit("EVAL: evaluation only. RESUME: checkpoint validation. REVIEW: saved work blocked.", width)
    hosts = set()
    snapshots = {}  # one snapshot per root, reused by render_nodes; this screen never shows GPUs
    for root in switch:
        if not (root / "switch.json").is_file():
            lines += ["", f"{label(root)}: not prepared"]
            continue
        try:
            data = snapshots[switch_status, root] = switch_status.snapshot(root, now=now, local_gpus=False)
        except Exception as exc:  # noqa: BLE001 - one unreadable root must not hide the others
            snapshots[switch_status, root] = None
            lines += ["", f"{label(root)}: unreadable ({exc})"]
            continue
        hosts |= {t.get("host") for t in execution_tasks(data.get("tasks", []))
                  if t.get("host") and t["status"] == "RUNNING"}
        lines += ["", *render_root(root, data, width=width, kind="switch")]
    for root in mopps:
        try:
            data = snapshots[mopps_status, root] = mopps_status.snapshot(root, now=now, local_gpus=False)
        except Exception as exc:  # noqa: BLE001
            snapshots[mopps_status, root] = None
            lines += ["", f"{label(root)}: unreadable ({exc})"]
            continue
        hosts |= {t.get("host") for t in execution_tasks(data.get("tasks", []))
                  if t.get("status") == "RUNNING" and t.get("host")}
        lines += ["", *render_root(root, data, width=width, kind="mopps")]
    if not switch and not mopps:
        lines += ["", "no prepared experiment root under " + str(Path(work) / "runs")]
    elif roots and not any((root / "switch.json").is_file() for root in switch):
        lines += ["", "none of the selected roots is prepared yet"]
    lines[at:at] = fit(f"NODES TRAINING NOW  {len(hosts)}  (recorded host labels; duplicate names may share a label)", width)
    lines += ["", *render_nodes(switch, mopps, width=width, now=now, snapshots=snapshots)]
    return "\n".join(lines)


def render_nodes(switch, mopps, *, width, now, snapshots=None):
    """Every node with launcher evidence or a running task: state, task, phase, silence.

    snapshots maps (module, root) to the snapshot render already took (None: unreadable),
    so each root is read once per frame."""
    tasks, snapshots = [], snapshots or {}
    for root, module in [(r, switch_status) for r in switch] + [(r, mopps_status) for r in mopps]:
        if (module, root) in snapshots:
            data = snapshots[module, root]
        else:
            try:
                data = module.snapshot(root, now=now, local_gpus=False)
            except Exception:  # noqa: BLE001
                data = None
        if data is None:
            continue
        for task in execution_tasks(data.get("tasks", [])):
            if task.get("status") in {"RUNNING", "STALE"} and task.get("host"):
                tasks.append({**task, "arm": f"{label(root)}: {task.get('arm', '')}"})
    anchor = switch[0] if switch else (mopps[0] if mopps else None)
    if anchor is None:
        return ["NODES  none"]
    view = switch_status.node_view
    nodes = view.listed(view.launcher_nodes(anchor, tasks, now=now))
    lines = [f"NODES  {view.render_summary(nodes)[7:]}"]
    rows = [(item["state"], clip(item["host"], 24), clip(item["phase"] or "", 12),
             "" if item["last_age"] is None else f"{int(item['last_age'])//60}m",
             item["task"] or ("between passes" if item["state"] == "HOLD" else item["reason"] or ""))
            for item in nodes]
    host_width, phase_width = 24, 12
    if width < 100 and any(len(f"  {s:<6} {h:<24} {p:<12} {a:>4} {w}") > width for s, h, p, a, w in rows):
        # Narrow terminal: size the node and phase columns to their longest entry first ...
        host_width, phase_width = max(len(r[1]) for r in rows), max(len(r[2]) for r in rows)
    for state, host, phase, age, what in rows:
        head = f"  {state:<6} {host:<{host_width}} {phase:<{phase_width}} {age:>4}"
        if width >= 100 or len(f"{head} {what}") <= width:
            lines.append(clip(f"{head} {what}", width))
        else:
            # ... then give a task that still does not fit its own line instead of clipping it.
            lines += [clip(head.rstrip(), width)] + ([clip(f"{'':9}{what}", width)] if what else [])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(os.environ.get("OM_WORK", "")))
    parser.add_argument("--width", type=int, default=None,
                        help="columns (default: COLUMNS or this terminal's width, else 80)")
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
            # Re-read every frame so a rotated phone or resized pane gets its own width.
            width = args.width if args.width is not None else shutil.get_terminal_size((80, 24)).columns
            text = render(args.work, width=width, roots=args.root)
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
