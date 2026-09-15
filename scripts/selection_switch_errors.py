#!/usr/bin/env python3
"""Print existing switch worker failures without starting or modifying a run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def log_tail(path, lines):
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        return "\n".join(handle.read().decode("utf-8", errors="replace").splitlines()[-lines:])


def utc_time(value):
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        return "unknown"
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OverflowError, OSError):
        return "unknown"


def show_admissions(root, limit):
    seen = set()
    paths = sorted(root.glob("node-preflight/*/admission.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"admission record is outside the switch root: {path}")
        try:
            report = read_json(path)
            if not isinstance(report, dict) or not isinstance(report.get("host"), str):
                raise ValueError("admission record has no host")
        except (OSError, ValueError) as exc:
            print(f"[unreadable node check] {path.relative_to(root)}: {exc}", flush=True)
            continue
        host = report["host"]
        if host in seen:
            continue
        seen.add(host)
        print(f"\n[last node check] host={host} state={report.get('state', 'unknown')} "
              f"runtime={report.get('runtime_commit', 'unknown')} saved_utc={utc_time(path.stat().st_mtime)}", flush=True)
        print(f"[evidence] {path.relative_to(root)}; probe outcome only, not current training status", flush=True)
        if report.get("diagnosis"):
            print(str(report["diagnosis"])[:2000], flush=True)
        if len(seen) >= limit:
            break


def show_errors(root, *, limit=3, lines=120, phase=None):
    root = root.resolve()
    if phase is None:
        logs = sorted(root.glob("logs/launcher.*.log"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for log in logs[:limit]:
            if not log.resolve().is_relative_to(root):
                raise ValueError(f"launcher log is outside the switch root: {log}")
            tail = log_tail(log, lines)
            if "[launcher-start]" in tail:
                tail = "[launcher-start]" + tail.rsplit("[launcher-start]", 1)[1]
            print(f"\n[launcher-log] {log.relative_to(root)} (tail; not a diagnosis of the exit cause)", flush=True)
            print(tail or "[empty launcher log]", flush=True)
        show_admissions(root, limit)
    paths = [*root.glob("prefixes/seed-*/segment-*/failure.json"),
             *root.glob("states/*/points/*/*/failure.json"),
             *root.glob("states/*/*/failure.json")]
    shown = 0
    for path in sorted(paths, key=lambda item: item.stat().st_mtime_ns, reverse=True):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"failure record is outside the switch root: {path}")
        directory = path.parent
        failure = read_json(path)
        progress = read_json(directory / "progress.json")
        current_phase = progress.get("phase")
        if phase is not None and current_phase != phase:
            continue
        print(f"\n[failure] {directory.relative_to(root)}", flush=True)
        print(f"[recorded failure] host={failure.get('host', 'unknown')} utc={utc_time(failure.get('time'))}; "
              "saved error; this display does not establish a new failure", flush=True)
        if progress:
            print(f"[last progress] host={progress.get('host', 'unknown')} pid={progress.get('pid', 'unknown')} "
                  f"state={progress.get('state', 'unknown')} phase={current_phase or 'unknown'} "
                  f"utc={utc_time(progress.get('updated'))}; heartbeat alone does not confirm a live owner", flush=True)
        print(str(failure.get("error", "unknown failure"))[:4000], flush=True)
        if isinstance(current_phase, str) and re.fullmatch(r"[A-Za-z0-9_-]+", current_phase):
            logs = sorted(directory.glob(f"{current_phase}-*.log"))
        else:
            logs = []
        if not logs:
            print("[no worker log] This failure may have occurred before a worker started.")
        for log in logs[:4]:
            if not log.resolve().is_relative_to(root):
                raise ValueError(f"worker log is outside the switch root: {log}")
            print(f"[worker-log] {log.relative_to(root)} (last {lines} lines, at most 64 KiB)", flush=True)
            print(log_tail(log, lines) or "[empty worker log]", flush=True)
        shown += 1
        if shown == limit:
            break
    if not shown:
        print(f"[no matching recorded failures] {root}")
    return shown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--lines", type=int, default=120)
    parser.add_argument("--phase")
    args = parser.parse_args()
    if args.limit < 1 or not 1 <= args.lines <= 2000:
        parser.error("limit must be positive; lines must be between 1 and 2000")
    try:
        if not args.root.is_dir():
            raise ValueError(f"switch root does not exist: {args.root}")
        show_errors(args.root, limit=args.limit, lines=args.lines, phase=args.phase)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"[cannot read failure logs] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
