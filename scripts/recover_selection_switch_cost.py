#!/usr/bin/env python3
"""Inspect or close an interrupted selection-switch cost event using evidence."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import selection_gate as core
import selection_gate_gpu as base


def read_events(directory):
    raw = (directory / "cost.jsonl").read_bytes()
    return raw, [json.loads(line) for line in raw.splitlines() if line.strip()]


def event_progress(directory, start):
    event_id = start["event_id"]
    if not event_id or Path(event_id).name != event_id or event_id in {".", ".."}:
        raise ValueError("invalid cost event ID")
    candidates = []
    archive = directory / "pending-costs" / f"{event_id}.json"
    if archive.exists():
        saved = core.read(archive)
        if saved.get("start") != start:
            raise ValueError("archived cost start differs from the open event")
        if saved.get("progress"):
            candidates.append(saved["progress"])
    path = directory / "progress.json"
    if path.exists():
        current = core.read(path)
        if current.get("event_id") == event_id:
            candidates.append(current)
    for progress in candidates:
        if any(progress.get(key) != start.get(key) for key in ("event_id", "phase", "ledger", "gpus", "gpu_type", "host")):
            raise ValueError("progress record allocation differs from the open event")
    return max(candidates, key=lambda row: core.number(row.get("seconds", 0.), "last recorded duration", 0.), default={})


def inspect(root):
    pending = []
    for path in sorted(root.rglob("cost.jsonl")):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"cost ledger is outside the switch root: {path}")
        _, events = read_events(path.parent)
        summary = core.cost_summary(events)
        if summary["missing_starts"]:
            raise ValueError(f"missing cost starts: {path}")
        for event_id in summary["incomplete_events"]:
            start = next(row for row in events if row["event_id"] == event_id and row["state"] == "started")
            progress = event_progress(path.parent, start)
            pending.append({"directory": str(path.parent.relative_to(root)), "start": start,
                            "progress": progress if progress.get("event_id") == event_id else None,
                            "finish_receipt": (path.parent / "cost-events" / f"{event_id}.json").is_file()})
    return pending


def recover(root, directory, event_id, *, seconds=None, reason=None):
    root = root.resolve()
    directory = (root / directory).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise ValueError("recovery directory must be inside the selection-switch root")
    relative = directory.relative_to(root).parts
    if len(relative) == 3 and relative[0] == "prefixes" and relative[2].startswith("segment-"):
        owner_lock = directory.parent / ".prefix.lock"
    elif len(relative) == 5 and relative[0] == "states" and relative[2] == "points":
        owner_lock = directory / (".measurement.lock" if directory.name in {"measurement", "gate_measurement"} else ".task.lock")
    else:
        raise ValueError("expected a switch prefix segment or continuation/measurement directory")
    if not event_id or Path(event_id).name != event_id or event_id in {".", ".."}:
        raise ValueError("invalid cost event ID")
    with contextlib.ExitStack() as locks:
        locks.enter_context(base.lease(owner_lock))
        locks.enter_context(base.lease(directory / ".cost.lock"))
        raw, events = read_events(directory)
        summary = core.cost_summary(events)
        if summary["missing_starts"]:
            raise ValueError("cannot repair a ledger with missing start records")
        starts = [row for row in events if row["event_id"] == event_id and row["state"] == "started"]
        if not starts:
            raise ValueError("cost event has no start record")
        start = starts[0]
        finished = next((row for row in events if row["event_id"] == event_id and row["state"] == "finished"), None)
        if finished is not None:
            if seconds is not None and finished["seconds"] != seconds:
                raise ValueError("event is already closed with a different duration")
            return {"status": "already_closed", "event_id": event_id}
        if any((directory / name).exists() for name in ("result.json", "initial.json")):
            raise ValueError("cannot change costs already bound to a published result or measurement")
        progress = event_progress(directory, start)
        pid = progress.get("pid", start.get("pid"))
        receipt_path = directory / "cost-events" / f"{event_id}.json"
        lower_bound = core.number(progress.get("seconds", 0.), "last recorded duration", 0.)
        if receipt_path.exists():
            finish = core.read(receipt_path)
            if finish.get("event_id") != event_id or finish.get("state") != "finished":
                raise ValueError("finish receipt belongs to a different event")
            if seconds is not None and seconds != finish["seconds"]:
                raise ValueError("reported duration differs from the completed event receipt")
            evidence = {"kind": "atomic_finish_receipt", "sha256": base.digest(receipt_path)}
        else:
            if start.get("host") == socket.gethostname() and type(pid) is int and pid > 0:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError(f"recorded owner PID {pid} is still alive; verify the stopped job first")
            if seconds is None or not reason or not reason.strip():
                raise ValueError(f"no completed event receipt; last heartbeat is only a lower bound ({lower_bound}s). "
                                 "For a confirmed stopped job, supply --seconds from its termination log and --reason identifying that evidence")
            core.number(seconds, "reported duration", lower_bound)
            finish = {**start, "state": "finished", "time": core.number(start["time"], "start time", 0.) + seconds,
                      "seconds": seconds, "allocated_gpu_seconds": seconds * start["gpus"], "exit_code": 130}
            evidence = {"kind": "operator_reported_stopped_job", "reason": reason.strip(),
                        "last_recorded_seconds": lower_bound}
        if finish["seconds"] < lower_bound:
            raise ValueError("finish duration precedes the last recorded progress")
        finish = {**finish, "recovery": {**evidence, "recorded_at": time.time(),
                  "prior_ledger_sha256": hashlib.sha256(raw).hexdigest(),
                  "tool_sha256": base.digest(Path(__file__))}}
        repaired = core.cost_summary([*events, finish])
        if event_id in repaired["incomplete_events"]:
            raise ValueError("recovery did not close the selected cost event")
        base.journal(directory / "cost.jsonl", finish)
        return {"status": "recovered", "directory": str(directory), "event_id": event_id,
                "seconds": finish["seconds"], "allocated_gpu_seconds": finish["allocated_gpu_seconds"],
                "evidence": evidence, "remaining_open_events": repaired["incomplete_events"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--event-id")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--reason")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "switch.json").is_file():
        parser.error("root must contain switch.json")
    if args.directory is None:
        if args.event_id is not None or args.seconds is not None or args.reason is not None:
            parser.error("event recovery requires --directory and --event-id")
        print(json.dumps({"open_events": inspect(root)}, indent=2))
        return 0
    if args.event_id is None:
        parser.error("event recovery requires --event-id")
    try:
        result = recover(root, args.directory, args.event_id, seconds=args.seconds, reason=args.reason)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"[recovery blocked] {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
