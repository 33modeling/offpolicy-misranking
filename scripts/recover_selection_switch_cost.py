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


def recover(root, directory, event_id, *, seconds=None, reason=None, evidence_kind="operator_reported_stopped_job", evidence_extra=None):
    root = root.resolve()
    directory = (root / directory).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise ValueError("recovery directory must be inside the selection-switch root")
    relative = directory.relative_to(root).parts
    if len(relative) == 3 and relative[0] == "prefixes" and relative[2].startswith("segment-"):
        owner_lock = directory.parent / ".prefix.lock"
    elif len(relative) == 5 and relative[0] == "states" and relative[2] == "points":
        owner_lock = directory / (".measurement.lock" if directory.name in {"measurement", "gate_measurement"} else ".task.lock")
    elif (root / "mopps.json").is_file() and len(relative) == 3 and relative[0] == "states" and relative[2] in {"mopps", "random_online"}:
        owner_lock = directory / ".task.lock"
    elif (root / "mopps.json").is_file() and len(relative) == 3 and relative[0] == "states" and relative[2] == "import-cost":
        owner_lock = directory.parent / ".import.lock"
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
            evidence = {"kind": evidence_kind, "reason": reason.strip(),
                        "last_recorded_seconds": lower_bound, **(evidence_extra or {})}
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


STALE_MARGIN_SECONDS = 60.


def stale_end_time(directory, start, progress, events):
    """Latest evidence of the interrupted attempt still running.

    Evidence is the meter heartbeat and the ranks' own phase-log writes. Log
    files can be reused by a later attempt of the same branch, so evidence is
    capped at the earliest later start recorded in the same ledger.
    """
    started = core.number(start["time"], "start time", 0.)
    later = [core.number(row.get("time", 0.), "later start time", 0.) for row in events
             if row.get("state") == "started" and row.get("event_id") != start["event_id"]
             and core.number(row.get("time", 0.), "later start time", 0.) > started]
    cap = min(later) if later else float("inf")
    candidates = [core.number(progress.get("updated", 0.), "heartbeat time", 0.)] if progress else []
    for path in directory.glob(f"{start.get('phase', '')}-*.log"):
        try:
            candidates.append(path.stat().st_mtime)
        except OSError:
            continue
    return min(max(candidates, default=started), cap)


def close_stale(root, *, min_age=900., now=None):
    """Close open events whose owner has shown no life for at least min_age seconds.

    Operator decision (2026-09-15): hard-killed attempts never write a finish
    receipt, and this cluster kills jobs routinely, so an open event with a
    silent owner is closed from evidence instead of blocking the branch forever.
    The charged duration is the last observed evidence of the attempt running
    (meter heartbeat or the ranks' phase-log writes, capped at any later attempt's
    start) minus the recorded start, plus STALE_MARGIN_SECONDS so the estimate
    over-counts rather than under-counts the interrupted attempt. Events with an
    atomic finish receipt are closed from the receipt. Recent evidence and live
    local owners are left alone.
    """
    core.number(min_age, "minimum stale age", 0.)
    now = core.number(time.time() if now is None else now, "inspection time", 0.)
    outcome = []
    for item in inspect(root):
        directory = root / item["directory"]
        start = item["start"]
        event_id = start["event_id"]
        row = {"directory": item["directory"], "event_id": event_id}
        try:
            if item["finish_receipt"]:
                row.update(recover(root, directory, event_id))
            else:
                progress = item["progress"] or {}
                _, events = read_events(directory)
                end = stale_end_time(directory, start, progress, events)
                age = now - end
                if age < min_age:
                    row.update(status="skipped", reason=f"last evidence of the job is {age:.0f}s old (< {min_age:.0f}s)")
                else:
                    heartbeat = core.number(progress.get("seconds", 0.), "last recorded duration", 0.)
                    started = core.number(start["time"], "start time", 0.)
                    seconds = max(heartbeat, min(end, now) - started) + STALE_MARGIN_SECONDS
                    row.update(recover(root, directory, event_id, seconds=seconds,
                        reason=(f"owner silent for {age:.0f}s; charged last observed evidence minus start "
                                f"plus {STALE_MARGIN_SECONDS:.0f}s margin (over-count, never under-count)"),
                        evidence_kind="stale_owner_last_evidence",
                        evidence_extra={"last_evidence_time": end, "silent_seconds": age,
                                        "heartbeat_seconds": heartbeat, "margin_seconds": STALE_MARGIN_SECONDS}))
        except (ValueError, OSError, BlockingIOError) as exc:
            row.update(status="blocked", reason=str(exc))
        outcome.append(row)
    return outcome


def brief(label, closed, remaining):
    """One line of counts, then one line per event that stayed open and why."""
    recovered = sum(row.get("status") == "recovered" for row in closed)
    lines = [f"[recover-cost] {label}: {recovered} stale event(s) closed, {len(remaining)} still open"]
    for row in closed:
        if row.get("status") != "recovered":
            lines.append(f"[recover-cost]   {row.get('directory')} {str(row.get('event_id', ''))[:8]}: "
                         f"{row.get('status')} - {str(row.get('reason', ''))[:200]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--event-id")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--reason")
    parser.add_argument("--stale", action="store_true",
                        help="close open events whose owner has shown no life for --min-age seconds: receipt if present, else last observed evidence plus a 60s over-count margin")
    parser.add_argument("--min-age", type=float, default=900.)
    parser.add_argument("--brief", action="store_true",
                        help="with --stale: one summary line plus one line per event that was not closed, instead of JSON")
    args = parser.parse_args()
    if args.stale and args.directory is not None:
        parser.error("--stale cannot be combined with a single-event recovery")
    root = args.root.resolve()
    if not any((root / name).is_file() for name in ("switch.json", "mopps.json")):
        parser.error("root must contain switch.json or mopps.json")
    if args.directory is None:
        if args.event_id is not None or args.seconds is not None or args.reason is not None:
            parser.error("event recovery requires --directory and --event-id")
        if args.stale:
            try:
                closed = close_stale(root, min_age=args.min_age)
            except (ValueError, OSError) as exc:
                parser.exit(2, f"[recovery blocked] {exc}\n")
            remaining = inspect(root)
            if args.brief:
                print(brief(root.name, closed, remaining))
            else:
                print(json.dumps({"stale_closure": closed, "open_events": remaining}, indent=2))
            return 2 if any(row["status"] == "blocked" for row in closed) else 0
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
