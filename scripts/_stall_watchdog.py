#!/usr/bin/env python3
"""Stop a GPU phase on this node whose worker logs stopped moving.

A rank that dies with a CUDA fault leaves the trainer hung; the meter would wait
until the phase's allocation limit and charge the whole allocation. This
watchdog scans every running phase on this host (progress.json written by the
meter), and when the phase's worker logs have been silent for --stall-seconds it
terminates that phase's worker processes (found by the cost-event marker the
meter puts in their environment), records the host under --faults-dir so the
launchers refuse further GPU work here, and writes stalled.json next to the
phase. The meter then records a failed attempt of a few minutes, which the queue
retries elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

DEFAULT_PHASES = ("train", "prefix-train")


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def running_phases(roots, host, phases):
    for root in roots:
        for progress in Path(root).rglob("progress.json"):
            p = read_json(progress)
            if p.get("state") == "running" and p.get("host") == host and p.get("phase") in phases and p.get("event_id"):
                yield progress.parent, p


def log_silence(directory, phase, now):
    """Seconds since the phase's newest worker-log write; None without logs."""
    newest = None
    for path in directory.glob(f"{phase}-*.log"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        newest = mtime if newest is None else max(newest, mtime)
    return None if newest is None else now - newest


def workers_of(event_id, me=None):
    """Own processes carrying the meter's cost-event marker for this phase."""
    marker = f"OM_SELECTION_COST_{event_id}=1".encode()
    me = os.getpid() if me is None else me
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        pid = int(proc.name)
        if pid == me:
            continue
        try:
            if marker in (proc / "environ").read_bytes().split(b"\0"):
                found.append(pid)
        except OSError:
            continue
    return found


def stop(pids, grace=30.):
    groups = set()
    for pid in pids:
        try:
            groups.add(os.getpgid(pid))
        except OSError:
            continue
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline and any(Path(f"/proc/{pid}").exists() for pid in pids):
        time.sleep(1)
    for pgid in groups:
        if any(Path(f"/proc/{pid}").exists() for pid in pids):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
    return sorted(groups)


def scan(roots, faults_dir, *, host=None, stall_seconds=1500., phases=DEFAULT_PHASES, now=None, dry_run=False):
    """One pass; returns the phases it stopped."""
    host = (os.environ.get("EXPERIMENTS_NODE_ID") or socket.gethostname()) if host is None else host
    now = time.time() if now is None else now
    stopped = []
    for directory, p in running_phases(roots, host, phases):
        silence = log_silence(directory, p["phase"], now)
        if silence is None or silence < stall_seconds:
            continue
        pids = workers_of(p["event_id"])
        record = {"host": host, "phase": p["phase"], "event_id": p["event_id"], "directory": str(directory),
                  "silent_seconds": silence, "pids": pids, "time": now}
        print(f"[stall] {directory}: {p['phase']} logs silent for {silence:.0f}s; "
              f"stopping {len(pids)} worker process(es) of event {p['event_id'][:8]}", flush=True)
        if not dry_run:
            record["process_groups"] = stop(pids) if pids else []
            (directory / "stalled.json").write_text(json.dumps(record, indent=2) + "\n")
            faults_dir = Path(faults_dir)
            faults_dir.mkdir(parents=True, exist_ok=True)
            fault_path = faults_dir / f"{host}.json"
            previous = read_json(fault_path) or {}
            handled = list(previous.get("events") or ([previous["event_id"]] if previous.get("event_id") else []))
            if p["event_id"] in handled:
                # The meter died with its ranks, so progress.json keeps saying "running":
                # the same stall is seen every scan. One stall is one strike.
                print(f"[stall] host {host}: event {p['event_id'][:8]} already recorded (strike {previous.get('strikes', 1)})", flush=True)
            else:
                record["strikes"] = int(previous.get("strikes", 0) or 0) + 1
                record["events"] = handled + [p["event_id"]]
                fault_path.write_text(json.dumps(record, indent=2) + "\n")
                print(f"[stall] host {host} recorded under {faults_dir} (strike {record['strikes']}); launchers refuse GPU "
                      "work here until the record expires or an operator restart clears it", flush=True)
        stopped.append(record)
    return stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--faults-dir", type=Path, required=True)
    parser.add_argument("--stall-seconds", type=float, default=1500.)
    parser.add_argument("--interval", type=float, default=60.)
    parser.add_argument("--phases", default=",".join(DEFAULT_PHASES))
    parser.add_argument("--host", default=None, help="node identity to match progress records and record faults under")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    phases = tuple(p for p in args.phases.split(",") if p)
    parent = os.getppid()
    print(f"[watchdog] pid={os.getpid()} host={args.host or os.environ.get('EXPERIMENTS_NODE_ID') or socket.gethostname()} roots={[str(r) for r in args.roots]} "
          f"stall={args.stall_seconds:.0f}s phases={phases}", flush=True)
    while True:
        try:
            scan(args.roots, args.faults_dir, host=args.host, stall_seconds=args.stall_seconds, phases=phases)
        except Exception as exc:  # the watchdog must outlive one bad file
            print(f"[watchdog] scan error: {exc}", flush=True)
        if args.once or os.getppid() != parent:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
