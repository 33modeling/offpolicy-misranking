"""Recover local Pair CUDA leftovers using event identity and the writer lease."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cleanup_run_processes as cleanup

MARKER = re.compile(r"OM_SELECTION_COST_([a-f0-9]{32})\Z")


def same_allocation(pid, *, proc=Path("/proc")):
    """A shared UID or hostname does not prove ownership of another job."""
    try:
        own, other = proc / "self", proc / str(pid)
        for namespace in ("pid", "mnt"):
            a, b = (own / "ns" / namespace).stat(), (other / "ns" / namespace).stat()
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                return False
        return (own / "cgroup").read_bytes() == (other / "cgroup").read_bytes()
    except OSError:
        return False


def live_controllers(root, processes):
    controllers = []
    for process in processes.values():
        words = process.argv
        if (len(words) == 5 and Path(words[1]).name in {
                "selector_pair_gpu.py", "queue_selector_pair_gpu.py"}
                and words[2] in {"run", "develop", "test", "freeze"}
                and words[3] == "--root" and same_allocation(process.pid)):
            try:
                cwd = (Path("/proc") / str(process.pid) / "cwd").resolve(strict=True)
                if (cwd / words[4]).resolve() == root:
                    controllers.append(process.pid)
            except OSError:
                continue
    return controllers


def event_record(directory, event):
    for path in (directory / "progress.json", directory / "cost-events" / f"{event}.json"):
        try:
            value = json.loads(path.read_text())
            if isinstance(value, dict) and value.get("event_id") == event:
                return True
        except (OSError, ValueError):
            pass
    return False


def candidates(root, processes):
    """Require local same-user processes, exact root, event marker and log evidence."""
    found = {}
    for process in processes.values():
        out = process.environ.get("OUT_ROOT", "")
        if not out or Path(out).resolve() != root or not same_allocation(process.pid):
            continue
        events = [match[1] for key, value in process.environ.items()
                  if value == "1" and (match := MARKER.fullmatch(key))]
        for name in process.open_files:
            path = Path(name)
            if not path.is_absolute() or path.suffix != ".log":
                continue
            directory = path.parent.resolve()
            if not directory.is_relative_to(root):
                continue
            for event in events:
                if event_record(directory, event):
                    found[(directory, event, out)] = None
    return found


def wait_release(targets, timeout=15.):
    owned = {p.pid for p in targets}
    deadline = time.monotonic() + timeout
    while owned:
        report = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
            timeout=max(.1, min(3., deadline - time.monotonic())))
        rows = [row.strip() for row in report.stdout.splitlines() if row.strip()]
        if any(not row.isdigit() for row in rows):
            raise RuntimeError("cannot verify CUDA PID release")
        remaining = owned & {int(row) for row in rows}
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"driver still reports old CUDA PIDs: {sorted(remaining)}")
        print(f"[cleanup] waiting for CUDA release: {sorted(remaining)} [pair]", flush=True)
        time.sleep(.25)


def recover(root):
    root = root.resolve()
    total = 0
    processes = cleanup._snapshot()
    controllers = live_controllers(root, processes)
    if controllers:
        print(f"[cleanup] live Pair controllers preserved: {controllers}; no process stopped [pair]", flush=True)
        return 0
    for directory, event, out in candidates(root, processes):
        try:
            lease = (directory / ".cost.lock").open("r+")
        except FileNotFoundError:
            continue  # Missing ownership evidence is not permission to kill.
        with lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"[cleanup] live phase preserved: {directory} [pair]", flush=True)
                continue
            # Hold the existing writer lease throughout teardown. A new worker
            # cannot enter while old ranks are releasing their CUDA contexts.
            if not event_record(directory, event):
                continue
            if live_controllers(root, cleanup._snapshot()):
                print("[cleanup] Pair controller became active; no further processes stopped [pair]", flush=True)
                break
            required = (("OUT_ROOT", out), (f"OM_SELECTION_COST_{event}", "1"))
            targets = cleanup.list_processes("/unused-pair-event-scope",
                command_patterns=("",), required_environment=required)
            if any(not same_allocation(process.pid) for process in targets):
                print(f"[cleanup] other or unknown allocation preserved: event={event} [pair]", flush=True)
                continue
            print(f"[cleanup] recovering ended phase: {directory} event={event} [pair]", flush=True)
            targets = cleanup.terminate("/unused-pair-event-scope", timeout=3,
                command_patterns=("",), required_environment=required, compact=True)
            remaining = cleanup.list_processes("/unused-pair-event-scope",
                command_patterns=("",), required_environment=required)
            if remaining:
                raise RuntimeError(f"owned workers remain: {[p.pid for p in remaining]}")
            wait_release(targets)
            total += len(targets)
    print(f"[cleanup] previous Pair phase processes stopped={total}; live/unknown owners preserved [pair]",
          flush=True)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        recover(args.root)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[blocked] Pair GPU recovery incomplete: {exc}; no GPU reset performed [pair]", flush=True)
        return 75
    return 0


if __name__ == "__main__":
    sys.exit(main())
