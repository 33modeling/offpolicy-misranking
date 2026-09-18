"""Conservative, read-only ownership evidence for interrupted cost events."""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selection_gate_gpu as base


def _vanished(directory):
    try:
        directory.stat()
    except FileNotFoundError:
        return True
    except OSError:
        pass
    return False


def local_event_owner(start, progress, *, proc_root=Path("/proc")):
    """Return live/stopped/unknown/remote without stopping any process.

    A missing local meter PID is insufficient: its detached GPU children carry
    the event marker and can outlive it without holding the meter's file locks.
    An incomplete process inspection must never certify that those children
    have stopped. Callers still acquire and recheck the worker/cost leases.
    """
    host = start.get("host")
    if not isinstance(host, str) or not host:
        return "unknown"
    if host != base.node_id():
        return "remote"
    recorded_pid = start.get("pid")
    current_pid = (progress or {}).get("pid")
    if recorded_pid is not None and current_pid is not None and recorded_pid != current_pid:
        return "unknown"
    pid = (progress or {}).get("pid", start.get("pid"))
    if type(pid) is not int or pid <= 0:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    except (OSError, OverflowError):
        return "unknown"
    else:
        # A reused PID is also live/unverified, never proof of a dead owner.
        return "live"

    event_id = start.get("event_id")
    if (not isinstance(event_id, str) or not event_id
            or any(char in event_id for char in ("\0", "="))):
        return "unknown"
    marker = f"OM_SELECTION_COST_{event_id}=1".encode()
    uncertain = False
    try:
        for directory in Path(proc_root).iterdir():
            if not directory.name.isdigit():
                continue
            try:
                if directory.stat().st_uid != os.getuid():
                    continue
                raw_stat = (directory / "stat").read_text()
                fields = raw_stat.rsplit(") ", 1)[1].split()
                if len(fields) < 20 or int(raw_stat.split(" ", 1)[0]) != int(directory.name):
                    raise ValueError("incomplete process identity")
                if fields[0] == "Z":
                    continue
                int(fields[19])  # Refuse malformed process start-time evidence.
                environment = (directory / "environ").read_bytes().split(b"\0")
                if marker in environment:
                    return "live"
            except FileNotFoundError:
                # Only the disappearance of the PID itself proves a race with
                # exit; a surviving PID with unreadable files remains unknown.
                uncertain |= not _vanished(directory)
            except (OSError, ValueError, IndexError, UnicodeError):
                uncertain = True
    except OSError:
        return "unknown"
    return "unknown" if uncertain else "stopped"
