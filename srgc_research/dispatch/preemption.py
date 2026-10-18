"""Recognize an interrupted child using only this attempt's bounded log tail."""

import re

_KILLED = re.compile(r"SIGKILL|signal\s+9\b|exitcode\s*:\s*-9\b", re.IGNORECASE)
_OOM = re.compile(r"out\s+of\s+memory|OutOfMemoryError|oom[-_ ]?kill|"
                  r"killed process\s+\d+.*total-vm", re.IGNORECASE)


def log_position(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def interruption_code(code, path, start):
    """Return a signal exit code, leaving application errors and explicit OOM alone."""
    normalized = 128 - code if code < 0 else code
    if normalized in (129, 130, 143):
        return normalized
    if normalized not in (1, 137):
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            # run_child appends. Also tolerate a log rotated/truncated externally.
            handle.seek(max(start if start <= end else 0, end - 65536))
            tail = handle.read(65536).decode("utf-8", errors="replace")
    except OSError:
        tail = ""
    if _OOM.search(tail):
        return None
    # During error cleanup torchrun may kill sibling ranks. Its reported root
    # failure, when present, must itself be SIGKILL before refunding an exit 1.
    root_failure = tail.rsplit("Root Cause (first observed failure):", 1)[-1]
    if normalized == 137 or _KILLED.search(root_failure):
        # torchrun commonly returns 1 while reporting a rank's actual -9 exit.
        return 137
    return None
