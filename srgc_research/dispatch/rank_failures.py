"""Bounded, attempt-local diagnostics that prefer the original rank exception."""

import os
import re
from pathlib import Path

EXCEPTION = re.compile(r"^(?:\[rank\d+\]:\s*)?(?:[\w.]*?(?:Error|Exception)|KeyboardInterrupt|SystemExit):")
WRAPPER = re.compile(r"ChildFailedError|ProcessRaisedException|ProcessExitedException")


def attempt_lines(path, *, start=0, max_bytes=256 * 1024):
    with Path(path).open("rb") as handle:
        end = handle.seek(0, os.SEEK_END)
        offset = max(start if start <= end else 0, end - max_bytes)
        handle.seek(offset)
        raw = handle.read(max_bytes)
    if offset > start and b"\n" in raw:
        raw = raw.split(b"\n", 1)[1]
    return raw.decode("utf-8", errors="replace").splitlines()


def original_errors(lines):
    # An appended log may contain many attempts. Prefer the last launch's
    # evidence, including its elastic traceback, rather than a historical OOM.
    # All ranks emit a marker before running; retain the whole launch. The
    # launcher boundary is present even when no rank manages to import Torch.
    boundaries = [i for i, line in enumerate(lines) if line.startswith("SRGC CHILD START ")]
    if boundaries:
        lines = lines[boundaries[-1]:]
    return list(dict.fromkeys(line.strip() for line in lines
                             if EXCEPTION.match(line.strip()) and not WRAPPER.search(line)))


def last_error_line(path):
    try:
        lines = attempt_lines(path)
    except (OSError, TypeError):
        return None
    errors = original_errors(lines)
    if errors:
        return errors[-1][:400]
    root = "\n".join(lines).rsplit("Root Cause (first observed failure):", 1)[-1]
    signal = re.search(r"Signal \d+ \([^\n]+\)|exitcode\s*:\s*-\d+[^\n]*", root)
    if signal:
        return signal.group()[:400]
    for line in reversed(lines):
        if re.search(r"Error|Exception|error:|out of memory|Killed|TIMEOUT|LAUNCH ERROR", line):
            return line.strip()[:400]
    return None


def failure_details(path, *, start=0):
    try:
        lines = attempt_lines(path, start=start)
    except OSError as error:
        return f"RANK FAILURE log={path}\nlog unavailable: {error}"
    errors = original_errors(lines)
    context = [line for line in lines if "SRGC RANK START " in line or line.startswith("SRGC CHILD START ")]
    root_index = next((i for i in range(len(lines) - 1, -1, -1)
                       if "Root Cause (first observed failure):" in lines[i]), None)
    tail = lines[root_index:][-60:] if root_index is not None else lines[-40:]
    return "\n".join([f"RANK FAILURE log={path}", *context[-5:],
                      "Original rank exceptions:" if errors else "No Python rank exception recorded; exit/signal evidence follows:",
                      *errors[-12:], *tail])
