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


OOM_NUMBERS = re.compile(r"Tried to allocate ([\d.]+ \w+)\. GPU (\d+) has a total capacity of ([\d.]+ \w+) "
                         r"of which ([\d.]+ \w+) is free\.")
FRAME = re.compile(r'File "([^"]+)", line \d+, in (\w+)')


def oom_summary(lines):
    """One status line from a CUDA OOM: sizes, other GPU processes and our call site."""
    index = next((i for i in range(len(lines) - 1, -1, -1) if "CUDA out of memory" in lines[i]), None)
    if index is None:
        return None
    text = lines[index]
    parts = []
    numbers = OOM_NUMBERS.search(text)
    if numbers:
        wanted, gpu, total, free = numbers.groups()
        parts.append(f"tried {wanted}, GPU{gpu} total {total}, free {free}")
    mine = re.search(r"this process has ([\d.]+ \w+) memory in use", text)
    if mine:
        parts.append(f"this process {mine.group(1)}")
    allocated = re.search(r"([\d.]+ \w+) is allocated by PyTorch", text)
    if allocated:
        parts.append(f"PyTorch allocated {allocated.group(1)}")
    others = re.findall(r"Process \d+ has ([\d.]+ \w+) memory in use", text)
    if numbers or mine:
        parts.append(f"other processes {len(others)}" + (f" ({', '.join(others)})" if others else ""))
    # Our own frames of the traceback that ends at this OOM, innermost last.
    frames = []
    for line in reversed(lines[max(0, index - 200):index]):
        if "Traceback (most recent call last)" in line:
            break
        match = FRAME.search(line)
        if match and "site-packages" not in match.group(1) and "/torch/" not in match.group(1):
            frames.append(match.group(2))
    sites = list(dict.fromkeys(reversed(frames)))[-4:]
    if sites:
        parts.append("at " + " > ".join(sites))
    return "CUDA OOM: " + ", ".join(parts) if parts else text.strip()[:400]


def last_error_line(path):
    try:
        lines = attempt_lines(path)
    except (OSError, TypeError):
        return None
    errors = original_errors(lines)
    if errors and "out of memory" in errors[-1]:
        boundaries = [i for i, line in enumerate(lines) if line.startswith("SRGC CHILD START ")]
        return oom_summary(lines[boundaries[-1]:] if boundaries else lines) or errors[-1][:400]
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


def previous_attempt_line(path, *, max_bytes=4 * 1024 * 1024):
    """Why the launch before the current one ended, for a task that is running again."""
    try:
        lines = attempt_lines(path, max_bytes=max_bytes)
    except (OSError, TypeError):
        return None
    boundaries = [i for i, line in enumerate(lines) if line.startswith("SRGC CHILD START ")]
    if len(boundaries) < 2:
        return None
    lines = lines[boundaries[-2]:boundaries[-1]]
    errors = original_errors(lines)
    if errors and "out of memory" in errors[-1]:
        return oom_summary(lines) or errors[-1][:400]
    if errors:
        return errors[-1][:400]
    for line in reversed(lines):
        if re.search(r"Error|Exception|error:|out of memory|Killed|TIMEOUT|LAUNCH ERROR|GPU WAIT|"
                     r"Signal \d+|exitcode\s*:\s*-\d+|Watchdog", line):
            return line.strip()[:400]
    return "no error recorded in its log"


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
