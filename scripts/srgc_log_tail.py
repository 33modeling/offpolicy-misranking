"""Bounded log reads for worker failures and live status."""

import os
from pathlib import Path

MAX_TAIL_BYTES = 256 * 1024


def tail_lines(path, lines=40, *, max_bytes=MAX_TAIL_BYTES):
    """Return at most ``lines`` from the final byte window, not the whole file.

    A partial first line is omitted when the window starts inside a long log.
    If one line exceeds the byte budget, retain its bounded suffix. Decode with
    replacement because a truncated UTF-8 character or child output is possible.
    """
    if lines < 0 or max_bytes < 1:
        raise ValueError("lines must be nonnegative and max_bytes must be positive")
    if lines == 0:
        return []
    with Path(path).open("rb") as handle:
        end = handle.seek(0, os.SEEK_END)
        start = max(0, end - max_bytes)
        handle.seek(start)
        raw = handle.read(max_bytes)
    if start and b"\n" in raw[:-1]:
        raw = raw.split(b"\n", 1)[1]
    return raw.decode("utf-8", errors="replace").splitlines()[-lines:]
