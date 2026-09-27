"""Rank-local completed work, never a launcher heartbeat masquerading as progress."""

import os
from pathlib import Path
import time

from .runtime import atomic_json


def record(stage, **details):
    directory = os.environ.get("SRGC_PROGRESS_DIR")
    if directory:
        atomic_json(Path(directory) / f"rank-{os.environ.get('RANK', '0')}.json",
                    {"stage": stage, "updated": time.time(), **details})


def signature(directory):
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                 for p in sorted(directory.glob("rank-*.json")))
