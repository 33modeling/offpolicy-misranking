"""Reload only the read-only status viewer when its loaded code changes."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


class StatusWatch:
    def __init__(self, files=None):
        root = Path(__file__).resolve().parents[1]
        self.files = list(files) if files is not None else [root / path for path in (
            "scripts/_status_watch.py", "scripts/_node_view.py", "scripts/_status_summary.py",
            "scripts/_switch_state_point.py", "scripts/experiments_status.py", "scripts/experiments_progress.py",
            "scripts/selection_switch_status.py", "scripts/mopps_comparison_status.py",
            "src/selection_switch.py", "src/selection_gate.py",
        )]
        self.loaded = self.fingerprint()

    def fingerprint(self):
        hashes = []
        for path in self.files:
            try:
                hashes.append(hashlib.sha256(path.read_bytes()).digest())
            except OSError:
                hashes.append(None)
        return tuple(hashes)

    def refresh(self):
        if self.fingerprint() == self.loaded:
            return
        print("[status-reload] viewer code changed; refreshing status only, workers untouched", file=sys.stderr, flush=True)
        os.execv(sys.executable, [sys.executable, *sys.argv])
