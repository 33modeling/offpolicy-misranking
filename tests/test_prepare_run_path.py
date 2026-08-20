"""Run paths initialized by another revision must be preserved and replaced."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from prepare_run_path import prepare_run_path


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    quarantine = tmp / "quarantine"

    current = tmp / "v4-27b-s0"
    current.mkdir()
    (current / "run_config.json").write_text(json.dumps({"git": "current"}))
    assert prepare_run_path(current, "current", quarantine) is None
    assert current.exists()

    stale = tmp / "v4-27b-s1"
    stale.mkdir()
    (stale / "run_config.json").write_text(json.dumps({"git": "old"}))
    (stale / "partial.json").write_text("preserve me")
    destination = prepare_run_path(stale, "current", quarantine)
    assert destination is not None
    assert not stale.exists()
    assert (destination / "partial.json").read_text() == "preserve me"

    malformed = tmp / "v4-7b-smoke"
    malformed.mkdir()
    (malformed / "run_config.json").write_text("not json")
    destination = prepare_run_path(malformed, "current", quarantine)
    assert destination is not None
    assert "unreadable" in destination.name

print("PASS incompatible run paths are preserved outside the active namespace")
