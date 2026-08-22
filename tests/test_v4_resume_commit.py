"""Selection of the immutable commit used to resume interrupted v4 runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from v4_resume_commit import select_resume_commit

with tempfile.TemporaryDirectory() as raw_tmp:
    root = Path(raw_tmp)
    assert select_resume_commit(root, "current") == "current"

    for name in ("v4-27b-s0", "v4-7b-s4-math500"):
        run = root / name
        run.mkdir()
        (run / "run_config.json").write_text(
            json.dumps({"git": "generation"}), encoding="utf-8"
        )
    assert select_resume_commit(root, "current") == "generation"

    mixed = root / "v4-27b-s2"
    mixed.mkdir()
    (mixed / "run_config.json").write_text(
        json.dumps({"git": "other"}), encoding="utf-8"
    )
    try:
        select_resume_commit(root, "current")
    except ValueError as exc:
        assert "mixed v4 generation commits" in str(exc)
        assert "v4-27b-s2" in str(exc)
    else:
        raise AssertionError("mixed generation commits were accepted")

print("PASS v4 resume commit selection")


def test_v4_resume_commit_selection() -> None:
    pass
