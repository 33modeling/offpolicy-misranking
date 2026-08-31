from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime_resume_commit import choose_generation_commit


def write_config(root: Path, name: str, commit: str) -> None:
    run = root / name
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(
        json.dumps({"git": commit}), encoding="utf-8"
    )


def test_empty_matrix_uses_current_commit(tmp_path: Path) -> None:
    assert choose_generation_commit(tmp_path, "current") == "current"


def test_partial_matrix_reuses_its_single_commit(tmp_path: Path) -> None:
    write_config(tmp_path, "run-a", "recorded")
    write_config(tmp_path, "run-b", "recorded")
    assert choose_generation_commit(tmp_path, "current") == "recorded"


def test_mixed_generation_commits_are_rejected(tmp_path: Path) -> None:
    write_config(tmp_path, "run-a", "first")
    write_config(tmp_path, "run-b", "second")
    with pytest.raises(ValueError, match="mixed generation commits"):
        choose_generation_commit(tmp_path, "current")
