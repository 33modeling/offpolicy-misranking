from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime_resume_commit import bind_generation_commit, choose_generation_commit


def write_config(root: Path, name: str, commit: str) -> None:
    run = root / name
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(
        json.dumps({"git": commit}), encoding="utf-8"
    )


def test_empty_matrix_uses_current_commit(tmp_path: Path) -> None:
    current = "a" * 40
    assert choose_generation_commit(tmp_path, current) == current


def test_partial_matrix_reuses_its_single_commit(tmp_path: Path) -> None:
    recorded = "b" * 40
    write_config(tmp_path, "run-a", recorded)
    write_config(tmp_path, "run-b", recorded)
    assert choose_generation_commit(tmp_path, "a" * 40) == recorded


def test_mixed_generation_commits_are_rejected(tmp_path: Path) -> None:
    write_config(tmp_path, "run-a", "a" * 40)
    write_config(tmp_path, "run-b", "b" * 40)
    with pytest.raises(ValueError, match="mixed generation commits"):
        choose_generation_commit(tmp_path, "c" * 40)


def test_invalid_generation_commit_is_rejected(tmp_path: Path) -> None:
    write_config(tmp_path, "run-a", "not-a-full-commit")
    with pytest.raises(ValueError, match="invalid generation commit"):
        choose_generation_commit(tmp_path, "a" * 40)


def test_marker_binds_empty_matrix_to_first_commit(tmp_path: Path) -> None:
    marker = tmp_path / ".queue/generation.git"
    first = "a" * 40
    second = "b" * 40

    assert bind_generation_commit(tmp_path, marker, first) == first
    assert bind_generation_commit(tmp_path, marker, second) == first
    assert marker.read_text(encoding="utf-8") == f"{first}\n"


def test_marker_must_match_existing_run_configs(tmp_path: Path) -> None:
    marker = tmp_path / ".queue/generation.git"
    marker.parent.mkdir()
    marker.write_text(f"{'a' * 40}\n", encoding="utf-8")
    write_config(tmp_path, "run-a", "b" * 40)

    with pytest.raises(ValueError, match="run configs use"):
        bind_generation_commit(tmp_path, marker, "a" * 40)


def test_malformed_marker_is_rejected(tmp_path: Path) -> None:
    marker = tmp_path / ".queue/generation.git"
    marker.parent.mkdir()
    marker.write_text("not-a-commit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid generation commit"):
        bind_generation_commit(tmp_path, marker, "a" * 40)
