from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime_resume_commit import (
    bind_generation_commit,
    bind_suite_generation_commit,
    choose_generation_commit,
    repair_generation_commit,
)


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


def test_failed_preflight_marker_can_advance_before_any_family(tmp_path: Path) -> None:
    marker = tmp_path / ".queue/generation.git"
    first = "a" * 40
    current = "b" * 40
    assert bind_generation_commit(tmp_path, marker, first) == first

    assert (
        bind_generation_commit(tmp_path, marker, current, advance_empty=True)
        == current
    )
    assert marker.read_text(encoding="utf-8") == f"{current}\n"


def test_generation_marker_never_advances_after_family_creation(tmp_path: Path) -> None:
    marker = tmp_path / ".queue/generation.git"
    first = "a" * 40
    current = "b" * 40
    assert bind_generation_commit(tmp_path, marker, first) == first
    (tmp_path / "family-mbpp-s0").mkdir()

    assert bind_generation_commit(
        tmp_path, marker, current, advance_empty=True
    ) == first


def test_marker_must_match_existing_run_configs(tmp_path: Path) -> None:
    marker = tmp_path / ".queue/generation.git"
    marker.parent.mkdir()
    marker.write_text(f"{'a' * 40}\n", encoding="utf-8")
    write_config(tmp_path, "run-a", "b" * 40)

    with pytest.raises(ValueError, match="run configs use"):
        bind_generation_commit(tmp_path, marker, "a" * 40)


def test_repair_quarantines_conflicting_checkpoint_suffix(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    marker = root / ".queue/generation.git"
    marker.parent.mkdir(parents=True)
    marker.write_text(f"{'b' * 40}\n", encoding="utf-8")
    target = "a" * 40

    def write_run(name: str, commit: str, dataset: str, seed: int, drift: int) -> None:
        run = root / name
        run.mkdir(parents=True)
        (run / "run_config.json").write_text(
            json.dumps(
                {"git": commit, "dataset": dataset, "seed": seed, "drift": drift}
            ),
            encoding="utf-8",
        )

    write_run("math-d0", target, "math500", 0, 0)
    write_run("math-d25", "b" * 40, "math500", 0, 25)
    write_run("math-d100", target, "math500", 0, 100)
    write_run("code-d0", target, "mbpp", 0, 0)

    selected, moved = repair_generation_commit(
        root, marker, target, tmp_path / "quarantine"
    )

    assert selected == target
    assert marker.read_text(encoding="utf-8") == f"{target}\n"
    assert (root / "math-d0/run_config.json").is_file()
    assert (root / "code-d0/run_config.json").is_file()
    assert not (root / "math-d25").exists()
    assert not (root / "math-d100").exists()
    assert {path.name.split("-mixed-")[0] for path in moved} == {
        "math-d25",
        "math-d100",
    }
    assert choose_generation_commit(root, "c" * 40) == target


def test_malformed_marker_is_rejected(tmp_path: Path) -> None:
    marker = tmp_path / ".queue/generation.git"
    marker.parent.mkdir()
    marker.write_text("not-a-commit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid generation commit"):
        bind_generation_commit(tmp_path, marker, "a" * 40)


def test_suite_reuses_commit_from_partial_matrix_for_empty_peer(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    replication = tmp_path / "replication"
    recorded = "b" * 40
    write_config(primary, "run-a", recorded)
    marker = tmp_path / ".rlvr-generation.git"

    assert (
        bind_suite_generation_commit(
            [primary, replication], marker, current="a" * 40
        )
        == recorded
    )
    assert marker.read_text(encoding="utf-8") == f"{recorded}\n"


def test_suite_reads_matrix_marker_before_first_run_config(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    replication = tmp_path / "replication"
    matrix_marker = replication / ".queue/generation.git"
    matrix_marker.parent.mkdir(parents=True)
    matrix_marker.write_text(f"{'c' * 40}\n", encoding="utf-8")

    assert bind_suite_generation_commit(
        [primary, replication],
        tmp_path / ".rlvr-generation.git",
        current="a" * 40,
    ) == "c" * 40


def test_suite_rejects_mixed_matrix_commits(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    replication = tmp_path / "replication"
    write_config(primary, "run-a", "a" * 40)
    write_config(replication, "run-b", "b" * 40)

    with pytest.raises(ValueError, match="mixed generation commits across RLVR suite"):
        bind_suite_generation_commit(
            [primary, replication],
            tmp_path / ".rlvr-generation.git",
            current="c" * 40,
        )


def test_suite_marker_must_match_later_matrix_artifacts(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    replication = tmp_path / "replication"
    suite_marker = tmp_path / ".rlvr-generation.git"
    suite_marker.write_text(f"{'a' * 40}\n", encoding="utf-8")
    write_config(replication, "run-b", "b" * 40)

    with pytest.raises(ValueError, match="mixed generation commits across RLVR suite"):
        bind_suite_generation_commit(
            [primary, replication], suite_marker, current="c" * 40
        )
