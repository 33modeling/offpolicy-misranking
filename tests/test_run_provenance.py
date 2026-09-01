"""Run provenance is immutable and partitioned by generation commit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_provenance import (
    RunProvenanceError,
    generation_commit,
    partition_by_generation,
    require_single_generation,
)


def make_run(root: Path, name: str, commit: str) -> Path:
    run = root / name
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"git": commit}))
    (run / "manifest.json").write_text(json.dumps({"git": commit}))
    (run / "score_protocol.json").write_text(
        json.dumps({"source_run_git": commit})
    )
    return run


def test_generation_provenance_matches_linked_artifacts(tmp_path: Path) -> None:
    commit = "a" * 40
    run = make_run(tmp_path, "run-a", commit)
    assert generation_commit(run) == commit

    (run / "manifest.json").write_text(json.dumps({"git": "b" * 40}))
    with pytest.raises(RunProvenanceError, match="manifest"):
        generation_commit(run)


def test_runs_are_partitioned_without_pooling_commits(tmp_path: Path) -> None:
    run_a = make_run(tmp_path, "run-a", "a" * 40)
    run_b = make_run(tmp_path, "run-b", "b" * 40)
    partitions = partition_by_generation([run_b, run_a])
    assert partitions == {"a" * 40: [run_a], "b" * 40: [run_b]}
    with pytest.raises(RunProvenanceError, match="mixed generation commits"):
        require_single_generation([run_a, run_b])
