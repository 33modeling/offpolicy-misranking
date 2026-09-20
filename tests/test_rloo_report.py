"""Partial paper exports must not alter training or canonical completion."""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rloo_report as reporting
from test_rloo_experiment import fixture


@pytest.fixture
def measured(tmp_path, monkeypatch):
    out = tmp_path / "math500-d0/s0"
    out.mkdir(parents=True)
    (out / "experiment.json").write_text("{}")
    monkeypatch.setattr(reporting.experiment, "validate", lambda out: (
        {"eval_n": 8, "source": {"seed": 0, "drift": 0}}, {}))

    def rows(out, arm, shard):
        return [{"prompt_idx": i, "reward": {"before": 0., "random": .25,
                 "passrate_beta": .5, "fresh_r": .75}[arm]}
                for i in range(shard * 2, (shard + 1) * 2)]

    monkeypatch.setattr(reporting.experiment, "checked_rows", rows)
    return out


def seal(out, arm, shards=range(4)):
    directory = out / arm / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        (directory / f"shard-{shard}.done.json").write_text("{}")


def test_missing_before_keeps_cached_comparison(measured):
    seal(measured, "passrate_beta")
    seal(measured, "fresh_r")
    result = reporting.point_report(measured)
    assert result["status"] == "incomplete"
    assert result["missing_arms"] == ["before", "random"]
    fresh = result["rows"][-1]
    assert fresh["vs_passrate_beta"]["mean"] == .25
    assert "vs_before" not in fresh and "vs_random" not in fresh
    assert fresh["missing_references"] == ["before", "random"]
    assert not (measured / "results.json").exists()


def test_partial_shards_are_explicit_and_unsealed_rows_ignored(measured):
    seal(measured, "before", [0])
    (measured / "before/evaluation/shard-1.jsonl").write_text("unfinished")
    result = reporting.point_report(measured)
    partial = result["evaluations"][0]
    assert partial["measured_prompts"] == 2
    assert partial["observed_mean_reward"] == 0.
    assert partial["missing_shards"] == [1, 2, 3]
    assert not partial["complete"] and result["rows"] == []
    assert result["evaluations"][1]["observed_mean_reward"] is None


def test_complete_matches_canonical_report(measured):
    for arm in ("before", *reporting.experiment.ARMS):
        seal(measured, arm)
    expected = reporting.experiment.report(measured)
    before = (measured / "results.json").read_bytes()
    actual = reporting.point_report(measured)
    assert actual["status"] == "complete"
    assert [{k: v for k, v in row.items() if k != "missing_references"}
            for row in actual["rows"]] == expected["rows"]
    assert (measured / "results.json").read_bytes() == before


def test_invalid_point_does_not_hide_other_points(measured, monkeypatch):
    other = measured.parent / "s1"
    seal(other, "random")
    (other / "experiment.json").write_text("{}")
    seal(measured, "fresh_r")
    original = reporting.experiment.checked_rows

    def checked(out, arm, shard):
        if out == measured:
            raise ValueError("evaluation seal mismatch")
        return original(out, arm, shard)

    monkeypatch.setattr(reporting.experiment, "checked_rows", checked)
    result = reporting.report(measured.parent.parent)
    assert result["points"][0]["status"] == "invalid"
    assert result["points"][0]["rows"] == []
    assert result["points"][1]["rows"][0]["arm"] == "random"
    assert result["points"][2]["status"] == "unprepared"
    assert not result["complete"]


def test_real_prepared_point_without_evaluation_is_exportable(tmp_path):
    _, out, _ = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    result = reporting.point_report(out)
    assert result["missing_arms"] == ["before", *reporting.experiment.ARMS]
    assert result["rows"] == []
    assert before == {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_launcher_exports_incomplete_matrix_without_gpu(tmp_path):
    import os
    root = tmp_path / "rloo"
    root.mkdir()
    process = subprocess.run(["bash", "scripts/run_rloo.sh", "report"],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True,
        env={**os.environ, "HOME": str(tmp_path), "RLOO_ROOT": str(root), "RLOO_PYTHON": sys.executable})
    assert process.returncode == 0, process.stderr
    exports = list(tmp_path.glob("rloo-results.txt"))
    assert len(exports) == 1
    assert "unprepared" in exports[0].read_text()
    assert "TXT saved:" in process.stdout


def test_missing_root_is_not_created(tmp_path):
    with pytest.raises(ValueError, match="no RLOO root"):
        reporting.report(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()
