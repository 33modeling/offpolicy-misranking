"""CPU-only checks for the read-only Pair cost export arithmetic."""
import pytest

import selector_pair_adaptive_costs as export


def test_reporting_selection_is_included_but_evaluation_is_not():
    closed = [(None, {"phase": "train", "ledger": "deployment", "allocated_gpu_seconds": 80.}),
              (None, {"phase": "fresh-r-candidate", "ledger": "reporting",
                      "allocated_gpu_seconds": 12.}),
              (None, {"phase": "evaluate", "ledger": "reporting",
                      "allocated_gpu_seconds": 40.})]
    assert export.inclusive_branch_seconds(closed, {"fresh-r-candidate"}) == 92.


def test_missing_endpoint_is_not_zero_cost(tmp_path):
    entry = (tmp_path, tmp_path, {}, {}, {})
    row = export.measure(entry, "selection_full", "sr_gc", .35, 3600.)
    assert row["reward_percent"] is None
    assert row["inclusive_endpoint_gpu_hours"] is None
    assert row["inclusive_target_gpu_hours"] is None
    assert row["reason"] == "endpoint_result_missing"


def test_endpoint_cost_adds_diagnosis_once_without_claiming_target_time(tmp_path, monkeypatch):
    directory = tmp_path / "selection_full"
    directory.mkdir()
    (directory / "result.json").write_text("{}")
    monkeypatch.setattr(export.pair_gpu.switch.runtime, "validate_result",
                        lambda *_: {"complete": True, "rewards": {"a": .4, "b": .2}})
    monkeypatch.setattr(export.pair_gpu.base, "read_cost_events", lambda _: ({}, []))
    monkeypatch.setattr(export.pair_gpu.pair, "finished_events", lambda _: [
        (None, {"phase": "train", "ledger": "deployment", "allocated_gpu_seconds": 80.}),
        (None, {"phase": "fresh-r-candidate", "ledger": "reporting",
                "allocated_gpu_seconds": 12.}),
        (None, {"phase": "evaluate", "ledger": "reporting",
                "allocated_gpu_seconds": 40.})])
    monkeypatch.setattr(export.pair_gpu.switch, "SCORING_PHASES", ("fresh-r-candidate",))
    row = export.measure((tmp_path, tmp_path, {}, {}, {}), "selection_full", "sr_gc",
                         .35, 3600.)
    assert row["reward_percent"] == pytest.approx(30.)
    assert row["branch_gpu_hours"] == 92 / 3600
    assert row["inclusive_endpoint_gpu_hours"] == 1 + 92 / 3600
    assert row["inclusive_target_gpu_hours"] is None
    assert row["reason"] == "curve_summary_missing"
