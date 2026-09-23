import pytest
import json

import selector_pair_srgc_temporal as temporal


def series(*values):
    return [{"step": 25 * (i + 1), "d": value, "status": "measured"}
            for i, value in enumerate(values)]


def report(*values):
    points = series(*values)
    return {"schema": temporal.SOURCE_SCHEMA, "status": "complete", "interval": 25,
            "trajectories": [{"state": "s3-t25", "scheduled_steps": [p["step"] for p in points],
                              "points": points, "pending": [], "errors": []}]}


def test_two_consecutive_negative_checks_signal_without_third():
    row = temporal.evaluate(report(2., -3., -1., 4.))["trajectories"][0]
    assert row["signal_step"] == 75
    assert [p["action"] for p in row["checks"]] == [
        "retain_on", "candidate", "switch_signal", "post_signal_diagnostic"]
    assert row["checks"][2]["window_mean"] == -2.
    assert row["executed_switch"] is False and row["switched_policy_rewards"] is None


def test_positive_bounce_needs_negative_third_and_negative_mean():
    row = temporal.evaluate(report(-4., 1., -2.))["trajectories"][0]
    assert row["signal_step"] == 75
    assert row["checks"][-1]["window_mean"] == pytest.approx(-5 / 3)
    row = temporal.evaluate(report(-1., 10., -1., -2.))["trajectories"][0]
    assert row["signal_step"] == 100
    assert [p["action"] for p in row["checks"]] == [
        "candidate", "await_third_check", "candidate", "switch_signal"]


def test_positive_third_cancels_candidate_and_zero_is_nonnegative():
    row = temporal.evaluate(report(-1., 0., 2., -3.))["trajectories"][0]
    assert row["signal_step"] is None and row["pending_confirmation"] is True
    assert [p["action"] for p in row["checks"]] == [
        "candidate", "await_third_check", "retain_on", "candidate"]


def test_partial_or_missing_d_never_produces_decision():
    partial = report(1., -2.)
    partial["status"] = "partial"
    with pytest.raises(ValueError, match="complete all-D"):
        temporal.evaluate(partial)
    complete = report(1., -2.)
    complete["trajectories"][0]["scheduled_steps"].append(75)
    with pytest.raises(ValueError, match="all scheduled"):
        temporal.evaluate(complete)
    with pytest.raises(ValueError, match="finite and consecutive"):
        temporal.decide([{"step": 25, "d": 1.}, {"step": 75, "d": -1.}], 25)


def test_crlf_export_is_read_without_changing_source_data():
    raw = b"HEADER\r\nDATA_JSON\r\n" + json.dumps(report(1., -2.)).encode() + b"\r\n"
    assert temporal.parse_export(raw) == report(1., -2.)
