"""MBPP's three status blocks must show saved work and current workers together."""

import sys

import pytest
from test_status_saved_random_integration import six_suites as suite_fixture
from test_status_saved_random_integration import switch_status as status

six_suites = suite_fixture


def test_compact_on_policy_keeps_21_random_results_visible(six_suites):
    roots, _, now = six_suites
    root = roots["selection-switch-mbpp-v1"]
    output = status.render_compact(status.snapshot(root, now=now))
    assert "SUITE MBPP on-policy" in output and f"ROOT {root}" in output
    assert "DONE 21/48 branches" in output
    assert "TRAINING RESULTS  21/48 published" in output
    assert "RF DONE 6/6" in output and "RR DONE 15/15" in output
    assert "CURRENT RUN 0" in output and "No current RUN heartbeat" in output
    assert "MOPPS" not in output and "MATH" not in output
    assert len(output.splitlines()) < 15


def test_compact_quality_shows_every_current_worker_and_training_step(six_suites):
    roots, _, now = six_suites
    data = status.snapshot(roots["selection-switch-mbpp-quality-v1"], now=now)
    current = [task for task in data["tasks"] if task["status"] == "RUNNING"]
    for index, task in enumerate(current):
        task["training_step"] = 120 + index
    output = status.render_compact(data)
    assert "CURRENT RUN 4" in output and "RUNNING 4" in output
    assert "TRAINING RESULTS  0/48 published" in output
    for index, task in enumerate(current):
        assert task["host"] in output and f"step={120 + index}" in output
    rows = [line for line in output.splitlines() if line.startswith("RUN ")]
    assert len(rows) == 4
    assert all("random_reduced" in row and "phase=train" in row and "elapsed=2m03s" in row for row in rows)
    assert rows == sorted(rows)


def test_compact_saved_final_and_checkpoint_counts_are_not_ready(six_suites):
    roots, _, now = six_suites
    data = status.snapshot(roots["selection-switch-long-v1"], now=now)
    output = status.render_compact(data)
    assert "EVAL 2" in output and "RESUME 2" in output
    assert "HISTORY 1" in output


def test_missing_suite_is_explicit_not_a_fresh_ready_grid(tmp_path):
    output = status.render_compact({"prepared": False, "root": str(tmp_path / "selection-switch-mbpp-difficulty-v1")})
    assert "NOT PREPARED" in output and "mbpp-difficulty" in output
    assert "READY" not in output and "DONE 0" not in output
    assert len(output.splitlines()) == 3


def test_compact_attention_is_bounded_and_current_work_stays_above_it(six_suites):
    roots, _, now = six_suites
    data = status.snapshot(roots["selection-switch-mbpp-quality-v1"], now=now)
    data["notices"] = [{"path": f"old-{index}", "error": "historical context " * 20} for index in range(7)]
    output = status.render_compact(data, width=90)
    assert sum(line.startswith("ATTENTION ") for line in output.splitlines()) == 3
    assert "4 additional attention items" in output
    assert output.index("CURRENT RUN 4") < output.index("ATTENTION")
    assert all(len(line) <= 90 for line in output.splitlines())


@pytest.mark.parametrize("arguments,compact", [([], True), (["--all"], False)])
def test_cli_uses_compact_only_with_mbpp_marker_and_without_all(six_suites, monkeypatch, capsys, arguments, compact):
    roots, _, now = six_suites
    root = roots["selection-switch-mbpp-v1"]
    data = status.snapshot(root, now=now)
    monkeypatch.setattr(status, "snapshot", lambda _: data)
    monkeypatch.setenv("SWITCH_STATUS_COMPACT", "1")
    monkeypatch.setattr(sys, "argv", ["status", "--root", str(root), *arguments])
    assert status.main() == 0
    output = capsys.readouterr().out
    assert ("SUITE MBPP on-policy" in output) is compact
    assert ("CONTINUATIONS" in output) is not compact
