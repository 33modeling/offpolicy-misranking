"""One MBPP-only dashboard preserves completions while showing every live task."""

import importlib.util
import re
import sys
from pathlib import Path

from test_status_saved_random_integration import six_suites as suite_fixture

six_suites = suite_fixture
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mbpp_status.py"
SPEC = importlib.util.spec_from_file_location("mbpp_status_dashboard", SCRIPT)
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def mbpp_roots(roots):
    return [roots["selection-switch-mbpp-v1"], roots["selection-switch-mbpp-quality-v1"],
            roots["selection-switch-mbpp-v1"].with_name("selection-switch-mbpp-difficulty-v1")]


def test_three_suites_have_one_summary_full_matrices_and_one_run_list(six_suites):
    roots, _, now = six_suites
    report = dashboard.snapshot(mbpp_roots(roots), now=now)
    output = dashboard.render(report)
    assert output.count("MBPP EXPERIMENTS") == 1 and output.count("CURRENT RUN") == 1
    assert "DONE/TOTAL" in output and "LEFT" in output and "PREFIX" in output
    on_policy = next(line for line in output.splitlines() if line.startswith("on-policy "))
    assert "21/48" in on_policy and "27" in on_policy and "15/15" in on_policy
    assert "CURRENT RUN 4" in output and "difficulty: NOT PREPARED" in output
    assert "CONTINUATIONS" not in output and "MOPPS" not in output and "MATH" not in output
    assert "ROOT " not in output
    assert output.count("FULL STATUS —") == 3
    assert len(re.findall(r"^s\d/t\d+\s", output, re.MULTILINE)) == 30
    for index in range(4):
        assert f"live-node-{index}" in output
    assert all(line.count("/48") <= 1 for line in output.splitlines())


def test_full_matrix_shows_each_completed_random_control_and_unfinished_arm(six_suites):
    roots, _, now = six_suites
    output = dashboard.render(dashboard.snapshot(mbpp_roots(roots), now=now))
    section = output.split("FULL STATUS — on-policy", 1)[1].split("FULL STATUS — quality", 1)[0]
    rows = [line.split() for line in section.splitlines() if re.match(r"^s\d/t\d+\s", line)]
    assert len(rows) == 15
    assert all(row[2] == "DONE" and row[4] == "DONE" for row in rows)
    held_out = [row for row in rows if row[1] == "TEST"]
    assert len(held_out) == 6 and all(row[6] == "DONE" for row in held_out)
    assert all(row[3] != "DONE" and row[7] != "DONE" for row in rows)
    assert "READY" in section and "WAIT" in section


def test_all_45_state_rows_are_visible_by_default_for_three_prepared_suites(six_suites):
    from test_selection_switch_status import prepared

    roots, _, now = six_suites
    wanted = mbpp_roots(roots)
    prepared(wanted[2])
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in wanted[0].parent.rglob("*") if p.is_file()}
    output = dashboard.render(dashboard.snapshot(wanted, now=now))
    rows = re.findall(r"^s\d/t\d+\s.*$", output, re.MULTILINE)
    assert len(rows) == 45
    for seed in range(5):
        for step in (25, 50, 100):
            assert sum(row.startswith(f"s{seed}/t{step} ") for row in rows) == 3
    assert all(status in output for status in ("DONE", "RUN", "READY", "WAIT"))
    assert output.count("CURRENT RUN 4") == 1
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in wanted[0].parent.rglob("*") if p.is_file()}


def test_matrix_preserves_evaluation_resume_and_failure_states(six_suites):
    roots, _, now = six_suites
    report = dashboard.snapshot(mbpp_roots(roots), now=now)
    branches = [task for task in report["suites"][0]["tasks"] if task["kind"] == "branch"]
    for task, state in zip(branches, ("EVAL", "RESUME", "FAILED", "REVIEW")):
        task["status"] = state
    output = dashboard.render(report)
    section = output.split("FULL STATUS — on-policy", 1)[1].split("FULL STATUS — quality", 1)[0]
    assert all(state in section for state in ("EVAL", "RESUME", "FAIL", "REVIEW"))


def test_all_shows_exact_roots_and_status_never_mutates_files(six_suites):
    roots, _, now = six_suites
    wanted = mbpp_roots(roots)
    runs = wanted[0].parent
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in runs.rglob("*") if p.is_file()}
    output = dashboard.render(dashboard.snapshot(wanted, now=now), all_tasks=True, width=200)
    assert all(f"ROOT {root}" in output for root in wanted)
    assert "DONE states/" in output and "RUNNING states/" in output
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in runs.rglob("*") if p.is_file()}


def test_unreadable_suite_does_not_hide_other_completions_or_workers(six_suites, monkeypatch):
    roots, _, now = six_suites
    wanted = mbpp_roots(roots)
    snapshot = dashboard.switch_status.snapshot

    def unreadable(root, **kwargs):
        if root == wanted[2]:
            raise ValueError("fixture broken manifest")
        return snapshot(root, **kwargs)

    monkeypatch.setattr(dashboard.switch_status, "snapshot", unreadable)
    output = dashboard.render(dashboard.snapshot(wanted, now=now))
    assert "21/48" in output and "CURRENT RUN 4" in output
    assert "difficulty: ERROR fixture broken manifest" in output


def test_cli_accepts_repeated_roots_and_missing_suite_without_initializing(six_suites, monkeypatch, capsys):
    roots, _, _ = six_suites
    wanted = mbpp_roots(roots)
    args = ["mbpp_status"]
    for root in wanted:
        args += ["--root", str(root)]
    monkeypatch.setattr(sys, "argv", args)
    assert dashboard.main() == 0
    assert "difficulty: NOT PREPARED" in capsys.readouterr().out
    assert not wanted[2].exists()


def test_dashboard_does_not_query_local_gpu_driver(six_suites, monkeypatch):
    roots, _, now = six_suites

    def forbidden():
        raise AssertionError("MBPP status must not wait for the local GPU driver")

    monkeypatch.setattr(dashboard.switch_status.node_view, "local_gpus", forbidden)
    output = dashboard.render(dashboard.snapshot(mbpp_roots(roots), now=now))
    assert "21/48" in output and "CURRENT RUN 4" in output


def test_cli_reports_unreadable_root_as_failure_without_hiding_other_roots(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mbpp_status", "--root", "/missing", "--root", "/broken"])
    monkeypatch.setattr(dashboard, "snapshot", lambda roots: {
        "updated": 1000,
        "suites": [{"root": "/missing", "prepared": False},
                   {"root": "/broken", "prepared": False, "error": "unreadable manifest"}],
    })
    assert dashboard.main() == 1
    output = capsys.readouterr().out
    assert "missing: NOT PREPARED" in output and "broken: ERROR unreadable manifest" in output
