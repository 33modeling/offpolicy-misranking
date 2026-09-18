"""One MBPP-only dashboard preserves completions while showing every live task."""

import importlib.util
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


def test_three_suites_have_one_summary_one_arm_table_and_one_run_list(six_suites):
    roots, _, now = six_suites
    report = dashboard.snapshot(mbpp_roots(roots), now=now)
    output = dashboard.render(report)
    assert output.count("MBPP EXPERIMENTS") == 1 and output.count("CURRENT RUN") == 1
    assert "DONE/TOTAL" in output and "LEFT" in output and "PREFIX" in output
    on_policy = next(line for line in output.splitlines() if line.startswith("on-policy "))
    assert "21/48" in on_policy and "27" in on_policy and "15/15" in on_policy
    assert "CURRENT RUN 4" in output and "difficulty: NOT PREPARED" in output
    assert "CONTINUATIONS" not in output and "MOPPS" not in output and "MATH" not in output
    assert "ROOT " not in output and len(output.splitlines()) <= 35
    for index in range(4):
        assert f"live-node-{index}" in output
    assert all(line.count("/48") <= 1 for line in output.splitlines())


def test_per_arm_progress_shows_which_random_controls_are_finished(six_suites):
    roots, _, now = six_suites
    output = dashboard.render(dashboard.snapshot(mbpp_roots(roots), now=now))
    rnd = next(line for line in output.splitlines() if line.startswith("RND "))
    full_random = next(line for line in output.splitlines() if line.startswith("FULL-R "))
    selection = next(line for line in output.splitlines() if line.startswith("SEL "))
    assert "15/15" in rnd and "0/15" in rnd
    assert "6/6" in full_random
    assert "0/15" in selection


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
