"""One MBPP-only dashboard preserves completions while showing every live task."""

import importlib.util
import re
import sys
from copy import deepcopy
from pathlib import Path

import pytest

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
    assert "Progress" in output and "계획" in output and "남음" in output and "Remarks" in output
    on_policy = next(line for line in output.splitlines() if line.startswith("On-policy · 선택비용 포함 "))
    assert re.search(r"선택비용 포함\s+48\s+21\s+27\s+43\.8%", on_policy)
    assert "CURRENT RUN 4" in output and "기록 미확인 48개" in output
    assert "총 계획 144개 | 완료 확인 21개 | 남음 123개" in output
    assert "CONTINUATIONS" not in output and "MOPPS" not in output and "MATH" not in output
    assert "ROOT " not in output
    assert output.count("FULL STATUS —") == 3
    assert len(re.findall(r"^\d\s*/\s*\d+\s", output, re.MULTILINE)) == 30
    for index in range(4):
        assert f"live-node-{index}" in output
    assert all(line.count("/48") <= 1 for line in output.splitlines())


def test_full_matrix_shows_each_completed_random_control_and_unfinished_arm(six_suites):
    roots, _, now = six_suites
    output = dashboard.render(dashboard.snapshot(mbpp_roots(roots), now=now))
    section = output.split("FULL STATUS — On-policy · 선택비용 포함", 1)[1].split("FULL STATUS — On-policy · 선택비용 별도", 1)[0]
    rows = [line.split() for line in section.splitlines() if re.match(r"^\d\s*/\s*\d+\s", line)]
    assert len(rows) == 15
    assert all(row[4] == "DONE" and row[6] == "DONE" for row in rows)
    held_out = [row for row in rows if row[3] == "검증"]
    assert len(held_out) == 6 and all(row[8] == "DONE" for row in held_out)
    assert all(row[5] != "DONE" and row[9] != "DONE" for row in rows)
    assert "READY" in section and "WAIT" in section


def test_all_45_state_rows_are_visible_by_default_for_three_prepared_suites(six_suites):
    from test_selection_switch_status import prepared

    roots, _, now = six_suites
    wanted = mbpp_roots(roots)
    prepared(wanted[2])
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in wanted[0].parent.rglob("*") if p.is_file()}
    output = dashboard.render(dashboard.snapshot(wanted, now=now))
    rows = re.findall(r"^(\d)\s*/\s*(\d+)\s", output, re.MULTILINE)
    assert len(rows) == 45
    for seed in range(5):
        for step in (25, 50, 100):
            assert rows.count((str(seed), str(step))) == 3
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
    section = output.split("FULL STATUS — On-policy · 선택비용 포함", 1)[1].split("FULL STATUS — On-policy · 선택비용 별도", 1)[0]
    joined = " ".join(section.split())
    assert all(dashboard.REMARKS[state] in joined for state in ("EVAL", "RESUME", "FAILED", "REVIEW"))
    assert "WAIT" in section
    assert not re.search(r"\b(?:EVAL|RESUME|FAIL|FAILED|REVIEW)\b", section)


def test_all_shows_exact_roots_and_status_never_mutates_files(six_suites):
    roots, _, now = six_suites
    wanted = mbpp_roots(roots)
    runs = wanted[0].parent
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in runs.rglob("*") if p.is_file()}
    output = dashboard.render(dashboard.snapshot(wanted, now=now), all_tasks=True, width=200)
    assert all(f"ROOT {root}" in output for root in wanted)
    assert "DONE states/" in output and "RUN states/" in output
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
    assert "Difficulty · 선택비용 포함: 설정 읽기 실패: fixture broken manifest" in output


def test_cli_accepts_repeated_roots_and_missing_suite_without_initializing(six_suites, monkeypatch, capsys):
    roots, _, _ = six_suites
    wanted = mbpp_roots(roots)
    args = ["mbpp_status"]
    for root in wanted:
        args += ["--root", str(root)]
    monkeypatch.setattr(sys, "argv", args)
    assert dashboard.main() == 0
    assert "계획 48개 | 완료 확인 0/48 | 남음 48개" in capsys.readouterr().out
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
    assert "실험 설정 확인 불가" in output and "broken: 설정 읽기 실패: unreadable manifest" in output
    assert "총 계획 96개 | 완료 확인 0개 | 남음 96개" in output


def test_only_four_display_statuses_full_names_and_evaluation_budget_remarks(six_suites):
    roots, _, now = six_suites
    report = dashboard.snapshot(mbpp_roots(roots), now=now)
    branches = [task for task in report["suites"][0]["tasks"] if task["kind"] == "branch"]
    for task, state in zip(branches, ("EVAL", "BUDGET", "RESUME", "FAILED", "REVIEW", "INVALID", "SAVING", "STALE")):
        task["status"] = state
    before = deepcopy(report)
    output = dashboard.render(report, width=200)
    assert all(name in output for name in ("Selection", "Random", "Full selection", "Full random", "Gate policy"))
    assert "평가·결과 저장 남음" in output and "GPU 시간 한도 도달; 결과 미완료" in output
    assert "Remarks" in output and "WAIT" in output
    assert not re.search(r"\b(?:SEL|RND|FULL-S|FULL-R|DEV|TEST|EVAL|RESUME|FAIL|FAILED|REVIEW|INVALID|SAVING|STALE|BUDGET|RUNNING)\b", output)
    assert report == before


def test_progress_is_completed_fraction_not_elapsed_or_timeout(six_suites):
    roots, _, now = six_suites
    report = dashboard.snapshot(mbpp_roots(roots), now=now)
    for elapsed in (1, 1000000):
        for suite in report["suites"]:
            for task in suite.get("tasks", []):
                task.update(seconds=elapsed, timeout=10)
        output = dashboard.render(report)
        summary = next(line for line in output.splitlines() if line.startswith("On-policy · 선택비용 포함 "))
        assert re.search(r"선택비용 포함\s+48\s+21\s+27\s+43\.8%", summary)
    assert dashboard.completion(report["suites"][0]) == ("43.8%", 21, 48)


def test_full_korean_remarks_wrap_without_exceeding_terminal_columns(six_suites):
    roots, _, now = six_suites
    report = dashboard.snapshot(mbpp_roots(roots), now=now)
    branches = [task for task in report["suites"][0]["tasks"] if task["kind"] == "branch"]
    branches[0]["status"] = "EVAL"
    branches[1]["status"] = "BUDGET"
    for width in (80, 100, 120):
        output = dashboard.render(report, width=width)
        assert all(dashboard.columns(line) <= width for line in output.splitlines())
        assert len(re.findall(r"^\d\s*/\s*\d+\s", output, re.MULTILINE)) == 30
    for remark in ("평가·결과 저장 남음", "GPU 시간 한도 도달; 결과 미완료"):
        wrapped = dashboard.wrap(remark, 8)
        assert "".join(wrapped).replace(" ", "") == remark.replace(" ", "")
        assert all(dashboard.columns(line) <= 8 for line in wrapped)


def test_condition_names_show_cost_accounting_without_renaming_saved_roots(six_suites):
    roots, _, now = six_suites
    report = dashboard.snapshot(mbpp_roots(roots), now=now)
    before = deepcopy(report)
    output = dashboard.render(report, width=160)
    for name in ("On-policy · 선택비용 포함", "On-policy · 선택비용 별도", "Difficulty · 선택비용 포함"):
        assert f"FULL STATUS — {name}" in output
    assert "quality" not in output and "fresh_r" not in output
    assert "별도도 총 GPU 비용에는 포함" in output
    assert "보상 0점이 아닙니다" in output
    assert report == before


def test_custom_root_uses_frozen_selector_accounting_and_gate_metadata(tmp_path):
    import selection_gate as core
    from test_selection_switch_status import prepared

    root = tmp_path / "unchanged-custom-storage"
    prepared(root)
    protocol = core.read(root / "switch.json")
    protocol.update(dataset="mbpp", selector="fresh_r", accounting="matched", gate="convergence")
    core.atomic_json(root / "switch.json", protocol)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    report = dashboard.snapshot([root])
    output = dashboard.render(report, width=160)
    assert "FULL STATUS — On-policy · 선택비용 별도" in output
    assert "Gate policy 판단: 비용 보정 학습 효율 기준" in output
    assert report["suites"][0]["protocol"] == {
        "dataset": "mbpp", "selector": "fresh_r", "accounting": "matched", "gate": "convergence"}
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}


def test_counts_show_plan_done_remaining_for_each_condition_including_missing(six_suites):
    roots, _, now = six_suites
    report = dashboard.snapshot(mbpp_roots(roots), now=now)
    before = deepcopy(report)
    values = [dashboard.counts(suite) for suite in report["suites"]]
    assert [(value["planned"], value["done"], value["remaining"], value["unknown"])
            for value in values] == [(48, 21, 27, 0), (48, 0, 48, 0), (48, 0, 48, 48)]
    output = dashboard.render(report, width=160)
    assert "남음 123개 = RUN 4개 + READY 59개 + WAIT 60개" in output
    assert "총 계획 144개 | 완료 확인 21개 | 남음 123개" in output
    assert "준비 전" not in output
    assert report == before
    for value in values:
        assert value["remaining"] == sum(value["states"][key] for key in ("READY", "RUN", "WAIT"))


@pytest.mark.parametrize("state", ["BUDGET", "EVAL", "RESUME", "FAILED"])
def test_unfinished_result_states_never_count_as_done(six_suites, state):
    roots, _, now = six_suites
    report = dashboard.snapshot(mbpp_roots(roots)[:1], now=now)
    suite = report["suites"][0]
    task = next(task for task in suite["tasks"] if task["kind"] == "branch" and task["status"] == "DONE")
    task["status"] = state
    count = dashboard.counts(suite)
    assert (count["planned"], count["done"], count["remaining"]) == (48, 20, 28)
    assert "총 계획 48개 | 완료 확인 20개 | 남음 28개" in dashboard.render(report)


def test_registered_slots_prevent_partial_duplicate_and_phase_counts_from_inflating_progress():
    task = {"kind": "branch", "seed": 0, "step": 25, "arm": "random_reduced", "status": "DONE"}
    extras = [{**task, "kind": "prefix"}, {**task, "kind": "phase"},
              {**task, "seed": 100}, {**task, "arm": "gated"}, {**task, "step": 76}]
    for tasks in ([task], [task] * 48, [task, *extras]):
        count = dashboard.counts({"tasks": tasks})
        assert (count["planned"], count["done"], count["remaining"], count["unknown"]) == (48, 1, 47, 47)
        assert count["progress"] == "2.1%" and count["states"]["WAIT"] == 47
    conflict = dashboard.counts({"tasks": [task, {**task, "status": "FAILED"}]})
    assert conflict["done"] == 0 and conflict["unknown"] == 48


def test_unreadable_record_shows_numeric_unverified_counts_not_a_reset():
    suite = {"root": "/broken", "prepared": False, "error": "fixture unreadable"}
    output = dashboard.render({"updated": 1000, "suites": [suite]}, width=160)
    assert "총 계획 48개 | 완료 확인 0개 | 남음 48개" in output
    assert "기록 미확인 48개" in output and "완료 여부 미확인" in output
    assert "기록 없음은 삭제·미실행의 증거가 아닙니다" in output
    assert "준비 전" not in output


def test_repeated_root_is_counted_once_in_summary(six_suites):
    roots, _, now = six_suites
    root = mbpp_roots(roots)[0]
    report = dashboard.snapshot([root, root / ".", str(root)], now=now)
    assert len(report["suites"]) == 1
    assert "총 계획 48개 | 완료 확인 21개 | 남음 27개" in dashboard.render(report)


def test_retained_quality_workers_remain_visible_without_inflating_new_plan(six_suites):
    roots, _, now = six_suites
    base = roots["selection-switch-mbpp-v1"]
    quality = roots["selection-switch-mbpp-quality-v1"]
    wanted = [base, base.with_name("selection-switch-mbpp-difficulty-v1"), base.with_name("selection-switch-mbpp-long-v1")]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in base.parent.rglob("*") if p.is_file()}
    report = dashboard.snapshot(wanted, retained_roots=[quality], now=now)
    output = dashboard.render(report, width=200)
    assert len(report["suites"]) == 3 and len(report["retained_suites"]) == 1
    assert "총 계획 144개 | 완료 확인 21개 | 남음 123개" in output
    assert "총 계획 192" not in output
    assert "FULL STATUS — On-policy · 장시간 예산" in output
    assert "기본 실행 제외 — 기존 기록 보존" in output and "CURRENT RUN 4" in output
    assert "On-policy · 선택비용 별도 (기본 실행 제외)" in output
    for index in range(4):
        assert f"live-node-{index}" in output
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in base.parent.rglob("*") if p.is_file()}


def test_retained_absent_or_already_selected_roots_are_not_double_counted(six_suites):
    roots, _, now = six_suites
    base = roots["selection-switch-mbpp-v1"]
    report = dashboard.snapshot([base], retained_roots=[base, base.with_name("missing")], now=now)
    assert not report.get("retained_suites")
    assert "총 계획 48개 | 완료 확인 21개 | 남음 27개" in dashboard.render(report)


def test_mbpp_training_comparison_shows_48_and_keeps_old_21_done_visible(six_suites):
    roots, _, now = six_suites
    base = roots["selection-switch-mbpp-v1"]
    quality = roots["selection-switch-mbpp-quality-v1"]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in base.parent.rglob("*") if p.is_file()}
    report = dashboard.snapshot([quality], retained_roots=[base], now=now)
    report['suites'][0]['protocol'].update(accounting='matched', budget_gpu_seconds=28380)
    output = dashboard.render(report, width=200)
    assert "총 계획 48개 | 완료 확인 0개 | 남음 48개" in output
    assert "다른 조건의 완료 결과 21개 보존" in output
    assert "학습 한도 공통; 선택 비용 별도 기록" in output
    assert "진단·학습 시간 한도: 28380 GPU-seconds" in output
    assert "선택 비용은 위 학습 한도에서 차감하지 않으며" in output
    assert "CURRENT RUN 4" in output
    assert "On-policy · 선택비용 포함 (기본 실행 제외)" not in output  # No live legacy branch in this fixture.
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in base.parent.rglob("*") if p.is_file()}
