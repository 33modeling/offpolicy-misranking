"""MBPP completion follows saved result/curve evidence, never elapsed time."""

import pytest

import selection_gate as core
from test_mbpp_status_dashboard import dashboard
from test_selection_switch_status import (
    completed_prefix, convergence_root, point, published, published_curve, running,
)


def inventory(root):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("arm", tuple(dashboard.ARM_NAMES))
def test_final_evaluation_then_curve_becomes_done_without_restart(tmp_path, arm):
    now = 2000000000
    convergence_root(tmp_path)
    completed_prefix(tmp_path, seed=3)
    directory = point(tmp_path, seed=3) / arm
    published(directory)
    core.atomic_json(directory / "failure.json", {"error": "historical failure", "time": now - 1000})
    running(directory, "node-finished-training", now=now - 300, phase="evaluate")
    before = inventory(tmp_path)

    report = dashboard.snapshot([tmp_path], now=now)
    task = next(t for t in report["suites"][0]["tasks"] if t["directory"] == str(directory.relative_to(tmp_path)))
    assert dashboard.display_state(task) == "WAIT"
    assert dashboard.remark(task) == "최종 평가 저장됨; 곡선 평가 남음 (재학습 없음)"
    assert "곡선 평가 남음 1개" in " ".join(dashboard.render(report, width=200).split())
    assert dashboard.counts(report["suites"][0])["done"] == 0
    assert inventory(tmp_path) == before

    running(directory / "curve", "node-curve", now=now, phase="curve")
    report = dashboard.snapshot([tmp_path], now=now)
    suite = report["suites"][0]
    task = next(t for t in suite["tasks"] if t["directory"] == str(directory.relative_to(tmp_path)))
    running_dirs = [t["directory"] for t in suite["tasks"] if dashboard.active(t)]
    assert dashboard.display_state(task, running_dirs) == "RUN"
    assert dashboard.counts(suite)["done"] == 0

    published_curve(directory)
    before = inventory(tmp_path)
    report = dashboard.snapshot([tmp_path], now=now)
    suite = report["suites"][0]
    task = next(t for t in suite["tasks"] if t["directory"] == str(directory.relative_to(tmp_path)))
    assert dashboard.display_state(task, running_dirs) == "DONE"
    assert dashboard.counts(suite)["done"] == 1
    assert "곡선 평가 남음" not in dashboard.remark(task)
    assert inventory(tmp_path) == before


def test_all_48_published_branches_count_done_without_model_or_restart(tmp_path):
    convergence_root(tmp_path)
    rule = dashboard.switch_status.rule
    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        for step in rule.STEPS:
            completed_prefix(tmp_path, seed=seed, step=step)
            for arm in rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS:
                directory = point(tmp_path, seed=seed, step=step) / arm
                published(directory)
                published_curve(directory)
                core.atomic_json(directory / "failure.json", {"error": "old failure"})
    before = inventory(tmp_path)
    report = dashboard.snapshot([tmp_path])
    count = dashboard.counts(report["suites"][0])
    assert (count["done"], count["remaining"], count["progress"]) == (48, 0, "100.0%")
    assert "총 계획 48개 | 완료 확인 48개 | 남음 0개" in dashboard.render(report)
    assert inventory(tmp_path) == before


def test_posthoc_recovery_is_visible_but_never_canonical_done(tmp_path):
    import hashlib
    convergence_root(tmp_path)
    manifest = core.read(tmp_path / "switch.json")
    core.atomic_json(tmp_path / "switch.json", {**manifest, "dataset": "mbpp"})
    completed_prefix(tmp_path, seed=3)
    directory = point(tmp_path, seed=3) / "random_full"
    result = directory / "budget-recovery/result.json"
    core.atomic_json(result, {"schema": "mbpp-budget-recovery/v1", "evaluation_complete": True,
                              "canonical_complete": False})
    core.atomic_json(result.with_suffix(".sha256.json"), {"sha256": hashlib.sha256(result.read_bytes()).hexdigest()})
    before = inventory(tmp_path)
    report = dashboard.snapshot([tmp_path])
    suite = report["suites"][0]
    task = next(t for t in suite["tasks"] if t["directory"] == str(directory.relative_to(tmp_path)))
    assert dashboard.display_state(task) == "WAIT"
    assert "복구 평가 저장됨" in dashboard.remark(task)
    assert not task["retryable"] and dashboard.counts(suite)["done"] == 0
    assert inventory(tmp_path) == before
