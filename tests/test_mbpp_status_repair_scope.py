"""Default MBPP status follows a verified repair without scheduling another plan."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_mbpp_status_dashboard import dashboard
from test_selection_switch_status import (
    completed_prefix, convergence_root, point, published, published_curve, running,
)

REPO = Path(__file__).resolve().parents[1]


def original_root(work):
    root = work / "runs/selection-switch-mbpp-quality-v1"
    convergence_root(root)
    path = root / "switch.json"
    protocol = json.loads(path.read_text())
    protocol.update(dataset="mbpp", accounting="matched", selector="fresh_r")
    path.write_text(json.dumps(protocol))
    completed_prefix(root)
    published(point(root) / "random_reduced")
    published_curve(point(root) / "random_reduced")
    return root


def repair_root(source, target=None):
    target = target or source.with_name("selection-switch-mbpp-quality-repair-v1")
    shutil.copytree(source, target)
    receipt = {"schema": "mbpp-repair/v1", "source_root": str(source), "root": str(target),
               "source_switch_sha256": hashlib.sha256((source / "switch.json").read_bytes()).hexdigest()}
    (target / "repair.json").write_text(json.dumps(receipt))
    return target


def status(work, *options, **overrides):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("SWITCH_MBPP_") and key != "MBPP_REPAIR_ROOT"}
    env.update(OM_WORK=str(work), SWITCH_PYTHON=sys.executable, **overrides)
    return subprocess.run(["bash", "scripts/run_mbpp_experiments.sh", "status", *options],
                          cwd=REPO, env=env, text=True, capture_output=True, timeout=15)


@pytest.mark.parametrize("custom", [False, True])
def test_default_bash_status_selects_repair_keeps_history_and_one_plan(tmp_path, custom):
    work = tmp_path / "work"
    source = original_root(work)
    repair = repair_root(source, work / "runs/custom-repair" if custom else None)
    running(point(repair) / "selection_reduced", "repair-active-node", now=time.time(), phase="train")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in work.rglob("*") if path.is_file()}
    result = status(work, **({"MBPP_REPAIR_ROOT": str(repair)} if custom else {}))
    assert result.returncode == 0, result.stdout + result.stderr
    text = result.stdout
    assert "현재 조회 범위 1개 조건 | 총 계획 48개 | 완료 확인 1개 | 남음 47개" in text
    assert "현재 실행: 분기 RUN 1개" in text and "repair-active-node" in text
    assert "기존 계획 48개 | 완료 확인 1개 | 남음 47개" in text
    assert "다른 조건의 완료 결과 1개 보존" in text
    assert "조회 합계 제외" in text and "기본 실행 제외" not in text
    compact = "".join(text.split())
    assert "복구조회:" + str(repair) in compact
    assert "원본기록:" + str(source) in compact
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in work.rglob("*") if path.is_file()}


def test_original_only_and_explicit_quality_status_keep_original_scope(tmp_path):
    work = tmp_path / "work"
    source = original_root(work)
    running(point(source) / "selection_reduced", "original-active-node", now=time.time(), phase="train")
    result = status(work)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "현재 실행: 분기 RUN 1개" in result.stdout and "original-active-node" in result.stdout
    repair = repair_root(source)
    running(point(repair) / "selection_reduced", "repair-active-node", now=time.time(), phase="train")
    result = status(work, "quality")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "original-active-node" in result.stdout and "repair-active-node" not in result.stdout
    assert "기존 계획" not in result.stdout


@pytest.mark.parametrize("broken", ["schema", "source", "root", "digest", "source_changed",
                                     "clone_changed", "missing_switch", "json", "array"])
def test_invalid_repair_identity_never_replaces_original_scope(tmp_path, broken):
    source = original_root(tmp_path)
    repair = repair_root(source)
    path = repair / "repair.json"
    receipt = json.loads(path.read_text())
    if broken in {"schema", "source", "root", "digest"}:
        key = {"schema": "schema", "source": "source_root", "root": "root",
               "digest": "source_switch_sha256"}[broken]
        receipt[key] = "wrong"
        path.write_text(json.dumps(receipt))
    elif broken in {"source_changed", "clone_changed"}:
        manifest = (source if broken == "source_changed" else repair) / "switch.json"
        manifest.write_text(manifest.read_text() + " ")
    elif broken == "missing_switch":
        (repair / "switch.json").unlink()
    else:
        path.write_text("{" if broken == "json" else "[]")
    report = dashboard.snapshot([source], repair_root=repair)
    assert [suite["root"] for suite in report["suites"]] == [str(source)]
    assert not report.get("retained_suites")


def test_repair_cannot_be_original_or_contain_it(tmp_path):
    source = original_root(tmp_path)
    for candidate in (source, source.parent, source / "nested"):
        assert dashboard.verified_repair_root(source, candidate) is None


@pytest.mark.parametrize("location", ["candidate", "source_root", "root"])
def test_cyclic_repair_identity_paths_preserve_original_scope(tmp_path, location):
    source = original_root(tmp_path)
    repair = repair_root(source)
    loop = tmp_path / "cyclic-root"
    loop.symlink_to(loop.name)
    if location == "candidate":
        repair = loop
    else:
        path = repair / "repair.json"
        receipt = json.loads(path.read_text())
        receipt[location] = str(loop)
        path.write_text(json.dumps(receipt))
    report = dashboard.snapshot([source], repair_root=repair)
    assert [suite["root"] for suite in report["suites"]] == [str(source)]
    assert not report.get("retained_suites") and "repair_source" not in report


def test_repair_view_labels_original_live_work_as_excluded_from_totals(tmp_path):
    source = original_root(tmp_path)
    repair = repair_root(source)
    running(point(source) / "selection_reduced", "original-active-node", now=time.time(), phase="train")
    report = dashboard.snapshot([source], repair_root=repair)
    output = dashboard.render(report, width=300)
    assert "On-policy · 선택비용 별도 (조회 합계 제외)" in output
    assert "기본 실행 제외" not in output and "original-active-node" in output
