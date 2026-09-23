"""The shared MBPP/Pair/RLOO renderer follows the real terminal width (phones: 60-79 columns)."""

import importlib.util
from pathlib import Path

import pytest

import selection_gate as core

from test_rloo_status import prepared as rloo_prepared
from test_status_fix_mbpp import NOW, branch, mbpp_root, published
from test_rloo_status import status as rloo
from test_status_saved_random_integration import six_suites as suite_fixture

six_suites = suite_fixture
rloo_root = rloo_prepared
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mbpp_status_layout", ROOT / "scripts/mbpp_status.py")
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)
PAIR = importlib.util.spec_from_file_location("pair_status_layout", ROOT / "scripts/selector_pair_status.py")
pair = importlib.util.module_from_spec(PAIR)
PAIR.loader.exec_module(pair)


def mbpp_report(roots, now):
    return dashboard.snapshot([roots["selection-switch-mbpp-v1"], roots["selection-switch-mbpp-quality-v1"]], now=now)


def fits(output, width):
    return max(dashboard.columns(line) for line in output.splitlines()) <= width


@pytest.mark.parametrize("width", [40, 60, 70, 79])
def test_mbpp_phone_width_is_honoured_with_stacked_blocks(six_suites, width):
    roots, _, now = six_suites
    report = mbpp_report(roots, now)
    output = dashboard.render(report, width=width, all_tasks=True)
    assert fits(output, width) and output != dashboard.render(report, width=80, all_tasks=True)
    lines = output.splitlines()
    # One suite per block: numbers and Remarks stay labelled, never interleaved columns.
    block = lines.index("On-policy · 선택비용 포함")
    assert lines[block + 1].startswith("  계획 48 | DONE 21 | 남음 27")
    assert any(line.startswith("Seed / Step 0 / 25") for line in lines)


@pytest.mark.parametrize("width", [80, 90])
def test_narrow_remarks_move_under_the_row(tmp_path, width):
    root = mbpp_root(tmp_path / "selection-switch-mbpp-quality-v1")
    published(branch(root, "selection_full", 3, 25))
    output = dashboard.render(dashboard.snapshot([root], now=NOW), width=width)
    lines = output.splitlines()
    assert fits(output, width)
    header = next(index for index, line in enumerate(lines) if line.startswith("Experiment "))
    assert "Remarks" not in lines[header] and lines[header + 1] == "  Remarks"
    # Remarks are full-width lines under their row, not a one-word-per-line ribbon.
    def under(prefix):
        index = next(index for index, line in enumerate(lines) if line.startswith(prefix))
        notes = []
        for line in lines[index + 1:]:
            if not line.startswith("  "):
                break
            notes.append(line.strip())
        return notes
    assert " ".join(under("On-policy · 선택비용 별도 ")) == (
        "학습 한도 공통; 선택 비용 별도 기록; 공통 학습 15/15; 최종 평가 저장됨·곡선 남음 1개")
    assert " ".join(under("3 / 25 ")) == (
        "Full selection: 최종 평가 저장됨; 곡선 평가 남음 (재학습 없음); Gate policy: development gate")
    assert len(under("On-policy · 선택비용 별도 ")) <= 2 and len(under("3 / 25 ")) <= 2


@pytest.mark.parametrize("width", [100, 120, 160])
def test_wide_terminal_keeps_the_single_table(six_suites, width):
    roots, _, now = six_suites
    output = dashboard.render(mbpp_report(roots, now), width=width)
    assert fits(output, width)
    header = next(line for line in output.splitlines() if line.startswith("Experiment "))
    assert header.rstrip().endswith("Remarks") and "  Remarks" not in output.splitlines()


@pytest.mark.parametrize("width", [40, 60, 70, 80, 90])
def test_pair_dashboard_fits_phone_width(tmp_path, width):
    data = pair.snapshot(tmp_path / "missing", now=1000.)
    output = pair.render(data, width=width, all_tasks=True)
    assert fits(output, width) and output != pair.render(data, width=100, all_tasks=True)
    assert "계획 42개 | 완료 확인 0/42 | 남음 42개" in " ".join(output.split())


@pytest.mark.parametrize("width", [40, 60, 70])
def test_rloo_dashboard_fits_phone_width(rloo_root, width):
    root, out = rloo_root
    core.atomic_json(out / "random/progress.json", {"state": "running", "updated": 1800000000. - 5,
                                                   "host": "h100-node-17", "phase": "train", "seconds": 10,
                                                   "timeout": 86400})
    data = rloo.snapshot(root, now=1800000000.)
    output = rloo.display.render(data, width=width, all_tasks=True)
    assert fits(output, width) and output != rloo.display.render(data, width=80, all_tasks=True)
    assert "1. h100-node-17" in output.splitlines() and "0 / step 0 / Random" in output
