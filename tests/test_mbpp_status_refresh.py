"""Published outcomes and finished phases agree across MBPP status sections."""

import hashlib

import pytest

import selection_gate as core
from test_mbpp_status_dashboard import dashboard
from test_selection_switch_status import completed_prefix, convergence_root, point, published, published_curve, running

NOW = 2000000000.


def setup(root):
    convergence_root(root)
    p = core.read(root / "switch.json")
    core.atomic_json(root / "switch.json", {**p, "dataset": "mbpp"})
    completed_prefix(root)
    return point(root) / "random_reduced"


def finished_receipt(directory, *, code=0, mismatch=None):
    progress = core.read(directory / "progress.json")
    progress.update(ledger="reporting", gpus=4, gpu_type="H100")
    core.atomic_json(directory / "progress.json", progress)
    receipt = {**progress, "state": "finished", "time": NOW - 1,
               "exit_code": code, "allocated_gpu_seconds": progress["seconds"] * 4}
    if mismatch:
        receipt[mismatch] = "different"
    core.atomic_json(directory / "cost-events" / f"{progress['event_id']}.json", receipt)


@pytest.mark.parametrize("phase_dir", ["", "curve", "curve/step-30", "budget-recovery"])
def test_finished_event_does_not_stay_run_for_heartbeat_ttl(tmp_path, phase_dir):
    directory = setup(tmp_path)
    published(directory)
    published_curve(directory)
    running(directory / phase_dir, "finished-node", now=NOW, phase="evaluate")
    finished_receipt(directory / phase_dir)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    data = dashboard.snapshot([tmp_path], now=NOW)
    text = dashboard.render(data, all_tasks=True, width=240)
    assert "CURRENT RUN 0" in text
    assert dashboard.counts(data["suites"][0])["done"] == 1
    assert not any(node["assignments"] for node in dashboard.node_assignments(data))
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("field", ["event_id", "phase", "host", "ledger", "gpus", "gpu_type"])
def test_other_event_receipt_cannot_hide_live_work(tmp_path, field):
    directory = setup(tmp_path)
    running(directory, "active-node", now=NOW, phase="train")
    finished_receipt(directory, mismatch=field)
    data = dashboard.snapshot([tmp_path], now=NOW)
    assert "CURRENT RUN 1" in dashboard.render(data)
    assert dashboard.counts(data["suites"][0])["states"]["RUN"] == 1


def test_failed_finished_phase_does_not_count_as_done_or_run(tmp_path):
    directory = setup(tmp_path)
    running(directory, "failed-node", now=NOW, phase="train")
    finished_receipt(directory, code=1)
    data = dashboard.snapshot([tmp_path], now=NOW)
    counts = dashboard.counts(data["suites"][0])
    assert counts["done"] == 0 and counts["states"]["RUN"] == 0
    assert "CURRENT RUN 0" in dashboard.render(data)


def test_live_nested_evaluation_agrees_in_matrix_totals_and_all_listing(tmp_path):
    directory = setup(tmp_path)
    published(directory)
    running(directory / "curve", "curve-node", now=NOW, phase="curve")
    data = dashboard.snapshot([tmp_path], now=NOW)
    text = dashboard.render(data, all_tasks=True, width=240)
    assert dashboard.counts(data["suites"][0])["states"]["RUN"] == 1
    assert f"RUN {directory.relative_to(tmp_path)} " in text
    assert "CURRENT RUN 1" in text


def test_recovered_evaluation_count_updates_without_inventing_canonical_done(tmp_path):
    directory = setup(tmp_path)
    before = dashboard.render(dashboard.snapshot([tmp_path], now=NOW))
    assert "복구 평가 완료 1개" not in before
    path = directory / "budget-recovery/result.json"
    core.atomic_json(path, {"schema": "mbpp-budget-recovery/v1", "evaluation_complete": True,
                           "canonical_complete": False})
    core.atomic_json(path.with_suffix(".sha256.json"), {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data = dashboard.snapshot([tmp_path], now=NOW + 1)
    counts = dashboard.counts(data["suites"][0])
    assert counts["done"] == 0 and counts["recovered"] == 1
    text = dashboard.render(data)
    assert "복구 평가 완료 1개" in text
    assert "동일예산 DONE 제외" in text
    assert "복구 평가 저장됨; 동일예산 완료 아님" in " ".join(text.split())


def test_running_to_done_refreshes_without_viewer_restart(tmp_path):
    directory = setup(tmp_path)
    running(directory, "training-node", now=NOW, phase="train")
    first = dashboard.snapshot([tmp_path], now=NOW)
    assert dashboard.counts(first["suites"][0])["states"]["RUN"] == 1
    finished_receipt(directory)
    published(directory)
    published_curve(directory)
    second = dashboard.snapshot([tmp_path], now=NOW + 1)
    count = dashboard.counts(second["suites"][0])
    assert count["done"] == 1 and count["states"]["RUN"] == 0
    assert "CURRENT RUN 0" in dashboard.render(second)
