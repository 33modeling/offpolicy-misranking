"""Node activity includes leased tasks and live launchers, without invented training."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from srgc_rebuttal.runtime import atomic_json, lease
from srgc_research.dispatch import model_chain, model_status, qwen_status
from srgc_research.tests.test_model_results import make_cohort, saved


@pytest.fixture(params=("gemma4_it", "qwen35", "llama31"))
def cohort(request, tmp_path):
    return make_cohort(request.param, tmp_path)


def coordinator(cohort, **changes):
    now = time.time()
    row = {"worker_id": "launcher", "host": socket.gethostname(), "pid": 123,
           "started": now - 20, "heartbeat": now, "status": "running",
           "model": "gemma4" if cohort.model == "gemma4_it" else cohort.model,
           "status_model": cohort.model, "root": str(cohort.root),
           "dataset": "all", "active_dataset": "math", "devices": "0,1,2,3",
           "order": list(model_chain.MODELS), **changes}
    path = cohort.work / "srgc-rebuttal/.model-chain/workers/launcher.json"
    atomic_json(path, row)
    return path


def inspect(cohort, capsys, *, as_json=True):
    with patch.dict(os.environ, cohort.env, clear=True), \
            patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        code = model_status.main([cohort.model, "--root", str(cohort.root),
                                  *(["--json"] if as_json else [])])
    output = capsys.readouterr()
    assert code == 0, output.out + output.err
    return json.loads(output.out) if as_json else output.out


@pytest.mark.parametrize("state,expected", [("running", "launching"),
                                           ("waiting", "waiting"),
                                           ("switching", "switching")])
def test_live_launcher_is_counted_before_dataset_worker_exists(cohort, capsys, state, expected):
    path = coordinator(cohort, status=state, idle_reason="waiting on peer cache")
    before, receipt = saved(cohort), path.read_bytes()
    value = inspect(cohort, capsys)
    assert len(value["nodes"]) == 1
    node = value["nodes"][0]
    assert node["status"] == expected and node["source"] == "model_chain"
    assert node["task"] is None and node["task_row"] is None
    assert all(not row["workers"] for row in value["datasets"])
    assert not any(task["status"] == "running" for row in value["datasets"] for task in row["tasks"])
    text = inspect(cohort, capsys, as_json=False)
    assert "NODES  1 active | 0 stopped | RUN 0" in text
    assert "no worker records" not in text
    assert f"* {socket.gethostname()}" in text
    assert saved(cohort) == before and path.read_bytes() == receipt


@pytest.mark.parametrize("worker_receipt", [False, True])
def test_live_task_lease_keeps_local_node_active_when_heartbeat_is_missing_or_stale(
        cohort, capsys, worker_receipt):
    queue = cohort.root / "runs/math/.queue"
    atomic_json(queue / "tasks/seed-5.cache.json", {
        "status": "running", "task": "seed-5.cache", "worker_id": "child",
        "host": socket.gethostname(), "attempt": 1})
    if worker_receipt:
        atomic_json(queue / "workers/child.json", {
            "status": "running", "worker_id": "child", "host": socket.gethostname(),
            "heartbeat": time.time() - 3600, "task": "seed-5.cache"})
    with lease(queue / "leases/seed-5.cache.lock"):
        before = saved(cohort)
        value = inspect(cohort, capsys)
        assert len(value["nodes"]) == 1
        node = value["nodes"][0]
        assert node["status"] == "running" and node["task"] == "seed-5.cache"
        assert node["source"] == "task_lease"
        assert node["heartbeat_stale"] is worker_receipt
        text = inspect(cohort, capsys, as_json=False)
        assert "NODES  1 active | 0 stopped | RUN 1" in text
        assert "초기 응답 캐시 작업 확보" in text
        assert saved(cohort) == before


def test_coordinator_and_dataset_worker_count_one_node_and_keep_task_progress(cohort, capsys):
    coordinator(cohort, status="waiting")
    queue = cohort.root / "runs/math/.queue"
    atomic_json(queue / "tasks/seed-5.cache.json", {
        "status": "running", "task": "seed-5.cache", "worker_id": "child",
        "host": socket.gethostname(), "attempt": 1})
    with lease(queue / "leases/seed-5.cache.lock"):
        value = inspect(cohort, capsys)
    assert len(value["nodes"]) == 1
    assert value["nodes"][0]["source"] == "task_lease"
    assert value["nodes"][0]["task"] == "seed-5.cache"


@pytest.mark.parametrize("changes", [{"root": "/other-experiment"},
                                     {"status_model": "gemma4_text"}])
def test_another_root_or_protocol_cannot_activate_this_experiment(tmp_path, capsys, changes):
    cohort = make_cohort("gemma4_it", tmp_path)
    coordinator(cohort, **changes)
    assert inspect(cohort, capsys)["nodes"] == []


@pytest.mark.parametrize("state,seconds", [("running", 120), ("stopped", 0), ("failed", 0)])
def test_stale_or_ended_coordinator_never_becomes_active(tmp_path, capsys, state, seconds):
    cohort = make_cohort("gemma4_it", tmp_path)
    coordinator(cohort, status=state, heartbeat=time.time() - seconds)
    value = inspect(cohort, capsys)
    assert all(row["status"] not in qwen_status.LIVE_WORKERS for row in value["nodes"])
    assert "NODES  1 active" not in inspect(cohort, capsys, as_json=False)


def test_legacy_coordinator_is_visible_without_claiming_an_experiment_task(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    path = coordinator(cohort)
    record = json.loads(path.read_text())
    for key in ("root", "status_model"):
        record.pop(key)
    atomic_json(path, record)
    value = inspect(cohort, capsys)
    assert value["nodes"][0]["task"] is None
    assert "experiment root not recorded" in inspect(cohort, capsys, as_json=False)


def test_public_gemma_status_shortcut_and_dataset_view_keep_waiting_launcher(tmp_path):
    cohort = make_cohort("gemma4_it", tmp_path)
    coordinator(cohort, status="waiting", active_dataset="mbpp", idle_reason="peer owns MBPP cache")
    script = Path(__file__).resolve().parents[2] / "scripts/run_srgc_gemma4.sh"
    for args in (["status"], ["math", "status"]):
        result = subprocess.run(["sh", str(script), *args, "--json"], env=cohort.env,
                                cwd="/tmp", capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(result.stdout)
        assert len(value["nodes"]) == 1 and value["nodes"][0]["status"] == "waiting"
        assert value["nodes"][0]["label"] == "MBPP"
        assert not any(task["status"] == "running" for row in value["datasets"] for task in row["tasks"])


def test_new_coordinator_receipt_records_actual_protocol_and_custom_root(tmp_path, monkeypatch):
    root = tmp_path / "custom-it-root"
    monkeypatch.setenv("GROUP_VOLUME", str(tmp_path))
    monkeypatch.setenv("OM_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("SRGC_GEMMA_IT_PROTOCOL", "1")

    def drain(order, dataset, initial, environment, update, **options):
        update("running", model="gemma4", active_dataset="math")
        return 0

    with patch.object(model_chain, "drain", side_effect=drain):
        assert model_chain.main(["--start", "gemma4", "all", "run", "--root", str(root)]) == 0
    row = model_chain.snapshot()[0]
    assert row["status_model"] == "gemma4_it" and row["root"] == str(root)


def test_long_coordinator_wait_refreshes_heartbeat_without_changing_wait_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("GROUP_VOLUME", str(tmp_path))
    monkeypatch.setenv("OM_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("SRGC_MODEL_POLL_SECONDS", "60")
    clock, ages = [100.0], []
    monkeypatch.setattr(model_chain.time, "time", lambda: clock[0])

    def sleeping(seconds):
        clock[0] += seconds
        row = model_chain.snapshot(now=clock[0])[0]
        ages.append(row["heartbeat_age_seconds"])
        assert row["status"] == "waiting"
        assert row["idle_reason"] == "peer owns cache"

    def drain(order, dataset, initial, environment, update, **options):
        update("waiting", model="gemma4", active_dataset="math", idle_reason="peer owns cache")
        options["sleep"](60)
        return 0

    monkeypatch.setattr(model_chain.time, "sleep", sleeping)
    with patch.object(model_chain, "drain", side_effect=drain):
        assert model_chain.main(["--start", "gemma4", "all"]) == 0
    assert ages and max(ages) <= 5


def test_launcher_without_dataset_worker_keeps_status_short(cohort, capsys):
    path = coordinator(cohort, child_pid=456, launch_attempt=3, launch_started=time.time() - 125,
                       previous_exit_code=76, previous_exit_at=time.time() - 126)
    before = path.read_bytes()
    text = inspect(cohort, capsys, as_json=False)
    assert 'worker 시작 미확인' in text
    # Per-launcher record lines are no longer printed (2026-10-31 user request).
    assert 'launcher records' not in text and 'MODEL QUEUE' not in text
    assert path.read_bytes() == before


@pytest.mark.parametrize("stage,label", [("training", "Prefix 업데이트"), ("checkpoint_save", "체크포인트 저장"),
                                        (None, "실행 단계 미기록")])
def test_prefix_lease_describes_actual_recorded_stage_and_unknown_progress(stage, label):
    row = {"task": "seed-5.prefix", "task_row": {"arm": "prefix", "phase_stage": stage}}
    text = qwen_status.leased_task_text(row)
    assert "공통 Prefix 작업 확보" in text and label in text
    assert "worker 응답 미기록, 진행 미확인" in text
    assert "학습 중" not in text


def test_lease_after_endpoint_is_saved_is_an_exit_wait_not_a_new_training_claim():
    row = {"task": "seed-5.switch", "heartbeat_age_seconds": 1,
           "task_row": {"arm": "switch", "endpoint_ready": True}}
    assert "결과 저장 완료 · 작업 종료/잠금 반환 대기" in qwen_status.leased_task_text(row)


@pytest.mark.parametrize("code,label", [(76, "배정 가능한 작업"), (75, "자원 점검/잠금"), (137, "자식 작업 중단")])
def test_launcher_wait_explains_recorded_child_exit_without_guessing_a_prefix(code, label):
    text = qwen_status.launcher_text({"status": "waiting", "previous_exit_code": code})
    assert label in text and "재시도 대기" in text
    assert "Prefix" not in text
