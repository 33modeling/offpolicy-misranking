"""Read-only Korean status view for V8 queues (scripts/v8_status.py)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("v8_status_under_test", REPO / "scripts/v8_status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, now):
    base = tmp_path / "v8"
    q = base / "olmo_math" / ".multi-node"
    write(q / "state.json", {"phase": "train", "failure": None})
    write(q / "jobs" / "a.json", {"state": "done", "task": {"kind": "train", "stage": "bank-trajectory"}, "attempts": []})
    write(q / "jobs" / "b.json", {"state": "running",
                                   "task": {"kind": "train", "stage": "bank-trajectory", "method_id": "on_monitored",
                                            "training_seed": 5, "run_id": "traj-5", "horizon": 275},
                                   "attempts": [{"id": "att1", "worker": "node01.gpu0123", "started_at": now - 1800}]})
    write(q / "jobs" / "c.json", {"state": "failed", "task": {"kind": "evaluate", "stage": "eval-final", "training_seed": 6},
                                   "attempts": [{"id": "att2", "error": "CUDA out of memory"}]})
    write(q / "jobs" / "d.json", {"state": "pending", "task": {"kind": "train"}, "attempts": []})
    log = q / "logs" / "att1.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("Loading weights: 100%\nPROGRESS 12:00:00 stage=generation records=40\n")
    os.utime(log, (now - 40, now - 40))
    run = base / "olmo_math" / "runs" / "traj-5"
    write(run / "manifest.json", {"attempts": [{"attempt_id": "x"}]})
    (run / "attempts" / "x").mkdir(parents=True)
    (run / "attempts" / "x" / "updates.jsonl").write_text('{"completed_updates": 37}\n')
    return base


def test_snapshot_shows_counts_running_task_and_failure(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    text = status.render([base], now=now, gpus=[("0", 97, 61000), ("1", 95, 60000)])
    assert "olmo_math  단계: 학습" in text
    assert "완료 1 · 진행 1 · 대기 1 · 실패 1 · 중단 0" in text
    assert "bank-trajectory on_monitored seed 5" in text and "node01" in text
    assert "update 37/275" in text and "출력 40초 전" in text and "30분째" in text
    assert "PROGRESS 12:00:00 stage=generation" in text
    assert "실패  eval-final seed 6: CUDA out of memory" in text
    assert "qwen_mbpp  단계: 시작 전" in text and "이 경로에 큐가 아직 없어요" in text
    assert "이 노드 GPU: 0: 97% 59G | 1: 95% 58G" in text
    assert "멈춤 의심" not in text


def test_stall_hint_only_when_silent_and_gpus_idle(tmp_path):
    now = time.time()
    base = fixture(tmp_path, now)
    log = base / "olmo_math/.multi-node/logs/att1.log"
    os.utime(log, (now - 3600, now - 3600))
    assert "멈춤 의심" in status.render([base], now=now, gpus=[("0", 0, 500)])
    assert "멈춤 의심" not in status.render([base], now=now, gpus=[("0", 98, 60000)])


def test_missing_nvidia_smi_is_reported(tmp_path):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    assert status.gpu_rows(run=missing) is None


def test_ago_formatting():
    assert status.ago(5) == "5초" and status.ago(600) == "10분" and status.ago(7300) == "2시간 1분" and status.ago(None) == "?"
