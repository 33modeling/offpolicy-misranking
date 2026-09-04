from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "src/rlzero_status.py"
TAG = "olmo3-test"


def status_command(root: Path, **overrides: int) -> list[str]:
    values = {
        "probe_seconds": 0,
        "stuck_seconds": 30,
        "worker_stale_seconds": 30,
        "heartbeat_stale_seconds": 30,
        "expected_workers": 1,
        "generation_batch": 8,
        "gradient_micro_batch": 4,
        "logprob_micro_batch": 4,
        "min_recovery_generation_batch": 2,
        "log_lines": 5,
        "error_lines": 5,
        **overrides,
    }
    return [
        sys.executable,
        str(STATUS),
        "--profile",
        "h100",
        "--root",
        str(root),
        "--results",
        str(root.parent / "results"),
        "--model-tag",
        TAG,
        "--datasets",
        "math500",
        "--seeds",
        "0",
        "--drifts",
        "0",
        "--probe-seconds",
        str(values["probe_seconds"]),
        "--stuck-seconds",
        str(values["stuck_seconds"]),
        "--worker-stale-seconds",
        str(values["worker_stale_seconds"]),
        "--heartbeat-stale-seconds",
        str(values["heartbeat_stale_seconds"]),
        "--expected-workers",
        str(values["expected_workers"]),
        "--config-sha",
        "test-config",
        "--model-revision",
        "test-model",
        "--generation-batch",
        str(values["generation_batch"]),
        "--gradient-micro-batch",
        str(values["gradient_micro_batch"]),
        "--logprob-micro-batch",
        str(values["logprob_micro_batch"]),
        "--min-recovery-generation-batch",
        str(values["min_recovery_generation_batch"]),
        "--log-lines",
        str(values["log_lines"]),
        "--error-lines",
        str(values["error_lines"]),
    ]


def active_family(root: Path) -> tuple[Path, Path, object]:
    family = root / "family-math500-s0"
    run = family / f"{TAG}-s0-math500-d0"
    logs = run / "logs"
    logs.mkdir(parents=True)
    (root / ".families").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (root / ".families/math500-s0.owner.json").write_text(
        '{"worker":"worker-1","host":"node-1"}\n', encoding="utf-8"
    )
    (root / "logs/worker-1.log").write_text("worker running\n", encoding="utf-8")
    (logs / "fresh-shard0.log").write_text("rollout 1/100\n", encoding="utf-8")
    partial = run / "rollouts_fresh_train.shard0.partial"
    partial.write_text("{}\n", encoding="utf-8")
    lock = (root / ".families/math500-s0.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX)
    return run, partial, lock


def write_worker_heartbeat(root: Path, worker: str = "worker-1") -> Path:
    path = root / ".workers" / f"{worker}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "offpolicy-worker-heartbeat/v1",
                "worker": worker,
                "state": "running",
                "heartbeat_at_ns": time.time_ns(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_pipeline_activity(run: Path, state: str, idle_seconds: int = 0) -> Path:
    path = run / ".pipeline-activity.json"
    path.write_text(
        json.dumps(
            {
                "schema": "offpolicy-pipeline-activity/v1",
                "observed_at_epoch": int(time.time()),
                "state": state,
                "runner_pid": 123,
                "cpu_delta_seconds": 0,
                "gpu_peak_percent": 0,
                "idle_seconds": idle_seconds,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_attempt_manifest(run: Path, offsets: dict[str, int]) -> Path:
    path = run / "logs/regime-attempt-2.log.start.json"
    path.write_text(
        json.dumps(
            {
                "schema": "offpolicy-pipeline-attempt/v1",
                "started_at_ns": time.time_ns(),
                "attempt_log": "regime-attempt-2.log",
                "log_offsets": offsets,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run_status(root: Path, **overrides: int) -> str:
    result = subprocess.run(
        status_command(root, **overrides),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_status_observes_real_progress_and_scans_all_active_logs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs" / TAG
    run, partial, lock = active_family(root)
    old_error = run / "logs/regime-attempt-1.log"
    old_error.write_text(
        "RuntimeError: CUDA error: unspecified launch failure\n", encoding="utf-8"
    )
    old = time.time() - 5
    os.utime(old_error, (old, old))

    def advance() -> None:
        time.sleep(0.2)
        with partial.open("a", encoding="utf-8") as stream:
            stream.write("{}\n")

    updater = threading.Thread(target=advance)
    updater.start()
    try:
        output = run_status(root, probe_seconds=1)
    finally:
        updater.join()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    assert "verdict=PROGRESSING reason=artifact_or_log_changed_during_probe" in output
    assert "observed_changes=rollouts_fresh_train.shard0.partial:" in output
    assert "error_evidence_from_all_checked_logs:" in output
    assert "unspecified launch failure" in output
    assert "worker=worker-1 state=CLAIMED claims=math500/s0" in output
    assert "workers_observed=1/1" in output
    assert "overall_verdict=RUNNING" in output


def test_status_distinguishes_alive_unknown_confirmed_stuck_and_dead(
    tmp_path: Path,
) -> None:
    alive_root = tmp_path / "alive" / TAG
    _, _, alive_lock = active_family(alive_root)
    try:
        alive = run_status(alive_root)
    finally:
        fcntl.flock(alive_lock, fcntl.LOCK_UN)
        alive_lock.close()
    assert "verdict=ALIVE" in alive
    assert "overall_verdict=RUNNING" in alive

    unknown_root = tmp_path / "unknown" / TAG
    _, _, unknown_lock = active_family(unknown_root)
    old = time.time() - 120
    for path in (
        unknown_root / ".families/math500-s0.owner.json",
        unknown_root / "logs/worker-1.log",
        *list((unknown_root / "family-math500-s0").rglob("*")),
    ):
        if path.is_file():
            os.utime(path, (old, old))
    try:
        unknown = run_status(
            unknown_root, stuck_seconds=10, worker_stale_seconds=10
        )
    finally:
        fcntl.flock(unknown_lock, fcntl.LOCK_UN)
        unknown_lock.close()
    assert "verdict=UNKNOWN reason=shared_activity_" in unknown
    assert "overall_verdict=UNKNOWN" in unknown

    stuck_root = tmp_path / "stuck" / TAG
    stuck_run, _, stuck_lock = active_family(stuck_root)
    write_worker_heartbeat(stuck_root)
    write_pipeline_activity(stuck_run, "terminating-idle", idle_seconds=60)
    try:
        stuck = run_status(stuck_root, stuck_seconds=10, worker_stale_seconds=10)
    finally:
        fcntl.flock(stuck_lock, fcntl.LOCK_UN)
        stuck_lock.close()
    assert "verdict=STUCK reason=pipeline_confirmed_idle_for_60s" in stuck
    assert "overall_verdict=STOPPED" in stuck

    dead_root = tmp_path / "dead" / TAG
    active_family(dead_root)[2].close()
    dead = run_status(dead_root)
    assert "math500/s0 stale-owner" in dead
    assert "verdict=DEAD" in dead
    assert "overall_verdict=STOPPED" in dead


def test_status_marks_missing_workers_as_degraded(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    _, _, lock = active_family(root)
    try:
        output = run_status(root, expected_workers=3)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert "workers_observed=1/3" in output
    assert "overall_verdict=DEGRADED" in output
    assert (
        "recommended_action=inspect_STUCK_DEAD_families_and_missing_workers" in output
    )


def test_status_reports_worker_preflight_before_first_claim(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    write_worker_heartbeat(root)
    output = run_status(root)
    assert "worker=worker-1 state=AVAILABLE claims=none" in output
    assert "workers_observed=1/1" in output
    assert "overall_verdict=STARTING" in output
    assert "recommended_action=wait_for_worker_preflight_or_queue_claim" in output


def test_status_uses_heartbeat_when_worker_log_is_quiet(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    _, _, lock = active_family(root)
    old = time.time() - 120
    os.utime(root / "logs/worker-1.log", (old, old))
    write_worker_heartbeat(root)
    try:
        output = run_status(
            root,
            stuck_seconds=10,
            worker_stale_seconds=10,
            heartbeat_stale_seconds=10,
        )
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert "liveness_evidence=heartbeat" in output
    assert "workers_observed=1/1" in output
    assert "overall_verdict=RUNNING" in output


def test_status_reports_computing_from_fresh_pipeline_telemetry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs" / TAG
    run, _, lock = active_family(root)
    old = time.time() - 120
    for path in (
        root / ".families/math500-s0.owner.json",
        root / "logs/worker-1.log",
        *list(run.rglob("*")),
    ):
        if path.is_file():
            os.utime(path, (old, old))
    write_worker_heartbeat(root)
    write_pipeline_activity(run, "cpu-active")
    try:
        output = run_status(root, stuck_seconds=10, worker_stale_seconds=10)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert "verdict=COMPUTING reason=pipeline_telemetry_cpu-active" in output
    assert "pipeline_telemetry=" in output
    assert "state=cpu-active" in output
    assert "overall_verdict=RUNNING" in output


def test_status_does_not_call_failed_telemetry_a_stall(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    run, _, lock = active_family(root)
    old = time.time() - 120
    for path in (
        root / ".families/math500-s0.owner.json",
        root / "logs/worker-1.log",
        *list(run.rglob("*")),
    ):
        if path.is_file():
            os.utime(path, (old, old))
    write_worker_heartbeat(root)
    write_pipeline_activity(run, "telemetry-error")
    try:
        output = run_status(root, stuck_seconds=10, worker_stale_seconds=10)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert "verdict=UNKNOWN reason=pipeline_activity_probe_failed_kill_suppressed" in output
    assert "verdict=STUCK" not in output
    assert "overall_verdict=UNKNOWN" in output


def test_status_rejects_malformed_pipeline_telemetry(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    run, _, lock = active_family(root)
    write_worker_heartbeat(root)
    telemetry = write_pipeline_activity(run, "gpu-active")
    record = json.loads(telemetry.read_text(encoding="utf-8"))
    record["schema"] = "unexpected"
    telemetry.write_text(json.dumps(record) + "\n", encoding="utf-8")
    try:
        output = run_status(root)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert "verdict=UNKNOWN reason=pipeline_telemetry_schema_invalid" in output
    assert "overall_verdict=UNKNOWN" in output


def test_status_separates_historical_and_current_attempt_errors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs" / TAG
    run, _, lock = active_family(root)
    stale = run / "logs/fresh-shard0.log"
    stale.write_text("CUDA error: stale failure\n", encoding="utf-8")
    attempt = run / "logs/regime-attempt-2.log"
    attempt.write_text("attempt started\n", encoding="utf-8")
    write_attempt_manifest(
        run,
        {
            stale.name: stale.stat().st_size,
            attempt.name: 0,
        },
    )
    try:
        historical = run_status(root)
        with stale.open("a", encoding="utf-8") as stream:
            stream.write("torch.OutOfMemoryError: current failure\n")
        current = run_status(root)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert "current_attempt_error_matches=0" in historical
    assert "error_assessment=historical_only_not_current_attempt" in historical
    assert "current_attempt_error_matches=1" in current
    assert "error_assessment=current_attempt_errors_present_but_activity_continues" in current
    assert "current_attempt_error_evidence:" in current


def test_status_reports_runtime_batch_contract_violation(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    run, _, lock = active_family(root)
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "gen_batch": "1",
                "gradient_micro_batch": 1,
                "grpo_logprob_micro_batch": 1,
            }
        ),
        encoding="utf-8",
    )
    try:
        output = run_status(root)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert "gen_batch:1!=8" in output
    assert "gradient_micro_batch:1!=4" in output
    assert "grpo_logprob_micro_batch:1!=4" in output
    assert "overall_verdict=INVALID" in output


def test_status_requires_exact_family_completion_stamp(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    run = root / "family-math500-s0" / f"{TAG}-s0-math500-d0"
    run.mkdir(parents=True)
    (run / "DONE").write_text("done\n", encoding="utf-8")
    (root / ".queue").mkdir()
    (root / ".queue/generation.git").write_text(
        "test-generation\n", encoding="utf-8"
    )
    stamp = root / "family-math500-s0/.family-complete"
    stamp.write_text("stale contract\n", encoding="utf-8")
    stale = run_status(root)
    assert "math500/s0 partial" in stale
    assert "math500/s0 complete" not in stale

    stamp.write_text(
        "test-generation test-config test-model math500 0\n", encoding="utf-8"
    )
    complete = run_status(root)
    assert "math500/s0 complete" in complete
    assert "overall_verdict=COMPLETE" in complete
