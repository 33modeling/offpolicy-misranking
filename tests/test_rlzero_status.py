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


def status_command(root: Path, verbose: bool = True, **overrides: int) -> list[str]:
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
    return ([
        sys.executable,
        str(STATUS),
    ] + (["--verbose"] if verbose else [])) + [
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


def run_status(root: Path, verbose: bool = True, **overrides: int) -> str:
    result = subprocess.run(
        status_command(root, verbose=verbose, **overrides),
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


def test_a_shortened_message_keeps_the_end_where_the_mismatch_is_named() -> None:
    """`config={...}, expected {...}` names the differing field last. Head-only
    truncation showed the operator the same 240 identical characters all night
    and never the one key that differed (2026-09-07)."""
    import rlzero_status

    message = (
        "config={'advantage_epsilon': 0.0001, 'checkpoint_every': 5, 'group_size': 8, "
        "'lora_rank': 16, 'max_grad_norm': 1.0}, expected {'advantage_epsilon': 0.0001, "
        "'checkpoint_every': 5, 'group_size': 8, 'lora_rank': 16, 'max_grad_norm': 1.0, "
        "'weight_decay': 0.0}"
    )
    short = rlzero_status.elide(message, 140)
    assert len(short) <= 140
    assert short.startswith("config={'advantage_epsilon'")
    assert short.endswith("'weight_decay': 0.0}")
    assert " ... " in short
    # a message that fits is returned whole, with runs of whitespace collapsed
    assert rlzero_status.elide("  a   b  ", 40) == "a b"


def test_status_reports_a_finished_point_that_the_completion_check_refuses(
    tmp_path: Path,
) -> None:
    """Busy GPUs are not progress. A point that finished and was refused is
    re-run from the start, so rollout bytes and GRPO steps keep moving and every
    liveness measure says TRAINING while nothing can ever be accepted
    (2026-09-07 night). Status must call that an error, not "ok"."""
    root = tmp_path / "runs" / TAG
    run, partial, lock = active_family(root)
    # the refused point is a FINISHED one (it has DONE); the worker is meanwhile
    # re-running it, which is what current_point() reports as active
    refused = run.parent / f"{TAG}-s0-math500-d25"
    (refused / "logs").mkdir(parents=True)
    (refused / "DONE").write_text("done\n", encoding="utf-8")
    (refused / "logs/supervisor.log").write_text(
        "[2026-09-08 01:00:00] [done-but-incomplete] ValueError: invalid GRPO policy lineage: config=...\n"
        "[2026-09-08 02:00:00] [done-but-incomplete] ValueError: invalid GRPO policy lineage: config=...\n",
        encoding="utf-8",
    )

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

    assert "overall_verdict=REDOING_REJECTED_WORK" in output
    assert "the completion check refused it (2x on math500/s0 d25)" in output
    assert "invalid GRPO policy lineage" in output
    assert "d25 finished and the completion check refused it 2x, so the worker keeps redoing it" in output


def test_rejections_before_the_point_was_accepted_are_not_counted(tmp_path: Path) -> None:
    """Once the cause is fixed the point is accepted; old rejection lines must not
    keep a healthy node reading as REDOING."""
    import rlzero_status

    run = tmp_path / "point"
    (run / "logs").mkdir(parents=True)
    (run / "logs/supervisor.log").write_text(
        "[2026-09-08 01:00:00] [done-but-incomplete] invalid GRPO policy lineage: config=...\n"
        "[2026-09-08 02:00:00] [done-but-incomplete] invalid GRPO policy lineage: config=...\n"
        "[2026-09-08 10:00:00] [point-accepted] completion check passed\n",
        encoding="utf-8",
    )
    assert rlzero_status.rejected_completions(run) == (0, "")

    with (run / "logs/supervisor.log").open("a", encoding="utf-8") as stream:
        stream.write("[2026-09-08 11:00:00] [done-but-incomplete] something else\n")
    count, reason = rlzero_status.rejected_completions(run)
    assert count == 1 and reason == "something else"
    assert rlzero_status.rejected_completions(None) == (0, "")
    assert rlzero_status.rejected_completions(tmp_path / "nope") == (0, "")


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


def test_status_default_is_one_screen_table(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "logs").mkdir(parents=True)
    output = run_status(root, verbose=False)
    lines = output.splitlines()
    assert lines[0].startswith("OLMo-3 RL-Zero h100")
    assert lines[1].startswith("DECISION ")
    assert lines[2].startswith("PROGRESS NOT STARTED")
    assert lines[3].startswith("STATE   NOT STARTED")
    assert any(line.startswith(" waiting") and "math500/s0" in line for line in lines)
    # the machine-readable verdict lines stay for scripts; the evidence dump does not
    assert "overall_verdict=NOT_STARTED" in output
    assert "== worker diagnostics ==" not in output
    assert len(lines) < 20


def test_short_worker_shows_node_and_job() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from rlzero_status import short_worker

    assert short_worker("run278140-first-rlvr-1-718eebe8-184b-4a22-aa34-2a2248932778") == "rlvr-1 run278140"
    assert short_worker("run277819-first-rlvr-5-ec778c50-2b79-4823-ae79-2d160dba9b05") == "rlvr-5 run277819"
    assert short_worker("worker-1") == "worker-1"
    assert len(short_worker("x" * 40)) == 18


def _status_with(root: Path, extra: list[str]) -> str:
    result = subprocess.run(
        status_command(root) + extra,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_live_worker_without_durable_writes_is_not_training(tmp_path: Path) -> None:
    """A fresh heartbeat and a held lock are liveness; with nothing durable
    written for longer than the stall window the first line must say so."""
    root = tmp_path / "runs" / TAG
    run, partial, lock = active_family(root)
    write_worker_heartbeat(root)
    (run / "run_config.json").write_text(
        json.dumps({"gen_batch": "8", "gradient_micro_batch": 4, "grpo_logprob_micro_batch": 4}),
        encoding="utf-8",
    )
    try:
        fresh = run_status(root, verbose=False)
        assert "PROGRESS TRAINING" in fresh
        assert "NOT TRAINING" not in fresh
        old = time.time() - 2 * 3600
        for path in list(run.rglob("*")) + [run]:
            os.utime(path, (old, old))
        stale = run_status(root, verbose=False)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    lines = stale.splitlines()
    assert lines[1].startswith("DECISION ERROR: NOT TRAINING for 2h"), lines[1]
    assert lines[2].startswith("PROGRESS NOT TRAINING for 2h")
    assert lines[3].startswith("STATE   NOT TRAINING")
    assert "overall_verdict=NOT_TRAINING" in stale


def test_unowned_family_mismatch_is_a_note_and_loop_marker_count_is_read(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    run = root / "family-math500-s0" / f"{TAG}-s0-math500-d0"
    (run / "logs").mkdir(parents=True)
    (root / ".families").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (run / "run_config.json").write_text(
        json.dumps({"gen_batch": "8", "gradient_micro_batch": 4, "grpo_logprob_micro_batch": 4}),
        encoding="utf-8",
    )
    (run / "rollouts_fresh_train.shard0.partial").write_text("{}\n", encoding="utf-8")
    # unowned (no lock held, no owner): the old batch values are repaired at the next claim
    output = _status_with(root, ["--dataset-generation-batch", "math500=32"])
    assert "runtime_contract_errors=0" in output
    assert "runtime_contract_notes_unowned=1" in output
    assert "~ contract (unowned runtime settings; checked on claim): math500/s0/d0:gen_batch:8!=32" in output
    assert "config/contract mismatch" not in output and "overall_verdict=INVALID" not in output
    # the launcher's loop marker: one line of pairs, then last_error and marked_at_utc
    (root / ".families/math500-s0.loop").write_text(
        "family=math500/s0 worker=w host=h consecutive_failures=4 last_rc=1\n"
        "last_error=RuntimeError: CUDA error: unspecified launch failure\nmarked_at_utc=2026-09-07T03:00:00Z\n",
        encoding="utf-8",
    )
    output = run_status(root, verbose=False)
    assert "failed 4 times in a row" in output
    assert "unspecified launch failure" in output


def test_finished_point_history_is_not_a_contract_error(tmp_path: Path) -> None:
    """2026-09-06: one complete point whose old recovery once ran at batch 1
    made status print ERROR/INVALID ("do not restart") for a whole day."""
    root = tmp_path / "runs" / TAG
    run = root / "family-math500-s0" / f"{TAG}-s0-math500-d0"
    run.mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (run / "run_config.json").write_text(
        json.dumps({"gen_batch": "8", "gradient_micro_batch": 4, "grpo_logprob_micro_batch": 4}),
        encoding="utf-8",
    )
    (run / "rollout_recovery.jsonl").write_text(
        json.dumps({"recovery_generation_batch": 1, "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    (run / "DONE").write_text("done\n", encoding="utf-8")
    output = run_status(root)
    assert "recovery_batch_below_floor" not in output
    assert "runtime_contract_errors=0" in output
    assert "overall_verdict=INVALID" not in output
    assert "config/contract mismatch" not in output


def test_per_dataset_runtime_expectations(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    run, _, lock = active_family(root)
    (run / "run_config.json").write_text(
        json.dumps({"gen_batch": "32", "gradient_micro_batch": 1, "grpo_logprob_micro_batch": 4}),
        encoding="utf-8",
    )
    try:
        plain = run_status(root)
        scoped = _status_with(
            root,
            ["--dataset-generation-batch", "math500=32", "--dataset-gradient-micro-batch", "math500=1"],
        )
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert "gen_batch:32!=8" in plain and "gradient_micro_batch:1!=4" in plain
    assert "runtime_contract_errors=0" in scoped
    assert "generation_batch=32/32" in scoped and "gradient_batch=1/1" in scoped
    assert "runtime_per_dataset math500:gen_batch=32,gradient_micro_batch=1" in scoped
    bad = subprocess.run(
        status_command(root) + ["--dataset-generation-batch", "gsm8k=32"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert bad.returncode != 0


def test_a_rescored_family_reads_as_rescore_pending_not_failed(tmp_path: Path) -> None:
    """After scripts/rescore_math500.sh the family has no DONE and its pinned scoring
    is parked; the status must say so instead of 'failed attempt'."""
    import rlzero_status

    family = tmp_path / "family-math500-s0"
    point = family / f"{TAG}-s0-math500-d25"
    (point / "pinned-scoring" / "20260909T000000Z").mkdir(parents=True)
    assert rlzero_status.rescore_pending(family)
    (point / "DONE").write_text("done\n")
    assert not rlzero_status.rescore_pending(family)
    assert not rlzero_status.rescore_pending(tmp_path / "missing")


def park_done(run: Path) -> None:
    parking = run / "pinned-scoring" / "20260909T000000Z"
    parking.mkdir(parents=True)
    (parking / "DONE").write_text("pinned\n")
    (run / "run_config.json").write_text("{}")


def test_unowned_rescore_has_separate_counts_and_gpu_queue(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run = root / "family-math500-s0" / f"{TAG}-s0-math500-d0"
    park_done(run)
    output = run_status(root)
    assert "prior completion 1/1 (current or archived DONE); final accepted 0/1; re-evaluation pending 1" in output
    assert "verdict=RESCORE_WAITING" in output
    assert any(line.split()[:2] == ["math500/s0", "R"] for line in output.splitlines())
    assert "GPU evaluation queued" in output
    assert 'OM_RLZERO_ONLY_FAMILIES="math500/s0"' in output
    assert "overall_verdict=EVALUATION_PENDING" in output
    assert "recovering by itself" not in output
    assert "AUTO: failed attempt" not in output
    assert "~" not in output.split("ACTION")[0]


def test_empty_parking_directory_does_not_invent_prior_completion(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run = root / "family-math500-s0" / f"{TAG}-s0-math500-d0"
    (run / "pinned-scoring").mkdir(parents=True)
    (run / "run_config.json").write_text("{}")
    output = run_status(root)
    assert "prior completion 0/1" in output


def test_live_rescore_shows_gpu_evaluation_not_training(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run, _, lock = active_family(root)
    park_done(run)
    write_worker_heartbeat(root)
    write_pipeline_activity(run, "gpu-active")
    try:
        output = run_status(root)
    finally:
        lock.close()
    assert "PROGRESS EVALUATION" in output
    assert "GPU RE-EVALUATION" in output
    assert "GPU evaluation queued" not in output
    assert "state=CLAIMED claims=math500/s0" in output


def test_rescore_does_not_hide_live_runtime_errors(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run, _, lock = active_family(root)
    park_done(run)
    (run / "run_config.json").write_text(json.dumps({"gen_batch": 1}))
    try:
        output = run_status(root)
    finally:
        lock.close()
    assert "overall_verdict=INVALID" in output
    assert "gen_batch:1!=8" in output


def test_completed_rescore_is_not_counted_twice(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run = root / "family-math500-s0" / f"{TAG}-s0-math500-d0"
    park_done(run)
    (run / "DONE").write_text("new\n")
    (root / ".queue").mkdir()
    (root / ".queue/generation.git").write_text("test-generation\n")
    (run.parent / ".family-complete").write_text("test-generation test-config test-model math500 0\n")
    output = run_status(root)
    assert "points 1/1 done" in output
    assert "overall_verdict=COMPLETE" in output
    assert "GPU EVALUATION QUEUED" not in output


def test_reopened_evaluation_does_not_call_busy_primary_a_failed_retry(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    primary, _, lock = active_family(root)
    write_worker_heartbeat(root)
    write_pipeline_activity(primary, "gpu-active")
    queued = root / "family-math500-s1" / f"{TAG}-s1-math500-d0"
    park_done(queued)
    try:
        output = _status_with(root, ["--seeds", "0", "1"])
    finally:
        lock.close()
    assert "overall_verdict=RUNNING_WITH_EVALUATION_PENDING" in output
    assert 'OM_RLZERO_ONLY_FAMILIES="math500/s1"' in output
    assert "verdict=COMPUTING" in output
    assert "recovering by itself" not in output


def test_rescore_waiting_does_not_hide_another_hung_family(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    primary, _, lock = active_family(root)
    old = time.time() - 8 * 3600
    for path in primary.rglob("*"):
        if path.is_file():
            os.utime(path, (old, old))
    write_worker_heartbeat(root)
    write_pipeline_activity(primary, "gpu-active")
    park_done(root / "family-math500-s1" / f"{TAG}-s1-math500-d0")
    try:
        output = _status_with(root, ["--seeds", "0", "1"])
    finally:
        lock.close()
    assert "verdict=HUNG" in output
    assert "overall_verdict=HUNG" in output


def test_flat_training_counters_do_not_trigger_restart_for_evaluation_only(tmp_path: Path, monkeypatch, capsys) -> None:
    import rlzero_status
    root = tmp_path / "runs"
    park_done(root / "family-math500-s0" / f"{TAG}-s0-math500-d0")
    write_worker_heartbeat(root, "idle-worker")
    signature = rlzero_status.training_progress.probe(root, 1)
    monkeypatch.setattr(rlzero_status.training_progress, "verdict", lambda *a, **k: ("NOT TRAINING", "NOT TRAINING for 2h", signature))
    monkeypatch.setattr(sys, "argv", status_command(root)[1:])
    rlzero_status.main()
    output = capsys.readouterr().out
    assert "overall_verdict=EVALUATION_PENDING" in output
    assert "Ctrl-C" not in output
