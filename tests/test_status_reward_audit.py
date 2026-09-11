"""Behavioral regressions for the September 6 follow-up audit."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from test_rlzero_status import (
    TAG,
    active_family,
    run_status,
    write_pipeline_activity,
    write_worker_heartbeat,
)

from data import extract_answer, normalize_math_answer, reward
from rlzero_status import Family, current_point


@pytest.mark.parametrize("answer", ["(1,234)", "[12,345]", "{6,789}", "12,3456,789"])
def test_answer_normalization_preserves_structural_commas(answer):
    assert normalize_math_answer(answer) == answer


@pytest.mark.parametrize(
    "answer, expected",
    [("1,234", "1234"), ("-12,345.67", "-12345.67"), ("1,234,567", "1234567")],
)
def test_answer_normalization_accepts_complete_grouped_numbers(answer, expected):
    assert normalize_math_answer(answer) == expected


def test_final_hash_answer_wins_over_earlier_attempt():
    assert extract_answer("First try: #### 8\nCorrection:\n#### 9") == "9"


def test_structural_comma_fix_reaches_reward_verifier():
    assert reward("#### (1234)", "(1,234)") == 0.0
    assert reward("#### (1,234)", "(1, 234)") == 1.0
    assert reward("#### 8\n#### 9", "9") == 1.0


def test_status_history_is_not_a_worker(tmp_path):
    root = tmp_path / "runs"
    (root / "logs").mkdir(parents=True)
    (root / "logs/status-history.log").write_text("status called\n")
    output = run_status(root)
    assert "workers_observed=0/1" in output
    assert "overall_verdict=NOT_STARTED" in output


def test_recovery_cause_does_not_overwrite_active_point_kind(tmp_path):
    root = tmp_path / "runs"
    run, _, lock = active_family(root)
    (run / "rollout_recovery.jsonl").write_text(
        json.dumps({"status": "failed", "failure_kind": "cuda-oom"}) + "\n"
    )
    try:
        output = run_status(root, verbose=False)
    finally:
        lock.close()
    row = next(line for line in output.splitlines() if line.startswith(" math500/s0"))
    assert "d0" in row
    assert row.split()[1] == "*"
    assert "CUDA recovery failed once (cuda-oom" in row


def test_deferred_d0_does_not_hide_completed_points_or_current_training(tmp_path):
    args = SimpleNamespace(root=tmp_path, model_tag=TAG, drifts=[0, 25, 100, 400])
    family = Family("math500", 0)
    for drift in [0, 25, 100]:
        run = tmp_path / "family-math500-s0" / f"{TAG}-s0-math500-d{drift}"
        (run / "logs").mkdir(parents=True)
        (run / "logs/main.log").write_text(f"drift {drift}\n")
        os.utime(run / "logs/main.log", (1000 + drift, 1000 + drift))
        if drift == 25:
            (run / "DONE").write_text("done\n")
    drift, _, kind, done = current_point(args, family)
    assert done == [25]
    assert (drift, kind) == (100, "active")


@pytest.mark.parametrize("state", ["idle-suspected", "terminating-idle"])
def test_invalid_idle_telemetry_does_not_trigger_stall_diagnosis(tmp_path, state):
    root = tmp_path / "runs"
    run, _, lock = active_family(root)
    import json

    path = write_pipeline_activity(run, state, idle_seconds=120)
    record = json.loads(path.read_text())
    record["schema"] = "invalid"
    path.write_text(json.dumps(record))
    write_worker_heartbeat(root)
    try:
        output = run_status(root)
    finally:
        lock.close()
    assert "verdict=UNKNOWN reason=pipeline_telemetry_schema_invalid" in output


def qwen_status(
    tmp_path,
    *,
    active=False,
    exit_code=None,
    empty_done=False,
    done_count=0,
    history_failure=False,
    error_text="",
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ["status_qwen35.sh", "setup_env.sh"]:
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    # the full-matrix renderer and its two imports; the status falls back to
    # its short table when they are missing, so copy them to test the real path
    (repo / "src").mkdir()
    for name in ["matrix_status.py", "training_progress.py", "point_key_numbers.py"]:
        shutil.copy2(ROOT / "src" / name, repo / "src" / name)
    work = tmp_path / "work"
    logs = work / "console-logs"
    logs.mkdir(parents=True)
    text = "[launch] utc=2026-09-06T00:00:00Z\n[stage] run\n"
    if exit_code is not None:
        text += f"[exit] rc={exit_code}\n"
    (logs / "additional-qwen35-test.log").write_text(text)
    run_id = "qwen35-9b-posttrained-math-code-grpo-v1"
    run = work / "runs" / run_id / "qwen35" / f"{run_id}-grpo-qwen35-s0-math500-d0"
    (run / "logs").mkdir(parents=True)
    (run / "logs/main.log").write_text("[progress] test  1/8 prep\n" + error_text)
    if empty_done:
        (run / "DONE").touch()
    for index in range(done_count):
        point = run if index == 0 else run.parent / f"test-s{index}-math500-d0"
        point.mkdir(exist_ok=True)
        (point / "DONE").write_text("done\n")
    bins = tmp_path / "bin"
    bins.mkdir()
    pgrep = bins / "pgrep"
    pgrep.write_text(f"#!/bin/sh\nexit {0 if active else 1}\n")
    pgrep.chmod(0o755)
    env = {
        **os.environ,
        "OM_WORK": str(work),
        "OM_REPO": str(repo),
        "STATUS_HISTORY_ACTIVE": "1",
        "PATH": str(bins) + os.pathsep + os.environ["PATH"],
    }
    if history_failure:
        env.pop("STATUS_HISTORY_ACTIVE")
        tee = bins / "tee"
        tee.write_text("#!/bin/sh\ncat\nexit 7\n")
        tee.chmod(0o755)
    return subprocess.run(
        ["bash", str(scripts / "status_qwen35.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_qwen_no_failures_has_valid_integer_count(tmp_path):
    result = qwen_status(tmp_path, active=True)
    assert result.returncode == 0
    assert result.stderr == ""
    assert "family failures this session: 0\n" in result.stdout


def test_qwen_status_prints_the_whole_matrix_like_olmo(tmp_path):
    result = qwen_status(tmp_path, active=True)
    assert result.returncode == 0, result.stdout + result.stderr
    # all ten families, not just the newest six points
    for dataset in ("math500", "mbpp"):
        for seed in range(5):
            assert f" {dataset}/s{seed} " in result.stdout
    assert "families 10:" in result.stdout
    assert "overall_verdict=" in result.stdout
    assert "recommended_action=" in result.stdout
    assert "KEY NUMBERS per scored point" in result.stdout
    assert "(full status renderer failed" not in result.stdout


@pytest.mark.parametrize("marker", ["[abort]", "GPU0 ✘ job rc=1"])
def test_qwen_error_search_matches_actual_abort_and_legacy_markers(tmp_path, marker):
    result = qwen_status(
        tmp_path, active=True, error_text=marker + "\nRuntimeError: test failure\n"
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert "ERROR (current): RuntimeError: test failure" in result.stdout


def test_qwen_successful_launcher_is_not_complete_with_missing_points(tmp_path):
    result = qwen_status(tmp_path, exit_code=0, empty_done=True)
    assert "DECISION DONE:" not in result.stdout
    assert "points   0 done" in result.stdout
    assert "DECISION WARNING:" in result.stdout


def test_qwen_complete_matrix_still_reports_done(tmp_path):
    result = qwen_status(tmp_path, exit_code=0, done_count=40)
    assert result.returncode == 0
    assert "DECISION DONE:" in result.stdout


def test_qwen_history_writer_failure_is_not_hidden(tmp_path):
    result = qwen_status(tmp_path, history_failure=True)
    assert result.returncode == 7


@pytest.mark.parametrize("name", ["run_qwen35_9b.sh", "run_olmo3_rlzero.sh"])
@pytest.mark.parametrize("git_available", [True, False])
@pytest.mark.parametrize("active_e5", [True, False])
def test_status_never_updates_code_or_requires_a_local_launcher(
    tmp_path, name, git_available, active_e5
):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source = (ROOT / "scripts" / name).read_text()
    if name == "run_olmo3_rlzero.sh":
        # Exercise the real entrypoint through its status revision report,
        # stopping before cluster provisioning and the full status renderer.
        source = source.split("export OM_ONLINE=", 1)[0] + 'echo STATUS_RENDERED\n'
    else:
        (scripts / "status_qwen35.sh").write_text('#!/bin/bash\necho "STATUS_RENDERED $*"\n')
    (scripts / name).write_text(source)
    work = tmp_path / "work"
    if active_e5:
        (work / "runs/e5-reduced/math500-d400/s0/logs").mkdir(parents=True)
        (work / "runs/e5-reduced/math500-d400/s0/logs/launcher-remote.log").write_text("[train] g11\n")
    bins = tmp_path / "bin"
    bins.mkdir()
    calls = tmp_path / "calls"
    git = bins / "git"
    git.write_text(
        '#!/bin/bash\nprintf "git %s\\n" "$*" >> "$CALLS"\n'
        + ('[ "$*" = "rev-parse --short HEAD" ] || exit 99\necho fixture\n'
           if git_available else 'exit 127\n')
    )
    git.chmod(0o755)
    pgrep = bins / "pgrep"
    pgrep.write_text('#!/bin/bash\necho pgrep >> "$CALLS"\nexit 1\n')
    pgrep.chmod(0o755)
    result = subprocess.run(
        ["bash", str(scripts / name), "status", "h100", "verbose"],
        env={**os.environ, "OM_WORK": str(work), "CALLS": str(calls),
             "PATH": str(bins) + os.pathsep + os.environ["PATH"]},
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STATUS_RENDERED" in result.stdout
    assert "read-only status; no automatic update" in result.stdout
    assert calls.read_text().splitlines() == ["git rev-parse --short HEAD"]
    if name == "run_qwen35_9b.sh":
        assert "STATUS_RENDERED verbose" in result.stdout
    if not git_available:
        assert "[code] unknown" in result.stdout
