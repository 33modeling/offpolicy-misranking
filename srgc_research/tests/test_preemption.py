"""Launcher exits must distinguish preemption from actual training failures."""

import pytest

from srgc_research.dispatch.preemption import interruption_code, log_position


@pytest.mark.parametrize(("code", "message", "expected"), [
    (-9, "", 137), (137, "", 137),
    (1, "exitcode  : -9 (pid: 123)\nSignal 9 (SIGKILL) received", 137),
    (1, "RuntimeError: ordinary error", None),
    (9, "", None), (124, "TIMEOUT", None),
    (137, "CUDA out of memory", None),
    (1, "OutOfMemoryError\nexitcode : -9", None),
    (-9, "slurmstepd: Detected 1 oom_kill event", None),
    (1, "Sending process closing signal SIGKILL\nRoot Cause (first observed failure):\nexitcode : 1\nValueError: bad input", None),
    (1, "Sending process closing signal SIGTERM\nRoot Cause (first observed failure):\nexitcode : -9\nSignal 9 (SIGKILL)", 137),
    (-15, "", 143), (-2, "", 130), (-1, "", 129), (129, "", 129),
])
def test_signal_and_oom_classification(tmp_path, code, message, expected):
    log = tmp_path / "job.log"
    log.write_text(message)
    assert interruption_code(code, log, 0) == expected


@pytest.mark.parametrize(("old", "current", "code", "expected"), [
    ("Signal 9 (SIGKILL)", "RuntimeError: verification error", 1, None),
    ("OutOfMemoryError", "exitcode : -9", 1, 137),
    ("CUDA out of memory", "", -9, 137),
])
def test_previous_attempt_does_not_change_current_classification(tmp_path, old, current, code, expected):
    log = tmp_path / "job.log"
    log.write_text(old + "\n")
    start = log_position(log)
    with log.open("a") as handle:
        handle.write(current + "\n")
    assert interruption_code(code, log, start) == expected


def test_missing_log_and_large_log_tail(tmp_path):
    log = tmp_path / "job.log"
    assert log_position(log) == 0
    assert interruption_code(-9, log, 0) == 137
    log.write_text("x" * 200000 + "\nSignal 9 (SIGKILL)")
    assert interruption_code(1, log, 0) == 137
