"""CUDA rollout recovery keeps full batching unless the current failure is OOM."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recovery_policy import (
    classify_cuda_failure,
    configured_generation_batch,
    select_recovery_batch,
)


def test_runtime_failure_reuses_configured_batch() -> None:
    assert classify_cuda_failure("CUDA error: unspecified launch failure") == "runtime"
    assert select_recovery_batch("runtime", configured=8, oom_batch=1) == 8


def test_oom_reduces_batch() -> None:
    assert classify_cuda_failure("torch.OutOfMemoryError: CUDA out of memory") == "oom"
    assert select_recovery_batch("oom", configured=8, oom_batch=1) == 1
    assert select_recovery_batch("oom", configured=1, oom_batch=4) == 1


def test_configuration_requires_a_positive_batch() -> None:
    assert configured_generation_batch({"gen_batch": "8"}) == 8
    with pytest.raises(ValueError, match="positive gen_batch"):
        configured_generation_batch({"gen_batch": None})


def test_cli_uses_only_the_current_attempt_log(tmp_path: Path) -> None:
    config = tmp_path / "run_config.json"
    current = tmp_path / "regime-attempt-2.log"
    stale = tmp_path / "fresh-shard0.log"
    config.write_text(json.dumps({"gen_batch": "8"}), encoding="utf-8")
    current.write_text("watchdog stopped after inactivity\n", encoding="utf-8")
    stale.write_text("CUDA error: unspecified launch failure\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "src/recovery_policy.py"),
            "--run-config",
            str(config),
            "--attempt-log",
            str(current),
            "--oom-batch",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""

    current.write_text("CUDA error: illegal memory access\n", encoding="utf-8")
    result = subprocess.run(result.args, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert result.stdout.strip() == "runtime 8 8"
