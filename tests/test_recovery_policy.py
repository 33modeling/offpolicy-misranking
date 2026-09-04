"""CUDA rollout recovery keeps full batching unless the current failure is OOM."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recovery_policy import (
    ATTEMPT_MANIFEST_SCHEMA,
    classify_cuda_failure,
    configured_generation_batch,
    current_attempt_evidence,
    failed_oom_batches,
    select_recovery_batch,
)


def write_attempt_manifest(path: Path, logs: Path, offsets: dict[str, int]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": ATTEMPT_MANIFEST_SCHEMA,
                "started_at_ns": 1,
                "attempt_log": "regime-attempt-2.log",
                "log_offsets": offsets,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_runtime_failure_reuses_configured_batch() -> None:
    assert classify_cuda_failure("CUDA error: unspecified launch failure") == "runtime"
    assert select_recovery_batch("runtime", configured=8, minimum=2) == 8


def test_oom_reduces_batch_geometrically_without_falling_below_floor() -> None:
    assert classify_cuda_failure("torch.OutOfMemoryError: CUDA out of memory") == "oom"
    assert select_recovery_batch("oom", configured=8, minimum=2) == 4
    assert select_recovery_batch("oom", configured=8, minimum=2, failed_batches=[4]) == 2
    with pytest.raises(ValueError, match="refusing batch 1"):
        select_recovery_batch(
            "oom", configured=8, minimum=2, failed_batches=[4, 2]
        )


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
    manifest = tmp_path / "regime-attempt-2.log.start.json"
    write_attempt_manifest(manifest, tmp_path, {stale.name: stale.stat().st_size})
    recovery = tmp_path / "rollout_recovery.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "src/recovery_policy.py"),
            "--run-config",
            str(config),
            "--attempt-log",
            str(current),
            "--attempt-manifest",
            str(manifest),
            "--logs-root",
            str(tmp_path),
            "--stage",
            "rollout-fresh",
            "--recovery-log",
            str(recovery),
            "--min-oom-batch",
            "2",
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


def test_cli_reads_only_new_bytes_from_current_stage_shards(tmp_path: Path) -> None:
    config = tmp_path / "run_config.json"
    attempt = tmp_path / "regime-attempt-2.log"
    fresh = tmp_path / "fresh-shard0.log"
    behavior = tmp_path / "beta-shard0.log"
    manifest = tmp_path / "regime-attempt-2.log.start.json"
    recovery = tmp_path / "rollout_recovery.jsonl"
    config.write_text(json.dumps({"gen_batch": 8}), encoding="utf-8")
    attempt.write_text("[stage-fail] pid=123 rc=1\n", encoding="utf-8")
    fresh.write_text("stale CUDA error: illegal memory access\n", encoding="utf-8")
    behavior.write_text("old behavior output\n", encoding="utf-8")
    write_attempt_manifest(
        manifest,
        tmp_path,
        {
            attempt.name: 0,
            fresh.name: fresh.stat().st_size,
            behavior.name: behavior.stat().st_size,
        },
    )
    with behavior.open("a", encoding="utf-8") as stream:
        stream.write("torch.OutOfMemoryError\n")
    assert (
        current_attempt_evidence(attempt, manifest, tmp_path, "rollout-fresh")
        == "[stage-fail] pid=123 rc=1\n"
    )

    with fresh.open("a", encoding="utf-8") as stream:
        stream.write("torch.OutOfMemoryError: allocation failed\n")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "src/recovery_policy.py"),
            "--run-config",
            str(config),
            "--attempt-log",
            str(attempt),
            "--attempt-manifest",
            str(manifest),
            "--logs-root",
            str(tmp_path),
            "--stage",
            "rollout-fresh",
            "--recovery-log",
            str(recovery),
            "--min-oom-batch",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "oom 4 8"


def test_failed_oom_history_is_stage_specific(tmp_path: Path) -> None:
    recovery = tmp_path / "rollout_recovery.jsonl"
    recovery.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "status": "failed",
                        "stage": "rollout-fresh",
                        "failure_kind": "oom",
                        "recovery_generation_batch": 4,
                    }
                ),
                json.dumps(
                    {
                        "status": "failed",
                        "stage": "rollout-behavior",
                        "failure_kind": "oom",
                        "recovery_generation_batch": 2,
                    }
                ),
                json.dumps(
                    {
                        "status": "completed",
                        "stage": "rollout-fresh",
                        "failure_kind": "oom",
                        "recovery_generation_batch": 2,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert failed_oom_batches(recovery, "rollout-fresh") == [4]
