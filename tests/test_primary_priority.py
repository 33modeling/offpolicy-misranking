"""Cluster-wide OLMo-first admission, without GPU or model processes."""

import hashlib
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
TAG = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"


def completed_primary(repo, work):
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts/require_olmo3_complete.sh", repo / "scripts")
    config = repo / "configs/olmo3_rlzero_h100.json"
    shutil.copy2(ROOT / "configs/olmo3_rlzero_h100.json", config)
    generation = "a" * 40
    binding = f"{generation} {hashlib.sha256(config.read_bytes()).hexdigest()} {'b' * 40}"
    root = work / "runs" / TAG
    results = work / "results" / TAG
    (root / ".queue").mkdir(parents=True)
    (root / ".queue/generation.git").write_text(generation + "\n")
    results.mkdir(parents=True)
    for name in ("REGIME.json", "REGIME.csv", "REGIME_SUMMARY.csv", "FINAL_REPORT.md"):
        (results / name).write_text("fixture\n")
    (results / "COMPLETE").write_text(binding + "\n")
    for dataset in ("math500", "mbpp"):
        for seed in range(5):
            family = root / f"family-{dataset}-s{seed}"
            family.mkdir()
            (family / ".family-complete").write_text(f"{binding} {dataset} {seed}\n")
            for drift in (0, 25, 100, 400):
                point = family / f"{TAG}-s{seed}-{dataset}-d{drift}"
                point.mkdir()
                (point / "DONE").write_text("done\n")
    return root, results


@pytest.mark.parametrize("damage", [None, "missing-complete", "generation", "config",
                                    "family", "point", "report", "malformed"])
def test_primary_gate_checks_full_shared_completion(tmp_path, damage):
    repo, work = tmp_path / "repo", tmp_path / "work"
    root, results = completed_primary(repo, work)
    if damage == "missing-complete":
        (results / "COMPLETE").unlink()
    elif damage == "generation":
        (root / ".queue/generation.git").write_text("c" * 40)
    elif damage == "config":
        (repo / "configs/olmo3_rlzero_h100.json").write_text("{}")
    elif damage == "family":
        (root / "family-mbpp-s4/.family-complete").write_text("stale")
    elif damage == "point":
        (root / f"family-mbpp-s4/{TAG}-s4-mbpp-d400/DONE").unlink()
    elif damage == "report":
        (results / "FINAL_REPORT.md").unlink()
    elif damage == "malformed":
        (results / "COMPLETE").write_text("not a contract")
    before = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
    result = subprocess.run(
        ["bash", "-euc", "source scripts/require_olmo3_complete.sh; require_olmo3_complete"],
        cwd=repo, env={**os.environ, "OM_WORK": str(work)},
        text=True, capture_output=True, timeout=5,
    )
    assert result.returncode == (75 if damage else 0), result.stdout + result.stderr
    assert before == {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}


@pytest.mark.parametrize("command", [
    ["scripts/run_qwen35_9b.sh", "check"],
    ["scripts/run_additional_experiments.sh", "--run", "qwen35"],
    ["scripts/run_additional_experiments.sh", "--run", "qwen38"],
    ["scripts/run_additional_experiments.sh", "--run", "qwen35_2b"],
    ["scripts/run_additional_experiments.sh", "--run", "qwen35_4b"],
    ["scripts/run_additional_experiments.sh", "--run", "olmo3_domains"],
    ["scripts/run_available_experiments.sh", "--once", "--first", "qwen35"],
])
def test_incomplete_primary_blocks_direct_and_rotation_compute(tmp_path, command):
    from test_generalization_launcher import checkout

    repo, env = checkout(tmp_path)
    work = Path(env["TEST_WORK"])
    (work / "results" / TAG / "COMPLETE").unlink()
    doctor = repo / "scripts/doctor_qwen35.sh"
    doctor.write_text('#!/bin/sh\necho unexpected > "$TEST_WORK/doctor-called"\n')
    doctor.chmod(0o755)
    for name in ("qwen35_2b_grpo.json", "qwen35_4b_grpo.json", "olmo3_domains_grpo.json"):
        (repo / "configs" / name).write_text("{}\n")
    result = subprocess.run(["bash", *command], cwd=repo, env=env,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 75, result.stdout + result.stderr
    assert "[primary-pending]" in result.stdout + result.stderr
    assert not (work / "gpu-preflights").exists()
    assert not (work / "phases").exists()
    assert not (work / "doctor-called").exists()
    assert "[fallback] starting" not in result.stdout


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("args", [[], ["run"], ["run-idle"]])
def test_explicit_qwen_defaults_to_idle_node_admission(tmp_path, blocked, args):
    from test_generalization_launcher import checkout

    repo, env = checkout(tmp_path)
    work = Path(env["TEST_WORK"])
    (work / "results" / TAG / "COMPLETE").unlink()
    local = Path(env["OM_LOCAL_LOCK_DIR"])
    local.mkdir()
    with (local / "primary.lock").open("a") as lock:
        if blocked:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", "scripts/run_qwen35_9b.sh", *args], cwd=repo,
            env={**env, "OM_WAIT_PRIMARY": "1"}, text=True, capture_output=True, timeout=15,
        )
    assert result.returncode == (1 if blocked else 0), result.stdout + result.stderr
    assert "explicit Qwen 9B exception" in result.stdout
    if blocked:
        assert not (work / "gpu-preflights").exists()
        assert not (work / "phases").exists()
    else:
        assert len((work / "phases").read_text().splitlines()) == 1
        assert (work / "phases").read_text().startswith("qwen35-9b-")
    assert not (work / "results" / TAG / "COMPLETE").exists()


def test_idle_node_exception_cannot_enable_rotation_or_another_host(tmp_path):
    from test_generalization_launcher import checkout

    repo, env = checkout(tmp_path)
    work = Path(env["TEST_WORK"])
    (work / "results" / TAG / "COMPLETE").unlink()
    for command, host in [
        (["scripts/run_available_experiments.sh", "--once"], os.uname().nodename),
        (["scripts/run_additional_experiments.sh", "--run", "qwen35"], "another-node"),
        (["scripts/run_additional_experiments.sh", "--run", "qwen38"], os.uname().nodename),
    ]:
        result = subprocess.run(["bash", *command], cwd=repo,
                                env={**env, "OM_QWEN_IDLE_NODE": host},
                                text=True, capture_output=True, timeout=10)
        assert result.returncode == 75, result.stdout + result.stderr
    assert not (work / "gpu-preflights").exists()
    assert not (work / "phases").exists()


def test_restart_idle_stops_only_old_local_qwen_scope(tmp_path):
    from test_generalization_launcher import checkout

    repo, env = checkout(tmp_path)
    work = Path(env["TEST_WORK"])
    (work / "results" / TAG / "COMPLETE").unlink()
    ready = work / "old-qwen-ready"
    old = subprocess.Popen(
        ["bash", "-c", 'sleep 120 & echo $! > "$TEST_WORK/old-child"; '
         'touch "$TEST_WORK/old-qwen-ready"; wait'],
        env={**env, "OM_WORK": str(work),
             "REGIME_ROOT": str(work / "runs/qwen35-9b-posttrained-math-code-grpo-v1/m1")},
        start_new_session=True,
    )
    olmo = subprocess.Popen(["sleep", "120"], env={**env, "OM_WORK": str(work),
                            "REGIME_ROOT": str(work / "runs" / TAG)}, start_new_session=True)
    other_work = subprocess.Popen(["sleep", "120"], env={**env, "OM_WORK": str(work / "other"),
        "REGIME_ROOT": str(work / "other/runs/qwen35-9b-posttrained-math-code-grpo-v1/m1")},
        start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        result = subprocess.run(["bash", "scripts/run_qwen35_9b.sh", "restart-idle"],
                                cwd=repo, env=env, text=True, capture_output=True, timeout=25)
        assert result.returncode == 0, result.stdout + result.stderr
        assert old.wait(timeout=5) != 0
        child = Path("/proc") / (work / "old-child").read_text().strip()
        assert not child.exists() or child.joinpath("stat").read_text().rsplit(") ", 1)[1].startswith("Z ")
        assert olmo.poll() is None
        assert other_work.poll() is None
        assert len((work / "phases").read_text().splitlines()) == 1
    finally:
        for process in (old, olmo, other_work):
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
