"""Cluster-wide OLMo-first admission, without GPU or model processes."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

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
    ["scripts/run_qwen35_9b.sh"],
    ["scripts/run_qwen35_9b.sh", "check"],
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
