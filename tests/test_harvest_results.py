"""The final harvest is validated, compact, atomic, and content-addressed."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("gsm8k", "math500")
DRIFTS = (0, 25, 100, 400)
POLICIES = ("stale_g00", "stale_g10", "stale_g01", "stale_g11", "passrate_beta")
ANALYSIS_FILES = ("REGIME.json", "REGIME.csv", "REGIME_SUMMARY.csv", "FINAL_REPORT.md")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh_marker(result: Path) -> None:
    lines = ["0" * 64]
    for name in ANALYSIS_FILES:
        digest = hashlib.sha256((result / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {result / name}")
    (result / ".regime_analysis.key").write_text("\n".join(lines) + "\n")


def write_analysis(
    result: Path,
    model: str,
    seeds: tuple[int, ...],
    generation_git: str = "a" * 40,
) -> None:
    result.mkdir(parents=True)
    rows = [
        {
            "run": f"run-{dataset}-s{seed}-d{drift}",
            "generation_git": generation_git,
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "drift": drift,
            "stratum": "all",
            "policy": policy,
            "final_resampling": True,
            "utility_gain": 0.25,
        }
        for dataset in DATASETS
        for seed in seeds
        for drift in DRIFTS
        for policy in POLICIES
    ]
    summary = [
        {
            "generation_git": generation_git,
            "model": model,
            "dataset": dataset,
            "drift": drift,
            "stratum": "all",
            "policy": policy,
            "seeds": len(seeds),
            "status": "effective",
        }
        for dataset in DATASETS
        for drift in DRIFTS
        for policy in POLICIES
    ]
    document = {
        "schema": "offpolicy-regime-map/v4",
        "analysis_git": "d" * 40,
        "generation_commits": [generation_git],
        "topk_frac": 0.10,
        "retention_threshold": 0.50,
        "replication_fraction": 0.80,
        "first_bootstrap": 10_000,
        "minimum_final_bootstrap": 10_000,
        "rows": rows,
        "summary": summary,
    }
    (result / "REGIME.json").write_text(json.dumps(document), encoding="utf-8")
    write_csv(result / "REGIME.csv", rows)
    write_csv(result / "REGIME_SUMMARY.csv", summary)
    (result / "FINAL_REPORT.md").write_text(f"# {model}\n", encoding="utf-8")
    refresh_marker(result)


def checkout(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    checkout_root = tmp_path / "checkout"
    work = tmp_path / "shared"
    (checkout_root / "scripts").mkdir(parents=True)
    (checkout_root / "src").mkdir()
    shutil.copy2(
        ROOT / "scripts/harvest_results.sh",
        checkout_root / "scripts/harvest_results.sh",
    )
    shutil.copy2(
        ROOT / "src/harvest_results.py", checkout_root / "src/harvest_results.py"
    )
    (checkout_root / "scripts/setup_env.sh").write_text(
        'export OM_WORK="$TEST_WORK"\nexport VENV_DIR="$TEST_WORK/venv"\n'
    )
    (work / "venv/bin").mkdir(parents=True)
    (work / "venv/bin/python").symlink_to(sys.executable)
    write_analysis(
        work / "results/regime-qwen2.5-7b-grpo-v1",
        "/models/Qwen2.5-7B-Instruct",
        (0, 1, 2),
    )
    write_analysis(
        work / "results/regime-qwen3.8-27b-grpo-v1",
        "/models/Qwen3.8-27B-BF16",
        (0, 1, 2, 3, 4),
    )
    subprocess.run(["git", "init", "-q"], cwd=checkout_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.com"],
        cwd=checkout_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "fixture"], cwd=checkout_root, check=True
    )
    subprocess.run(["git", "add", "."], cwd=checkout_root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout_root, check=True)
    return checkout_root, work, {**os.environ, "TEST_WORK": str(work)}


def run_harvest(
    checkout_root: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/harvest_results.sh"],
        cwd=checkout_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_harvest_publishes_exact_bundle_and_skips_unchanged_inputs(
    tmp_path: Path,
) -> None:
    checkout_root, work, env = checkout(tmp_path)

    first = run_harvest(checkout_root, env)
    assert first.returncode == 0, first.stdout + first.stderr
    bundle = work / "readouts/rlvr-grpo"
    assert {path.name for path in bundle.iterdir()} == {
        "REPORT.md",
        "RESULTS.json",
        "RESULTS.csv",
        "MANIFEST.sha256",
    }
    document = json.loads((bundle / "RESULTS.json").read_text())
    assert set(document) == {
        "schema",
        "generation_git",
        "harvest_git",
        "input_digest",
        "primary_27b",
        "replication_7b",
    }
    assert document["schema"] == "offpolicy-rlvr-harvest/v3"
    assert document["generation_git"] == "a" * 40

    second = run_harvest(checkout_root, env)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "inputs unchanged; reuse" in second.stdout

    (bundle / "stale.txt").write_text("must not survive\n")
    third = run_harvest(checkout_root, env)
    assert third.returncode == 0, third.stdout + third.stderr
    assert "published" in third.stdout
    assert not (bundle / "stale.txt").exists()

    script = checkout_root / "scripts/harvest_results.sh"
    script.write_text(script.read_text() + "\n# cache-key change\n")
    fourth = run_harvest(checkout_root, env)
    assert fourth.returncode == 0, fourth.stdout + fourth.stderr
    changed = json.loads((bundle / "RESULTS.json").read_text())
    assert changed["input_digest"] != document["input_digest"]
    assert [path.name for path in (work / "readouts").iterdir()] == ["rlvr-grpo"]


def test_harvest_replaces_bundle_and_removes_legacy_layout(tmp_path: Path) -> None:
    checkout_root, work, env = checkout(tmp_path)
    legacy = work / "readouts/rlvr-grpo-20260831-010203-deadbeef"
    legacy.mkdir(parents=True)
    (legacy / "old.txt").write_text("old\n")
    pointer = work / "results/.rlvr-harvest-current"
    pointer.write_text(f"old\n{legacy}\n")

    result = run_harvest(checkout_root, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not legacy.exists()
    assert not pointer.exists()
    assert {path.name for path in (work / "readouts/rlvr-grpo").iterdir()} == {
        "REPORT.md",
        "RESULTS.json",
        "RESULTS.csv",
        "MANIFEST.sha256",
    }


def test_concurrent_harvesters_publish_one_valid_bundle(tmp_path: Path) -> None:
    checkout_root, work, env = checkout(tmp_path)
    workers = [
        subprocess.Popen(
            ["bash", "scripts/harvest_results.sh"],
            cwd=checkout_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(3)
    ]
    outputs = []
    for worker in workers:
        output, _ = worker.communicate(timeout=10)
        outputs.append(output)
        assert worker.returncode == 0, output
    assert sum("published" in output for output in outputs) == 1
    assert sum("inputs unchanged; reuse" in output for output in outputs) == 2
    bundle = work / "readouts/rlvr-grpo"
    subprocess.run(
        ["sha256sum", "-c", "MANIFEST.sha256"],
        cwd=bundle,
        check=True,
        capture_output=True,
    )


def test_harvest_rejects_stale_analysis_marker_without_replacing_bundle(
    tmp_path: Path,
) -> None:
    checkout_root, work, env = checkout(tmp_path)
    first = run_harvest(checkout_root, env)
    assert first.returncode == 0, first.stdout + first.stderr
    bundle = work / "readouts/rlvr-grpo"
    before = {path.name: path.read_bytes() for path in bundle.iterdir()}

    source = work / "results/regime-qwen3.8-27b-grpo-v1/REGIME.csv"
    source.write_text(source.read_text() + "corrupt,row\n")
    failed = run_harvest(checkout_root, env)
    assert failed.returncode != 0
    assert "content differs from analysis marker" in failed.stderr
    assert {path.name: path.read_bytes() for path in bundle.iterdir()} == before


def test_harvest_rejects_csv_that_disagrees_with_json(tmp_path: Path) -> None:
    checkout_root, work, env = checkout(tmp_path)
    result = work / "results/regime-qwen3.8-27b-grpo-v1"
    rows = list(csv.DictReader((result / "REGIME.csv").open()))
    rows[0]["utility_gain"] = "999"
    write_csv(result / "REGIME.csv", rows)
    refresh_marker(result)

    failed = run_harvest(checkout_root, env)
    assert failed.returncode != 0
    assert "row differs from REGIME.json" in failed.stderr
    assert not (work / "readouts/rlvr-grpo").exists()


def test_harvest_rejects_incomplete_registered_matrix(tmp_path: Path) -> None:
    checkout_root, work, env = checkout(tmp_path)
    result = work / "results/regime-qwen3.8-27b-grpo-v1"
    document = json.loads((result / "REGIME.json").read_text())
    document["rows"] = [
        row
        for row in document["rows"]
        if not (
            row["dataset"] == "math500" and row["seed"] == 4 and row["drift"] == 400
        )
    ]
    (result / "REGIME.json").write_text(json.dumps(document))
    write_csv(result / "REGIME.csv", document["rows"])
    refresh_marker(result)

    failed = run_harvest(checkout_root, env)
    assert failed.returncode != 0
    assert "incomplete registered matrix" in failed.stderr
    assert not (work / "readouts/rlvr-grpo").exists()


def test_harvest_rejects_swapped_primary_and_replication_roots(tmp_path: Path) -> None:
    checkout_root, work, env = checkout(tmp_path)
    env.update(
        {
            "RLVR_RESULTS_7B": str(work / "results/regime-qwen3.8-27b-grpo-v1"),
            "RLVR_RESULTS_27B": str(work / "results/regime-qwen2.5-7b-grpo-v1"),
        }
    )
    failed = run_harvest(checkout_root, env)
    assert failed.returncode != 0
    assert (
        "incomplete registered matrix" in failed.stderr
        or "outside the registered matrix" in failed.stderr
    )
    assert not (work / "readouts/rlvr-grpo").exists()


def test_harvest_rejects_different_generation_commits(tmp_path: Path) -> None:
    checkout_root, work, env = checkout(tmp_path)
    replication = work / "results/regime-qwen2.5-7b-grpo-v1"
    shutil.rmtree(replication)
    write_analysis(
        replication,
        "/models/Qwen2.5-7B-Instruct",
        (0, 1, 2),
        generation_git="b" * 40,
    )

    failed = run_harvest(checkout_root, env)
    assert failed.returncode != 0
    assert "different generation commits" in failed.stderr
    assert not (work / "readouts/rlvr-grpo").exists()
