"""The final harvest is compact and content-addressed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_harvest_publishes_four_files_and_skips_unchanged_inputs(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    work = tmp_path / "shared"
    (checkout / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/harvest_results.sh", checkout / "scripts/harvest_results.sh")
    (checkout / "scripts/setup_env.sh").write_text(
        'export OM_WORK="$TEST_WORK"\nexport VENV_DIR="$TEST_WORK/venv"\n'
    )
    (work / "venv/bin").mkdir(parents=True)
    (work / "venv/bin/python").symlink_to(sys.executable)
    for tag in ("qwen2.5-7b-grpo-v1", "qwen3.8-27b-grpo-v1"):
        result = work / "results" / f"regime-{tag}"
        result.mkdir(parents=True)
        (result / "REGIME.json").write_text(json.dumps({"model": tag, "rows": []}))
        (result / "REGIME.csv").write_text("dataset,seed,drift\ngsm8k,0,25\n")
        (result / "REGIME_SUMMARY.csv").write_text("dataset,count\ngsm8k,1\n")
        (result / "FINAL_REPORT.md").write_text(f"# {tag}\n")
        (result / ".regime_analysis.key").write_text("key\n")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout, check=True)
    env = {**os.environ, "TEST_WORK": str(work)}

    first = subprocess.run(
        ["bash", "scripts/harvest_results.sh"], cwd=checkout, env=env,
        text=True, capture_output=True, check=False,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    bundle = work / "readouts/rlvr-grpo"
    assert {path.name for path in bundle.iterdir()} == {
        "REPORT.md", "RESULTS.json", "RESULTS.csv", "MANIFEST.sha256"
    }
    document = json.loads((bundle / "RESULTS.json").read_text())
    assert set(document) == {"schema", "git", "input_digest", "primary_27b", "replication_7b"}

    second = subprocess.run(
        ["bash", "scripts/harvest_results.sh"], cwd=checkout, env=env,
        text=True, capture_output=True, check=False,
    )
    assert second.returncode == 0
    assert "inputs unchanged; reuse" in second.stdout
    assert [path.name for path in (work / "readouts").iterdir()] == ["rlvr-grpo"]


def test_harvest_replaces_bundle_and_removes_legacy_layout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    work = tmp_path / "shared"
    (checkout / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/harvest_results.sh", checkout / "scripts/harvest_results.sh")
    (checkout / "scripts/setup_env.sh").write_text(
        'export OM_WORK="$TEST_WORK"\nexport VENV_DIR="$TEST_WORK/venv"\n'
    )
    (work / "venv/bin").mkdir(parents=True)
    (work / "venv/bin/python").symlink_to(sys.executable)
    for tag in ("qwen2.5-7b-grpo-v1", "qwen3.8-27b-grpo-v1"):
        result = work / "results" / f"regime-{tag}"
        result.mkdir(parents=True)
        (result / "REGIME.json").write_text(json.dumps({"model": tag, "rows": []}))
        (result / "REGIME.csv").write_text("dataset,seed,drift\ngsm8k,0,25\n")
        (result / "REGIME_SUMMARY.csv").write_text("dataset,count\ngsm8k,1\n")
        (result / "FINAL_REPORT.md").write_text(f"# {tag}\n")
        (result / ".regime_analysis.key").write_text("key\n")
    legacy = work / "readouts/rlvr-grpo-20260831-010203-deadbeef"
    legacy.mkdir(parents=True)
    (legacy / "old.txt").write_text("old\n")
    pointer = work / "results/.rlvr-harvest-current"
    pointer.write_text(f"old\n{legacy}\n")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout, check=True)

    result = subprocess.run(
        ["bash", "scripts/harvest_results.sh"], cwd=checkout,
        env={**os.environ, "TEST_WORK": str(work)}, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not legacy.exists()
    assert not pointer.exists()
    assert {path.name for path in (work / "readouts/rlvr-grpo").iterdir()} == {
        "REPORT.md", "RESULTS.json", "RESULTS.csv", "MANIFEST.sha256"
    }
