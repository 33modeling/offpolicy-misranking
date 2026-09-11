"""Progress diagnostics must preserve frozen E5 code and work without CUDA."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import downstream_status as status

ROOT = Path(__file__).resolve().parents[1]


def contract(out):
    out.mkdir(parents=True)
    (out / "experiment.json").write_text(json.dumps({
        "seed": 0, "drift": 400, "steps": 100, "eval_prompts": 300,
        "eval_k": 8, "selectors": ["random", "fresh_r", "g11"],
    }))


def test_progress_only_changes_preserve_the_running_experiment_hash():
    # a1004ee/e6bf22e/ab21fee outputs bind this complete file, including status.
    expected = "14649d8ab4e1d1768c91d7c7f14aea0ae5bf04aa7719f1e33173abf5490de342"
    assert hashlib.sha256((ROOT / "src/evidence_downstream.py").read_bytes()).hexdigest() == expected


def test_status_cli_without_site_packages_or_prepared_output(tmp_path):
    result = subprocess.run([sys.executable, "-S", str(ROOT / "src/downstream_status.py"),
                             "--out", str(tmp_path)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0 and result.stdout.strip() == "not prepared"


def test_status_preserves_training_checkpoint_details(tmp_path):
    contract(tmp_path / "out")
    out = tmp_path / "out"
    policy = out / "g11/policy"
    (policy / "checkpoint-000405").mkdir(parents=True)
    (policy / "checkpoint-incomplete").mkdir()
    (policy / "grpo_stats.jsonl").write_text("{}\n{}\n")
    text = status.train_state(out, "g11")
    assert "2/100 updates" in text and "step 405" in text
    assert status.arm_state(out, "random") == "not started"


def test_status_preserves_completed_and_partial_shard_counts(tmp_path):
    contract(tmp_path / "out")
    out = tmp_path / "out"
    target = out / "before/evaluation"
    target.mkdir(parents=True)
    (target / "shard-0.done.json").write_text("{}")
    (target / "shard-1.jsonl.partial").write_text("{}\n{}\n")
    lines = status.shard_states(out, "before")
    assert "done (600 responses)" in lines[0]
    assert "2/600 responses" in lines[1]
    assert status.arm_state(out, "before") == "evaluating (1/4 shards done)"


def test_log_tail_is_bounded_and_keeps_the_latest_nonempty_line(tmp_path):
    log = tmp_path / "log"
    log.write_text("x" * 100_000 + "\n[progress] latest\n\n")
    assert status._tail(log) == "[progress] latest"


def test_existing_shell_status_command_reaches_the_separate_diagnostics(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/run_e5.sh", repo / "scripts")
    (repo / "scripts/setup_env.sh").write_text("true\n")
    (repo / "src").symlink_to(ROOT / "src", target_is_directory=True)
    work = tmp_path / "work"
    out = work / "runs/e5-reduced/math500-d400/s0"
    contract(out)
    (out / "logs").mkdir()
    (out / "logs/eval-before-0.log").write_text("loading model\n")
    result = subprocess.run(["bash", "scripts/run_e5.sh", "status"], cwd=repo,
                            env={**os.environ, "OM_WORK": str(work), "E5_SEEDS": "0",
                                 "VENV_DIR": str(Path(sys.executable).parent.parent),
                                 "DATASETS_DIR": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "shard 0: started, no responses yet" in result.stdout
    assert "seed 0 d400 steps=100 eval_k=8 test=300" in result.stdout
