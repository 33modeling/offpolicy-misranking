"""Canonical launcher locks hardware and both registered matrices."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def checkout(tmp_path: Path, gpu_count: int = 4) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "checkout"
    work = tmp_path / "shared"
    fake_bin = tmp_path / "bin"
    (root / "scripts").mkdir(parents=True)
    (work / "venv/bin").mkdir(parents=True)
    for model in ("Qwen2.5-7B-Instruct", "Qwen3.8-27B-BF16"):
        directory = work / "models" / model
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}\n")
        (directory / "tokenizer_config.json").write_text("{}\n")
    fake_bin.mkdir()
    (root / "scripts/run_rlvr.sh").write_text(
        (ROOT / "scripts/run_rlvr.sh").read_text()
    )
    (work / "venv/bin/python").symlink_to("/usr/bin/python3")
    (root / "scripts/setup_env.sh").write_text(
        'export GROUP_VOLUME="$TEST_WORK"\n'
        'export OM_WORK="$TEST_WORK"\n'
        'export VENV_DIR="$TEST_WORK/venv"\n'
        'export MODELS_DIR="$TEST_WORK/models"\n'
        'export DATASETS_DIR="$TEST_WORK/data"\n'
    )
    executable(root / "scripts/check_data.sh", "#!/bin/sh\nexit 0\n")
    (root / "scripts/check_27b_fla.py").write_text("print('fla-ok')\n")
    executable(
        root / "scripts/run_matrix.sh",
        "#!/usr/bin/env bash\n"
        '[ "${TEST_MATRIX_DELAY:-0}" = 0 ] || /bin/sleep 0.25\n'
        "python3 - <<'PY'\n"
        "import json, os\n"
        "p=os.environ['TEST_WORK']+'/phases.jsonl'\n"
        "row={k:os.environ[k] for k in ('MODEL_PATH','REGIME_MODEL_TAG','REGIME_DATASETS','REGIME_SEEDS','REGIME_DRIFTS','GRPO_WORLD_SIZE','GRPO_GROUP_SIZE','OM_ALLOW_ANALYSIS_UPGRADE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE')}\n"
        "row['hf_token_present']='HF_TOKEN' in os.environ or 'HUGGING_FACE_HUB_TOKEN' in os.environ\n"
        "with open(p,'a') as f: f.write(json.dumps(row)+'\\n')\n"
        "PY\n"
        'exit "${TEST_MATRIX_RC:-0}"\n',
    )
    executable(root / "scripts/harvest_results.sh", "#!/bin/sh\nprintf harvested\\n\n")
    executable(fake_bin / "hostname", "#!/bin/sh\nprintf cloned-cluster\\n\n")
    executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  *memory.used*) " + "printf '0\\n'\n" * gpu_count + " ;;\n"
        "  *) " + "printf 'NVIDIA H100 80GB HBM3\\n'\n" * gpu_count + " ;;\n"
        "esac\n",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.com"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_WORK": str(work),
        "OM_LOCAL_LOCK_DIR": str(tmp_path / "local-locks"),
    }
    return root, env


def test_one_command_runs_exact_7b_and_27b_matrices(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    env["HF_TOKEN"] = "must-not-reach-compute"
    env["HUGGING_FACE_HUB_TOKEN"] = "must-not-reach-compute"
    result = subprocess.run(
        ["/bin/bash", "scripts/run_rlvr.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    phases = [
        json.loads(line)
        for line in (Path(env["TEST_WORK"]) / "phases.jsonl").read_text().splitlines()
    ]
    assert len(phases) == 2
    assert phases[0]["REGIME_MODEL_TAG"] == "qwen3.8-27b-grpo-v1"
    assert phases[0]["REGIME_SEEDS"] == "0 1 2 3 4"
    assert phases[0]["REGIME_DRIFTS"] == "0 25 100 400"
    assert phases[1]["REGIME_MODEL_TAG"] == "qwen2.5-7b-grpo-v1"
    assert phases[1]["REGIME_SEEDS"] == "0 1 2"
    assert phases[1]["REGIME_DRIFTS"] == "0 25 100 400"
    assert all(row["GRPO_WORLD_SIZE"] == "4" for row in phases)
    assert all(row["GRPO_GROUP_SIZE"] == "8" for row in phases)
    assert all(row["OM_ALLOW_ANALYSIS_UPGRADE"] == "1" for row in phases)
    assert all(row["hf_token_present"] is False for row in phases)
    assert all(row["HF_HUB_OFFLINE"] == "1" for row in phases)
    assert all(row["TRANSFORMERS_OFFLINE"] == "1" for row in phases)
    assert all(row["HF_DATASETS_OFFLINE"] == "1" for row in phases)
    assert "harvested" in result.stdout


def test_launcher_rejects_non_four_h100_node(tmp_path: Path) -> None:
    root, env = checkout(tmp_path, gpu_count=3)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_rlvr.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "exactly four H100 GPUs required" in result.stdout
    assert not (Path(env["TEST_WORK"]) / "phases.jsonl").exists()


def test_three_same_hostname_clusters_all_enter_shared_matrix_queue(
    tmp_path: Path,
) -> None:
    root, base_env = checkout(tmp_path)
    workers = []
    for index in range(3):
        env = {
            **base_env,
            "TEST_MATRIX_DELAY": "1",
            "OM_LOCAL_LOCK_DIR": str(tmp_path / f"cluster-{index}/local-locks"),
        }
        workers.append(
            subprocess.Popen(
                ["/bin/bash", "scripts/run_rlvr.sh"],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )

    outputs = []
    for worker in workers:
        output, _ = worker.communicate(timeout=20)
        outputs.append(output)
        assert worker.returncode == 0, output
    phases = [
        json.loads(line)
        for line in (Path(base_env["TEST_WORK"]) / "phases.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(phases) == 6
    assert sum(row["REGIME_MODEL_TAG"] == "qwen3.8-27b-grpo-v1" for row in phases) == 3
    assert sum(row["REGIME_MODEL_TAG"] == "qwen2.5-7b-grpo-v1" for row in phases) == 3
    assert all("shared_queue=" in output for output in outputs)


def test_launcher_rejects_a_shared_node_lock_directory(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    env["OM_LOCAL_LOCK_DIR"] = str(Path(env["TEST_WORK"]) / "locks/local")
    result = subprocess.run(
        ["/bin/bash", "scripts/run_rlvr.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "OM_LOCAL_LOCK_DIR must be node-local" in result.stdout
    assert not (Path(env["TEST_WORK"]) / "phases.jsonl").exists()


def test_permanent_contract_failure_is_not_retried(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    env["TEST_MATRIX_RC"] = "43"
    result = subprocess.run(
        ["/bin/bash", "scripts/run_rlvr.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 43, result.stdout + result.stderr
    phases = (Path(env["TEST_WORK"]) / "phases.jsonl").read_text().splitlines()
    assert len(phases) == 1
    assert "permanent prompt/contract failure; not retrying" in result.stdout


def test_two_workers_on_one_physical_node_do_not_oversubscribe_gpus(
    tmp_path: Path,
) -> None:
    root, env = checkout(tmp_path)
    env["TEST_MATRIX_DELAY"] = "1"
    first = subprocess.Popen(
        ["/bin/bash", "scripts/run_rlvr.sh"],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_dir = Path(env["TEST_WORK"]) / "console-logs"
    for _ in range(100):
        if list(log_dir.glob("rlvr-*.log")):
            break
        time.sleep(0.01)
    second = subprocess.run(
        ["/bin/bash", "scripts/run_rlvr.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    first_output, _ = first.communicate(timeout=20)
    assert first.returncode == 0, first_output
    assert second.returncode != 0
    assert "already running on this physical node" in second.stdout
