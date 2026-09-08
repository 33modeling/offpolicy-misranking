"""The generalization launcher binds and runs every configured model matrix."""

from __future__ import annotations

import fcntl
import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def checkout(tmp_path: Path, gpu_count: int = 4) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "checkout"
    work = tmp_path / "shared"
    fake_bin = tmp_path / "bin"
    (root / "scripts").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "configs").mkdir()
    (root / "scripts/launch_logging.sh").write_text(
        (ROOT / "scripts/launch_logging.sh").read_text(), encoding="utf-8"
    )
    (root / "src/diagnose_launch_failure.py").write_text(
        (ROOT / "src/diagnose_launch_failure.py").read_text(), encoding="utf-8"
    )
    (work / "venv/bin").mkdir(parents=True)
    (work / "models/m1").mkdir(parents=True)
    (work / "models/m2").mkdir(parents=True)
    fake_bin.mkdir()
    (root / "scripts/run_additional_experiments.sh").write_text(
        (ROOT / "scripts/run_additional_experiments.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for name in (
        "generalization_logic.json",
        "generalization_science.json",
        "generalization_knowledge.json",
        "qwen38_27b_grpo.json",
        "qwen35_9b_grpo.json",
    ):
        (root / "configs" / name).write_text("{}\n", encoding="utf-8")
    (root / "scripts/setup_env.sh").write_text(
        'export GROUP_VOLUME="$TEST_WORK"\n'
        'export OM_WORK="$TEST_WORK"\n'
        'export VENV_DIR="$TEST_WORK/venv"\n'
        'export MODELS_DIR="$TEST_WORK/models"\n'
        'export DATASETS_DIR="$TEST_WORK/data"\n',
        encoding="utf-8",
    )
    executable(
        work / "venv/bin/python",
        "#!/usr/bin/env bash\n"
        "script=$1; shift\n"
        'case "$script" in\n'
        "  src/model_matrix.py)\n"
        '    case " $* " in\n'
        "      *'qwen35_9b_grpo.json list-models '*) printf 'm1\\n' ;;\n"
        "      *'qwen35_9b_grpo.json experiment-field datasets '*) printf 'math500 mbpp\\n' ;;\n"
        "      *'qwen38_27b_grpo.json list-models '*) printf 'm1\\n' ;;\n"
        "      *'qwen38_27b_grpo.json experiment-field datasets '*) printf 'math500 mbpp\\n' ;;\n"
        "      *' runtime-field '*) printf '1\\n' ;;\n"
        "      *' list-models '*) printf 'm1\\nm2\\n' ;;\n"
        "      *'generalization_logic.json experiment-field datasets '*) printf 'kk\\n' ;;\n"
        "      *'generalization_science.json experiment-field datasets '*) printf 'arc-challenge\\n' ;;\n"
        "      *'generalization_knowledge.json experiment-field datasets '*) printf 'mmlu-pro-nonmath\\n' ;;\n"
        "      *' dataset-n-train '*) printf '512\\n' ;;\n"
        "      *' experiment-field policy_method '*) printf 'grpo\\n' ;;\n"
        "      *' experiment-field seeds '*) printf '0 1 2\\n' ;;\n"
        "      *' experiment-field drifts '*) printf '0 25 100 400\\n' ;;\n"
        "      *' experiment-field n_train '*) printf '512\\n' ;;\n"
        "      *' experiment-field n_val '*) printf '100\\n' ;;\n"
        "      *' experiment-field behavior_k '*) printf '8\\n' ;;\n"
        "      *' experiment-field fresh_k '*) printf '32\\n' ;;\n"
        "      *' experiment-field val_k '*) printf '8\\n' ;;\n"
        "      *' experiment-field micro_group '*) printf '4\\n' ;;\n"
        "      *' experiment-field max_new_tokens '*) printf '512\\n' ;;\n"
        "      *' experiment-field proj_dim '*) printf '4096\\n' ;;\n"
        "      *' experiment-field grad_layers '*) printf '4\\n' ;;\n"
        "      *' experiment-field clip_cap '*) printf '10.0\\n' ;;\n"
        "      *' experiment-field topk_frac '*) printf '0.1\\n' ;;\n"
        "      *' experiment-field temperature '*) printf '1.0\\n' ;;\n"
        "      *' experiment-field first_bootstrap '*) printf '10000\\n' ;;\n"
        "      *' experiment-field top_p '*) printf '1.0\\n' ;;\n"
        "      *' experiment-field thinking '*) printf 'off\\n' ;;\n"
        "      *' experiment-field attn '*) printf 'eager\\n' ;;\n"
        "      *' experiment-field skip_hybrid '*) printf '1\\n' ;;\n"
        "      *' grpo-field world_size '*) printf '4\\n' ;;\n"
        "      *' grpo-field group_size '*) printf '8\\n' ;;\n"
        "      *' grpo-field clip_epsilon '*) printf '0.2\\n' ;;\n"
        "      *' grpo-field learning_rate '*) printf '1e-5\\n' ;;\n"
        "      *' grpo-field reference_kl_beta '*) printf '0.0\\n' ;;\n"
        "      *' grpo-field epochs_per_batch '*) printf '1\\n' ;;\n"
        "      *' grpo-field max_grad_norm '*) printf '1.0\\n' ;;\n"
        "      *' grpo-field advantage_epsilon '*) printf '1e-4\\n' ;;\n"
        "      *' grpo-field lora_rank '*) printf '16\\n' ;;\n"
        "      *' grpo-field lora_alpha '*) printf '32\\n' ;;\n"
        "      *' field m1 path '*) printf '%s/models/m1\\n' \"$TEST_WORK\" ;;\n"
        "      *' field m2 path '*) printf '%s/models/m2\\n' \"$TEST_WORK\" ;;\n"
        "      *' field '*' lora_targets '*) printf 'q_proj,v_proj\\n' ;;\n"
        "      *' field '*' prompt_format '*) printf 'verifiable_completion\\n' ;;\n"
        "      *' check '*) printf '[check] pass\\n' ;;\n"
        "      *) printf 'unexpected model_matrix args: %s\\n' \"$*\" >&2; exit 2 ;;\n"
        "    esac ;;\n"
        "  src/qualify_domain_data.py)\n"
        '    printf \'%s\\n\' "$*" >> "$TEST_WORK/qualifications"\n'
        '    while [ $# -gt 0 ]; do [ "$1" = --output ] && { printf \'{}\\n\' > "$2"; break; }; shift; done ;;\n'
        "  src/transfer_smoke.py)\n"
        '    [ "${TEST_SMOKE_HANG:-0}" = 0 ] || sleep 30\n'
        '    [ "${TEST_SMOKE_FAIL:-0}" = 0 ] || { echo "synthetic smoke traceback" >&2; exit 17; }\n'
        '    while [ $# -gt 0 ]; do [ "$1" = --marker ] && { mkdir -p "$(dirname "$2")"; printf \'{}\\n\' > "$2"; break; }; shift; done ;;\n'
        "  scripts/check_27b_fla.py) exit 0 ;;\n"
        "  src/locate_uploaded_snapshot.py)\n"
        '    [ "${TEST_MODEL_MISSING:-0}" = 0 ] || { echo "model missing" >&2; exit 1; }\n'
        "    case \" $* \" in *'--model-key m2 '*) printf '%s/models/m2\\n' \"$TEST_WORK\" ;; *) printf '%s/models/m1\\n' \"$TEST_WORK\" ;; esac ;;\n"
        "  src/bootstrap_math_verify.py) printf '%s/runtime-deps/math-verify-test\\n' \"$TEST_WORK\" ;;\n"
        "  src/training_progress.py)\n"
        '    case " $* " in *" --watch "*) echo $$ > "$TEST_WORK/progress.pid"; exec sleep 300 ;; esac ;;\n'
        "  -c) exit 0 ;;\n"
        "  src/regime_contract.py)\n"
        '    while [ $# -gt 0 ]; do [ "$1" = --matrix ] && { mkdir -p "$(dirname "$2")"; printf \'{}\\n\' > "$2"; break; }; shift; done ;;\n'
        "  *) printf 'unexpected script: %s\\n' \"$script\" >&2; exit 2 ;;\n"
        "esac\n",
    )
    executable(
        root / "scripts/run_matrix.sh",
        "#!/usr/bin/env bash\n"
        '[ -z "${HF_TOKEN+x}" ] || { printf "HF_TOKEN leaked\\n" >&2; exit 91; }\n'
        '[ -z "${HUGGING_FACE_HUB_TOKEN+x}" ] || { printf "legacy token leaked\\n" >&2; exit 92; }\n'
        '[ "$HF_HUB_OFFLINE" = 1 ] && [ "$TRANSFORMERS_OFFLINE" = 1 ] && '
        '[ "$HF_DATASETS_OFFLINE" = 1 ] || { printf "offline flags missing\\n" >&2; exit 93; }\n'
        'printf \'%s\\n\' "$REGIME_MODEL_TAG|$MODEL_PATH|$REGIME_DATASETS|$REGIME_SEEDS|$REGIME_DRIFTS|$REGIME_MATRIX|$REGIME_N_TRAIN_BY_DATASET|$OM_PROMPT_FORMAT" >> "$TEST_WORK/phases"\n'
        'sleep 0.05\nexit "${TEST_MATRIX_RC:-0}"\n',
    )
    executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  *memory.used*) "
        + "printf '0\\n'\n" * gpu_count
        + "    ;;\n  *) "
        + "printf 'NVIDIA H100 80GB HBM3\\n'\n" * gpu_count
        + "    ;;\nesac\n",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.com"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root, {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_WORK": str(work),
        "ADDITIONAL_NODE_TAG": "test-node",
        "ADDITIONAL_SKIP_PROVISION": "1",
        "OM_LOCAL_LOCK_DIR": str(tmp_path / "local-locks"),
    }


def test_qwen_check_does_not_launch_matrix(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    result = subprocess.run(
        ["bash", "scripts/run_additional_experiments.sh", "--check", "qwen38"],
        cwd=root, env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (Path(env["TEST_WORK"]) / "phases").exists()
    assert list((Path(env["TEST_WORK"]) / "contracts").glob("*qwen38*smoke*"))


@pytest.mark.parametrize("rc", [1, 43, 75])
def test_failed_matrix_stops_progress_before_draining_logs(tmp_path, rc):
    root, env = checkout(tmp_path)
    env.update(TEST_MATRIX_RC=str(rc), ADDITIONAL_MAX_RESTARTS="0")
    process = subprocess.Popen(
        ["bash", "scripts/run_additional_experiments.sh", "--run", "qwen38"],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        out, err = process.communicate(timeout=10)
        assert process.returncode == rc, out + err
        pid = int((Path(env["TEST_WORK"]) / "progress.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        for name in ("primary.lock", "additional-suite.lock"):
            with (Path(env["OM_LOCAL_LOCK_DIR"]) / name).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def test_hung_smoke_has_a_deadline_and_never_starts_matrix(tmp_path):
    root, env = checkout(tmp_path)
    env.update(TEST_SMOKE_HANG="1", ADDITIONAL_SMOKE_TIMEOUT="1")
    result = subprocess.run(
        ["bash", "scripts/run_additional_experiments.sh", "--run", "qwen38"],
        cwd=root, env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 124, result.stdout + result.stderr
    assert "preflight=smoke exceeded 1s" in result.stdout + result.stderr
    assert not (Path(env["TEST_WORK"]) / "phases").exists()


def test_missing_model_yields_before_dataset_qualification(tmp_path):
    root, env = checkout(tmp_path)
    env["TEST_MODEL_MISSING"] = "1"
    result = subprocess.run(
        ["bash", "scripts/run_additional_experiments.sh", "--run", "qwen38"],
        cwd=root, env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 43, result.stdout + result.stderr
    assert not (Path(env["TEST_WORK"]) / "qualifications").exists()
    assert not (Path(env["TEST_WORK"]) / "phases").exists()


def test_qwen35_logs_failure_stderr_and_exit_status(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    env["TEST_SMOKE_FAIL"] = "1"
    result = subprocess.run(
        ["bash", "scripts/run_additional_experiments.sh", "--check", "qwen35"],
        cwd=root, env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    logs = list((Path(env["TEST_WORK"]) / "console-logs").glob("additional-qwen35-check-*.log"))
    assert len(logs) == 1
    content = logs[0].read_text()
    assert "synthetic smoke traceback" in content
    assert "rc=17 stage=smoke-m1" in content
    assert "[launch] utc=" in content
    assert not (Path(env["TEST_WORK"]) / "phases").exists()


def test_qwen35_prepare_and_run_have_separate_session_logs(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    for mode in ("--prepare", "--run"):
        result = subprocess.run(
            ["bash", "scripts/run_additional_experiments.sh", mode, "qwen35"],
            cwd=root, env=env, capture_output=True, text=True, timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    work = Path(env["TEST_WORK"])
    assert (work / "phases").read_text().startswith("qwen35-9b-posttrained-")
    logs = list((work / "console-logs").glob("additional-qwen35-*.log"))
    assert len(logs) == 2
    assert all("rc=0" in log.read_text() for log in logs)


def test_qwen_run_uses_separate_matrix(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    result = subprocess.run(
        ["bash", "scripts/run_additional_experiments.sh", "--run", "qwen38"],
        cwd=root, env=env, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    phases = (Path(env["TEST_WORK"]) / "phases").read_text().splitlines()
    assert len(phases) == 1
    assert phases[0].startswith("qwen38-27b-posttrained-math-code-grpo-v1-grpo-m1|")
    assert "|math500 mbpp|" in phases[0]


def test_launcher_runs_all_generalization_strata_in_order(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_additional_experiments.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    phases = (Path(env["TEST_WORK"]) / "phases").read_text().splitlines()
    assert len(phases) == 6
    assert phases[0].startswith("generalization-logic-grpo-v1-grpo-m1|")
    assert phases[1].startswith("generalization-logic-grpo-v1-grpo-m2|")
    assert phases[2].startswith("generalization-science-grpo-v1-grpo-m1|")
    assert phases[3].startswith("generalization-science-grpo-v1-grpo-m2|")
    assert phases[4].startswith("generalization-knowledge-grpo-v1-grpo-m1|")
    assert phases[5].startswith("generalization-knowledge-grpo-v1-grpo-m2|")
    assert all("|0 1 2|0 25 100 400|" in row for row in phases)
    assert all(row.endswith("|verifiable_completion") for row in phases)
    assert "|kk|0 1 2|0 25 100 400|" in phases[0]
    assert "|arc-challenge|0 1 2|0 25 100 400|" in phases[2]
    assert "|mmlu-pro-nonmath|0 1 2|0 25 100 400|" in phases[4]
    qualifications = (
        Path(env["TEST_WORK"]) / "qualifications"
    ).read_text().splitlines()
    assert len(qualifications) == 3
    assert "kk --data-root" in qualifications[0]
    assert "arc-challenge --data-root" in qualifications[1]
    assert "mmlu-pro-nonmath --data-root" in qualifications[2]


def test_launcher_rejects_non_four_h100_node(tmp_path: Path) -> None:
    root, env = checkout(tmp_path, gpu_count=3)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_additional_experiments.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "exactly four H100 GPUs required" in result.stdout
    assert not (Path(env["TEST_WORK"]) / "phases").exists()


def test_launcher_refuses_when_primary_owns_the_node(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    lock_path = Path(env["OM_LOCAL_LOCK_DIR"]) / "primary.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("w")
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_additional_experiments.sh"],
        cwd=root, env=env, text=True, capture_output=True, timeout=20, check=False,
    )
    assert result.returncode != 0
    assert "OLMo primary launcher is running on THIS node" in result.stdout + result.stderr
    assert not (Path(env["TEST_WORK"]) / "phases").exists()
    fcntl.flock(lock_stream, fcntl.LOCK_UN)
    lock_stream.close()


def test_launcher_does_nothing_until_primary_lock_is_released(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    env["OM_WAIT_PRIMARY"] = "1"
    lock_path = Path(env["OM_LOCAL_LOCK_DIR"]) / "primary.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("w")
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    process = subprocess.Popen(
        ["/bin/bash", "scripts/run_additional_experiments.sh"],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(0.2)
    assert process.poll() is None
    assert not (Path(env["TEST_WORK"]) / "phases").exists()

    fcntl.flock(lock_stream, fcntl.LOCK_UN)
    lock_stream.close()
    output, _ = process.communicate(timeout=20)
    assert process.returncode == 0, output
    assert len((Path(env["TEST_WORK"]) / "phases").read_text().splitlines()) == 6


def test_prepare_mode_needs_neither_primary_lock_nor_gpus(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    lock_path = Path(env["OM_LOCAL_LOCK_DIR"]) / "primary.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("w")
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    nvidia_smi = Path(env["PATH"].split(":", 1)[0]) / "nvidia-smi"
    executable(nvidia_smi, "#!/bin/sh\nexit 99\n")

    result = subprocess.run(
        ["/bin/bash", "scripts/run_additional_experiments.sh", "--prepare"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    fcntl.flock(lock_stream, fcntl.LOCK_UN)
    lock_stream.close()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "snapshots prepared and qualified" in result.stdout
    assert not (Path(env["TEST_WORK"]) / "phases").exists()
    qualifications = (Path(env["TEST_WORK"]) / "qualifications").read_text().splitlines()
    assert len(qualifications) == 3
    assert "--dataset-n-train kk=512" in qualifications[0]
    assert "--dataset-n-train arc-challenge=512" in qualifications[1]
    assert "--dataset-n-train mmlu-pro-nonmath=512" in qualifications[2]


def test_run_mode_never_enters_snapshot_download(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    env.pop("ADDITIONAL_SKIP_PROVISION")
    env["HF_TOKEN"] = "must-not-reach-compute"
    env["HUGGING_FACE_HUB_TOKEN"] = "must-not-reach-compute"
    result = subprocess.run(
        ["/bin/bash", "scripts/run_additional_experiments.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unexpected model_matrix args" not in result.stdout + result.stderr
    assert len((Path(env["TEST_WORK"]) / "phases").read_text().splitlines()) == 6
