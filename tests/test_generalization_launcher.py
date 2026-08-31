"""The generalization launcher binds and runs every configured model matrix."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

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
    (work / "venv/bin").mkdir(parents=True)
    (work / "models/m1").mkdir(parents=True)
    (work / "models/m2").mkdir(parents=True)
    fake_bin.mkdir()
    (root / "scripts/run_generalization.sh").write_text(
        (ROOT / "scripts/run_generalization.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "configs/domain_transfer.json").write_text("{}\n", encoding="utf-8")
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
        "case \"$script\" in\n"
        "  src/model_matrix.py)\n"
        "    case \" $* \" in\n"
        "      *' list-models '*) printf 'm1\\nm2\\n' ;;\n"
        "      *' experiment-field datasets '*) printf 'gsm8k mbpp\\n' ;;\n"
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
        "      *' grpo-field epochs_per_batch '*) printf '2\\n' ;;\n"
        "      *' grpo-field max_grad_norm '*) printf '1.0\\n' ;;\n"
        "      *' grpo-field advantage_epsilon '*) printf '1e-4\\n' ;;\n"
        "      *' grpo-field lora_rank '*) printf '16\\n' ;;\n"
        "      *' grpo-field lora_alpha '*) printf '32\\n' ;;\n"
        "      *' field m1 path '*) printf '%s/models/m1\\n' \"$TEST_WORK\" ;;\n"
        "      *' field m2 path '*) printf '%s/models/m2\\n' \"$TEST_WORK\" ;;\n"
        "      *' field '*' lora_targets '*) printf 'q_proj,v_proj\\n' ;;\n"
        "      *' check '*) printf '[check] pass\\n' ;;\n"
        "      *) printf 'unexpected model_matrix args: %s\\n' \"$*\" >&2; exit 2 ;;\n"
        "    esac ;;\n"
        "  src/qualify_domain_data.py)\n"
        "    while [ $# -gt 0 ]; do [ \"$1\" = --output ] && { printf '{}\\n' > \"$2\"; break; }; shift; done ;;\n"
        "  src/transfer_smoke.py)\n"
        "    while [ $# -gt 0 ]; do [ \"$1\" = --marker ] && { mkdir -p \"$(dirname \"$2\")\"; printf '{}\\n' > \"$2\"; break; }; shift; done ;;\n"
        "  src/regime_contract.py)\n"
        "    while [ $# -gt 0 ]; do [ \"$1\" = --matrix ] && { mkdir -p \"$(dirname \"$2\")\"; printf '{}\\n' > \"$2\"; break; }; shift; done ;;\n"
        "  *) printf 'unexpected script: %s\\n' \"$script\" >&2; exit 2 ;;\n"
        "esac\n",
    )
    executable(
        root / "scripts/run_matrix.sh",
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$REGIME_MODEL_TAG|$MODEL_PATH|$REGIME_DATASETS|$REGIME_SEEDS|$REGIME_DRIFTS|$REGIME_MATRIX\" >> \"$TEST_WORK/phases\"\n",
    )
    executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        + "printf 'NVIDIA H100 80GB HBM3\\n'\n" * gpu_count,
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root, {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_WORK": str(work),
    }


def test_launcher_runs_every_pinned_model_with_one_shared_matrix(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_generalization.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    phases = (Path(env["TEST_WORK"]) / "phases").read_text().splitlines()
    assert len(phases) == 2
    assert phases[0].startswith("generalization-grpo-v1-grpo-m1|")
    assert phases[1].startswith("generalization-grpo-v1-grpo-m2|")
    assert all("|gsm8k mbpp|0 1 2|0 25 100 400|" in row for row in phases)
    assert all("/contracts/generalization-grpo-v1-" in row for row in phases)


def test_launcher_rejects_non_four_h100_node(tmp_path: Path) -> None:
    root, env = checkout(tmp_path, gpu_count=3)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_generalization.sh"],
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
