"""The one-command H100 launcher must lock the canonical discovery run."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def checkout(
    tmp_path: Path,
    gpu_count: int = 4,
    runner_failures: int = 0,
    runner_exit: int = 7,
) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "checkout"
    work = tmp_path / "work"
    fake_bin = tmp_path / "bin"
    (root / "scripts").mkdir(parents=True)
    (work / "venv/bin").mkdir(parents=True)
    (work / "models/Qwen2.5-7B-Instruct").mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(REPO / "scripts/go_additional.sh", root / "scripts/go_additional.sh")
    (work / "venv/bin/python").symlink_to(Path("/bin/true"))
    (work / "models/Qwen2.5-7B-Instruct/config.json").write_text("{}\n", encoding="utf-8")
    (root / "scripts/setup_env.sh").write_text(
        'export OM_WORK="${OM_WORK:?}"\n'
        'export GROUP_VOLUME="$OM_WORK"\n'
        'export VENV_DIR="$OM_WORK/venv"\n'
        'export MODELS_DIR="$OM_WORK/models"\n'
        'export MODEL_QWEN25_7B="$MODELS_DIR/Qwen2.5-7B-Instruct"\n',
        encoding="utf-8",
    )
    executable(
        root / "scripts/check_data.sh",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$OM_WORK/data-checks"\n',
    )
    executable(
        root / "scripts/go_regime.sh",
        "#!/usr/bin/env bash\n"
        "python3 - <<'PY'\n"
        "import json,os\n"
        "counter=os.environ['OM_WORK']+'/runner-count'\n"
        "try:\n"
        "    count=int(open(counter).read())+1\n"
        "except FileNotFoundError:\n"
        "    count=1\n"
        "open(counter,'w').write(str(count))\n"
        "keys=[k for k in os.environ if k.startswith('REGIME_') or k.startswith('OM_') or k in ('MODEL_14B','HYBRID_PROMPTS','K_CELL','RADIUS_MODE')]\n"
        "open(os.environ['OM_WORK']+'/runner-env.json','w').write(json.dumps({k:os.environ[k] for k in keys},sort_keys=True))\n"
        f"raise SystemExit({runner_exit} if count <= {runner_failures} else 0)\n"
        "PY\n"
    )
    executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        "  *memory.used*) " + "printf '0\\n'\n" * gpu_count + " ;;\n"
        "  *) " + "printf 'NVIDIA H100 80GB HBM3\\n'\n" * gpu_count + " ;;\n"
        "esac\n",
    )
    executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "OM_WORK": str(work),
        "MODEL_14B": "/wrong/model",
        "OM_GPUS": "0",
        "OM_BEHAVIOR_SOURCE": "/wrong/behavior",
        "OM_POOL_FILE": "/wrong/pool",
        "OM_SKIP_HYBRID": "0",
        "OM_ATTN": "sdpa",
        "REGIME_DATASETS": "wrong",
        "REGIME_SEEDS": "99",
        "REGIME_DRIFTS": "7",
        "REGIME_N_TRAIN": "1",
        "REGIME_ROOT": "/wrong/root",
        "REGIME_RESULTS": "/wrong/results",
        "REGIME_MATRIX": "/wrong/matrix",
        "REGIME_FIRST_CALIBRATION": "/wrong/calibration",
    }
    return root, env


def test_launcher_locks_full_matrix_and_clears_incompatible_overrides(tmp_path: Path) -> None:
    root, env = checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/go_additional.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    captured = json.loads((Path(env["OM_WORK"]) / "runner-env.json").read_text())
    assert captured["MODEL_14B"].endswith("/models/Qwen2.5-7B-Instruct")
    assert captured["REGIME_DATASETS"] == "gsm8k math500"
    assert captured["REGIME_SEEDS"] == "0 1 2"
    assert captured["REGIME_DRIFTS"] == "0 25 100 400"
    assert captured["OM_SKIP_HYBRID"] == "1"
    assert captured["OM_ATTN"] == "eager"
    assert captured["HYBRID_PROMPTS"] == "24"
    assert captured["K_CELL"] == "8"
    assert captured["RADIUS_MODE"] == "gaussian"
    for key in (
        "OM_GPUS",
        "OM_BEHAVIOR_SOURCE",
        "OM_POOL_FILE",
        "REGIME_N_TRAIN",
        "REGIME_ROOT",
        "REGIME_RESULTS",
        "REGIME_MATRIX",
        "REGIME_FIRST_CALIBRATION",
    ):
        assert key not in captured
    assert (Path(env["OM_WORK"]) / "data-checks").read_text().splitlines() == [
        "gsm8k 512 100",
        "math500 400 100",
    ]


def test_launcher_rejects_non_four_h100_node(tmp_path: Path) -> None:
    root, env = checkout(tmp_path, gpu_count=3)
    result = subprocess.run(
        ["/bin/bash", "scripts/go_additional.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "four H100 GPUs required" in result.stdout
    assert not (Path(env["OM_WORK"]) / "runner-env.json").exists()


def test_launcher_propagates_worker_failure(tmp_path: Path) -> None:
    root, env = checkout(tmp_path, runner_failures=13)
    result = subprocess.run(
        ["/bin/bash", "scripts/go_additional.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 7
    assert (Path(env["OM_WORK"]) / "runner-count").read_text() == "13"
    assert "restart limit reached (12)" in result.stdout
    assert "failed after 12 worker restarts (rc=7)" in result.stdout


def test_launcher_restarts_worker_and_resumes(tmp_path: Path) -> None:
    root, env = checkout(tmp_path, runner_failures=2)
    result = subprocess.run(
        ["/bin/bash", "scripts/go_additional.sh"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (Path(env["OM_WORK"]) / "runner-count").read_text() == "3"
    assert "worker failed (rc=7)" in result.stdout
    assert "complete (worker restarts=2)" in result.stdout
