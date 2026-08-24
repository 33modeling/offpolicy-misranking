"""A failed worker must not become success when post-processing is skipped."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    checkout = tmp / "checkout"
    work = tmp / "work"
    fake_bin = tmp / "bin"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "src").mkdir()
    (work / "venv/bin").mkdir(parents=True)
    fake_bin.mkdir()

    shutil.copy(REPO / "scripts/go_v2.sh", checkout / "scripts/go_v2.sh")
    (checkout / "scripts/setup_env.sh").write_text(
        'export OM_WORK="${OM_WORK:?}"\n'
        'export VENV_DIR="$OM_WORK/venv"\n'
        'export MODELS_DIR="$OM_WORK/models"\n'
        'export TMPDIR="$OM_WORK/tmp"\n'
        'export HF_HOME="$OM_WORK/hf"\n'
        'mkdir -p "$TMPDIR" "$HF_HOME"\n',
        encoding="utf-8",
    )
    executable(work / "venv/bin/python", "#!/bin/sh\nexit 0\n")
    executable(
        fake_bin / "nvidia-smi",
        "#!/bin/sh\n"
        'if [ "${1:-}" = -L ]; then echo "GPU 0: fake"; exit 0; fi\n'
        'echo "0, fake, 0, 0, 80000"\n',
    )
    executable(
        fake_bin / "sleep",
        "#!/bin/sh\n"
        'if [ "${1:-}" = 15 ]; then exec /bin/sleep 0.01; fi\n'
        "exit 0\n",
    )
    executable(checkout / "scripts/run_14b.sh", "#!/bin/sh\nexit 7\n")
    executable(
        checkout / "scripts/diagnose_run_failure.sh", "#!/bin/sh\nexit 0\n"
    )
    (checkout / "src/cleanup_run_processes.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )

    base = work / "runs/v4-7b"
    smoke = Path(str(base) + "-smoke")
    smoke.mkdir(parents=True)
    for artifact in ("report.json", "score_protocol.json", "oracle_protocol.json"):
        (smoke / artifact).write_text("{}\n", encoding="utf-8")

    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "OM_WORK": str(work),
        "RUN_BASE": str(base),
        "RUN_LABEL": "exit-status-test",
        "SEEDS": "0",
        "DATASETS": "gsm8k",
        "OM_SKIP_HYBRID": "1",
        "OM_SKIP_POSTPROCESS": "1",
        "OM_MAX_RETRIES": "1",
        "OM_STALL_MINUTES": "100",
    })
    proc = subprocess.run(
        ["/bin/bash", "scripts/go_v2.sh"],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "[실패] 1개 run 미완료" in proc.stdout

print("PASS go_v2 preserves worker failure exit status")


def test_go_v2_exit_status() -> None:
    pass
