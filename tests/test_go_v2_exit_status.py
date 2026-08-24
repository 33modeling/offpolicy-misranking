"""go_v2 preserves failures and does not kill active silent computation."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

go_v2_source = (REPO / "scripts/go_v2.sh").read_text(encoding="utf-8")
assert '"$active_run"/logs/*.log' in go_v2_source
assert '"$BASE"*/logs/*.log' not in go_v2_source


def executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    checkout = tmp / "checkout"
    work = tmp / "work"
    fake_bin = tmp / "bin"
    snapshot = tmp / "snapshot"
    activity_capture = tmp / "pipeline-env.txt"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "src").mkdir()
    (work / "venv/bin").mkdir(parents=True)
    fake_bin.mkdir()
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "src").mkdir()

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
        'case "$*" in *utilization.gpu*) echo "${FAKE_GPU_UTIL:-0}"; exit 0;; esac\n'
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
    for artifact in (
        "DONE",
        "run_config.json",
        "manifest.json",
        "report.json",
        "score_protocol.json",
        "oracle_protocol.json",
    ):
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

    executable(
        snapshot / "scripts/run_14b.sh",
        "#!/bin/sh\n"
        'printf "%s|%s|%s\\n" "$OM_REPO" "$PYTHONPATH" "$OM_RETRY_INDEX" '
        '> "$ACTIVITY_CAPTURE"\n'
        "/bin/sleep 1\n"
        'mkdir -p "$OUT_ROOT"\n'
        'for artifact in DONE run_config.json manifest.json score_protocol.json '
        'oracle_protocol.json report.json; do printf \'{}\\n\' > "$OUT_ROOT/$artifact"; done\n',
    )
    active_base = work / "runs/v4-active"
    active_smoke = Path(str(active_base) + "-smoke")
    active_smoke.mkdir(parents=True)
    for artifact in (
        "DONE",
        "run_config.json",
        "manifest.json",
        "report.json",
        "score_protocol.json",
        "oracle_protocol.json",
    ):
        (active_smoke / artifact).write_text("{}\n", encoding="utf-8")
    active_env = env.copy()
    active_env.update({
        "RUN_BASE": str(active_base),
        "RUN_LABEL": "active-watchdog-test",
        "OM_PIPELINE_REPO": str(snapshot),
        "OM_PIPELINE_SCRIPT": str(snapshot / "scripts/run_14b.sh"),
        "ACTIVITY_CAPTURE": str(activity_capture),
        "FAKE_GPU_UTIL": "90",
        "OM_STALL_MINUTES": "1",
    })
    active = subprocess.run(
        ["/bin/bash", "scripts/go_v2.sh"],
        cwd=checkout,
        env=active_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert active.returncode == 0, active.stdout + active.stderr
    assert "계산 활동 확인" in active.stdout
    pipeline_repo, pythonpath, retry_index = (
        activity_capture.read_text(encoding="utf-8").strip().split("|")
    )
    assert pipeline_repo == str(snapshot)
    assert pythonpath == str(snapshot / "src")
    assert retry_index == "1"

print("PASS go_v2 preserves failures and active snapshot computation")


def test_go_v2_exit_status() -> None:
    pass
