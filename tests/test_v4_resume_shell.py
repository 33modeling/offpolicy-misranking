"""End-to-end check that v4 resume enters each run's recorded snapshot."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, env=env, text=True).strip()


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    checkout = tmp / "checkout"
    work = tmp / "work"
    failed_work = tmp / "failed-work"
    fake_bin = tmp / "bin"
    capture = tmp / "capture.txt"
    failed_capture = tmp / "failed-capture.txt"
    checkout.mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "src").mkdir()
    (work / "runs" / "v4-27b-s2").mkdir(parents=True)
    (failed_work / "runs" / "v4-27b-s2").mkdir(parents=True)
    fake_bin.mkdir()

    shutil.copy(REPO / "scripts/resume_v4.sh", checkout / "scripts/resume_v4.sh")
    shutil.copy(
        REPO / "src/v4_resume_commit.py", checkout / "src/v4_resume_commit.py"
    )
    (checkout / "src/cleanup_run_processes.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    (checkout / "scripts/setup_env.sh").write_text(
        'export OM_REPO="${OM_REPO:-$PWD}"\n'
        'export OM_WORK="${OM_WORK:?}"\n'
        'export VENV_DIR="$OM_WORK/no-venv"\n'
        'export MODELS_DIR="$OM_WORK/models"\n'
        'export TMPDIR="$OM_WORK/tmp"\n'
        'mkdir -p "$TMPDIR"\n'
        'export PYTHONPATH="$OM_REPO/src"\n',
        encoding="utf-8",
    )
    (checkout / "scripts/go_v2.sh").write_text(
        "#!/bin/sh\n"
        'suffix=""; [ "$DATASETS" = gsm8k ] || suffix="-$DATASETS"\n'
        'run="$RUN_BASE-s$SEEDS$suffix"\n'
        'printf "%s|%s|%s|%s|%s\\n" "$PWD" "$(git rev-parse HEAD)" '
        '"$OM_REPO" "$SEEDS" "$DATASETS" >> "$RESUME_CAPTURE"\n'
        '[ "$(basename "$run")" = "${RESUME_FAIL_RUN:-}" ] && exit 9\n'
        'mkdir -p "$run"\n'
        'head=$(git rev-parse HEAD)\n'
        '[ -s "$run/run_config.json" ] || printf \'{"git":"%s"}\\n\' "$head" '
        '> "$run/run_config.json"\n'
        'for artifact in manifest.json score_protocol.json oracle_protocol.json report.json; do '
        'printf \'{}\\n\' > "$run/$artifact"; done\n'
        'printf \'ok\\n\' > "$run/DONE"\n',
        encoding="utf-8",
    )

    run(["git", "init", "-q"], checkout)
    run(["git", "config", "user.name", "resume-test"], checkout)
    run(["git", "config", "user.email", "resume-test@example.invalid"], checkout)
    run(["git", "add", "."], checkout)
    run(["git", "commit", "-qm", "generation"], checkout)
    generation = run(["git", "rev-parse", "HEAD"], checkout)

    (checkout / "marker").write_text("new analysis\n", encoding="utf-8")
    run(["git", "add", "."], checkout)
    run(["git", "commit", "-qm", "analysis"], checkout)

    (work / "runs/v4-27b-s2/run_config.json").write_text(
        json.dumps({"git": generation}), encoding="utf-8"
    )
    (failed_work / "runs/v4-27b-s2/run_config.json").write_text(
        json.dumps({"git": generation}), encoding="utf-8"
    )
    fake_sleep = fake_bin / "sleep"
    fake_sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_sleep.chmod(fake_sleep.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update({
        "OM_WORK": str(work),
        "OM_REPO": "leaked-current-checkout",
        "PYTHONPATH": "leaked-current-src",
        "RESUME_CAPTURE": str(capture),
        "PATH": f"{fake_bin}:{env['PATH']}",
    })
    subprocess.run(
        ["/bin/bash", "scripts/resume_v4.sh", "2"],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = capture.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 6
    for row in rows:
        cwd, head, om_repo, _, _ = row.split("|")
        assert cwd.endswith(f"offpolicy-misranking-{generation[:12]}")
        assert head == generation
        assert om_repo == cwd

    failed_env = env.copy()
    failed_env.update({
        "OM_WORK": str(failed_work),
        "RESUME_CAPTURE": str(failed_capture),
        "RESUME_FAIL_RUN": "v4-27b-s2",
    })
    failed = subprocess.run(
        ["/bin/bash", "scripts/resume_v4.sh", "2"],
        cwd=checkout,
        env=failed_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode == 1, failed.stdout + failed.stderr
    failed_rows = failed_capture.read_text(encoding="utf-8").splitlines()
    assert len(failed_rows) == 8
    attempted = [(row.split("|")[-2], row.split("|")[-1]) for row in failed_rows]
    assert attempted.count(("2", "gsm8k")) == 3
    assert len(set(attempted)) == 6
    assert "나머지 run 계속 진행" in failed.stdout
    assert "supervisor 3회 뒤에도 미완료 run 존재" in failed.stderr

print("PASS v4 resume verifies completion and retries only failed runs")


def test_v4_resume_shell() -> None:
    pass
