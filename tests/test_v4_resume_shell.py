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
    fake_bin = tmp / "bin"
    capture = tmp / "capture.txt"
    checkout.mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "src").mkdir()
    (work / "runs" / "v4-27b-s2").mkdir(parents=True)
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
        'printf "%s|%s|%s|%s|%s\\n" "$PWD" "$(git rev-parse HEAD)" '
        '"$OM_REPO" "$SEEDS" "$DATASETS" >> "$RESUME_CAPTURE"\n',
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

print("PASS mixed v4 resume enters the recorded snapshot per run")


def test_v4_resume_shell() -> None:
    pass
