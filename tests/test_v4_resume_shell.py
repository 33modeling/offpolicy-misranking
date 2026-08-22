"""End-to-end check that v4 resume enters the recorded Git snapshot."""

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
    capture = tmp / "capture.json"
    checkout.mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "src").mkdir()
    (work / "runs" / "v4-27b-s0").mkdir(parents=True)
    fake_bin.mkdir()

    shutil.copy(REPO / "scripts/resume_v4.sh", checkout / "scripts/resume_v4.sh")
    shutil.copy(
        REPO / "src/v4_resume_commit.py", checkout / "src/v4_resume_commit.py"
    )
    (checkout / "scripts/setup_env.sh").write_text(
        'export OM_REPO="${OM_REPO:-$PWD}"\n'
        'export OM_WORK="${OM_WORK:?}"\n'
        'export VENV_DIR="$OM_WORK/no-venv"\n',
        encoding="utf-8",
    )
    (checkout / "scripts/go_v4.sh").write_text("old generation\n", encoding="utf-8")

    run(["git", "init", "-q"], checkout)
    run(["git", "config", "user.name", "resume-test"], checkout)
    run(["git", "config", "user.email", "resume-test@example.invalid"], checkout)
    run(["git", "add", "."], checkout)
    run(["git", "commit", "-qm", "generation"], checkout)
    generation = run(["git", "rev-parse", "HEAD"], checkout)

    (checkout / "scripts/go_v4.sh").write_text("new analysis\n", encoding="utf-8")
    run(["git", "add", "."], checkout)
    run(["git", "commit", "-qm", "analysis"], checkout)

    (work / "runs/v4-27b-s0/run_config.json").write_text(
        json.dumps({"git": generation}), encoding="utf-8"
    )
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\n"
        'python3 - "$RESUME_CAPTURE" "$@" <<\'PY\'\n'
        "import json, os, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({\n"
        "    'cwd': os.getcwd(),\n"
        "    'om_repo': os.environ.get('OM_REPO'),\n"
        "    'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "    'args': sys.argv[2:],\n"
        "}))\n"
        "PY\n",
        encoding="utf-8",
    )
    fake_bash.chmod(fake_bash.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "OM_WORK": str(work),
            "OM_REPO": "leaked-current-checkout",
            "PYTHONPATH": "leaked-current-src",
            "RESUME_CAPTURE": str(capture),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    subprocess.run(
        ["/bin/bash", "scripts/resume_v4.sh", "2"],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(capture.read_text(encoding="utf-8"))
    assert observed["cwd"].endswith(f"offpolicy-misranking-{generation[:12]}")
    assert observed["om_repo"] is None
    assert observed["pythonpath"] is None
    assert observed["args"] == ["scripts/go_v4.sh", "2"]

    current = run(["git", "rev-parse", "HEAD"], checkout)
    (work / "runs/v4-27b-s0/run_config.json").write_text(
        json.dumps({"git": current}), encoding="utf-8"
    )
    capture.unlink()
    subprocess.run(
        ["/bin/bash", "scripts/resume_v4.sh", "1"],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(capture.read_text(encoding="utf-8"))
    assert observed["cwd"] == str(checkout)
    assert observed["om_repo"] is None
    assert observed["pythonpath"] is None
    assert observed["args"] == ["scripts/go_v4.sh", "1"]

print("PASS v4 resume enters the recorded snapshot without environment leakage")


def test_v4_resume_shell() -> None:
    pass
