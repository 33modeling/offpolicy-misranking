from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def checkout(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout"
    work = tmp_path / "work"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    work.mkdir()
    shutil.copy2(REPO / "scripts/go_offpolicy.sh", scripts / "go_offpolicy.sh")
    executable(
        scripts / "setup_env.sh",
        '#!/usr/bin/env bash\nexport OM_WORK="${OM_WORK:?}"\n',
    )
    executable(
        scripts / "go_v4.sh",
        '#!/usr/bin/env bash\nprintf "%s\\n" v4 >> "$OM_WORK/stages"\n'
        'if [ "${HOLD_V4:-0}" = 1 ]; then\n'
        '  touch "$OM_WORK/v4-started"\n'
        '  while [ ! -e "$OM_WORK/v4-release" ]; do sleep 0.01; done\n'
        'fi\n',
    )
    for name, label in (("go_additional.sh", "regime"), ("collect_v4.sh", "collect")):
        executable(
            scripts / name,
            f'#!/usr/bin/env bash\nprintf "%s\\n" "{label}" >> "$OM_WORK/stages"\n',
        )
    return checkout, work


def test_end_to_end_runner_executes_each_stage_once(tmp_path: Path) -> None:
    checkout_dir, work = checkout(tmp_path)

    result = subprocess.run(
        ["/bin/bash", "scripts/go_offpolicy.sh", "2"],
        cwd=checkout_dir,
        env={**os.environ, "OM_WORK": str(work)},
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (work / "stages").read_text(encoding="utf-8").splitlines() == [
        "v4",
        "regime",
        "collect",
    ]


def test_duplicate_slot_is_rejected_before_starting_work(tmp_path: Path) -> None:
    checkout_dir, work = checkout(tmp_path)
    env = {**os.environ, "OM_WORK": str(work), "HOLD_V4": "1"}
    first = subprocess.Popen(
        ["/bin/bash", "scripts/go_offpolicy.sh", "1"],
        cwd=checkout_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for _ in range(200):
        if (work / "v4-started").exists():
            break
        time.sleep(0.01)
    else:
        first.kill()
        raise AssertionError("first workflow did not acquire its lock")

    duplicate = subprocess.run(
        ["/bin/bash", "scripts/go_offpolicy.sh", "1"],
        cwd=checkout_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert duplicate.returncode != 0
    assert "already running" in duplicate.stderr
    assert (work / "stages").read_text(encoding="utf-8").splitlines() == ["v4"]

    (work / "v4-release").touch()
    output, _ = first.communicate(timeout=10)
    assert first.returncode == 0, output
