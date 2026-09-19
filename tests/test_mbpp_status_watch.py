"""The user-facing MBPP status command must never fall back to the math view."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import test_mbpp_experiments_launcher as launcher_fixtures
from test_experiments_status import mbpp_results

ROOT = Path(__file__).resolve().parents[1]
launcher = launcher_fixtures.launcher


def two_frame_sleep(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    sleep = binaries / "sleep"
    sleep.write_text('''#!/usr/bin/env bash
"$TEST_PYTHON" - "$@" <<'PY'
import json, os, sys
from pathlib import Path
path = Path(os.environ["SLEEP_LOG"])
rows = json.loads(path.read_text()) if path.exists() else []
rows.append(sys.argv[1:])
path.write_text(json.dumps(rows))
sys.exit(143 if len(rows) == 2 else 0)
PY
''')
    sleep.chmod(0o755)
    nvidia = binaries / "nvidia-smi"
    nvidia.write_text("#!/usr/bin/env bash\nexit 0\n")
    nvidia.chmod(0o755)
    return {"PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "SLEEP_LOG": str(tmp_path / "sleeps.json"), "TEST_PYTHON": sys.executable}


@pytest.mark.parametrize("options,interval", [(["--watch"], "15"), (["--watch", "1", "--all"], "1")])
def test_watch_visits_every_mbpp_root_each_frame_without_starting_workers(launcher, tmp_path, options, interval):
    run, env = launcher
    extra = two_frame_sleep(tmp_path)
    result = run("status", *options, **extra)
    assert result.returncode == 143, result.stdout + result.stderr
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    expected = ["selection-switch-mbpp-v1", "selection-switch-mbpp-quality-v1", "selection-switch-mbpp-difficulty-v1"]
    assert [Path(call["root"]).name for call in calls] == expected * 2
    assert all(call["args"] == ["status", *(["--all"] if "--all" in options else [])] for call in calls)
    assert len({call["dashboard_pid"] for call in calls[:3]}) == 1
    assert len({call["dashboard_pid"] for call in calls[3:]}) == 1
    assert calls[0]["dashboard_pid"] != calls[3]["dashboard_pid"]
    assert all("SWITCH_ROOT" not in call["env"] for call in calls)
    assert all(call["env"]["EXPERIMENTS_COMBINED"] == "0" for call in calls)
    assert json.loads(Path(extra["SLEEP_LOG"]).read_text()) == [[interval], [interval]]
    assert result.stdout.count("MBPP EXPERIMENTS") == 2
    assert not Path(env["AUDIT_LOG"]).exists() and not Path(env["CHECK_LOG"]).exists()


@pytest.mark.parametrize("options", [["--watch", "0"], ["--watch", "-1"], ["--watch", "nan"], ["--oops"]])
def test_bad_status_options_fail_before_any_launch(launcher, options):
    run, env = launcher
    assert run("status", *options).returncode == 2
    assert not Path(env["CALLS"]).exists() and not Path(env["AUDIT_LOG"]).exists()


def test_real_mbpp_watch_keeps_saved_results_and_live_work_visible_not_math(tmp_path):
    from test_selection_switch_status import point, running

    work = tmp_path / "work"
    root = mbpp_results(work)
    import time
    running(point(root, 4, 100) / "selection_reduced", "mbpp-active-node", now=time.time(), phase="train")
    extra = two_frame_sleep(tmp_path)
    env = {**os.environ, **extra, "OM_WORK": str(work), "SWITCH_PYTHON": sys.executable,
           "SWITCH_ROOT": "/wrong/math-root", "OUT_ROOT": "/wrong/math-root", "EXPERIMENTS_COMBINED": "1"}
    for key in ("SWITCH_MBPP_ROOT", "SWITCH_MBPP_QUALITY_ROOT", "SWITCH_MBPP_DIFFICULTY_ROOT"):
        env.pop(key, None)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in work.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", "scripts/run_mbpp_experiments.sh", "status", "--watch", "1"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 143, result.stdout + result.stderr
    assert result.stdout.count("MBPP EXPERIMENTS") == 2
    assert len(re.findall(r"^On-policy · 선택비용 포함\s+43\.8%\s+21/48\s+\d+\s+21\s+\d+\s+1\b", result.stdout, re.MULTILINE)) == 2
    assert result.stdout.count("CURRENT RUN 1") == 2
    assert result.stdout.count("mbpp-active-node") >= 2
    assert "wrong/math-root" not in result.stdout and "MOPPS COMPARISON" not in result.stdout
    assert "On-policy · 선택비용 별도: 준비 전" in result.stdout and "Difficulty · 선택비용 포함: 준비 전" in result.stdout
    assert "Remarks" in result.stdout and "Full selection" in result.stdout
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in work.rglob("*") if path.is_file()}
