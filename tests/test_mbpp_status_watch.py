"""The user-facing MBPP status command must never fall back to the math view."""
import json
import os
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
import hashlib, json, os, sys
from pathlib import Path
path = Path(os.environ["SLEEP_LOG"])
rows = json.loads(path.read_text()) if path.exists() else []
rows.append(sys.argv[1:])
path.write_text(json.dumps(rows))
if len(rows) == 1 and os.environ.get("TEST_COMPLETE_BRANCH"):
    branch = Path(os.environ["TEST_COMPLETE_BRANCH"])
    schema = json.loads(Path(os.environ["TEST_SWITCH_MANIFEST"]).read_text())["schema"]
    progress = json.loads((branch / "progress.json").read_text())
    progress["state"] = "finished"
    (branch / "progress.json").write_text(json.dumps(progress))
    result = branch / "result.json"
    result.write_text(json.dumps({"schema": "offpolicy-net-gain-gate/v3-1", "complete": True}))
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    (branch / "result.sha256.json").write_text(json.dumps({"sha256": digest}))
    (branch / "curve.json").write_text(json.dumps({"schema": schema, "result_sha256": digest, "points": {"25": {"reward": .1}}}))
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
    expected = ["selection-switch-mbpp-quality-v1"]
    assert [Path(call["root"]).name for call in calls] == expected * 2
    assert all(call["args"] == ["status", *(["--all"] if "--all" in options else [])] for call in calls)
    assert calls[0]["dashboard_pid"] != calls[1]["dashboard_pid"]
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
    for key in ("SWITCH_MBPP_ROOT", "SWITCH_MBPP_QUALITY_ROOT", "SWITCH_MBPP_DIFFICULTY_ROOT", "SWITCH_MBPP_LONG_ROOT"):
        env.pop(key, None)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in work.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", "scripts/run_mbpp_experiments.sh", "status", "--watch", "1"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 143, result.stdout + result.stderr
    assert result.stdout.count("MBPP EXPERIMENTS") == 2
    assert result.stdout.count("다른 조건의 완료 결과 21개 보존") == 2
    assert result.stdout.count("CURRENT RUN 1") == 2
    assert result.stdout.count("mbpp-active-node") >= 2
    assert "wrong/math-root" not in result.stdout and "MOPPS COMPARISON" not in result.stdout
    assert result.stdout.count("계획 48개 | 완료 확인 0/48 | 남음 48개") == 2
    assert "총 계획 48개 | 완료 확인 0개 | 남음 48개" in result.stdout
    assert "기존 계획 48개 | 완료 확인 21개 | 남음 27개" in result.stdout
    assert "Remarks" in result.stdout and "Full selection" in result.stdout
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in work.rglob("*") if path.is_file()}


def test_real_watch_updates_full_status_run_and_done_between_frames(tmp_path):
    import time
    from test_selection_switch_status import convergence_root, completed_prefix, point, running

    work = tmp_path / "work"
    root = work / "runs/selection-switch-mbpp-quality-v1"
    convergence_root(root)
    completed_prefix(root)
    branch = point(root) / "random_reduced"
    running(branch, "finishing-node", now=time.time(), phase="evaluate")
    extra = two_frame_sleep(tmp_path)
    env = {**os.environ, **extra, "OM_WORK": str(work), "SWITCH_PYTHON": sys.executable,
           "TEST_COMPLETE_BRANCH": str(branch), "TEST_SWITCH_MANIFEST": str(root / "switch.json")}
    for key in ("SWITCH_MBPP_ROOT", "SWITCH_MBPP_QUALITY_ROOT", "SWITCH_MBPP_DIFFICULTY_ROOT", "SWITCH_MBPP_LONG_ROOT"):
        env.pop(key, None)
    result = subprocess.run(["bash", "scripts/run_mbpp_experiments.sh", "status", "--watch", "1"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 143, result.stdout + result.stderr
    first, second = result.stdout.split("MBPP EXPERIMENTS")[1:]
    assert "완료 확인 0개" in first and "CURRENT RUN 1" in first
    assert "완료 확인 1개" in second and "CURRENT RUN 0" in second
    assert "완료 확인 1/48" in second
