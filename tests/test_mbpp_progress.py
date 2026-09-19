"""`run_mbpp_experiments.sh progress` shows the MBPP roots and nothing else.

The generic progress screen lists every prepared root under runs/; with three
math suites ahead of it, the MBPP root was buried and its unprepared sibling
suites (quality, difficulty) were absent, so the MBPP status could not be read.
"""
import os
import subprocess
import sys
from pathlib import Path

from test_experiments_status import mbpp_results

ROOT = Path(__file__).resolve().parents[1]


def run_progress(script, work, *args):
    env = {**os.environ, "OM_WORK": str(work), "SWITCH_PYTHON": sys.executable,
           "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(["bash", f"scripts/{script}", "progress", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


def workspace(tmp_path):
    from test_selection_switch_status import prepared
    work = tmp_path / "work"
    mbpp_results(work)                                   # MBPP on-policy root with saved results
    prepared(work / "runs/selection-switch-v1")          # math roots that used to crowd the screen
    prepared(work / "runs/selection-switch-difficulty-v1")
    prepared(work / "runs/selection-switch-mbpp-quality-v1")
    return work


def test_mbpp_progress_shows_only_mbpp_roots_including_unprepared_suites(tmp_path):
    work = workspace(tmp_path)
    result = run_progress("run_mbpp_experiments.sh", work)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    labels = [line.split(":", 1)[0] for line in out.splitlines() if ":" in line and not line.startswith(" ")]
    assert "MBPP On-policy · 선택비용 포함" in labels
    assert "MBPP On-policy · 선택비용 별도" in labels
    assert "MBPP Difficulty · 선택비용 포함" in labels, out
    assert "MBPP Difficulty · 선택비용 포함: not prepared" in out
    assert "on-policy" not in labels, out            # math suite must not appear
    assert "difficulty" not in labels, out           # math difficulty suite must not appear
    screen = out[out.index("PROGRESS"):]          # launcher banner lines carry root paths; judge the screen only
    assert screen.index("MBPP On-policy · 선택비용 포함:") < screen.index("MBPP On-policy · 선택비용 별도:") < screen.index("MBPP Difficulty · 선택비용 포함:")
    assert "MBPP roots only" in screen


def test_generic_progress_still_lists_every_prepared_root(tmp_path):
    work = workspace(tmp_path)
    result = run_progress("run_experiments.sh", work)
    assert result.returncode == 0, result.stderr
    labels = [line.split(":", 1)[0] for line in result.stdout.splitlines() if ":" in line and not line.startswith(" ")]
    assert {"on-policy", "difficulty", "MBPP On-policy · 선택비용 포함", "MBPP On-policy · 선택비용 별도"} <= set(labels), result.stdout
    assert "MBPP roots only" not in result.stdout


def test_progress_root_option_accepts_unprepared_root(tmp_path):
    work = workspace(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    missing = work / "runs/selection-switch-mbpp-difficulty-v1"
    result = subprocess.run([sys.executable, "scripts/experiments_progress.py", "--work", str(work),
                             "--root", str(work / "runs/selection-switch-mbpp-v1"), "--root", str(missing)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "MBPP Difficulty · 선택비용 포함: not prepared" in result.stdout
    assert "MBPP On-policy · 선택비용 포함:" in result.stdout
    assert "on-policy:" not in result.stdout
