"""`sh scripts/run_v8_nodes.sh status` (run_srgc.sh habit) must show the status table, not look for
configs/cohorts/status.json (user report 2026-11-08: 'status.json 못찾음')."""
from __future__ import annotations

from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_v8_nodes.sh"


def run(args, tmp_path):
    env = {"PATH": "/usr/bin:/bin", "V8_ROOT": str(tmp_path), "HOME": str(tmp_path)}
    return subprocess.run(["sh", str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=60)


def test_status_word_shows_table(tmp_path):
    out = run(["status"], tmp_path)
    assert out.returncode == 0, out.stderr
    assert "V8 status · " in out.stdout and "OLMo MATH" in out.stdout and "Qwen MBPP" in out.stdout
    assert "status.json" not in out.stdout + out.stderr
    out = run(["olmo_math", "status"], tmp_path)
    assert out.returncode == 0 and "OLMo MATH" in out.stdout and "Qwen MBPP" not in out.stdout


def test_unknown_word_gets_usage_not_traceback(tmp_path):
    out = run(["stauts"], tmp_path)
    assert out.returncode == 2
    assert "unknown command or cohort 'stauts'" in out.stderr and "run_v8_nodes.sh status" in out.stderr
    assert "Traceback" not in out.stderr


def fake_supervisor(tmp_path):
    """Copy the launcher next to a supervisor that only echoes its argv."""
    import shutil
    d = tmp_path / "scripts"
    d.mkdir()
    shutil.copy(SCRIPT, d / "run_v8_nodes.sh")
    (d / "v8_supervisor.py").write_text("import sys,json;print(json.dumps(sys.argv[1:]))")
    return d / "run_v8_nodes.sh"


def test_run_srgc_words_and_option_values_are_forwarded_intact(tmp_path):
    import json
    script = fake_supervisor(tmp_path)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    cases = {(): [], ("math",): ["olmo_math"], ("mbpp", "run"): ["qwen_mbpp"], ("all",): [],
             ("--gpus", "4,5,6,7"): ["--gpus", "4,5,6,7"],
             ("olmo_math", "--gpus", "4,5,6,7", "--new-queue"): ["--gpus", "4,5,6,7", "--new-queue", "olmo_math"],
             ("--root", "/a b/v8", "mbpp"): ["--root", "/a b/v8", "qwen_mbpp"],
             ("math", "--status"): ["--status", "olmo_math"]}
    for args, expected in cases.items():
        out = subprocess.run(["sh", str(script), *args], capture_output=True, text=True, env=env, timeout=30)
        assert out.returncode == 0, (args, out.stderr)
        assert json.loads(out.stdout) == expected, args


def test_mode_words(tmp_path):
    for args in (["math", "status"], ["mbpp", "status"]):
        out = run(args, tmp_path)
        assert out.returncode == 0 and "V8 status · " in out.stdout
    assert "Qwen MBPP" not in run(["math", "status"], tmp_path).stdout
    for args in (["status", "status"], ["math", "mbpp"], ["results"]):
        out = run(args, tmp_path)
        assert out.returncode == 2 and "usage:" in out.stderr and "Traceback" not in out.stderr
