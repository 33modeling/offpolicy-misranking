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
    for args in (["status"], ["olmo_math", "status"]):
        out = run(args, tmp_path)
        assert out.returncode == 0, out.stderr
        assert "V8 olmo_math · " in out.stdout and "V8 qwen_mbpp · " in out.stdout
        assert "status.json" not in out.stdout + out.stderr


def test_unknown_word_gets_usage_not_traceback(tmp_path):
    out = run(["stauts"], tmp_path)
    assert out.returncode == 2
    assert "unknown command or cohort 'stauts'" in out.stderr and "run_v8_nodes.sh status" in out.stderr
    assert "Traceback" not in out.stderr
