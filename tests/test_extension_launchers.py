"""Extension launchers (E1/E5/go_extensions) use the registered reward
function, refuse a node whose GPUs are taken, and report failures."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("scripts/go_extensions.sh", "scripts/run_drift_curve.sh", "scripts/run_downstream_compare.sh")


def test_scripts_parse() -> None:
    for script in SCRIPTS:
        subprocess.run(["bash", "-n", str(ROOT / script)], check=True)


def test_every_gpu_launcher_uses_the_registered_math_verifier() -> None:
    """2026-09-07: the curve and the downstream update trained and scored on
    exact-match rewards while the registered matrix uses Math-Verify."""
    for script in SCRIPTS:
        text = (ROOT / script).read_text(encoding="utf-8")
        assert "OM_MATH_VERIFIER=math_verify" in text, script
        assert "bootstrap_math_verify.py" in text, script


def test_gpu_launchers_take_the_node_lock_and_defer_to_the_parent() -> None:
    for script in ("scripts/run_drift_curve.sh", "scripts/run_downstream_compare.sh"):
        text = (ROOT / script).read_text(encoding="utf-8")
        assert 'flock -n 8' in text and 'primary.lock' in text, script
        assert 'OM_NODE_LOCK_HELD' in text, script
    go = (ROOT / "scripts/go_extensions.sh").read_text(encoding="utf-8")
    assert "OM_NODE_LOCK_HELD=1 bash scripts/run_drift_curve.sh" in go
    assert "OM_NODE_LOCK_HELD=1 bash scripts/run_downstream_compare.sh" in go


def test_go_extensions_defaults_to_the_h100_profile_and_reports_failures() -> None:
    go = (ROOT / "scripts/go_extensions.sh").read_text(encoding="utf-8")
    assert "PROFILE=${1:-h100}" in go
    assert "[extensions] FAILED:" in go and "exit 1" in go
    assert "note_stage" in go


def test_go_extensions_rejects_an_unknown_profile(tmp_path: Path) -> None:
    env = {**os.environ, "OM_WORK": str(tmp_path / "work"), "GROUP_VOLUME": str(tmp_path / "none")}
    result = subprocess.run(["bash", "scripts/go_extensions.sh", "bogus"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "unknown profile" in result.stdout + result.stderr
