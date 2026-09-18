import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/node_fault_state.py"
SPEC = importlib.util.spec_from_file_location("node_fault_state", SCRIPT)
faults = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(faults)


def test_legacy_receipt_expires_from_mtime_not_each_read(tmp_path):
    path = tmp_path / "fault.json"
    path.write_text('{"phase":"train"}')
    os.utime(path, (1000, 1000))
    assert faults.inspect(path, 60, now=1030)[:2] == ("cooldown", 30)
    assert faults.inspect(path, 60, now=1060)[:2] == ("expired", 0)
    assert path.read_text() == '{"phase":"train"}'
    assert path.stat().st_mtime == 1000


@pytest.mark.parametrize("record", ['{', '{}', '[]', '{"strikes":0}', '{"strikes":"1"}',
                                   '{"time":null}', '{"time":NaN}', '{"time":Infinity}',
                                   '{"time":2000}', '{"time":0}'])
def test_corrupt_fault_blocks_instead_of_holding_forever(tmp_path, record):
    path = tmp_path / "fault.json"
    path.write_text(record)
    assert faults.inspect(path, 60, now=1000)[:2] == ("blocked", 0)
    assert path.read_text() == record


@pytest.mark.parametrize("ttl", [0, -1, float("nan"), float("inf")])
def test_bad_ttl_cannot_disable_protection(tmp_path, ttl):
    assert faults.inspect(tmp_path / "absent", ttl, now=1000)[0] == "blocked"


def test_second_strike_is_not_released_by_expiry(tmp_path):
    path = tmp_path / "fault.json"
    path.write_text(json.dumps({"time":1000, "strikes":2}))
    assert faults.inspect(path, 60, now=5000)[:2] == ("blocked", 0)
    assert faults.inspect(tmp_path / "absent", 60, now=1000)[:2] == ("ready", 0)


@pytest.mark.parametrize("mbpp,expected", [(True, 0), (False, 79)])
def test_mbpp_reprobes_after_sixty_seconds_and_legacy_policy_is_unchanged(tmp_path, mbpp, expected):
    import time
    path = tmp_path / "fault.json"
    path.write_text(json.dumps({"time":time.time() - 70, "strikes":1}))
    env = {k:v for k,v in os.environ.items() if k not in ("EXPERIMENTS_FAULT_TTL_SECONDS", "EXPERIMENTS_MBPP_SUITE")}
    if mbpp:
        env["EXPERIMENTS_MBPP_SUITE"] = "all"
    result = subprocess.run([sys.executable, str(SCRIPT), str(path)], env=env,
                            text=True, capture_output=True, timeout=5, check=False)
    assert result.returncode == expected, result.stderr
    assert ("admission probe required" if mbpp else "[cooldown]") in result.stdout
