"""Same hostname must not collapse independent Pair/RLOO node ownership."""

import fcntl
import os
from pathlib import Path
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = '''source scripts/_e5_node.sh
PY=true
hostname() { echo duplicate; }
e5_physical_node_id() { printf '%s' "$TEST_BOOT"; }
e5_acquire_shared_pair_node "$TEST_LOCKS" || exit $?
touch "$TEST_READY"
[ "$TEST_HOLD" = 1 ] && read -r unused
exit 0
'''


def launch(tmp_path, name, boot, hold=False):
    return subprocess.Popen(["bash", "-c", SCRIPT], cwd=ROOT, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "TEST_LOCKS": str(tmp_path), "TEST_BOOT": boot,
             "TEST_READY": str(tmp_path / name), "TEST_HOLD": str(int(hold)),
             "EXPERIMENTS_NODE_ID": name})


@pytest.mark.parametrize("same_boot", [False, True])
def test_same_hostname_uses_physical_identity_even_when_display_id_changes(tmp_path, same_boot):
    first = launch(tmp_path, "first", "boot-a", hold=True)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "first").exists():
            assert first.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.01)
        second = launch(tmp_path, "second", "boot-a" if same_boot else "boot-b")
        out, err = second.communicate(timeout=5)
        assert second.returncode == (75 if same_boot else 0), out + err
        assert first.poll() is None
    finally:
        first.communicate("stop\n", timeout=5)


def test_legacy_exclusive_hostname_owner_is_never_bypassed(tmp_path):
    with (tmp_path / "primary.duplicate.lock").open("w") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = launch(tmp_path, "new", "boot-new")
        out, err = process.communicate(timeout=5)
        assert process.returncode == 75, out + err
        assert "legacy hostname lock is held" in out


def test_updated_controller_keeps_legacy_launchers_out(tmp_path):
    process = launch(tmp_path, "first", "boot-a", hold=True)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "first").exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        with (tmp_path / "primary.duplicate.lock").open("r+") as lease:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        process.communicate("stop\n", timeout=5)
