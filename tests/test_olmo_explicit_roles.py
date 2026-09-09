"""Explicit GPU roles cannot steal training or terminate another node owner."""

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

from test_olmo3_launcher import fixture_checkout


TAG = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"


def existing_family(tmp_path):
    repo, env = fixture_checkout(tmp_path)
    python = Path(env["TEST_SHARED"]) / "work/venv/bin/python"
    python.write_text("\n".join(
        '      *owner.json) exec python3 - "$@" ;;' if line.lstrip().startswith("*.owner.json)") else line
        for line in python.read_text().splitlines()
    ) + "\n")
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    root = Path(env["TEST_SHARED"]) / "work/runs" / TAG
    (root / ".queue").mkdir(parents=True)
    (root / ".queue/generation.git").write_text(pin + "\n")
    family = root / "family-mbpp-s4"
    family.mkdir()
    (family / "saved.partial").write_bytes(b"expensive saved responses\n")
    return repo, env, root, family, pin


def command(role, *args):
    return ["bash", "scripts/run_olmo3_rlzero.sh", role, "h100", *(args or ("mbpp", "4"))]


def wait_for(text, path, process, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = path.read_text() if path.exists() else ""
        if text in output:
            return
        assert process.poll() is None, output
        time.sleep(0.03)
    pytest.fail(path.read_text())


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def test_helper_started_first_waits_then_only_evaluates_d0(tmp_path):
    repo, env, root, family, pin = existing_family(tmp_path)
    matrix = repo / "scripts/run_matrix.sh"
    matrix.write_text(matrix.read_text().replace('git=$(git -C "$OM_PIPELINE_REPO" rev-parse HEAD)', r'''
test "$REGIME_DATASETS/$REGIME_SEEDS" = mbpp/4 || exit 97
test "$REGIME_PARALLEL_CONTROL" = 1 || exit 98
owner="$TEST_SHARED/work/runs/$REGIME_MODEL_TAG/.families/$REGIME_DATASETS-s$REGIME_SEEDS"
if [ "$REGIME_CONTROL_ONLY" = 1 ]; then owner="$owner.control-owner.json"; else owner="$owner.owner.json"; fi
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["parallel_control"] is True; assert len(r["supervisor_git"]) == 40; assert r["role"] == ("control" if sys.argv[2] == "1" else "family")' "$owner" "$REGIME_CONTROL_ONLY" || exit 99
if [ "$REGIME_CONTROL_ONLY" = 1 ]; then
  echo helper >> "$TEST_SHARED/work/role-events"
  run="$REGIME_ROOT/$REGIME_MODEL_TAG-s$REGIME_SEEDS-$REGIME_DATASETS-d0"
  mkdir -p "$run"
  echo done > "$run/DONE"
  touch "$TEST_SHARED/work/helper-done"
  exit 0
fi
echo chain >> "$TEST_SHARED/work/role-events"
while [ ! -e "$TEST_SHARED/work/helper-done" ]; do /bin/sleep 0.05; done
git=$(git -C "$OM_PIPELINE_REPO" rev-parse HEAD)
'''))
    subprocess.run(["git", "add", "scripts/run_matrix.sh"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "upgraded supervisor fixture"], cwd=repo, check=True)
    outputs = [tmp_path / "helper.log", tmp_path / "chain.log"]
    processes = []
    with outputs[0].open("w") as helper_log, outputs[1].open("w") as chain_log:
        try:
            helper = subprocess.Popen(command("assist"), cwd=repo,
                env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "node-3")},
                stdout=helper_log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(helper)
            wait_for("[assist-wait] no shared training owner", outputs[0], helper)
            assert not (Path(env["TEST_SHARED"]) / "work/role-events").exists()
            chain = subprocess.Popen(command("resume-family"), cwd=repo,
                env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "node-4")},
                stdout=chain_log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(chain)
            assert helper.wait(timeout=30) == 0, outputs[0].read_text()
            assert chain.wait(timeout=30) == 0, outputs[1].read_text()
        finally:
            for process in processes:
                stop(process)
    events = (Path(env["TEST_SHARED"]) / "work/role-events").read_text().splitlines()
    assert events == ["chain", "helper"]
    assert "[control-assist] completed mbpp/s4/d0" in outputs[0].read_text()
    assert "[assist-complete]" in outputs[0].read_text()
    assert "role=resume-family families=mbpp/s4 parallel_control=1" in outputs[1].read_text()
    assert (family / "saved.partial").read_bytes() == b"expensive saved responses\n"
    assert (root / ".queue/generation.git").read_text().strip() == pin
    assert not (Path(env["TEST_SHARED"]) / f"work/results/{TAG}/COMPLETE").exists()
    assert all("[fixture-fallback]" not in path.read_text() for path in outputs)


@pytest.mark.parametrize("role", ["assist", "resume-family"])
def test_explicit_role_preserves_held_node_lock(tmp_path, role):
    repo, env, root, family, _ = existing_family(tmp_path)
    local = tmp_path / "node"
    local.mkdir()
    with (local / "primary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run(command(role), cwd=repo,
            env={**env, "OM_LOCAL_LOCK_DIR": str(local)}, text=True, capture_output=True, timeout=15)
        assert result.returncode == 75, result.stdout + result.stderr
        assert "No process was terminated" in result.stdout
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (family / "saved.partial").is_file()
    assert not list((root / "logs").glob("*.log"))


@pytest.mark.parametrize("role", ["assist", "resume-family"])
def test_explicit_role_does_not_kill_unrelated_gpu_process(tmp_path, role):
    repo, env, _, _, _ = existing_family(tmp_path)
    other = subprocess.Popen(["sleep", "300"])
    try:
        result = subprocess.run(command(role), cwd=repo,
            env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "node"), "TEST_GPU_PID": str(other.pid)},
            text=True, capture_output=True, timeout=15)
        assert result.returncode == 75, result.stdout + result.stderr
        assert "do not kill unrelated compute" in result.stdout
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=5)


def test_helper_cannot_attach_to_legacy_exclusive_owner(tmp_path):
    repo, env, root, _, _ = existing_family(tmp_path)
    queue = root / ".families"
    queue.mkdir()
    log = tmp_path / "helper.log"
    with (queue / "mbpp-s4.lock").open("a") as lock, log.open("w") as output:
        fcntl.flock(lock, fcntl.LOCK_EX)
        process = subprocess.Popen(command("assist"), cwd=repo,
            env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "node-3")},
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_for("exclusive family lease; no helper attached", log, process)
            assert not (Path(env["TEST_SHARED"]) / "work/claims").exists()
        finally:
            stop(process)


@pytest.mark.parametrize("args", [("unknown", "4"), ("mbpp", "99"), ("mbpp", "x")])
def test_invalid_target_aborts_before_compute(tmp_path, args):
    repo, env = fixture_checkout(tmp_path)
    result = subprocess.run(command("assist", *args), cwd=repo,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "node")},
        text=True, capture_output=True, timeout=15)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "[worker]" not in result.stdout


def test_existing_generation_is_required(tmp_path):
    repo, env = fixture_checkout(tmp_path)
    result = subprocess.run(command("resume-family"), cwd=repo,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "node")},
        text=True, capture_output=True, timeout=15)
    assert result.returncode == 2
    assert "no new matrix will be created" in result.stdout


def test_completed_control_does_not_acquire_gpu_or_collect(tmp_path):
    repo, env, root, family, _ = existing_family(tmp_path)
    d0 = family / f"{TAG}-s4-mbpp-d0"
    d0.mkdir()
    (d0 / "DONE").write_text("done\n")
    local = tmp_path / "node"
    result = subprocess.run(command("assist"), cwd=repo,
        env={**env, "OM_LOCAL_LOCK_DIR": str(local)}, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no GPU work started" in result.stdout
    assert not (local / "primary.lock").exists()
    assert not (family / ".family-complete").exists()


def test_cleanup_only_terminates_target_family_on_same_node(tmp_path):
    repo, env, _, family, _ = existing_family(tmp_path)
    local = tmp_path / "node-3"
    own = subprocess.Popen(["sleep", "300"], env={**os.environ,
        "REGIME_ROOT": str(family), "OM_NODE_NAMESPACE": str(local)})
    peer = subprocess.Popen(["sleep", "300"], env={**os.environ,
        "REGIME_ROOT": str(family), "OM_NODE_NAMESPACE": str(tmp_path / "node-4")})
    log = tmp_path / "helper.log"
    with log.open("w") as output:
        process = subprocess.Popen(command("assist"), cwd=repo,
            env={**env, "OM_LOCAL_LOCK_DIR": str(local)},
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_for("[assist-wait] no shared training owner", log, process)
            own.wait(timeout=3)
            assert peer.poll() is None
        finally:
            stop(process)
            for child in (own, peer):
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)
