"""OLMo d0 work sharing preserves the sequential GRPO chain and point leases."""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    start = source.index(name + "() {")
    return source[start:source.index("\n}\n", start) + 2]


def point_harness(tmp_path):
    source = (ROOT / "scripts/run_matrix.sh").read_text()
    return "\n".join(function(source, name) for name in
                     ("run_point", "run_point_unlocked", "run_family")) + r'''
QUEUE="$TEST_ROOT/.queue"
mkdir -p "$QUEUE"
PARALLEL_CONTROL=1
CONTRACT=''
MAX_RETRIES=1
DRIFTS=(0 25 100 400)
SEEDS=(4)
DATASETS=(fixture)
PY=echo
run_dir() { echo "$TEST_ROOT/d$3"; }
n_train_for_dataset() { echo 1; }
rollout_artifact_ready() { test -s "$TEST_ROOT/behavior"; }
run_complete() { test -s "$1/DONE"; }
reenter_runtime_fields() { echo "$1" >> "$TEST_ROOT/repairs"; }
note_point_accepted() { :; }
cleanup_active_pipeline() { :; }
run_pipeline_watchdog() {
  local run=$1
  mkdir -p "$run/logs"
  echo "${run##*/}:start" >> "$TEST_ROOT/events"
  if [ "${run##*/}" = d25 ] && [ "$TEST_BARRIER" = 1 ]; then
    touch "$TEST_ROOT/training-started"
    while [ ! -s "$TEST_ROOT/d0/DONE" ]; do sleep 0.02; done
  fi
  if [ "${run##*/}" = d0 ]; then sleep 0.2; fi
  echo done > "$run/DONE"
  echo "${run##*/}:end" >> "$TEST_ROOT/events"
}
CONTROL_ONLY=$TEST_CONTROL_ONLY
if [ "$CONTROL_ONLY" = point ]; then
  run_point fixture 4 0 '' '' ''
else
  run_family fixture 4
fi
'''


def test_control_overlaps_training_and_is_not_repeated(tmp_path):
    (tmp_path / "behavior").write_bytes(b"immutable behavior\n")
    script = point_harness(tmp_path)
    env = {**os.environ, "TEST_ROOT": str(tmp_path), "TEST_BARRIER": "1"}
    train = subprocess.Popen(["bash", "-c", script],
                             env={**env, "TEST_CONTROL_ONLY": "0"},
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "training-started").exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        control = subprocess.run(["bash", "-c", script],
                                 env={**env, "TEST_CONTROL_ONLY": "1"},
                                 text=True, capture_output=True, timeout=5)
        output, _ = train.communicate(timeout=5)
        assert control.returncode == 0, control.stdout + control.stderr
        assert train.returncode == 0, output
    finally:
        if train.poll() is None:
            train.kill()
            train.wait(timeout=5)
    events = (tmp_path / "events").read_text().splitlines()
    assert events.index("d25:start") < events.index("d0:start") < events.index("d25:end")
    assert events.count("d0:start") == 1
    assert events.index("d25:end") < events.index("d100:start") < events.index("d400:start")
    assert (tmp_path / "behavior").read_bytes() == b"immutable behavior\n"


def test_busy_point_yields_before_repair_or_generation(tmp_path):
    lock_path = tmp_path / ".queue/points/fixture-s4-d0.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", "-c", point_harness(tmp_path)],
            env={**os.environ, "TEST_ROOT": str(tmp_path), "TEST_CONTROL_ONLY": "point", "TEST_BARRIER": "0"},
            text=True, capture_output=True, timeout=5)
    assert result.returncode == 75, result.stdout + result.stderr
    assert not (tmp_path / "repairs").exists()
    assert not (tmp_path / "events").exists()


def test_control_requires_existing_valid_behavior(tmp_path):
    result = subprocess.run(["bash", "-c", point_harness(tmp_path)],
        env={**os.environ, "TEST_ROOT": str(tmp_path), "TEST_CONTROL_ONLY": "1", "TEST_BARRIER": "0"},
        text=True, capture_output=True, timeout=5)
    assert result.returncode == 75
    assert not (tmp_path / "events").exists()


def test_three_olmo_workers_share_one_family_without_duplicate_control(tmp_path):
    from test_olmo3_launcher import fixture_checkout

    repo, env = fixture_checkout(tmp_path)
    pinned = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    run_root = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-v1"
    (run_root / ".queue").mkdir(parents=True)
    (run_root / ".queue/generation.git").write_text(pinned + "\n")
    family = run_root / "family-mbpp-s4"
    family.mkdir()
    (family / "existing.partial").write_bytes(b"preserved expensive work\n")
    matrix = repo / "scripts/run_matrix.sh"
    text = matrix.read_text()
    text = text.replace('git=$(git -C "$OM_PIPELINE_REPO" rev-parse HEAD)', r'''
if [ "${REGIME_PARALLEL_CONTROL:-0}" = 1 ]; then
  if [ "$REGIME_CONTROL_ONLY" = 1 ]; then
    echo control >> "$TEST_SHARED/work/roles"
    run="$REGIME_ROOT/$REGIME_MODEL_TAG-s$REGIME_SEEDS-$REGIME_DATASETS-d0"
    mkdir -p "$run"
    echo done > "$run/DONE"
    touch "$TEST_SHARED/work/control-finished"
    exit 0
  fi
  if [ ! -e "$TEST_SHARED/work/control-finished" ]; then
    echo training >> "$TEST_SHARED/work/roles"
    touch "$TEST_SHARED/work/training-started"
    while [ ! -e "$TEST_SHARED/work/control-finished" ]; do /bin/sleep 0.05; done
  fi
fi
git=$(git -C "$OM_PIPELINE_REPO" rev-parse HEAD)
''')
    matrix.write_text(text)
    subprocess.run(["git", "add", "scripts/run_matrix.sh"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "parallel fixture"], cwd=repo, check=True)
    workers = []
    try:
        for i in range(3):
            workers.append(subprocess.Popen(
                ["bash", "scripts/run_olmo3_rlzero.sh", "run"], cwd=repo,
                env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / f"node-{i}"),
                     "OM_RLZERO_PARALLEL_CONTROL": "1", "OM_RLZERO_ONLY_FAMILIES": "mbpp/s4"},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True))
        outputs = []
        for worker in workers:
            try:
                output, _ = worker.communicate(timeout=45)
            except subprocess.TimeoutExpired as exc:
                pytest.fail(str(exc.output)[-8000:])
            assert worker.returncode == 0, output
            outputs.append(output)
        roles = (Path(env["TEST_SHARED"]) / "work/roles").read_text().splitlines()
        assert roles == ["training", "control"]
        assert (family / "existing.partial").read_bytes() == b"preserved expensive work\n"
        assert (run_root / ".queue/generation.git").read_text().strip() == pinned
        claims = (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()
        assert all(line.split("|")[2] == pinned for line in claims)
        assert any("[control-assist] completed mbpp/s4/d0" in output for output in outputs)
        assert all("[fixture-fallback]" not in output for output in outputs)
    finally:
        for worker in workers:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait(timeout=5)


def test_legacy_exclusive_family_lock_blocks_shared_lease(tmp_path):
    # Same lock inode is used by old EX and upgraded SH workers.
    lock_path = tmp_path / "family.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["flock", "-sn", str(lock_path), "true"], timeout=5)
        assert result.returncode != 0


def test_status_counts_control_helper_as_claimed_not_idle(tmp_path):
    from test_rlzero_status import active_family, status_command, write_worker_heartbeat

    _, _, train_lock = active_family(tmp_path)
    fcntl.flock(train_lock, fcntl.LOCK_UN)
    fcntl.flock(train_lock, fcntl.LOCK_SH)
    (tmp_path / ".queue").mkdir()
    generation = "a" * 40
    (tmp_path / ".queue/generation.git").write_text(generation)
    write_worker_heartbeat(tmp_path)
    write_worker_heartbeat(tmp_path, "worker-control")
    owner = tmp_path / ".families/math500-s0.control-owner.json"
    owner.write_text(json.dumps({"worker": "worker-control", "role": "control", "generation_git": generation}))
    try:
        with (tmp_path / ".families/math500-s0.control.lock").open("a") as control_lock:
            fcntl.flock(control_lock, fcntl.LOCK_EX)
            result = subprocess.run(status_command(tmp_path, expected_workers=2),
                                    text=True, capture_output=True, timeout=5)
        assert result.returncode == 0, result.stdout + result.stderr
        line = next(line for line in result.stdout.splitlines()
                    if line.startswith("worker=worker-control "))
        assert "state=CLAIMED" in line
        assert "claims=math500/s0/d0" in line
    finally:
        train_lock.close()
