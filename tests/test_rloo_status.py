"""Read-only RLOO status, using the same renderer and states as MBPP/Pair."""
import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rloo_status as status
from test_rloo_experiment import inputs

NOW = 1800000000.


@pytest.mark.parametrize('offset', [-3600, 3600])
def test_live_evaluation_with_clock_skew_remains_visible(prepared, offset):
    root, out = prepared
    directory = out / 'random'
    write(directory / 'progress.json', dict(host='rloo-peer', state='running', phase='evaluation',
          updated=NOW + offset, event_id='evaluation-1', seconds=123, timeout=86400))
    with (directory / '.cost.lock').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = files(root)
        data = status.snapshot(root, now=NOW)
        assert task(data)['status'] == 'RUNNING' and task(data)['owner_active']
        assert not task(data)['heartbeat_fresh']
        assert 'CURRENT RUN 1' in status.display.render(data)
        assert before == files(root)
    assert task(status.snapshot(root, now=NOW))['status'] == 'STALE'


def test_finished_cost_receipt_overrides_stale_running_meter(prepared):
    root, out = prepared
    directory = out / 'random'
    progress = dict(host='rloo-peer', state='running', phase='evaluation', updated=NOW-3600,
                    event_id='evaluation-1', ledger='research', gpus=4, gpu_type='test')
    write(directory / 'progress.json', progress)
    write(directory / 'cost-events/evaluation-1.json', dict(progress, state='finished',
          exit_code=1, seconds=30, allocated_gpu_seconds=120, time=NOW-3500))
    with (directory / '.cost.lock').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(root, now=NOW)
        assert task(data)['status'] == 'WAIT'
        assert not status.display.active(task(data))


@pytest.fixture
def prepared(tmp_path):
    run, evaluation = inputs(tmp_path / "inputs")
    root = tmp_path / "rloo"
    out = root / "math500-d0/s0"
    status.experiment.prepare(run, out, evaluation)
    return root, out


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def task(data, arm="random", drift=0, seed=0):
    return next(t for s in data["suites"] for t in s["tasks"]
                if t["arm"] == arm and t["seed"] == seed and t["step"] == drift)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(out, arm):
    manifest = out / arm / "policy/policy_train.json"
    if arm != "before":
        write(manifest, dict(training_objective="rloo", start_step=0, completed_steps=100, adapter_sha256='a' * 64))
    for shard in range(4):
        target = out / arm / "evaluation"
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"shard-{shard}.jsonl"
        path.write_text("".join(json.dumps(dict(prompt_idx=i, rollout_idx=j, reward=1)) + "\n"
                                for i in range(75 * shard, 75 * (shard + 1)) for j in range(8)))
        binding = dict(experiment_sha256=status.digest(out / "experiment.json"),
                       inputs_sha256=status.digest(out / "inputs.json"), arm=arm, shard=shard,
                       adapter_sha256=None if arm == "before" else 'a' * 64,
                       manifest_sha256=None if arm == "before" else status.digest(manifest))
        write(target / f"shard-{shard}.done.json", dict(binding=binding, rollouts_sha256=status.digest(path)))


def test_absent_root_has_18_training_slots_and_never_creates_files(tmp_path):
    root = tmp_path / "absent"
    data = status.snapshot(root, now=NOW)
    counts = [status.display.counts(suite) for suite in data["suites"]]
    assert sum(c["planned"] for c in counts) == 18
    assert sum(c["done"] for c in counts) == 0
    assert sum(c["states"]["WAIT"] for c in counts) == 18
    assert sum(c["unknown"] for c in counts) == 18
    text = status.display.render(data)
    for label in ("RLOO EXPERIMENTS", "FULL STATUS", "CURRENT RUN", "NODE ASSIGNMENTS", "Progress"):
        assert label in text
    assert "WAIT 0" not in text
    assert not root.exists()


def test_prepared_status_reads_without_creating_worker_locks(prepared):
    root, out = prepared
    before = files(root)
    data = status.snapshot(root, now=NOW)
    assert task(data)["status"] == "READY"
    assert task(data, seed=1)["status"] == "WAIT"
    assert before == files(root)
    assert not list(root.rglob(".worker.lock"))
    text = status.display.render(data, all_tasks=True)
    assert "Random" in text and "Cached" in text and "On-policy" in text and "Before" in text
    assert "0 / 0" in text and "0 / 25" not in text


def test_baseline_is_separate_and_done_requires_all_sealed_shards(prepared):
    root, out = prepared
    seal(out, "before")
    seal(out, "random")
    data = status.snapshot(root, now=NOW)
    assert task(data, "before")["status"] == "DONE"
    assert task(data)["status"] == "DONE"
    assert status.display.counts(data["suites"][0])["done"] == 1
    (out / "random/evaluation/shard-3.done.json").unlink()
    data = status.snapshot(root, now=NOW)
    assert task(data)["status"] == "EVAL" and task(data)["evaluation_shards"] == 3
    assert status.display.counts(data["suites"][0])["done"] == 0


@pytest.mark.parametrize("kind", ["rollout", "binding", "policy", "contract"])
def test_corrupt_publications_are_not_done(prepared, kind):
    root, out = prepared
    seal(out, "random")
    if kind == "rollout":
        (out / "random/evaluation/shard-0.jsonl").write_text("")
    elif kind == "binding":
        path = out / "random/evaluation/shard-0.done.json"
        value = status.read(path)
        value["binding"]["arm"] = "fresh_r"
        write(path, value)
    elif kind == "policy":
        write(out / "random/policy/policy_train.json", dict(training_objective="grpo"))
    else:
        path = out / "experiment.json"
        value = status.read(path)
        value["source"]["seed"] = 4
        write(path, value)
    data = status.snapshot(root, now=NOW)
    assert task(data)["status"] == "WAIT"
    assert task(data)["reason"]
    assert status.display.counts(data["suites"][0])["done"] == 0


def test_training_without_evaluation_is_not_done(prepared):
    root, out = prepared
    write(out / "random/policy/policy_train.json",
          dict(training_objective="rloo", start_step=0, completed_steps=100))
    assert task(status.snapshot(root, now=NOW))["status"] == "EVAL"


@pytest.mark.parametrize("state,age,expected", [("running", 5, "RUNNING"), ("running", 90, "STALE"),
                                              ("failed", 5, "WAIT"), ("finished", 5, "READY")])
def test_fresh_stale_and_failed_heartbeats(prepared, state, age, expected):
    root, out = prepared
    write(out / "random/progress.json", dict(state=state, updated=NOW-age, host="rloo-node-2", phase="train",
                                            seconds=300, timeout=86400))
    data = status.snapshot(root, now=NOW)
    assert task(data)["status"] == expected
    output = status.display.render(data, all_tasks=True)
    assert ("CURRENT RUN 1" in output) == (expected == "RUNNING")
    if expected == "RUNNING":
        assert "rloo-node-2" in output and "RLOO MATH d0 / seed 0 / step 0 / Random" in " ".join(output.split())


def test_lock_file_alone_does_not_prove_running(prepared):
    root, out = prepared
    write(out / "random/.worker.lock", {})
    assert task(status.snapshot(root, now=NOW))["status"] == "READY"


def test_queue_error_remains_visible_with_failed_phase(prepared):
    root, out = prepared
    write(out / "random/queue-attempt.json", dict(state="FAILED", error="specific branch validation failure"))
    write(out / "random/progress.json", dict(state="failed", phase="train", updated=NOW-5))
    observed = task(status.snapshot(root, now=NOW))
    assert observed["status"] == "WAIT"
    assert observed["reason"] == "specific branch validation failure"


def test_malformed_source_metadata_does_not_crash_dashboard(prepared):
    root, out = prepared
    c = status.read(out / "experiment.json")
    c["source"] = []
    write(out / "experiment.json", c)
    data = status.snapshot(root, now=NOW)
    assert data["suites"][0]["error"]
    assert task(data)["status"] == "WAIT"


def test_d400_matrix_and_node_assignment_keep_correct_checkpoint(prepared, tmp_path):
    root, out = prepared
    run, evaluation = inputs(tmp_path / "d400-inputs", drift=400)
    later = root / "math500-d400/s0"
    status.experiment.prepare(run, later, evaluation)
    for point, host in ((out, "early-node"), (later, "late-node")):
        write(point / "random/progress.json", dict(state="running", updated=NOW-5, host=host,
                                                  phase="train", seconds=10, timeout=86400))
    data = status.snapshot(root, now=NOW)
    output = " ".join(status.display.render(data, width=180).split())
    assert "0 / 400" in output
    assert "RLOO MATH d0 / seed 0 / step 0 / Random" in output
    assert "RLOO MATH d400 / seed 0 / step 400 / Random" in output
    assert "NODES 2 current" in output and "CURRENT RUN 2" in output


@pytest.mark.parametrize("width", [80, 100, 120, 180])
def test_same_table_fits_terminal_width(prepared, width):
    root, _ = prepared
    output = status.display.render(status.snapshot(root, now=NOW), width=width)
    assert all(status.display.columns(line) <= width for line in output.splitlines())


def test_launcher_json_and_all_are_cpu_read_only(tmp_path):
    env = {**os.environ, "RLOO_PYTHON": sys.executable, "RLOO_ROOT": str(tmp_path / "absent")}
    script = status.experiment.ROOT / "scripts/run_rloo.sh"
    result = subprocess.run(["bash", str(script), "status", "--json", "--all"], env=env,
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["subject"] == "RLOO"
    assert not (tmp_path / "absent").exists()
    result = subprocess.run(["bash", str(script), "status", "--watch", "0"], env=env,
                            text=True, capture_output=True)
    assert result.returncode == 2


def test_watch_refreshes_without_gpu_work(tmp_path):
    calls = tmp_path / "calls"
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$CALLS\"\n")
    fake.chmod(0o755)
    env = {**os.environ, "RLOO_PYTHON": str(fake), "RLOO_ROOT": str(tmp_path / "absent"), "CALLS": str(calls)}
    process = subprocess.Popen(["bash", str(status.experiment.ROOT / "scripts/run_rloo.sh"),
                                "status", "--watch", "1", "--all"], env=env)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if calls.exists() and len(calls.read_text().splitlines()) >= 2:
                break
            time.sleep(.05)
        rows = calls.read_text().splitlines()
        assert len(rows) >= 2 and all("scripts/rloo_status.py" in row and "--all" in row for row in rows)
        assert not (tmp_path / "absent").exists()
    finally:
        process.terminate()
        process.wait(timeout=3)
