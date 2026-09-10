"""Current/history separation and shared reference-log visibility, without GPUs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from test_olmo3_launcher import fixture_checkout
from test_rlzero_status import (
    TAG,
    active_family,
    status_command,
    write_worker_heartbeat,
)

CODE = Path(__file__).resolve().parents[1]


def test_concurrent_status_snapshots_include_reference_without_interleaving(tmp_path):
    checkout, env = fixture_checkout(tmp_path)
    shutil.copy2(CODE / "scripts/reference_status.sh", checkout / "scripts")
    command = ["bash", "scripts/run_olmo3_rlzero.sh", "status", "h100"]
    processes = [subprocess.Popen(command, cwd=checkout, env=env, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
    for process in processes:
        out, err = process.communicate(timeout=30)
        assert process.returncode == 0, out + err
        assert "REFERENCE EXPERIMENTS" in out
    logs = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2/logs"
    history = (logs / "status-history.log").read_text()
    snapshots = history.split("===== status ")[1:]
    assert len(snapshots) == 3
    for snapshot in snapshots:
        assert snapshot.count("status_checkout=") == 1
        assert snapshot.count("REFERENCE EXPERIMENTS") == 1
        assert "primary_generation=" in snapshot
        assert "reference_outputs=" in snapshot
    assert not list(logs.glob(".status.*"))


def test_d0_evaluation_is_not_labelled_training_and_has_no_zero_day_eta(tmp_path):
    run, _, lock = active_family(tmp_path)
    try:
        write_worker_heartbeat(tmp_path)
        (run / "run_config.json").write_text(json.dumps({
            "gen_batch": "8", "gradient_micro_batch": 4, "grpo_logprob_micro_batch": 4}))
        later = run.parent / f"{TAG}-s0-math500-d25"
        later.mkdir()
        (later / "DONE").write_text("done")
        command = status_command(tmp_path, verbose=False)
        idx = command.index("--drifts")
        command.insert(idx + 2, "25")
        output = subprocess.check_output(command, text=True)
        assert "PROGRESS EVALUATING" in output
        assert "training checkpoints complete" in output
        assert "~0 days" not in output
        assert "with_claim=1 without_claim=0" in output
    finally:
        lock.close()


def test_reference_status_reports_progress_without_claiming_remote_liveness(tmp_path):
    workers = tmp_path / "reference-workers"
    workers.mkdir()
    run = tmp_path / "runs/reference-axes/math500-fk32-vk8-s101288"
    (run / "logs").mkdir(parents=True)
    stage = run / "logs/fresh-shard2.log"
    stage.write_text("[02:30:27] rollout 91/128 (71%, ETA 1h39m)\n")
    old_stage = run / "logs/val-grads.log"
    old_stage.write_text("RuntimeError: CUDA error: historical failure\n")
    os.utime(old_stage, (1, 1))
    console = tmp_path / "console.log"
    console.write_text("[condition] fresh_k=32 val_k=8\n")
    record = ["RUNNING", "remote-test-node", "123", "math500", "0", "32", "8", "101288",
              str(run), str(console), str(int(time.time()) - 10)]
    (workers / "test.state").write_text("\t".join(record) + "\n")
    output = subprocess.check_output(["bash", str(CODE / "scripts/reference_status.sh"), str(tmp_path)], text=True)
    assert "replicate=0 host=remote-test-node" in output
    assert "liveness=remote-unverified" in output
    assert "rollout 91/128" in output
    assert "scope=modified-since-condition-start" in output
    assert "scope=prior-condition-history last=RuntimeError" in output
    assert "stage/shard only" in output


def test_reference_status_marks_a_dead_local_launcher_unclean(tmp_path):
    workers = tmp_path / "reference-workers"
    workers.mkdir()
    host = subprocess.check_output(["hostname"], text=True).strip()
    child = subprocess.Popen(["true"])
    child.wait()
    record = ["RUNNING", host, str(child.pid), "math500", "0", "32", "8", "101288",
              "-", "-", str(int(time.time()))]
    (workers / "dead.state").write_text("\t".join(record) + "\n")
    output = subprocess.check_output(["bash", str(CODE / "scripts/reference_status.sh"), str(tmp_path)], text=True)
    assert "reported_state=EXITED_UNCLEAN liveness=local-pid-absent" in output


def test_why_excludes_old_error_even_when_old_attempt_number_is_larger(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("why.sh", "setup_env.sh", "reference_status.sh"):
        shutil.copy2(CODE / "scripts" / name, repo / "scripts" / name)
    work = tmp_path / "work"
    root = work / "runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    run = root / "family-mbpp-s4/olmo3-1025-7b-base-rlzero-grpo-h100-v2-s4-mbpp-d0"
    logs = run / "logs"
    logs.mkdir(parents=True)
    old = logs / "regime-attempt-2.log"
    old.write_text("[2026-09-06 22:20:36] RuntimeError: CUDA error: OLD_FAILURE\n")
    os.utime(old, (1, 1))
    (logs / "regime-attempt-1.log").write_text("[2026-09-10 02:30:27] rollout 91/128\n")
    (logs / "fresh-shard2.log").write_text("rollout 91/128\n")
    env = {**os.environ, "OM_REPO": str(repo), "OM_WORK": str(work), "OM_OLMO3_ROOT": str(root),
           "GROUP_VOLUME": str(tmp_path / "absent"), "VENV_DIR": str(Path(sys.executable).parent.parent),
           "WHY_HISTORY": "0", "TMPDIR": str(work / "tmp")}
    subprocess.run(["bash", str(repo / "scripts/why.sh")], env=env, check=True, capture_output=True, timeout=20)
    report = next((work / "exports").glob("why-*.txt")).read_text()
    assert "LATEST ATTEMPT LOG" in report
    assert "regime-attempt-1.log" in report
    assert "OLD_FAILURE" not in report
    assert "REFERENCE EXPERIMENTS" in report
    assert "execution status UNKNOWN" in report


def test_default_status_does_not_print_old_unclaimed_worker_alarms(tmp_path):
    _, _, lock = active_family(tmp_path)
    try:
        write_worker_heartbeat(tmp_path)
        (tmp_path / "logs/ALERTS.log").write_text("2026-09-06T00:00:00Z [WORKER DEAD] old-unclaimed-node\n")
        output = subprocess.check_output(status_command(tmp_path, verbose=False), text=True)
        assert "old-unclaimed-node" not in output
        assert "overall_verdict=RUNNING" in output
    finally:
        lock.close()
