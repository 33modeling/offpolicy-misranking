"""Full Qwen/additional-matrix status (src/matrix_status.py): the flat
run_matrix.sh layout rendered like the OLMo `status h100` screen."""

import fcntl
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matrix_status

RUN_ID = "qwen35-9b-posttrained-math-code-grpo-v1"
MODEL_KEY = "qwen3.5-9b-posttrained"
TAG = f"{RUN_ID}-grpo-{MODEL_KEY}"


def point_dir(root: Path, dataset: str, seed: int, drift: int) -> Path:
    return root / RUN_ID / MODEL_KEY / f"{TAG}-s{seed}-{dataset}-d{drift}"


def make_point(root: Path, dataset: str, seed: int, drift: int, *, done=False, progress=None, main_extra="", age=0.0) -> Path:
    run = point_dir(root, dataset, seed, drift)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "run_config.json").write_text(json.dumps({"n_train": 400}))
    text = ""
    if progress:
        text += f"[2026-09-10 10:00:00] [progress] {run.name}  {progress}  +3min\n"
    text += main_extra
    (run / "logs/main.log").write_text(text)
    if done:
        (run / "DONE").write_text("done\n")
    if age:
        stamp = time.time() - age
        for path in run.rglob("*"):
            os.utime(path, (stamp, stamp))
        os.utime(run, (stamp, stamp))
    return run


def session_log(work: Path, name: str, *, host: str, pid: int, family: str = "", point: str = "", exit_rc=None, fails=0, waiting=False) -> Path:
    logs = work / "console-logs"
    logs.mkdir(parents=True, exist_ok=True)
    text = f"[launch] utc=2026-09-10T00:31:25Z profile=qwen35 mode=--run host={host} pid={pid} git=abc log=x\n"
    text += "[stage] 00:40:00Z  matrix-qwen3.5-9b-posttrained-attempt-1\n"
    if family:
        text += f"[progress] family={family} point={point} points_done=0/40 seeds=0 1 2 3 4 datasets=math500 mbpp\n"
    text += "[family-fail] math500/s9\n" * fails
    if waiting:
        text += "[queue] waiting for 3 families held by other workers\n"
    if exit_rc is not None:
        text += f"[exit] rc={exit_rc}\n"
    path = logs / name
    path.write_text(text)
    return path


def render(work: Path, **overrides) -> str:
    args = matrix_status.parse_args(
        [
            "--root", str(work / "runs" / RUN_ID),
            "--config", str(ROOT / "configs/qwen35_9b_grpo.json"),
            "--console-logs", str(work / "console-logs"),
            "--log-glob", "additional-qwen35-*.log",
        ]
        + [f"--{k.replace('_', '-')}" if v is True else f"--{k.replace('_', '-')}={v}" for k, v in overrides.items()]
    )
    lines, _ = matrix_status.render(args)
    return "\n".join(lines)


def test_every_family_is_listed_even_when_nothing_started(tmp_path):
    out = render(tmp_path / "work")
    for dataset in ("math500", "mbpp"):
        for seed in range(5):
            assert f" {dataset}/s{seed} " in out
    assert "families 10: NOT_STARTED=10" in out
    assert "overall_verdict=NOT_STARTED" in out
    assert "(no scored point yet)" in out


def test_claimed_partial_complete_and_pending_families_are_distinguished(tmp_path):
    work = tmp_path / "work"
    runs = work / "runs"
    # math500/s0: complete
    for drift in (0, 25, 100, 400):
        make_point(runs, "math500", 0, drift, done=True, progress="8/8 DONE")
    # mbpp/s0: claimed by a launcher on this node, at d25 GRPO with 10 steps done
    make_point(runs, "mbpp", 0, 0, done=True, progress="8/8 DONE")
    run = make_point(runs, "mbpp", 0, 25, progress="3/8 grpo steps 0->25 (see grpo.log)")
    (run / "policy_step_25").mkdir()
    (run / "policy_step_25/grpo_stats.jsonl").write_text("{}\n" * 10)
    queue = runs / RUN_ID / MODEL_KEY / ".queue"
    queue.mkdir(parents=True)
    (queue / "generation.git").write_text("0123456789abcdef0123456789abcdef01234567\n")
    lock = (queue / "mbpp-s0.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # math500/s1: started earlier, nobody holds it, a launcher is alive -> QUEUED
    make_point(runs, "math500", 1, 0, progress="2/8 behavior-rollout 400x8 on 4 GPUs", age=7200)
    session_log(work, "additional-qwen35-run-a.log", host=os.uname().nodename, pid=os.getpid(), family="mbpp/s0", point="d25")
    try:
        out = render(work)
    finally:
        lock.close()
    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith((" math500/", " mbpp/"))}
    assert "COMPLETE" in rows["math500/s0"] and "ok     ok     ok     ok" in rows["math500/s0"]
    assert "PROGRESSING" in rows["mbpp/s0"]
    assert " d25 " in rows["mbpp/s0"] and "3/8 grpo" in rows["mbpp/s0"] and "10/25" in rows["mbpp/s0"]
    assert "QUEUED" in rows["math500/s1"] and "2/8" in rows["math500/s1"]
    assert "QUEUED" in rows["mbpp/s1"] and "not started" in rows["mbpp/s1"]
    assert "ALIVE (this node)" in out
    assert "generation 0123456789ab" in out
    assert "overall_verdict=RUNNING" in out
    assert "points DONE 5/40, started 7" in out


def test_current_attempt_error_is_named_and_earlier_errors_are_history(tmp_path):
    work = tmp_path / "work"
    runs = work / "runs"
    run = make_point(runs, "math500", 0, 0, progress="1/8 prep", main_extra="[abort]\nRuntimeError: test failure\n")
    (run / "logs/regime-attempt-1.log").write_text("CUDA error: unspecified launch failure\n")
    (run / "logs/regime-attempt-2.log").write_text("attempt started\nrollout 3/100\n")
    queue = runs / RUN_ID / MODEL_KEY / ".queue"
    queue.mkdir(parents=True)
    lock = (queue / "math500-s0.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        out = render(work, verbose=True)
    finally:
        lock.close()
    row = next(line for line in out.splitlines() if line.startswith(" math500/s0 "))
    assert "ERROR (current): RuntimeError: test failure" in row
    assert "!1/8" in row
    assert "overall_verdict=DEGRADED" in out
    assert "read_the_ERROR_rows" in out
    # verbose: the per-point row names the attempt and counts both errors
    assert "math500/s0/d0  attempt 2" in out
    # earlier-only error: the current attempt is clean
    (run / "logs/main.log").write_text(f"[abort]\nRuntimeError: old\n[2026-09-10 10:00:00] [progress] {run.name}  4/8 fresh-rollout  +9min\n")
    lock = (queue / "math500-s0.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        out = render(work)
    finally:
        lock.close()
    row = next(line for line in out.splitlines() if line.startswith(" math500/s0 "))
    assert "ok (earlier attempt failed: RuntimeError: old)" in row
    assert "4/8" in row and "!4/8" not in row


MODEL_ALIAS_ERROR = (
    "ValueError: rollouts_behavior_train.shard0.manifest.json: "
    "model mismatch: expected 'uploaded-snapshot', recorded 'Qwen3.5-9B-pinned'"
)


def attempt_log(run, name, text, started_ns, *, main_offset=None):
    path = run / "logs" / name
    path.write_text(text)
    record = {
        "schema": "offpolicy-pipeline-attempt/v1",
        "attempt_log": name,
        "started_at_ns": started_ns,
        "log_offsets": {} if main_offset is None else {"main.log": main_offset},
    }
    path.with_name(name + ".start.json").write_text(json.dumps(record))
    return path


def test_alias_resume_does_not_report_the_behavior_failure_as_current(tmp_path):
    work = tmp_path / "work"
    run = make_point(work / "runs", "math500", 0, 0, progress="2/8 behavior-rollout",
                     main_extra=f"Traceback (most recent call last):\n{MODEL_ALIAS_ERROR}\n")
    main = run / "logs/main.log"
    before = main.read_bytes()
    attempt_log(run, "regime-attempt-1.log", f"Traceback:\n{MODEL_ALIAS_ERROR}\n", 100)
    progress = f"[progress] {run.name}  4/8 fresh-rollout 400x32 + val  +0min\n"
    main.write_bytes(before + progress.encode())
    attempt_log(run, "regime-attempt-1-alias-1.log", progress, 200, main_offset=len(before))
    before_status = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}

    point = matrix_status.inspect_point("math500", 0, 0, run, [0, 25, 100, 400])
    assert not point.current_error
    assert "Qwen3.5-9B-pinned" in point.earlier_error
    assert point.mark() == "4/8"
    out = render(work, verbose=True)
    assert "ERROR (current)" not in out
    assert "regime-attempt-1-alias-1.log" in out
    assert {p: p.read_bytes() for p in work.rglob("*") if p.is_file()} == before_status


@pytest.mark.parametrize("recorded_start", [True, False])
def test_new_session_attempt_one_supersedes_old_attempt_three(tmp_path, recorded_start):
    run = make_point(tmp_path, "math500", 0, 0, progress="4/8 fresh-rollout")
    old = attempt_log(run, "regime-attempt-3.log", f"Traceback:\n{MODEL_ALIAS_ERROR}\n", 100)
    new = attempt_log(run, "regime-attempt-1.log", "new launch\n", 200)
    if recorded_start:
        # A copied/touched historical log must not outrank the actual start record.
        os.utime(old, ns=(300, 300))
        os.utime(new, ns=(200, 200))
    else:
        for path in run.glob("logs/*.start.json"):
            path.unlink()
        os.utime(old, ns=(100, 100))
        os.utime(new, ns=(200, 200))
    point = matrix_status.inspect_point("math500", 0, 0, run, [0, 25])
    assert point.attempt == 1
    assert not point.current_error
    assert "Qwen3.5-9B-pinned" in point.earlier_error


def test_new_failure_in_alias_attempt_is_still_current(tmp_path):
    run = make_point(tmp_path, "math500", 0, 0, progress="4/8 fresh-rollout")
    attempt_log(run, "regime-attempt-1.log", "old attempt\n", 100)
    # A one-line ValueError must be visible even without a Traceback header.
    attempt_log(run, "regime-attempt-1-alias-2.log", MODEL_ALIAS_ERROR + "\n", 200)
    point = matrix_status.inspect_point("math500", 0, 0, run, [0, 25])
    assert "Qwen3.5-9B-pinned" in point.current_error
    assert point.mark() == "!4/8"


@pytest.mark.parametrize("new_failure", [False, True])
def test_main_log_errors_are_scoped_to_attempt_start_offset(tmp_path, new_failure):
    run = make_point(tmp_path, "math500", 0, 0, progress="2/8 behavior-rollout",
                     main_extra=f"Traceback:\n{MODEL_ALIAS_ERROR}\n")
    main = run / "logs/main.log"
    offset = main.stat().st_size
    attempt_log(run, "regime-attempt-1-alias-1.log", "preflight started\n", 200, main_offset=offset)
    if new_failure:
        with main.open("a") as stream:
            stream.write("[abort]\nRuntimeError: new fresh failure\n")
    point = matrix_status.inspect_point("math500", 0, 0, run, [0, 25])
    if new_failure:
        assert "new fresh failure" in point.current_error
    else:
        assert not point.current_error
        assert "Qwen3.5-9B-pinned" in point.earlier_error


@pytest.mark.parametrize("record_problem", ["bad-json", "wrong-name", "bad-offset", "truncated-main"])
def test_bad_start_record_or_truncated_main_does_not_hide_failure(tmp_path, record_problem):
    run = make_point(tmp_path, "math500", 0, 0, progress="4/8 fresh-rollout",
                     main_extra="RuntimeError: current failure\n")
    main = run / "logs/main.log"
    path = attempt_log(run, "regime-attempt-1-alias-1.log", "starting\n", 200,
                       main_offset=main.stat().st_size)
    sidecar = path.with_name(path.name + ".start.json")
    record = json.loads(sidecar.read_text())
    if record_problem == "bad-json":
        sidecar.write_text("{")
    elif record_problem == "wrong-name":
        record["attempt_log"] = "regime-attempt-9.log"
        sidecar.write_text(json.dumps(record))
    elif record_problem == "bad-offset":
        record["log_offsets"]["main.log"] = -1
        sidecar.write_text(json.dumps(record))
    else:
        main.write_text("RuntimeError: current failure\n")
    point = matrix_status.inspect_point("math500", 0, 0, run, [0, 25])
    assert "current failure" in point.current_error


def test_quiet_and_hung_claimed_families_and_stopped_without_launchers(tmp_path):
    work = tmp_path / "work"
    runs = work / "runs"
    make_point(runs, "math500", 0, 0, progress="4/8 fresh-rollout", age=3000)
    make_point(runs, "mbpp", 0, 0, progress="4/8 fresh-rollout", age=4 * 3600)
    make_point(runs, "math500", 1, 0, progress="2/8 behavior-rollout", age=600)
    queue = runs / RUN_ID / MODEL_KEY / ".queue"
    queue.mkdir(parents=True)
    locks = []
    for name in ("math500-s0.lock", "mbpp-s0.lock"):
        lock = (queue / name).open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    try:
        out = render(work)
    finally:
        for lock in locks:
            lock.close()
    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith((" math500/", " mbpp/"))}
    assert "QUIET" in rows["math500/s0"] and "check again in 30 min" in rows["math500/s0"]
    assert "HUNG" in rows["mbpp/s0"] and "NEEDS YOU" in rows["mbpp/s0"]
    assert "STOPPED" in rows["math500/s1"] and "no launcher anywhere" in rows["math500/s1"]
    assert "launchers: no session log found" in out
    # nothing is clearly progressing, one family is hung: the OLMo rule says HUNG
    assert "overall_verdict=HUNG" in out
    assert "verify_HUNG_node_and_stage_logs_before_interrupting" in out


def test_launcher_rows_report_exit_remote_liveness_and_failures(tmp_path):
    work = tmp_path / "work"
    session_log(work, "additional-qwen35-run-old.log", host="run1-first-rlvr-2", pid=4242, exit_rc=1, fails=3)
    fresh = session_log(work, "additional-qwen35-run-new.log", host="run2-first-rlvr-3", pid=4343, waiting=True)
    stale = session_log(work, "additional-qwen35-run-stale.log", host="run3-first-rlvr-4", pid=4444)
    old = time.time() - 7200
    os.utime(stale, (old, old))
    out = render(work)
    assert "rlvr-2" in out and "EXITED rc=1" in out and " 3     " in out
    assert "rlvr-3" in out and "RUNNING? (remote, log fresh)" in out
    assert "waiting for 3 families held by other workers" in out
    assert "rlvr-4" in out and "SILENT (remote, log 2h00m old)" in out
    assert "launchers live 1/3" in out
    assert "overall_verdict=STARTING" in out
    assert fresh.is_file()


def test_key_numbers_are_printed_for_scored_points(tmp_path):
    work = tmp_path / "work"
    run = make_point(work / "runs", "math500", 2, 100, done=True, progress="8/8 DONE")
    (run / "report.json").write_text(json.dumps({
        "k": 40, "noise_floor": 0.25, "certagrad": {"precision_vs_oracle": 0.175},
        "g00": {"precision": 0.1}, "g01": {"precision": 0.125}, "g10": {"precision": 0.15}, "g11": {"precision": 0.2},
    }))
    (run / "divergence_stats.json").write_text(json.dumps({"token_kl_beta_pi": 0.000327, "traj_ess_frac_g11": 0.5}))
    out = render(work)
    assert " math500 s2 d100 current floor=0.250 GATE-OK  fresh=0.175 g00=0.100 g01=0.125 g10=0.150 g11=0.200 KL=0.000327 ESS=0.500" in out


def test_renderer_never_creates_queue_files(tmp_path):
    work = tmp_path / "work"
    make_point(work / "runs", "math500", 0, 0, progress="1/8 prep")
    before = sorted(p.relative_to(work) for p in work.rglob("*"))
    render(work)
    after = sorted(p.relative_to(work) for p in work.rglob("*"))
    assert before == after


def test_default_output_lists_all_forty_points_including_done_and_pending(tmp_path):
    work = tmp_path / "work"
    make_point(work / "runs", "math500", 0, 0, done=True, progress="8/8 DONE")
    make_point(work / "runs", "math500", 0, 25, progress="3/8 grpo")
    out = render(work)
    table = out.split("ALL POINTS (40)", 1)[1].split("LAUNCHERS", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith((" math500/", " mbpp/"))]
    assert len(rows) == 40
    for dataset in ("math500", "mbpp"):
        for seed in range(5):
            for drift in (0, 25, 100, 400):
                assert sum(f" {dataset}/s{seed}/d{drift} " in row for row in rows) == 1
    assert "DONE" in rows[0] and "NOT_STARTED" in rows[-1]


def test_no_live_launcher_is_hidden_by_eight_newer_exit_logs(tmp_path):
    work = tmp_path / "work"
    active = session_log(work, "additional-qwen35-live.log", host=os.uname().nodename, pid=os.getpid())
    old = time.time() - 1800
    os.utime(active, (old, old))
    for index in range(10):
        session_log(work, f"additional-qwen35-exit-{index}.log", host=f"node-{index}", pid=4444, exit_rc=1)
    out = render(work)
    assert "ALIVE (this node)" in out
    assert "launchers live 1/11" in out
    assert "overall_verdict=STARTING" in out


def test_silent_remote_session_is_unverified_not_a_dead_local_process(tmp_path):
    work = tmp_path / "work"
    make_point(work / "runs", "math500", 0, 0, progress="4/8 fresh-rollout", age=7200)
    log = session_log(work, "additional-qwen35-remote.log", host="remote-node", pid=7777)
    old = time.time() - 7200
    os.utime(log, (old, old))
    out = render(work)
    assert "overall_verdict=UNVERIFIED" in out
    assert "STOPPED" not in next(line for line in out.splitlines() if line.startswith(" math500/s0 "))
    assert "Ctrl-C" not in out


def test_complete_design_ignores_failures_in_old_launcher_sessions(tmp_path):
    work = tmp_path / "work"
    for dataset in ("math500", "mbpp"):
        for seed in range(5):
            for drift in (0, 25, 100, 400):
                make_point(work / "runs", dataset, seed, drift, done=True)
    session_log(work, "additional-qwen35-failed.log", host="remote-node", pid=7777, exit_rc=1)
    out = render(work)
    assert out.startswith("DECISION DONE:")
    assert "overall_verdict=COMPLETE" in out


def test_live_session_with_a_current_error_is_not_reported_as_no_error(tmp_path):
    work = tmp_path / "work"
    make_point(work / "runs", "math500", 0, 0, progress="1/8 prep", main_extra="[abort]\nRuntimeError: test failure\n")
    session_log(work, "additional-qwen35-current.log", host=os.uname().nodename, pid=os.getpid())
    out = render(work)
    assert "DECISION NO ERROR" not in out
    assert "overall_verdict=DEGRADED" in out


def test_silent_preflight_is_not_proof_of_training_progress(tmp_path):
    work = tmp_path / "work"
    log = session_log(work, "additional-qwen35-preflight.log", host=os.uname().nodename, pid=os.getpid())
    old = time.time() - 7200
    os.utime(log, (old, old))
    out = render(work)
    assert "overall_verdict=DEGRADED" in out
    assert "pid_liveness_is_not_progress" in out
    assert "DECISION NO ERROR" not in out


def test_days_old_session_log_without_exit_does_not_keep_families_unverified(tmp_path):
    work = tmp_path / "work"
    make_point(work / "runs", "math500", 0, 0, progress="2/8 behavior-rollout", age=5 * 86400)
    fresh = session_log(work, "additional-qwen35-run-recent.log", host="run1-first-rlvr-2", pid=4242)
    stale = session_log(work, "additional-qwen35-run-lost.log", host="run2-first-rlvr-3", pid=4343)
    old = time.time() - 5 * 86400
    os.utime(stale, (old, old))
    out = render(work)
    row = next(line for line in out.splitlines() if line.startswith(" math500/s0 "))
    # the recent remote log (20 min < age < 3 d) keeps the family UNVERIFIED ...
    two_hours = time.time() - 7200
    os.utime(fresh, (two_hours, two_hours))
    out = render(work)
    row = next(line for line in out.splitlines() if line.startswith(" math500/s0 "))
    assert "UNVERIFIED" in row
    # ... but once every non-exited log is days old the family is STOPPED
    os.utime(fresh, (old, old))
    out = render(work)
    row = next(line for line in out.splitlines() if line.startswith(" math500/s0 "))
    assert "STOPPED" in row and "no launcher anywhere" in row
    assert "overall_verdict=STOPPED" in out
