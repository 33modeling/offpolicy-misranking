"""Full Qwen/additional-matrix status (src/matrix_status.py): the flat
run_matrix.sh layout rendered like the OLMo `status h100` screen."""

import fcntl
import json
import os
import sys
import time
from pathlib import Path

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
    assert "Ctrl-C_that_launcher" in out


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
