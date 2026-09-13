"""CPU contract for the one-screen queue overview (scripts/run_queue.sh status)."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import queue_status as qs

ROOT = Path(__file__).resolve().parents[1]
TAG = "tag"
SEEDS = [0, 1, 2]


def _point(root: Path, dataset: str, seed: int, drift: int, done: bool = True) -> Path:
    run = root / f"family-{dataset}-s{seed}" / f"{TAG}-s{seed}-{dataset}-d{drift}"
    run.mkdir(parents=True, exist_ok=True)
    if done:
        (run / "DONE").write_text("ok")
    return run


def _seed_dir(branch: Path, seed: int, selectors, steps: int = 100) -> Path:
    out = branch / f"s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.json").write_text(json.dumps({"seed": seed, "drift": 0, "steps": steps, "eval_k": 8,
                                                     "eval_prompts": 300, "selectors": list(selectors)}))
    return out


def _evaluated(out: Path, arm: str) -> None:
    target = out / arm / "evaluation"
    target.mkdir(parents=True, exist_ok=True)
    for s in range(4):
        (target / f"shard-{s}.done.json").write_text("{}")


def _trained(out: Path, arm: str, logged: int | None = None) -> None:
    policy = out / arm / "policy"
    policy.mkdir(parents=True, exist_ok=True)
    if logged is None:
        (policy / "policy_train.json").write_text("{}")
    else:
        (policy / "grpo_stats.jsonl").write_text("{}\n" * logged)


def _bench(out: Path, arm: str, sets, finished) -> None:
    for name in finished:
        d = out / arm / "benchmark" / name
        d.mkdir(parents=True, exist_ok=True)
        for s in range(4):
            (d / f"shard-{s}.done.json").write_text("{}")
    if not (out / "benchmarks.json").is_file():
        (out / "benchmarks.json").write_text(json.dumps({"sets": {n: {"prompts": 30} for n in sets}, "eval_k": 8}))


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    work = tmp_path / "work"
    root = work / "runs" / TAG
    for seed in SEEDS:
        for drift in (0, 400):
            _point(root, "math500", seed, drift)
    _point(root, "mbpp", 0, 0)
    return work, root


def test_fresh_tree_is_all_todo_or_waiting(tmp_path):
    work, root = _tree(tmp_path)
    rows = qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200)
    states = {name: state for name, state, _ in rows}
    assert states["mixed pool: pool"] == "TODO"
    assert states["mixed pool: point"] == "WAITING" and states["mixed pool: arms"] == "WAITING"
    assert states["reuse split-half d400"] == "TODO" and states["reuse split-half d0"] == "TODO"
    assert states["benchmarks d0"] == "WAITING" and states["d100 continuation"] == "TODO"
    assert states["analyses + export"] == "TODO"
    text = qs.render(rows, "HDR", "NODE")
    assert text.splitlines()[0] == "HDR" and "TODO 5" in text and "WAITING 5" in text
    assert text.index("NODES") < text.index("STEPS") < text.index("mixed pool: pool")


def test_progress_states_and_leases(tmp_path):
    work, root = _tree(tmp_path)
    # pool built, point in progress and leased
    pool = work / "inputs" / "mixed" / "pool-math500-mbpp-s0.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text('{"question": "q", "answer": "1", "split": "train", "source": "math500"}\n')
    point = _point(root, "math500mix", 0, 0, done=False)
    (point / "logs").mkdir()
    (point / "logs" / "main.log").write_text(
        "2026-09-13 10:00:00 [progress] tag-s0-math500mix-d0  1/8 prep  +0min\n"
        "2026-09-13 10:05:00 [progress] tag-s0-math500mix-d0  2/8 behavior-rollout (400x8 on 4 GPUs)  +5min\n")
    lease = Path(str(point) + ".lease")
    lease.write_text("")
    holder = os.open(lease, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    # reuse split-half: seed 0 scored, seed 1 two shards
    (root / "family-math500-s0" / f"{TAG}-s0-math500-d400" / "scores_stale_splithalf.json").write_text("{}")
    run1 = root / "family-math500-s1" / f"{TAG}-s1-math500-d400"
    for s in range(2):
        (run1 / f"scores_stale_splithalf.shard{s}.json").write_text("{}")
    # benchmarks d0: seed 0 before + random on all sets, difficulty on one set, fresh not trained
    e5 = work / "runs" / "e5-reduced"
    sets = ["aime24", "gsm8k"]
    out0 = _seed_dir(e5 / "math500-d0", 0, ["random", "passrate_beta", "fresh_r"])
    for arm in ("random", "passrate_beta"):
        _trained(out0, arm)
    _bench(out0, "before", sets, sets)
    _bench(out0, "random", sets, sets)
    _bench(out0, "passrate_beta", sets, sets[:1])
    for seed in (1, 2):
        _seed_dir(e5 / "math500-d0", seed, ["random"])
    # d100: seed 0 done, seed 1 training
    out100 = _seed_dir(e5 / "math500-d100", 0, ["random", "fresh_r"])
    for arm in ("before", "random", "fresh_r"):
        _evaluated(out100, arm)
    (out100 / "downstream_results.csv").write_text("selector\n")
    out101 = _seed_dir(e5 / "math500-d100", 1, ["random", "fresh_r"])
    _evaluated(out101, "before")
    _trained(out101, "random", logged=37)
    # an export bundle
    (work / "exports").mkdir()
    (work / "exports" / "e5-results-20260913T053944Z.txt").write_text("x")
    try:
        rows = qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200)
    finally:
        os.close(holder)
    by = {name: (state, lines) for name, state, lines in rows}
    assert by["mixed pool: pool"][0] == "DONE"
    assert by["mixed pool: point"][0] == "RUNNING" and by["mixed pool: point"][1][0].startswith("2/8 behavior-rollout +5min *[?]")
    assert by["mixed pool: point"][1][1].startswith("last file write 0m ago")
    assert by["mixed pool: arms"][0] == "WAITING"
    assert by["reuse split-half d400"] == ("PARTIAL", ["s0 ok  s1 2/4 shards  s2 -"])
    assert by["reuse split-half d0"][0] == "TODO"
    state, lines = by["benchmarks d0"]
    assert state == "PARTIAL"
    assert lines[0] == "s0: before ok | random ok | difficulty 1/2 | fresh wait"
    assert lines[1] == "s1: -" and lines[2] == "s2: -"
    state, lines = by["d100 continuation"]
    assert state == "PARTIAL" and lines[0] == "s0: DONE"
    assert lines[1] == "s1: before ok | random train 37/100 | fresh -" and lines[2] == "s2: not prepared"
    state, lines = by["analyses + export"]
    assert state == "PARTIAL" and lines[0].endswith("e5-results-20260913T053944Z.txt") and "gate-decision: none" in lines
    # after the lease is released the point counts as stopped
    rows = qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200)
    assert dict((n, s) for n, s, _ in rows)["mixed pool: point"] == "PARTIAL"
    text = qs.render(rows, "HDR", "NODE")
    assert max(len(line) for line in text.splitlines()) < 110
    assert "DONE 1" in text


def test_mixed_arms_and_gate_after_the_point(tmp_path):
    work, root = _tree(tmp_path)
    pool = work / "inputs" / "mixed" / "pool-math500-mbpp-s0.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text("{}\n")
    _point(root, "math500mix", 0, 0)
    out = _seed_dir(work / "runs" / "e5-reduced" / "math500mix-d0", 0, list(qs.MIX_ARMS), steps=200)
    for arm in ("before", *qs.MIX_ARMS):
        _evaluated(out, arm)
    (out / "downstream_results.csv").write_text("selector\n")
    rows = dict((n, (s, l)) for n, s, l in qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200))
    assert rows["mixed pool: point"][0] == "DONE"
    assert rows["mixed pool: arms (200 upd)"][0] == "DONE"
    assert rows["mixed pool: gate"][0] == "TODO"
    pilot = out / "gate_passrate" / "pilot"
    pilot.mkdir(parents=True)
    (pilot / "grpo_stats.jsonl").write_text("{}\n" * 4)
    (out / "gate_rule.json").write_text(json.dumps({"pilot_steps": 10}))
    rows = dict((n, (s, l)) for n, s, l in qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200))
    assert rows["mixed pool: gate"] == ("PARTIAL", ["s0: gate pilot 4/10"])


def test_cli_and_queue_script_syntax(tmp_path):
    work, root = _tree(tmp_path)
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "OM_WORK": str(work)}
    result = subprocess.run([sys.executable, str(ROOT / "src/queue_status.py"), "--tag", TAG], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("QUEUE STATUS") and "mixed pool: point" in result.stdout
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_queue.sh")], check=True)
    subprocess.run(["bash", "-n", str(ROOT / "scripts/_node_watch.sh")], check=True)
    env["OM_LOCAL_LOCK_DIR"] = str(tmp_path / "locks")
    result = subprocess.run([sys.executable, str(ROOT / "src/queue_status.py"), "--record"], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and result.stdout == ""
    import socket
    assert (work / "queue" / f"{socket.gethostname()}.seen.json").is_file()


def test_lease_notes_name_the_node_and_queue_notes_list_nodes(tmp_path):
    work, root = _tree(tmp_path)
    run1 = root / "family-math500-s1" / f"{TAG}-s1-math500-d400"
    (run1 / "scores_stale_splithalf.shard0.json").write_text("{}")
    lock = run1 / ".stale-splithalf.lock"
    lock.write_text("host=run280417-first-qw-7 pid=145 since=2026-09-13T12:03:00Z\n")
    holder = os.open(lock, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    notes = work / "queue"
    notes.mkdir()
    (notes / "run280417-first-qw-7.txt").write_text("host=run280417-first-qw-7 pid=1 step=run_stale_splithalf.sh since=2026-09-13T12:00:00Z\n")
    (notes / "run280417-first-qw-7.beat").write_text("2026-09-13T12:10:00Z\n")
    (notes / "run280417-first-qw-3.txt").write_text("host=run280417-first-qw-3 pid=2 step=run_mixed_pool.sh point since=2026-09-13T09:00:00Z\n")
    (notes / "run280417-first-qw-3.beat").write_text("x\n")
    os.utime(notes / "run280417-first-qw-3.beat", (1_000_000, 1_000_000))
    (notes / "run280417-first-qw-1.txt").write_text("host=run280417-first-qw-1 pid=3 step=done since=2026-09-13T11:00:00Z\n")
    try:
        rows = qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200)
        by = {name: (state, lines) for name, state, lines in rows}
        assert by["reuse split-half d400"] == ("RUNNING", ["s0 -  s1 1/4 shards*[qw-7] (no write yet)  s2 -"])
        assert qs.NODES == {"run280417-first-qw-7": ["reuse split-half d400 s1"]}
        text = qs.render(rows, "HDR", "NODE", qs.queue_notes(work))
    finally:
        os.close(holder)
    lines = text.splitlines()
    node = [l for l in lines if l.startswith("  qw-7 ")][0]
    assert "run_stale_splithalf.sh  since 09-13 12:00Z  alive (heartbeat" in node
    assert "     holds: reuse split-half d400 s1" in text
    assert [l for l in lines if l.startswith("  qw-3 ")][0].endswith("(killed?)") and "NO HEARTBEAT" in text
    assert [l for l in lines if l.startswith("  qw-1 ")][0].endswith("queue finished 09-13 11:00Z")
    assert "  busy nodes: 1 (qw-7)" in text
    assert "gone or silent" in text and "qw-1" in [l for l in lines if "gone or silent" in l][0]
    # a lease held without a note (job started before the notes existed)
    lock.write_text("")
    holder = os.open(lock, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        rows = qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200)
    finally:
        os.close(holder)
    assert dict((n, l) for n, s, l in rows)["reuse split-half d400"] == ["s0 -  s1 1/4 shards*[?] (no write yet)  s2 -"]


def test_this_node_lists_queue_processes_by_their_marker():
    assert qs.job_label("/w/runs/e5-reduced/math500-d0/.bench") == "benchmarks math500-d0"
    assert qs.job_label("/w/runs/tag/.stale-splithalf-d400") == "reuse split-half d400"
    assert qs.job_label("/w/runs/e5-reduced/math500mix-d0") == "E5 arms math500mix-d0"
    assert qs.job_label("/w/runs/tag/family-math500mix-s0/tag-s0-math500mix-d0") == "point tag-s0-math500mix-d0"
    child = subprocess.Popen(["sleep", "30"], env={"OUT_ROOT": "/w/runs/e5-reduced/math500-d100", "PATH": "/usr/bin:/bin"})
    try:
        import time
        time.sleep(0.2)
        assert "E5 arms math500-d100" in qs.node_jobs()
    finally:
        child.kill()
        child.wait()


def test_old_leases_are_attributed_from_launcher_logs(tmp_path):
    work, root = _tree(tmp_path)
    e5 = work / "runs" / "e5-reduced"
    out = _seed_dir(e5 / "math500-d100", 0, ["random", "fresh_r"])
    _evaluated(out, "before")
    _trained(out, "random", logged=12)
    (out / "logs").mkdir()
    (out / "logs" / "launcher-run280417-first-qw-4-20260913T100000Z.log").write_text("old\n")
    os.utime(out / "logs" / "launcher-run280417-first-qw-4-20260913T100000Z.log", (1_000_000, 1_000_000))
    (out / "logs" / "launcher-run280417-first-qw-2-20260913T120000Z.log").write_text("new\n")
    lock = out / ".random.lock"
    lock.write_text("")
    holder = os.open(lock, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        rows = qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200)
        by = {name: (state, lines) for name, state, lines in rows}
        assert by["d100 continuation"][0] == "RUNNING"
        assert by["d100 continuation"][1][0] == "s0: before ok | random train 12/100*[~qw-2] (write 0m ago) | fresh -"
        text = qs.render(rows, "HDR", "NODE", {})
    finally:
        os.close(holder)
    assert "  busy nodes: 1 (~qw-2)" in text
    assert "lease holder inferred" in text and "holds: d100 s0 random (train 12/100)" in text
    # no launcher log at all: the lease is listed under an unidentified node
    for log in (out / "logs").glob("launcher-*.log"):
        log.unlink()
    holder = os.open(lock, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        rows = qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200)
        text = qs.render(rows, "HDR", "NODE", {})
    finally:
        os.close(holder)
    assert "busy nodes: 0 (none) + 1 lease(s) on an unidentified node" in text
    assert "  ?          holds: d100 s0 random (train 12/100)" in text


def test_seen_records_attribute_old_leases_and_list_idle_nodes(tmp_path, monkeypatch):
    work, root = _tree(tmp_path)
    run1 = root / "family-math500-s1" / f"{TAG}-s1-math500-d400"
    (run1 / "scores_stale_splithalf.shard0.json").write_text("{}")
    lock = run1 / ".stale-splithalf.lock"
    lock.write_text("")
    holder = os.open(lock, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    notes = work / "queue"
    notes.mkdir()
    (notes / "run1-qw-9.seen.json").write_text(json.dumps({"host": "run1-qw-9", "jobs": ["reuse split-half d400"], "lock": True}))
    (notes / "run1-qw-8.seen.json").write_text(json.dumps({"host": "run1-qw-8", "jobs": [], "lock": False}))
    (notes / "run1-qw-6.seen.json").write_text(json.dumps({"host": "run1-qw-6", "jobs": ["E5 arms math500-d100"], "lock": True}))
    import time
    os.utime(notes / "run1-qw-6.seen.json", (time.time() - 3600, time.time() - 3600))
    qs.SEEN.clear()
    qs.SEEN.update(qs.seen_nodes(work))
    try:
        rows = qs.build_rows(work, root, TAG, SEEDS, "mbpp", 0, 200)
        text = qs.render(rows, "HDR", "NODE", {})
    finally:
        os.close(holder)
        qs.SEEN.clear()
    by = {name: (state, lines) for name, state, lines in rows}
    assert by["reuse split-half d400"][1] == ["s0 -  s1 1/4 shards*[~qw-9] (no write yet)  s2 -"]
    assert "  busy nodes: 1 (~qw-9)" in text
    assert "IDLE nodes (alive, nothing running; start the queue there): qw-8" in text
    assert "qw-6" in [l for l in text.splitlines() if "gone or silent" in l][0]
    assert "reported 0m ago from that node: running reuse split-half d400" in text
    # status leaves this node's own view behind
    monkeypatch.setenv("OM_LOCAL_LOCK_DIR", str(tmp_path / "locks"))
    qs.record_seen(work)
    import socket
    record = json.loads((notes / f"{socket.gethostname()}.seen.json").read_text())
    assert record["host"] == socket.gethostname() and isinstance(record["jobs"], list) and record["lock"] is False
