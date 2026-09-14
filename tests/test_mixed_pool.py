"""CPU contracts for the mixed candidate pool (positive control)."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import data
import mixed_pool as mp

ROOT = Path(__file__).resolve().parents[1]


def _run(root: Path, name: str, n_train: int, n_val: int, prefix: str) -> Path:
    run = root / f"family-{name}-s0" / f"tag-s0-{name}-d0"
    run.mkdir(parents=True)
    prompts = {"train": [{"question": f"{prefix} train {i}", "answer": str(i)} for i in range(n_train)],
               "val": [{"question": f"{prefix} val {i}", "answer": str(i)} for i in range(n_val)]}
    (run / "prompts.json").write_text(json.dumps(prompts))
    (run / "DONE").write_text("ok")
    (run / "run_config.json").write_text(json.dumps({
        "model": "/models/olmo", "dataset": name, "n_train": n_train, "n_val": n_val, "behavior_k": 8, "fresh_k": 32,
        "val_k": 8, "micro_group": 4, "hybrid_prompts": 24, "k_cell": 8, "max_new_tokens": 2048, "proj_dim": 4096,
        "grad_layers": 4, "clip_cap": 10.0, "temperature": 1.0, "topk_frac": 0.1, "radius_mode": "gaussian",
        "top_p": 1.0, "thinking": "off", "prompt_format": "olmo_rlzero_math", "attn": "eager", "gen_batch": "32",
        "lora_targets": "q_proj,v_proj", "skip_hybrid": "1", "grpo_world_size": 4, "grpo_group_size": 8,
        "grpo_clip_epsilon": 0.2, "grpo_learning_rate": 1e-5, "grpo_epochs_per_batch": 1, "grpo_max_grad_norm": 1.0,
        "grpo_advantage_epsilon": 1e-4, "grpo_lora_rank": 16, "grpo_lora_alpha": 32, "grpo_logprob_micro_batch": 1,
        "grpo_gradient_checkpointing": "1", "gradient_micro_batch": 1, "seed": 0, "drift": 0}))
    return run


def test_build_mixes_training_prompts_and_keeps_math_validation(tmp_path):
    math = _run(tmp_path, "math500", 400, 100, "math")
    other = _run(tmp_path, "mbpp", 400, 100, "code")
    out = tmp_path / "pool.jsonl"
    manifest = mp.build(math, other, out, 200, 200, 100, seed=0)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 500 and manifest["counts"] == {"train:math500": 200, "train:mbpp": 200, "val:math500": 100}
    train = [r for r in rows if r["split"] == "train"]
    assert {r["source"] for r in train} == {"math500", "mbpp"}
    assert [r["question"] for r in rows if r["split"] == "val"] == [f"math val {i}" for i in range(100)]
    assert sorted(r["question"] for r in train if r["source"] == "math500") == sorted(f"math train {i}" for i in range(200))
    assert train[0]["question"] != "math train 0"  # shuffled order
    again = mp.build(math, other, out, 200, 200, 100, seed=0)
    assert again["sha256"] == manifest["sha256"]
    with pytest.raises(ValueError, match="different content"):
        mp.build(math, other, out, 150, 250, 100, seed=0)
    with pytest.raises(ValueError, match="not enough distinct"):
        mp.build(math, other, tmp_path / "p2.jsonl", 500, 200, 100, seed=0)


def test_repeated_questions_are_skipped_not_fatal(tmp_path):
    math = _run(tmp_path, "math500", 400, 100, "math")
    other = _run(tmp_path, "mbpp", 400, 100, "code")
    prompts = json.loads((other / "prompts.json").read_text())
    prompts["train"][5] = dict(prompts["train"][0])          # the source repeats one candidate
    prompts["train"][9] = dict(prompts["train"][1])
    (other / "prompts.json").write_text(json.dumps(prompts))
    out = tmp_path / "pool.jsonl"
    manifest = mp.build(math, other, out, 200, 200, 100, seed=0)
    assert manifest["counts"] == {"train:math500": 200, "train:mbpp": 200, "val:math500": 100}
    assert manifest["repeated_questions_skipped"] == 2
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    questions = [r["question"] for r in rows]
    assert len(set(questions)) == len(questions) == 500


def test_loader_honours_the_pre_split_pool(tmp_path, monkeypatch):
    math = _run(tmp_path, "math500", 30, 10, "math")
    other = _run(tmp_path, "mbpp", 30, 10, "code")
    out = tmp_path / "pool.jsonl"
    mp.build(math, other, out, 20, 20, 10, seed=1)
    monkeypatch.setenv("OM_PROMPT_POOL_FILE", str(out))
    prompts = data.load_prompts("math500", 40, 10, seed=0)
    assert len(prompts["train"]) == 40 and len(prompts["val"]) == 10
    assert all(q["question"].startswith("math val") for q in prompts["val"])
    assert set(prompts["train"][0]) == {"question", "answer"}
    with pytest.raises(ValueError, match="train 40 < 50"):
        data.load_prompts("math500", 50, 10, seed=0)


def test_the_mixed_pool_does_not_declare_a_prescreened_pool():
    """OM_POOL_FILE makes run_point.sh requalify the pool (src/qualify_pool.py); the mixed pool
    is only a prompt list and must use OM_PROMPT_POOL_FILE, or every launch aborts there."""
    runner = (ROOT / "scripts/run_mixed_pool.sh").read_text()
    assert "OM_PROMPT_POOL_FILE=\"$POOL\"" in runner and "unset OM_POOL_FILE" in runner
    assert "OM_POOL_FILE=\"$POOL\"" not in runner
    point = (ROOT / "scripts/run_point.sh").read_text()
    assert 'if [ -n "${OM_POOL_FILE:-}" ]; then' in point and "qualify_pool.py" in point


def test_env_lines_cover_the_point_runner_configuration(tmp_path):
    math = _run(tmp_path, "math500", 400, 100, "math")
    lines = dict(l.split("=", 1) for l in mp.env_lines(math))
    assert lines["MODEL_PATH"] == "/models/olmo" and lines["N_TRAIN"] == "400" and lines["FRESH_K"] == "32"
    assert lines["OM_PROMPT_FORMAT"] == "olmo_rlzero_math" and lines["GRPO_EPOCHS_PER_BATCH"] == "1"
    assert lines["OM_LORA_TARGETS"] == "q_proj,v_proj" and lines["MAX_NEW_TOKENS"] == "2048"
    assert "DATASET" not in lines and "DRIFT" not in lines  # set by the wrapper, not copied


def test_cli_and_script_syntax(tmp_path):
    math = _run(tmp_path, "math500", 400, 100, "math")
    other = _run(tmp_path, "mbpp", 400, 100, "code")
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    cmd = [sys.executable, str(ROOT / "src/mixed_pool.py")]
    result = subprocess.run(cmd + ["build", "--math-run", str(math), "--other-run", str(other), "--out", str(tmp_path / "pool.jsonl")],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0 and "train:mbpp" in result.stdout, result.stdout + result.stderr
    assert (tmp_path / "pool.jsonl.manifest.json").is_file()
    result = subprocess.run(cmd + ["env", "--run", str(math)], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and "BEHAVIOR_K=8" in result.stdout
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_mixed_pool.sh")], check=True)
    subprocess.run(["bash", "-n", str(ROOT / "scripts/_pin_checkout.sh")], check=True)
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_e5.sh")], check=True)


FAKE_POINT = """#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "runner=%s cwd=$PWD retry=${OM_RETRY_INDEX:-0}"
mkdir -p "$OUT_ROOT/logs"
n=$(( $(cat "$OUT_ROOT/attempts" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$OUT_ROOT/attempts"
if [ -n "${FAKE_PERMANENT:-}" ]; then
  echo "[$(date '+%%F %%T')] [config-abort] existing artifacts use a different run config: ['gen_batch']" >> "$OUT_ROOT/logs/main.log"
  exit 2
fi
if [ "$n" -lt 2 ]; then echo "[$(date '+%%F %%T')] [stage-fail] pid=1 rc=1 fake shard crash" >> "$OUT_ROOT/logs/main.log"; exit 1; fi
echo done > "$OUT_ROOT/DONE"
"""


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, env=env).strip()


def test_point_step_reenters_the_pinned_commit_and_retries_transient_failures(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src").mkdir()
    for name in ("run_mixed_pool.sh", "_lease.sh", "_e5_node.sh", "_pin_checkout.sh"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    for name in ("mixed_pool.py", "cleanup_run_processes.py"):
        shutil.copy2(ROOT / "src" / name, repo / "src" / name)
    (repo / "scripts/setup_env.sh").write_text('export OM_WORK="$TEST_WORK" VENV_DIR="$TEST_VENV"\n')
    (repo / "scripts/run_point.sh").write_text(FAKE_POINT % "A")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "A")
    commit_a = _git(repo, "rev-parse", "HEAD")
    (repo / "scripts/run_point.sh").write_text(FAKE_POINT % "B")  # this checkout moved on
    _git(repo, "commit", "-q", "-am", "B")
    work = tmp_path / "work"
    root = work / "runs" / "tag"
    _run(root, "math500", 8, 4, "math")
    shutil.copytree(root / "family-math500-s0" / "tag-s0-math500-d0", root / "family-math500-s1" / "tag-s1-math500-d0")
    pool = work / "inputs" / "mixed"
    pool.mkdir(parents=True)
    for seed in (0, 1):
        (pool / f"pool-math500-mbpp-s{seed}.jsonl").write_text("{}\n")
        point = root / f"family-math500mix-s{seed}" / f"tag-s{seed}-math500mix-d0"
        point.mkdir(parents=True)
        (point / "run_config.json").write_text(json.dumps({"git": commit_a}))  # initialized under A
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp_path), "TEST_WORK": str(work),
           "TEST_VENV": str(Path(sys.executable).parent.parent), "OM_OLMO3_MODEL_TAG": "tag",
           "OM_LOCAL_LOCK_DIR": str(tmp_path / "locks"), "OM_PIPELINE_CACHE": str(tmp_path / "cache"),
           "MIX_RETRY_SLEEP": "0", "MIX_POINT_ATTEMPTS": "3"}
    result = subprocess.run(["bash", "scripts/run_mixed_pool.sh", "point"], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=120, check=False)
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    point = root / "family-math500mix-s0" / "tag-s0-math500mix-d0"
    assert "runner=A" in out and "runner=B" not in out, out
    assert f"cwd={tmp_path / 'cache' / 'clones' / commit_a}" in out, out
    assert "re-entering the partial point under its pinned commit" in out
    assert "point attempt 1 failed rc=1: [" in out and "[stage-fail] pid=1 rc=1 fake shard crash" in out
    assert (point / "attempts").read_text().strip() == "2" and (point / "DONE").is_file()
    assert "point complete" in out
    # a contract failure is not retried
    result = subprocess.run(["bash", "scripts/run_mixed_pool.sh", "point"], cwd=repo,
                            env={**env, "MIX_SEED": "1", "FAKE_PERMANENT": "1"}, capture_output=True, text=True,
                            timeout=120, check=False)
    out = result.stdout + result.stderr
    assert result.returncode == 2, out
    point = root / "family-math500mix-s1" / "tag-s1-math500mix-d0"
    assert (point / "attempts").read_text().strip() == "1" and not (point / "DONE").exists()
    assert "contract/config failure; not retrying" in out


def test_a_point_declaring_a_prescreened_pool_is_moved_aside_not_reused(tmp_path, monkeypatch):
    """Its run config is immutable and its qualification stage can only abort, so the runner keeps
    it under a superseded name and builds the point again."""
    runner = (ROOT / "scripts/run_mixed_pool.sh").read_text()
    body = runner.split("  point)", 1)[1].split("  e5)", 1)[0]
    assert body.index("flock -n 6") < body.index("superseded"), "quarantine must happen under the lease"
    assert 'mv -- "$POINT" "$superseded"' in body and "rm " not in body
    point = tmp_path / "point"
    point.mkdir()
    (point / "run_config.json").write_text(json.dumps({"pool": "/x/pool.jsonl", "dataset": "math500"}))
    check = [sys.executable, "-c", 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pool") else 1)',
             str(point / "run_config.json")]
    assert subprocess.run(check).returncode == 0          # declared pool -> quarantine
    (point / "run_config.json").write_text(json.dumps({"pool": None, "dataset": "math500"}))
    assert subprocess.run(check).returncode == 1          # no declared pool -> keep and resume


def test_a_stale_lease_is_replaced_only_after_the_point_stops_being_written(tmp_path):
    """The holder's node may be gone; flock cannot then be released, so the lease file is replaced
    once nothing under the point has been written for the idle window."""
    body = (ROOT / "scripts/run_mixed_pool.sh").read_text().split("  point)", 1)[1].split("  e5)", 1)[0]
    assert 'rm -f -- "$POINT.lease"' in body and "MIX_STALE_LEASE_SECONDS:-2700" in body
    assert body.index("the holder is still writing") < body.index('rm -f -- "$POINT.lease"')
    # replacing the file leaves the old holder's lock behind and lets a new lock be taken
    lease = tmp_path / "point.lease"
    lease.write_text("host=gone pid=1 since=x\n")
    holder = os.open(lease, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        second = os.open(lease, os.O_RDWR)
        with pytest.raises(BlockingIOError):
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(second)
        lease.unlink()
        lease.write_text("host=new pid=2 since=y\n")
        fresh = os.open(lease, os.O_RDWR)
        fcntl.flock(fresh, fcntl.LOCK_EX | fcntl.LOCK_NB)   # no longer blocked by the old holder
        os.close(fresh)
    finally:
        os.close(holder)
