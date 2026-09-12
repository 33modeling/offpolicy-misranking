"""CPU contracts for the mixed candidate pool (positive control)."""

from __future__ import annotations

import json
import os
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
    with pytest.raises(ValueError, match="not enough"):
        mp.build(math, other, tmp_path / "p2.jsonl", 500, 200, 100, seed=0)


def test_loader_honours_the_pre_split_pool(tmp_path, monkeypatch):
    math = _run(tmp_path, "math500", 30, 10, "math")
    other = _run(tmp_path, "mbpp", 30, 10, "code")
    out = tmp_path / "pool.jsonl"
    mp.build(math, other, out, 20, 20, 10, seed=1)
    monkeypatch.setenv("OM_POOL_FILE", str(out))
    prompts = data.load_prompts("math500", 40, 10, seed=0)
    assert len(prompts["train"]) == 40 and len(prompts["val"]) == 10
    assert all(q["question"].startswith("math val") for q in prompts["val"])
    assert set(prompts["train"][0]) == {"question", "answer"}
    with pytest.raises(ValueError, match="train 40 < 50"):
        data.load_prompts("math500", 50, 10, seed=0)


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
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_e5.sh")], check=True)
