"""CPU contracts for the split-half reuse-estimator scoring (tiny Llama, no CUDA)."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest
import torch

import stale_splithalf as ss
from gate_decision import signal_halves

ROOT = Path(__file__).resolve().parents[1]


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(3)
    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
                         attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).cpu().float().eval()


def fake_point(tmp_path: Path, n: int = 5, k: int = 4, duplicate_halves: bool = False) -> Path:
    run = tmp_path / "point-d0"
    run.mkdir()
    rng = random.Random(1)
    with (run / "rollouts_behavior_train.jsonl").open("w") as handle:
        for i in range(n):
            rows = []
            for j in range(k):
                if duplicate_halves and j >= k // 2:
                    row = dict(rows[j - k // 2], rollout_idx=j)
                else:
                    ids = [rng.randrange(1, 16) for _ in range(10)]
                    row = {"prompt_idx": i, "rollout_idx": j, "input_ids": ids, "resp_start": 4, "resp_end": 10,
                           "reward": float(j % 2) if not duplicate_halves else float(j % 2 == 0)}
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
    torch.save(torch.randn(8, 64), run / "val_groups.pt")
    (run / "run_config.json").write_text(json.dumps({"model": "tiny", "drift": 0, "proj_dim": 64, "grad_layers": 1,
                                                     "clip_cap": 10.0, "micro_batch": 2, "seed": 0, "dataset": "math500"}))
    (run / "scores_oracle.json").write_text(json.dumps({str(i): {"score": 0.0, "norm": 1.0} for i in range(n)}))
    (run / "prompts.json").write_text(json.dumps({"train": [{"question": f"q{i}", "answer": "1"} for i in range(n)], "val": []}))
    return run


def test_halves_are_scored_merged_and_readable(tmp_path):
    run = fake_point(tmp_path, n=5)
    model = tiny_model()
    loader = lambda base, adapter: (model, None)  # noqa: E731
    logs = []
    ss.compute_shard(run, 0, 2, check_full=2, loader=loader, log=logs.append)
    ss.compute_shard(run, 1, 2, check_full=0, loader=loader, log=logs.append)
    part = json.loads((run / "scores_stale_splithalf.shard0.json").read_text())
    assert set(part["halves"]["g11"]) == {"0", "2", "4"} and set(part["full_check"]["g11"]) == {"0", "2"}
    # the stored full scores are what the same machinery computes on the whole group
    stored = {est: {i: {"score": v, "norm": 1.0} for i, v in part["full_check"][est].items()} for est in ss.ESTIMATORS}
    (run / "scores_offpolicy.json").write_text(json.dumps(stored))
    target = ss.merge(run, 2)
    data = json.loads(target.read_text())
    assert all(set(data[est]) == {str(i) for i in range(5)} for est in ss.ESTIMATORS)
    assert all(-1 <= v["a"] <= 1 and -1 <= v["b"] <= 1 for est in ss.ESTIMATORS for v in data[est].values())
    protocol = json.loads((run / "scores_stale_splithalf.protocol.json").read_text())
    assert protocol["full_score_check"]["g11"]["max_abs_difference"] == pytest.approx(0.0, abs=1e-9)
    halves = signal_halves(run, "g11")
    assert set(halves) == set(range(5))
    rel = ss.reliability(run)
    assert set(rel) == set(ss.ESTIMATORS)
    assert any("ETA" in line for line in logs)
    # a second run of a finished shard is skipped, not recomputed
    ss.compute_shard(run, 0, 2, loader=loader, log=logs.append)
    assert "skipped" in logs[-1]


def test_identical_halves_give_identical_scores(tmp_path):
    run = fake_point(tmp_path, n=3, duplicate_halves=True)
    model = tiny_model()
    ss.compute_shard(run, 0, 1, loader=lambda base, adapter: (model, None), log=lambda *_: None)
    ss.merge(run, 1)
    data = json.loads((run / "scores_stale_splithalf.json").read_text())
    for est in ss.ESTIMATORS:
        for value in data[est].values():
            assert value["a"] == pytest.approx(value["b"], abs=1e-5)


def test_merge_rejects_missing_or_overlapping_shards(tmp_path):
    run = fake_point(tmp_path, n=4)
    model = tiny_model()
    ss.compute_shard(run, 0, 2, loader=lambda base, adapter: (model, None), log=lambda *_: None)
    with pytest.raises(FileNotFoundError):
        ss.merge(run, 2)
    ss.compute_shard(run, 1, 2, loader=lambda base, adapter: (model, None), log=lambda *_: None)
    shard1 = run / "scores_stale_splithalf.shard1.json"
    payload = json.loads(shard1.read_text())
    payload["halves"]["g00"]["0"] = {"a": 0.0, "b": 0.0}  # overlaps shard 0
    shard1.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="two shards"):
        ss.merge(run, 2)


def test_script_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_stale_splithalf.sh")], check=True)
