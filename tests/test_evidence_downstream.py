"""CPU contracts for the reduced independent-test E5 driver; no CUDA."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_additional_experiments import _write_run
from test_grpo_policy import _policy_artifact

import evidence_downstream as ed

ROOT = Path(__file__).resolve().parents[1]


def source_point(tmp_path, drift=400):
    run = _write_run(tmp_path / "source", n=40, drift=drift)
    config = ed.read(run / "run_config.json")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    config.update(model=str(model), topk_frac=0.1, grpo_world_size=4, grpo_epochs_per_batch=1,
                  grpo_group_size=8, grpo_clip_epsilon=0.2, grpo_learning_rate=1e-5,
                  grpo_max_grad_norm=1, grpo_advantage_epsilon=1e-4,
                  grpo_lora_rank=16, grpo_lora_alpha=32, grpo_logprob_micro_batch=1,
                  grpo_gradient_checkpointing=1, max_new_tokens=64, temperature=1,
                  prompt_format="olmo_rlzero_math")
    ed.atomic_json(run / "run_config.json", config)
    (run / "DONE").write_text("complete")
    parent = run / f"policy_step_{drift}"
    _policy_artifact(parent, completed_steps=drift)
    manifest = ed.read(parent / "policy_train.json")
    manifest.update(seed=0, prompt_format="olmo_rlzero_math",
                    optimizer_sha256=ed.digest(parent / "optimizer.pt"),
                    grpo_stats_sha256=ed.digest(parent / "grpo_stats.jsonl"))
    ed.atomic_json(parent / "policy_train.json", manifest)
    inp = tmp_path / "eval-input"
    inp.mkdir()
    evaluation = inp / "test.json"
    ed.atomic_json(evaluation, {"test": [{"question": f"new question {i}", "answer": "2"} for i in range(8)],
                               "provenance": {"dataset": "independent-math", "revision": "frozen", "split": "train"}})
    return run, evaluation


def test_independent_test_rejects_reuse_even_with_changed_answer_and_spacing():
    source = {"train": [{"question": "What is  1+1?", "answer": "2"}], "val": [{"question": "v", "answer": "1"}]}
    evaluation = {"test": [{"question": "What is 1+1?", "answer": "3"}],
                  "provenance": {"dataset": "d", "revision": "r", "split": "s"}}
    with pytest.raises(ValueError, match="overlaps"):
        ed.independent_test(source, evaluation)


def test_prepare_dry_run_uses_the_source_drift_and_creates_nothing(tmp_path):
    run, evaluation = source_point(tmp_path, drift=400)
    out = tmp_path / "output"
    contract = ed.prepare(run, out, evaluation, 100, 8, dry=True)
    assert not out.exists()
    assert contract["drift"] == 400
    assert contract["selectors"] == ["random", "fresh_r", "g11"]
    assert contract["power_verified"] is False


def test_prepare_is_idempotent_and_targets_drift_plus_steps(tmp_path):
    run, evaluation = source_point(tmp_path, drift=400)
    original = {str(p): ed.digest(p) for p in run.rglob("*") if p.is_file()}
    out = tmp_path / "output"
    first = ed.prepare(run, out, evaluation, 100, 8)
    assert ed.prepare(run, out, evaluation, 100, 8) == first
    assert original == {str(p): ed.digest(p) for p in run.rglob("*") if p.is_file()}
    args = (out / "subsets/train-g11.args").read_bytes().decode().rstrip("\0").split("\0")
    assert args[args.index("--target-steps") + 1] == "500"
    assert args[args.index("--start-step") + 1] == "400"
    assert args[args.index("--resume-optimizer") + 1] == str(run / "policy_step_400/optimizer.pt")
    with pytest.raises(ValueError, match="contract changed"):
        ed.prepare(run, out, evaluation, 200, 8)


def test_prepare_rejects_zero_drift_and_unknown_arms(tmp_path):
    run, evaluation = source_point(tmp_path, drift=0)
    with pytest.raises(ValueError, match="positive drift"):
        ed.prepare(run, tmp_path / "o", evaluation, 100, 8, dry=True)
    run, evaluation = source_point(tmp_path / "b", drift=100)
    with pytest.raises(ValueError, match="unknown selector"):
        ed.prepare(run, tmp_path / "o2", evaluation, 100, 8, ["fresh_r", "certagrad"], dry=True)


def test_output_cannot_write_into_source(tmp_path):
    run, evaluation = source_point(tmp_path)
    with pytest.raises(ValueError, match="refusing"):
        ed.prepare(run, run / "e5", evaluation, 100, 8, dry=True)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nan", "extra"])
def test_evaluation_requires_exact_finite_prompt_response_coverage(tmp_path, bad):
    path = tmp_path / "shard.jsonl"
    rows = [{"prompt_idx": i, "rollout_idx": j, "reward": 1.0} for i in range(2) for j in range(2)]
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows.append(rows[0])
    elif bad == "nan":
        rows[0]["reward"] = float("nan")
    elif bad == "extra":
        rows.append({"prompt_idx": 5, "rollout_idx": 0, "reward": 1.0})
    path.write_text("".join(json.dumps(r, allow_nan=True) + "\n" for r in rows))
    with pytest.raises(ValueError):
        ed.reward_rows(path, range(2), 2)
    good = [{"prompt_idx": i, "rollout_idx": j, "reward": 0.5} for i in range(2) for j in range(2)]
    path.write_text("".join(json.dumps(r) + "\n" for r in good))
    assert len(ed.reward_rows(path, range(2), 2)) == 4


def test_paired_prompt_interval_keeps_pairs():
    values = np.array([0.1, 0.1, 0.1, 0.1])
    assert ed.paired_interval(values, 0, reps=512) == (0.1, 0.1)


def test_test_preparation_excludes_the_whole_source_pool_and_is_frozen(tmp_path):
    run, _ = source_point(tmp_path)
    source = ed.read(run / "prompts.json")
    used = [r["question"] for r in source["train"] + source["val"]]
    pool = tmp_path / "pool.jsonl"
    rows = [{"problem": q, "answer": "1"} for q in used] + [{"problem": f"held-out {i}", "answer": "1"} for i in range(6)]
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "inputs/test.json"
    result = ed.prepare_test(pool, [run], out, 4, 1, "d", "r", "train")
    assert len(result["test"]) == 4
    assert not {r["question"] for r in result["test"]} & set(used)
    assert result["provenance"]["eligible_questions"] == 6
    again = ed.prepare_test(pool, [run], out, 4, 1, "d", "r", "train")
    assert again == result
    with pytest.raises(ValueError, match="contract changed"):
        ed.prepare_test(pool, [run], out, 5, 1, "d", "r", "train")


def test_scripts_parse():
    for name in ("scripts/run_downstream_independent.sh", "scripts/run_e5.sh", "scripts/fetch_math_train.sh"):
        subprocess.run(["bash", "-n", str(ROOT / name)], check=True)


def test_status_reports_progress_without_evaluation_files(tmp_path, capsys):
    import downstream_status as status

    run, evaluation = source_point(tmp_path)
    out = tmp_path / "output"
    ed.prepare(run, out, evaluation, 100, 8)
    assert ed.arm_state(out, "before") == "not evaluated"
    assert ed.arm_state(out, "g11") == "not started"
    (out / "logs").mkdir()
    (out / "logs" / "eval-before-0.log").write_text("loading\nrollout 3/2 (x)\n")
    (out / "before" / "evaluation").mkdir(parents=True)
    (out / "before" / "evaluation" / "shard-1.jsonl.partial").write_text("{}\n{}\n")
    assert status.arm_state(out, "before").startswith("evaluating (0/4")
    status.print_status(out)
    text = capsys.readouterr().out
    assert "shard 1: 2/16 responses" in text and "rollout 3/2" in text


def test_prepare_tolerates_driver_code_changes_but_not_design_changes(tmp_path):
    run, evaluation = source_point(tmp_path)
    out = tmp_path / "output"
    first = ed.prepare(run, out, evaluation, 100, 8)
    saved = ed.read(out / "experiment.json")
    saved["code_hashes"]["src/evidence_downstream.py"] = "0" * 64
    saved["runtime"]["packages"]["torch"] = "0.0"
    ed.atomic_json(out / "experiment.json", saved)
    again = ed.prepare(run, out, evaluation, 100, 8)
    assert again["code_hashes"]["src/evidence_downstream.py"] == "0" * 64
    assert ed.read(out / "experiment.json") == saved
    with pytest.raises(ValueError, match="contract changed"):
        ed.prepare(run, out, evaluation, 100, 8, ["random", "fresh_r"])
    assert first["selectors"] == ["random", "fresh_r", "g11"]
