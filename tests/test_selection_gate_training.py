"""Exercise integration with the existing trainer on CPU, without CUDA models."""

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import selection_gate as gate
import selection_gate_gpu as gpu
import train_selection_gate_grpo as driver

ROOT = Path(__file__).resolve().parents[1]


def source(tmp_path):
    from test_evidence_downstream import source_point
    run, evaluation = source_point(tmp_path)
    c = gate.read(run / "run_config.json")
    c.update(behavior_k=8, grpo_group_size=8)
    gate.atomic_json(run / "run_config.json", c)
    return run, evaluation


def test_prepare_random_input_contract_needs_no_scores_and_preserves_source(tmp_path):
    run, evaluation = source(tmp_path)
    for name in ("scores_offpolicy.json", "scores_oracle.json", "scores_splithalf.json", "rollouts_behavior_train.jsonl"):
        (run / name).unlink(missing_ok=True)
    before = {str(p): gpu.digest(p) for p in run.rglob("*") if p.is_file()}
    c = gpu.source_contract(run, gate.read(evaluation), budget=40000., gpu_type="H100",
                            role="development", selector="low_order", eval_k=8, max_steps=100000)
    assert c["n"] == 40
    assert before == {str(p): gpu.digest(p) for p in run.rglob("*") if p.is_file()}


def test_train_command_uses_new_driver_and_keeps_parent_for_every_arm(tmp_path, monkeypatch):
    run, evaluation = source(tmp_path)
    c = gpu.source_contract(run, gate.read(evaluation), budget=40000., gpu_type="H100",
                            role="development", selector="low_order", eval_k=8, max_steps=100000)
    monkeypatch.setenv("E5_RELIABILITY_LOG", "1")
    for arm in gpu.study.BRANCHES:
        args = gpu.train_command(tmp_path / "out", c, arm, 30000.)
        assert str(ROOT / "src/train_selection_gate_grpo.py") in args
        assert str(ROOT / "src/train_policy_grpo.py") not in args
        assert "--reliability-log" not in args
        assert args[args.index("--resume-adapter")+1] == str(run / "policy_step_400")
        assert args[args.index("--resume-optimizer")+1] == str(run / "policy_step_400/optimizer.pt")
        assert args[args.index("--prompts")+1].endswith(f"subset-{arm}.json")


def test_new_driver_expired_budget_stops_before_sampling(tmp_path, monkeypatch):
    import peft
    from torch.nn import parallel

    prompts = tmp_path / "prompts.json"
    gate.atomic_json(prompts, {"train": [{"question": "1+1?", "answer": "2"}]})
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(use_cache=True)
    model.enable_input_require_grads = lambda: None
    monkeypatch.setattr(driver, "_distributed_setup", lambda *a: (0, 0, 1))
    monkeypatch.setattr(driver, "_checkpoint_contract", lambda *a: {})
    monkeypatch.setattr(driver, "_latest_checkpoint", lambda *a: (None, 0))
    monkeypatch.setattr(driver, "load_model", lambda *a, **k: (model, object()))
    monkeypatch.setattr(driver, "_lora_targets", lambda: ["weight"])
    monkeypatch.setattr(peft, "get_peft_model", lambda model, config: model)
    monkeypatch.setattr(parallel, "DistributedDataParallel", lambda model, **k: model)
    monkeypatch.setattr(driver, "_sample_group", lambda *a, **k: pytest.fail("sampled after the deadline"))
    tensor = torch.tensor
    monkeypatch.setattr(driver.torch, "tensor", lambda *a, **k: tensor(*a, **{**k, "device": "cpu"}))
    args = argparse.Namespace(output=str(tmp_path / "out"), model=str(tmp_path / "model"), prompts=str(prompts),
                              wall_budget_deadline=0., budget_save_reserve=30., expected_world_size=1,
                              objective="grpo", group_size=8, clip_epsilon=.2, learning_rate=1e-5,
                              epochs_per_batch=1, max_grad_norm=1., advantage_epsilon=1e-4,
                              lora_rank=16, lora_alpha=32, checkpoint_every=5, logprob_micro_batch=1,
                              target_steps=10000, start_step=0, resume_adapter=None, resume_optimizer=None,
                              max_new_tokens=64, seed=0, disable_gradient_checkpointing=True)
    driver.train(args)
    stop = gate.read(tmp_path / "out/budget_stop.json")
    assert stop["completed_steps"] == 0 and stop["use_parent_policy"]
    assert stop["stop_reason"] == "no_block_fits"
    assert not (tmp_path / "out/policy_train.json").exists()


def test_budget_driver_imports_and_help_on_cpu():
    result = subprocess.run([sys.executable, str(ROOT / "src/train_selection_gate_grpo.py"), "--help"],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert "--wall-budget-deadline" in result.stdout


def test_one_shot_scoring_uses_existing_cpu_math_and_reuses_frozen_selection(tmp_path, monkeypatch):
    from test_low_order_backend import TinyLoRA
    from test_low_order_experiment import fixture

    import low_order_experiment as low

    run, evaluation, _ = fixture(tmp_path, monkeypatch)
    toy = TinyLoRA()
    optimizer = torch.optim.AdamW(toy.parameters(), lr=1e-5, weight_decay=0.)
    for parameter in toy.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.save(optimizer.state_dict(), run / "policy_step_100/optimizer.pt")
    parent = gate.read(run / "policy_step_100/policy_train.json")
    parent["optimizer_sha256"] = gpu.digest(run / "policy_step_100/optimizer.pt")
    gate.atomic_json(run / "policy_step_100/policy_train.json", parent)
    out = tmp_path / "gate-point"
    gate.atomic_json(out / "inputs/test.json", gate.read(evaluation))
    cfg = gate.read(run / "run_config.json")
    c = {"source_run": str(run), "config": cfg, "scope": {"gpu_type": "H100", "selector": "low_order"}, "eval_k": 8}
    monkeypatch.setattr(low.backend, "load_current", lambda *a: (TinyLoRA(), None))
    real_meter = gpu.meter
    def cpu_meter(directory, name, gpu_type, **kwargs):
        if kwargs.get("commands"):
            point = next(low.entries(out / "scoring"))
            for shard in range(4):
                (low.validation_worker if name == "validation" else low.score_worker)(point, shard)
            kwargs.pop("commands")
            kwargs.pop("env", None)
            kwargs.pop("timeout", None)
            kwargs["action"] = lambda: None
        return real_meter(directory, name, gpu_type, **kwargs)
    monkeypatch.setattr(gpu, "meter", cpu_meter)
    selected = gpu.select_once(out, c, out / "selection_reduced", 100000., {}, ["0", "1", "2", "3"])
    assert len(selected) == 4
    paid = gpu.spent(out / "selection_reduced")
    monkeypatch.setattr(low, "prepare", lambda *a, **k: pytest.fail("selection repeated"))
    assert gpu.select_once(out, c, out / "selection_reduced", 100000., {}, []) == selected
    assert gpu.spent(out / "selection_reduced") == paid


def test_gpu_results_export_valid_three_branch_training_labels(tmp_path):
    run, evaluation = source(tmp_path)
    root, out = tmp_path / "suite", tmp_path / "suite/points/pilot"
    c = gpu.source_contract(run, gate.read(evaluation), budget=40000., gpu_type="H100",
                            role="development", selector="low_order", eval_k=8, max_steps=100000)
    gate.atomic_json(out / "contract.json", c)
    gate.atomic_json(out / "evaluation.json", c["evaluation"])
    gate.atomic_json(root / "suite.json", {"schema": gpu.SCHEMA, "mode": "study",
                                          "points": [{"name": "pilot", "sha256": gpu.digest(out / "contract.json")}]})
    gate.atomic_json(out / "initial.json", {"gpu_seconds": 2., "payload": {
        "features": {f: .5 for f in gate.FEATURES}, "feature_step": 400, "full_pool_coverage": True}})
    for arm in gpu.study.BRANCHES:
        gpu.freeze_subset(out, c, arm)
        gate.atomic_json(out / arm / "policy/budget_stop.json", {"completed_steps": 400,
                         "stop_reason": "no_block_fits", "use_parent_policy": True})
        for shard in range(4):
            _, indices, binding = gpu.eval_contract(out, c, arm, shard)
            path = out / arm / "evaluation" / f"shard-{shard}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [{"prompt_idx": i, "rollout_idx": j, "reward": j%2} for i in indices for j in range(8)]
            import json
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            gate.atomic_json(path.with_suffix(".done.json"), {"binding": binding, "sha256": gpu.digest(path)})
        gate.atomic_json(out / arm / "result.json", {"rewards": gpu.rewards(out, c, arm), "used_gpu_seconds": 0.,
                         "budget_gpu_seconds": 40000. if arm == "random_full" else 39998.,
                         "complete": True, "stop_reason": "no_block_fits"})
    result = gpu.summarize(root)
    assert len(result["points"]) == 1 and not result["excluded"]
    assert gpu.study.validate_study(result)[0]["net_selection_gain"] == 0.
    value = gate.read(out / "selection_reduced/result.json")
    value["used_gpu_seconds"] = 40001.
    gate.atomic_json(out / "selection_reduced/result.json", value)
    assert not gpu.summarize(root)["points"], "inconsistent or over-budget results must not become gate labels"
