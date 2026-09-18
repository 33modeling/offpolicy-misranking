"""CPU fault injection at the publication boundaries used by the MBPP queue."""
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import net_gain_gate_gpu as runtime
import selection_gate as core
import selection_gate_gpu as base
import selection_switch_gpu as switch
from test_net_gain_gate_gpu import completed, protocol, source


class NodeLoss(BaseException):
    pass


def cpu_branch(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    monkeypatch.setattr(base, "verify", lambda _: c)
    monkeypatch.setattr(base, "policy", lambda *a: Path(c["source_run"]))
    monkeypatch.setattr(base, "rewards", lambda *a: {"q0": .5})
    monkeypatch.setattr(runtime, "select_once", lambda *a: [0, 1, 2, 3])
    completed(out, "selection_full")
    return out, protocol()


def run(out, p):
    runtime.run_arm(out, {"eval_timeout": 5}, p, "selection_full", list("0123"), {})


def interrupt_publication(out, p, target, monkeypatch):
    bind = base.bind
    def interrupted(path, value):
        bind(path, value)
        if path == target:
            raise NodeLoss()
    with monkeypatch.context() as patch:
        patch.setattr(base, "bind", interrupted)
        with pytest.raises(NodeLoss):
            run(out, p)
    assert target.exists() and not target.with_suffix(".sha256.json").exists()


@pytest.mark.parametrize("artifact", ["selection_full/execution.json", "subsets/subset-selection_full.json",
                                     "selection_full/result.json"])
def test_payload_without_seal_resumes_and_preserves_payload(tmp_path, monkeypatch, artifact):
    out, p = cpu_branch(tmp_path, monkeypatch)
    target = out / artifact
    interrupt_publication(out, p, target, monkeypatch)
    before = target.read_bytes()
    paid = base.spent(out / "selection_full")
    assert not switch.branch_finished({}, out / "selection_full")
    run(out, p)
    assert target.read_bytes() == before
    assert core.read(target.with_suffix(".sha256.json")) == {"sha256": base.digest(target)}
    assert runtime.validate_result(out, p, "selection_full")["complete"]
    assert switch.branch_finished({}, out / "selection_full")
    if target.name == "result.json":
        assert base.spent(out / "selection_full") == paid
    paid = base.spent(out / "selection_full")
    run(out, p)
    assert base.spent(out / "selection_full") == paid


@pytest.mark.parametrize("artifact", ["selection_full/execution.json", "subsets/subset-selection_full.json",
                                     "selection_full/result.json"])
def test_unsealed_payload_is_not_blindly_trusted(tmp_path, monkeypatch, artifact):
    out, p = cpu_branch(tmp_path, monkeypatch)
    target = out / artifact
    interrupt_publication(out, p, target, monkeypatch)
    value = core.read(target)
    if target.name == "execution.json":
        value["indices"] = [1, 2, 3, 4]
    elif target.name == "result.json":
        value["rewards"] = {"q0": 1.}
    else:
        value["train"][0]["question"] = "changed"
    core.atomic_json(target, value)
    with pytest.raises(ValueError):
        run(out, p)
    assert not target.with_suffix(".sha256.json").exists()


def test_existing_bad_seal_is_never_replaced(tmp_path, monkeypatch):
    out, p = cpu_branch(tmp_path, monkeypatch)
    run(out, p)
    seal = out / "selection_full/result.sha256.json"
    core.atomic_json(seal, {"sha256": "changed"})
    assert not switch.branch_finished({}, seal.parent)
    with pytest.raises(ValueError, match="result hash changed"):
        run(out, p)
    assert core.read(seal) == {"sha256": "changed"}


@pytest.mark.parametrize("artifact", ["execution.json", "result.json"])
def test_actual_sigkill_between_payload_and_seal_resumes(tmp_path, monkeypatch, artifact):
    out, p = cpu_branch(tmp_path, monkeypatch)
    child = """
import os, signal, sys
from pathlib import Path
import net_gain_gate_gpu as runtime
import selection_gate as core
import selection_gate_gpu as base
from test_net_gain_gate_gpu import protocol
out, name = Path(sys.argv[1]), sys.argv[2]
c = core.read(out / 'contract.json')
base.verify = lambda _: c
base.policy = lambda *a: Path(c['source_run'])
base.rewards = lambda *a: {'q0': .5}
runtime.select_once = lambda *a: [0, 1, 2, 3]
bind = base.bind
def interrupted(path, value):
    bind(path, value)
    if path == out / 'selection_full' / name:
        os.kill(os.getpid(), signal.SIGKILL)
base.bind = interrupted
runtime.run_arm(out, {'eval_timeout': 5}, protocol(), 'selection_full', list('0123'), {})
"""
    result = subprocess.run([sys.executable, "-c", child, str(out), artifact], cwd=base.ROOT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=20)
    assert result.returncode == -signal.SIGKILL, result.stdout + result.stderr
    target = out / "selection_full" / artifact
    before = target.read_bytes()
    assert not target.with_suffix(".sha256.json").exists()
    run(out, p)
    assert target.read_bytes() == before
    assert runtime.validate_result(out, p, "selection_full")["complete"]


def published_policy(tmp_path, monkeypatch):
    import train_selection_gate_grpo as trainer
    from artifact_contract import sha256_file
    from test_grpo_policy import _policy_artifact

    policy = tmp_path / "policy"
    _policy_artifact(policy)
    prompts = tmp_path / "prompts.json"
    core.atomic_json(prompts, {"train": [{"question": "q", "answer": "a"}], "val": []})
    model = tmp_path / "model"
    model.mkdir()
    config = asdict(trainer.GrpoConfig())
    manifest = core.read(policy / "policy_train.json")
    manifest.update({
        "base_model": str(model.resolve()), "seed": 7, "max_new_tokens": 512,
        "prompt_format": trainer.prompt_format(), "config": config,
        "samples_per_step": 4 * config["group_size"], "prompts_sha256": sha256_file(prompts),
        "optimizer_sha256": sha256_file(policy / "optimizer.pt"),
        "grpo_stats_sha256": sha256_file(policy / "grpo_stats.jsonl"),
        "training_budget": {"requested_target_steps": 1000, "completed_steps": 25,
                            "stop_reason": "budget_exhausted", "deadline_monotonic": 1234., "save_reserve_seconds": 30.},
    })
    core.atomic_json(policy / "policy_train.json", manifest)
    args = SimpleNamespace(**config, output=str(policy), model=str(model), prompts=str(prompts),
                           target_steps=1000, start_step=0, resume_adapter=None, resume_optimizer=None,
                           expected_world_size=4, objective="grpo", seed=7, max_new_tokens=512,
                           logprob_micro_batch=1, wall_budget_deadline=time.monotonic() + 100,
                           budget_save_reserve=30.)
    monkeypatch.setattr(trainer, "_distributed_setup", lambda _: (0, 0, 4))
    monkeypatch.setattr(trainer.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(trainer, "load_model", lambda *a, **kw: pytest.fail("must not retrain a published policy"))
    return trainer, args, manifest


def test_completed_policy_restores_original_budget_stop_without_training(tmp_path, monkeypatch):
    trainer, args, manifest = published_policy(tmp_path, monkeypatch)
    policy = Path(args.output)
    before = {p: p.read_bytes() for p in policy.iterdir()}
    for _ in range(2):
        trainer.train(args)  # Real lineage/hash validation, mocked distributed setup only.
        assert core.read(policy / "budget_stop.json") == {**manifest["training_budget"], "use_parent_policy": False}
        assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("damage", ["optimizer", "budget_record", "stop"])
def test_policy_recovery_rejects_inconsistent_artifacts(tmp_path, monkeypatch, damage):
    trainer, args, manifest = published_policy(tmp_path, monkeypatch)
    policy = Path(args.output)
    if damage == "optimizer":
        (policy / "optimizer.pt").write_bytes(b"corrupt")
    elif damage == "budget_record":
        manifest["training_budget"]["completed_steps"] = 24
        core.atomic_json(policy / "policy_train.json", manifest)
    else:
        core.atomic_json(policy / "budget_stop.json", {"completed_steps": 24})
    with pytest.raises(ValueError):
        trainer.train(args)
    if damage != "stop":
        assert not (policy / "budget_stop.json").exists()


@pytest.mark.parametrize("selector", ["difficulty", "hard"])
def test_cached_selector_recovers_missing_seal_from_unchanged_cache(tmp_path, monkeypatch, selector):
    from test_selection_switch_gpu import reward_cache
    out, c = source(tmp_path)
    c["scope"]["selector"] = selector
    reward_cache(Path(c["source_run"]) / "rollouts_behavior_train.jsonl")
    path = out / "selection_full/cached-select/selection.json"
    bind = base.bind
    def interrupted(target, value):
        bind(target, value)
        if target == path:
            raise NodeLoss()
    with monkeypatch.context() as patch:
        patch.setattr(base, "bind", interrupted)
        with pytest.raises(NodeLoss):
            switch.cached_select_once(out, c, protocol(), "selection_full", {})
    before = path.read_bytes()
    indices = switch.cached_select_once(out, c, protocol(), "selection_full", {})
    assert indices == core.read(path)["indices"] and path.read_bytes() == before
    assert core.read(path.with_suffix(".sha256.json")) == {"sha256": base.digest(path)}
