"""Exhausted training can yield audited measurements, never invented budget DONE."""

import json
import shutil
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from test_mbpp_node_queue import queue_worker
from test_saved_policy_publication import saved_policy, bytes_under
from test_net_gain_gate_gpu import protocol
from test_waive_stalled_attempts import event

recovery = queue_worker.recovery
base, core, switch = recovery.base, recovery.core, recovery.switch


def fixture(tmp_path, monkeypatch, *, checkpoint=False, remaining=-9):
    out, c, policy = saved_policy(tmp_path, monkeypatch, remaining=remaining)
    c.update(eval_k=8, eval_seed=123, evaluation={"val": ["q0", "q1", "q2", "q3"]})
    c["config"]["temperature"] = 1.
    core.atomic_json(out / "contract.json", c)
    directory = policy.parent
    shutil.rmtree(directory / "evaluation")
    p = {"dataset": "mbpp", "gate": "final"}
    net = protocol()
    monkeypatch.setattr(switch, "verify", lambda _: c)
    monkeypatch.setattr(switch, "protocol", lambda _: net)
    core.atomic_json(out.parent.parent / "suite.json", {"eval_timeout": 5})
    core.atomic_json(directory / "decision.json", {
        "binding": {"protocol_sha256": core.fingerprint(net), "contract_sha256": base.digest(out / "contract.json")},
        "budget_gpu_seconds": c["budget_gpu_seconds"], "measurement_gpu_seconds": 0})
    core.atomic_json(directory / "execution.json", {"action": "random", "indices": None})
    core.atomic_json(directory / "execution.sha256.json", {"sha256": base.digest(directory / "execution.json")})
    if checkpoint:
        candidate = policy / "checkpoint-150"
        candidate.mkdir()
        for name in recovery.FILES:
            (policy / name).rename(candidate / name)
        manifest = core.read(policy / "policy_train.json")
        (policy / "policy_train.json").unlink()
        state = {**recovery.checkpoint_contract(out, c, "random_full"), "completed_steps": 150,
                 **{key: manifest[key] for key in ("adapter_sha256", "optimizer_sha256", "grpo_stats_sha256")}}
        core.atomic_json(candidate / "checkpoint_state.json", state)
        policy = candidate
    return p, directory, c, policy


def fake_evaluation(monkeypatch, directory):
    calls = []

    def collect(model, tokenizer, prompts, k, tokens, temperature, path, *, idx_offset, sampling_seed_base):
        rows = [{"prompt_idx": i + idx_offset, "rollout_idx": j, "reward": float(j % 2)}
                for i in range(len(prompts)) for j in range(k)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setitem(sys.modules, "rollout", SimpleNamespace(
        load_policy=lambda model, adapter: (None, None), collect_rollouts=collect))

    def meter(target, phase, gpu, **kwargs):
        assert target == directory / "budget-recovery"
        assert phase == "evaluate" and kwargs["ledger"] == "reporting"
        assert "action" not in kwargs
        for command, device in kwargs["commands"]:
            index = int(command[command.index("--point") + 1])
            shard = int(command[command.index("--shard") + 1])
            calls.append((index, shard))
            recovery.evaluate(directory, index, shard)
        for state in ("started", "finished"):
            row = event(f"eval-{len(calls)}", phase, state, seconds=3)
            row["ledger"] = "reporting"
            base.journal(target / "cost.jsonl", row)

    monkeypatch.setattr(base, "meter", meter)
    return calls


@pytest.mark.parametrize("checkpoint", [False, True])
def test_saved_work_evaluates_once_without_training_or_original_writes(tmp_path, monkeypatch, checkpoint):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, checkpoint=checkpoint)
    before = bytes_under(directory.parent)
    calls = fake_evaluation(monkeypatch, directory)
    assert recovery.required(p, directory)
    with base.lease(directory / ".task.lock"):
        result = recovery.recover(p, directory, list("0123"), {})
    assert calls == [(0, i) for i in range(4)]
    assert result["evaluation_complete"] and not result["canonical_complete"]
    assert result["over_budget_gpu_seconds"] == 9
    assert result["recovery_cost"]["ledgers"]["reporting"]["gpu_seconds"] == 12
    assert not (directory / "result.json").exists()
    assert not (directory / "policy/budget_stop.json").exists()
    assert all(path.read_bytes() == value for path, value in before.items())
    assert not switch.branch_finished(p, directory)
    after = bytes_under(directory.parent)
    recovery.recover(p, directory, list("0123"), {})
    assert bytes_under(directory.parent) == after and len(calls) == 4


@pytest.mark.parametrize("damage", ["adapter_model.safetensors", "optimizer.pt", "grpo_stats.jsonl", "checkpoint_state.json"])
def test_invalid_checkpoint_never_starts_evaluation_or_parent_restart(tmp_path, monkeypatch, damage):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, checkpoint=True)
    (policy / damage).write_text("corrupt")
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("invalid checkpoint reached GPU work"))
    before = bytes_under(directory.parent)
    with pytest.raises(ValueError, match="no valid saved"):
        recovery.recover(p, directory, list("0123"), {})
    assert bytes_under(directory.parent) == before


def test_valid_older_checkpoint_used_when_newest_is_corrupt(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, checkpoint=True)
    bad = policy.with_name("checkpoint-155")
    shutil.copytree(policy, bad)
    (bad / "optimizer.pt").write_text("bad")
    _, plan = recovery.prepare(p, directory)
    assert plan["completed_steps"] == 150
    assert plan["points"][-1]["adapter"] == str(policy)


def test_latest_checkpoint_is_pinned_before_any_gpu_work(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, checkpoint=True)
    _, original = recovery.prepare(p, directory)
    other = policy.with_name("checkpoint-155")
    shutil.copytree(policy, other)
    _, resumed = recovery.prepare(p, directory)
    assert resumed == original


def test_partial_evaluation_resumes_only_missing_shards(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, checkpoint=True)
    calls = fake_evaluation(monkeypatch, directory)
    recovery.prepare(p, directory)
    recovery.evaluate(directory, 0, 0)
    recovery.evaluate(directory, 0, 2)
    recovery.recover(p, directory, list("0123"), {})
    assert calls == [(0, 1), (0, 3)]


def test_recovery_result_publication_resumes_without_gpu_work(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch)
    calls = fake_evaluation(monkeypatch, directory)
    recovery.recover(p, directory, list("0123"), {})
    seal = directory / "budget-recovery/result.sha256.json"
    original = seal.read_bytes()
    seal.unlink()
    recovery.recover(p, directory, list("0123"), {})
    assert seal.read_bytes() == original and len(calls) == 4


def test_missing_published_shard_does_not_charge_again(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch)
    calls = fake_evaluation(monkeypatch, directory)
    recovery.recover(p, directory, list("0123"), {})
    (directory / "budget-recovery/point-0/shard-0.done.json").unlink()
    with pytest.raises(ValueError, match="lost a shard"):
        recovery.recover(p, directory, list("0123"), {})
    assert len(calls) == 4


def test_curve_uses_frozen_k_seed_and_saved_steps(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, checkpoint=True)
    p.update(gate="convergence", curve={"points": 3, "k": 4})
    monkeypatch.setattr(switch, "curve_config", lambda _: {"points": 3, "k": 4})
    archive = directory / "policy/curve-checkpoints/step-125"
    archive.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "checkpoint_state.json"):
        shutil.copyfile(policy / name, archive / name)
    state = core.read(archive / "checkpoint_state.json")
    state["completed_steps"] = 125
    core.atomic_json(archive / "checkpoint_state.json", state)
    calls = fake_evaluation(monkeypatch, directory)
    result = recovery.recover(p, directory, list("0123"), {})
    assert [item["step"] for item in result["points"]] == [100, 125, 150]
    assert len(calls) == 12
    plan = core.read(directory / "budget-recovery/plan.json")
    assert [item["seed"] for item in plan["points"]] == [c["eval_seed"] + 7919 * 101,
                                                          c["eval_seed"] + 7919 * 126, c["eval_seed"]]
    assert not (directory / "curve.json").exists()


@pytest.mark.parametrize("dataset,remaining,expected", [("math500", -9, False), ("mbpp", 1, False),
                                                       ("mbpp", 0, False), ("mbpp", -9, True)])
def test_only_exhausted_mbpp_is_intercepted(tmp_path, monkeypatch, dataset, remaining, expected):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, remaining=remaining)
    p["dataset"] = dataset
    assert recovery.required(p, directory) is expected


def test_queue_recovery_returns_review_not_canonical_done(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, checkpoint=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    import additive_experiment as ae
    monkeypatch.setattr(ae, "model_environment", lambda _: {})
    fake_evaluation(monkeypatch, directory)
    with base.lease(directory / ".task.lock"):
        assert queue_worker.finish_exhausted(p, directory, lambda *args: False)
    assert (directory / "budget-recovery/result.sha256.json").is_file()
    assert not switch.branch_finished(p, directory)


def test_queue_invalid_policy_is_reviewed_without_gpu_work(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, checkpoint=True)
    (policy / "optimizer.pt").unlink()
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("missing checkpoint reached GPU work"))
    assert queue_worker.finish_exhausted(p, directory, lambda *args: False)
    assert core.read(directory / "budget-recovery/review.json")["training_restarted"] is False


def test_existing_final_evaluation_is_reused_without_gpu_work(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch)
    calls = fake_evaluation(monkeypatch, directory)
    _, plan = recovery.prepare(p, directory)
    for shard in range(4):
        item, indices, target, _ = recovery.shard_info(directory, c, plan, 0, shard)
        source = directory / "evaluation"
        source.mkdir(exist_ok=True)
        path = source / f"shard-{shard}.jsonl"
        path.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": .5}) + "\n"
                                for i in indices for j in range(c["eval_k"])))
        binding = {"experiment_sha256": base.digest(directory.parent / "contract.json"),
                   "adapter_sha256": base.digest(policy / "adapter_model.safetensors"),
                   "policy_manifest_sha256": base.digest(policy / "policy_train.json"),
                   "arm": directory.name, "shard": shard}
        core.atomic_json(path.with_suffix(".done.json"), {"binding": binding, "sha256": base.digest(path)})
    result = recovery.recover(p, directory, list("0123"), {})
    assert not calls
    assert set(result["points"][-1]["rewards"].values()) == {.5}


@pytest.mark.parametrize("target", ["decision.json", "execution.json", "cost.jsonl",
                                  "policy/checkpoint-150/adapter_model.safetensors"])
def test_frozen_recovery_refuses_changed_input(tmp_path, monkeypatch, target):
    p, directory, c, policy = fixture(tmp_path, monkeypatch, checkpoint=True)
    recovery.prepare(p, directory)
    calls = fake_evaluation(monkeypatch, directory)
    path = directory / target
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, KeyError)):
        recovery.recover(p, directory, list("0123"), {})
    assert not calls


def test_unknown_reporting_cost_cannot_publish_or_retry(tmp_path, monkeypatch):
    p, directory, c, policy = fixture(tmp_path, monkeypatch)
    calls = fake_evaluation(monkeypatch, directory)
    row = event("interrupted-eval", "evaluate", "started", seconds=2)
    row["ledger"] = "reporting"
    base.journal(directory / "budget-recovery/cost.jsonl", row)
    with pytest.raises(ValueError, match="unclosed cost"):
        recovery.recover(p, directory, list("0123"), {})
    assert not calls
    assert not (directory / "budget-recovery/result.json").exists()
