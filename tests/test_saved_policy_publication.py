"""Recover a saved final policy without GPU work or replacement by its parent."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from fake_trainer import parse, write_policy
from test_net_gain_gate_gpu import protocol, source
from test_waive_stalled_attempts import event

import evidence_downstream as ed
import net_gain_gate_gpu as runtime
import selection_gate as core
import selection_gate_gpu as base


def saved_policy(tmp_path, monkeypatch, remaining=80):
    monkeypatch.setenv("OM_PROMPT_FORMAT", "olmo_rlzero_math")
    out, c = source(tmp_path)
    model = tmp_path / "model"
    prompts = Path(c["source_run"]) / "prompts.json"
    parent = Path(c["source_run"]) / "policy_step_100"
    args = parse(["--model", str(model), "--prompts", str(prompts), "--output", str(parent),
                  "--target-steps", "100"])
    write_policy(args)
    c["config"].update({field: getattr(args, flag.replace("-", "_")) for flag, field in ed.TRAIN_FLAGS.items()})
    c["config"].update(model=str(model), max_new_tokens=args.max_new_tokens, prompt_format="olmo_rlzero_math")
    core.atomic_json(out / "contract.json", c)
    subset = base.freeze_subset(out, c, "random_full")
    policy = out / "random_full/policy"
    child_args = parse(["--model", str(model), "--prompts", str(subset), "--output", str(policy),
                        "--target-steps", "150", "--start-step", "100", "--resume-adapter", str(parent),
                        "--resume-optimizer", str(parent / "optimizer.pt")])
    write_policy(child_args)
    manifest = core.read(policy / "policy_train.json")
    manifest["training_budget"] = {
        "requested_target_steps": 100 + c["max_steps"], "completed_steps": 150,
        "stop_reason": "budget_exhausted", "deadline_monotonic": 12345., "save_reserve_seconds": 30.,
    }
    core.atomic_json(policy / "policy_train.json", manifest)
    for index in range(4):
        core.atomic_json(out / f"random_full/evaluation/shard-{index}.done.json", {})
    directory = out / "random_full"
    for state in ("started", "finished"):
        base.journal(directory / "cost.jsonl", event("saved-train", "train", state,
                                                    seconds=(c["budget_gpu_seconds"] - remaining) / 4))
    monkeypatch.setattr(base, "verify", lambda _: c)
    monkeypatch.setattr(base, "rewards", lambda *a: {"q0": .5})
    return out, c, policy


def bytes_under(root):
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("remaining", [0, 80, 200])
def test_saved_final_is_recovered_without_metering_or_training(tmp_path, monkeypatch, remaining):
    out, c, policy = saved_policy(tmp_path, monkeypatch, remaining)
    before = bytes_under(out)
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("completed policy was metered or trained again"))
    runtime.run_arm(out, {"eval_timeout": 5}, protocol(), "random_full", list("0123"), {})
    stop = core.read(policy / "budget_stop.json")
    assert stop["completed_steps"] == 150 and stop["use_parent_policy"] is False
    original_trainer_bytes = (json.dumps({**core.read(policy / "policy_train.json")["training_budget"],
                                         "use_parent_policy": False}, indent=2, sort_keys=True) + "\n").encode()
    assert (policy / "budget_stop.json").read_bytes() == original_trainer_bytes
    result = runtime.validate_result(out, protocol(), "random_full")
    assert result["completed_steps"] == 150
    assert result["used_gpu_seconds"] == c["budget_gpu_seconds"] - remaining
    assert {p: p.read_bytes() for p in before} == before
    finished = bytes_under(out)
    runtime.run_arm(out, {}, protocol(), "random_full", [], {})
    assert bytes_under(out) == finished


def test_missing_stop_with_existing_sealed_result_restores_exact_original_bytes(tmp_path, monkeypatch):
    out, _, policy = saved_policy(tmp_path, monkeypatch)
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("recovery metered GPU work"))
    runtime.run_arm(out, {"eval_timeout": 5}, protocol(), "random_full", list("0123"), {})
    stop = policy / "budget_stop.json"
    original = stop.read_bytes()
    stop.unlink()
    before = bytes_under(out)
    runtime.run_arm(out, {}, protocol(), "random_full", [], {})
    assert stop.read_bytes() == original
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("remaining", [0, 80])
@pytest.mark.parametrize("existing_stop", [False, True])
def test_saved_policy_runs_only_pending_reporting_evaluation(tmp_path, monkeypatch, remaining, existing_stop):
    out, c, policy = saved_policy(tmp_path, monkeypatch, remaining)
    directory = out / "random_full"
    if existing_stop:
        assert runtime.restore_budget_stop(out, c, "random_full") is True
    for shard in (1, 3):
        (directory / f"evaluation/shard-{shard}.done.json").unlink()
    before = bytes_under(out)
    ledger = directory / "cost.jsonl"
    mtimes = {path: path.stat().st_mtime_ns for path in before if path != ledger}
    calls = []

    def reporting_only(target, phase, gpu_type, **kwargs):
        assert target == directory
        assert phase == "evaluate", "saved training must not be verified, selected, or trained on the deployment ledger again"
        assert kwargs["ledger"] == "reporting"
        assert "action" not in kwargs
        commands = kwargs["commands"]
        shards = [(int(command[command.index("--shard") + 1]), device) for command, device in commands]
        assert shards == [(1, "1"), (3, "3")]
        calls.append(phase)
        for state in ("started", "finished"):
            row = event("pending-evaluation", phase, state, seconds=3)
            row["ledger"] = "reporting"
            base.journal(ledger, row)
        for shard, _ in shards:
            core.atomic_json(directory / f"evaluation/shard-{shard}.done.json", {})

    monkeypatch.setattr(base, "meter", reporting_only)
    monkeypatch.setattr(runtime, "select_once", lambda *a, **k: pytest.fail("saved policy restarted selection"))
    runtime.run_arm(out, {"eval_timeout": 5}, protocol(), "random_full", list("0123"), {})

    result = runtime.validate_result(out, protocol(), "random_full")
    assert calls == ["evaluate"]
    assert result["completed_steps"] == 150
    assert result["used_gpu_seconds"] == c["budget_gpu_seconds"] - remaining
    assert result["cost"]["ledgers"]["reporting"]["gpu_seconds"] == 12
    assert base.spent(directory) == c["budget_gpu_seconds"] - remaining
    assert core.read(policy / "budget_stop.json")["use_parent_policy"] is False
    assert ledger.read_bytes().startswith(before[ledger])
    for path, previous in before.items():
        if path != ledger:
            assert path.read_bytes() == previous
            assert path.stat().st_mtime_ns == mtimes[path]
    finished = bytes_under(out)
    runtime.run_arm(out, {}, protocol(), "random_full", [], {})
    assert calls == ["evaluate"]
    assert bytes_under(out) == finished


@pytest.mark.parametrize("name", ["adapter_model.safetensors", "optimizer.pt", "grpo_stats.jsonl"])
def test_existing_stop_does_not_publish_or_meter_a_damaged_final_policy(tmp_path, monkeypatch, name):
    out, c, policy = saved_policy(tmp_path, monkeypatch, remaining=0)
    assert runtime.restore_budget_stop(out, c, "random_full") is True
    (policy / name).write_bytes(b"corrupt")
    before = bytes_under(out)
    mtimes = {path: path.stat().st_mtime_ns for path in before}
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("damaged saved policy reached metered work"))
    with pytest.raises(ValueError):
        runtime.run_arm(out, {"eval_timeout": 5}, protocol(), "random_full", list("0123"), {})
    assert not (out / "random_full/result.json").exists()
    assert not (out / "random_full/result.sha256.json").exists()
    assert {path: path.read_bytes() for path in before} == before
    assert {path: path.stat().st_mtime_ns for path in before} == mtimes


@pytest.mark.parametrize("remaining", [-1, -80])
def test_existing_stop_over_budget_never_becomes_a_completed_result(tmp_path, monkeypatch, remaining):
    out, c, _ = saved_policy(tmp_path, monkeypatch, remaining)
    assert runtime.restore_budget_stop(out, c, "random_full") is True
    before = bytes_under(out)
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("over-budget saved policy was charged again"))
    with pytest.raises(ValueError, match="exceeded|exhausted"):
        runtime.run_arm(out, {"eval_timeout": 5}, protocol(), "random_full", list("0123"), {})
    assert not (out / "random_full/result.json").exists()
    assert not (out / "random_full/result.sha256.json").exists()
    assert base.spent(out / "random_full") == c["budget_gpu_seconds"] - remaining
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("damage", ["adapter", "optimizer", "stats", "parent", "prompts",
                                   "budget-target", "budget-step", "budget-reason", "budget-missing"])
def test_unverified_final_policy_never_writes_parent_stop_or_starts_work(tmp_path, monkeypatch, damage):
    out, _, policy = saved_policy(tmp_path, monkeypatch)
    if damage in {"adapter", "optimizer", "stats"}:
        name = {"adapter": "adapter_model.safetensors", "optimizer": "optimizer.pt", "stats": "grpo_stats.jsonl"}[damage]
        (policy / name).write_bytes(b"corrupt")
    elif damage == "parent":
        parent = tmp_path / "source/policy_step_100/optimizer.pt"
        parent.write_bytes(b"changed parent")
    elif damage == "prompts":
        core.atomic_json(out / "subsets/subset-random_full.json", {"train": []})
    else:
        manifest = core.read(policy / "policy_train.json")
        if damage == "budget-target":
            manifest["training_budget"]["requested_target_steps"] += 1
        elif damage == "budget-step":
            manifest["training_budget"]["completed_steps"] = 151
        elif damage == "budget-reason":
            manifest["training_budget"]["stop_reason"] = "updates_completed"
        else:
            manifest.pop("training_budget")
        core.atomic_json(policy / "policy_train.json", manifest)
    before = bytes_under(out)
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("invalid policy reached metered work"))
    with pytest.raises(ValueError):
        runtime.run_arm(out, {}, protocol(), "random_full", [], {})
    assert bytes_under(out) == before
    assert not (policy / "budget_stop.json").exists()


def test_saved_result_disagreement_blocks_stop_repair(tmp_path, monkeypatch):
    out, _, policy = saved_policy(tmp_path, monkeypatch)
    core.atomic_json(out / "random_full/result.json", {"artifact_hashes": {"random_full/policy/budget_stop.json": "0" * 64}})
    before = bytes_under(out)
    with pytest.raises(ValueError, match="disagree"):
        runtime.restore_budget_stop(out, core.read(out / "contract.json"), "random_full")
    assert bytes_under(out) == before
    assert not (policy / "budget_stop.json").exists()


@pytest.mark.parametrize("result_exists", [False, True])
def test_existing_parent_only_stop_cannot_hide_saved_final_policy(tmp_path, monkeypatch, result_exists):
    out, _, policy = saved_policy(tmp_path, monkeypatch)
    core.atomic_json(policy / "budget_stop.json", {"use_parent_policy": True, "completed_steps": 100,
                                                  "stop_reason": "no_block_fits"})
    if result_exists:
        core.atomic_json(out / "random_full/result.json", {"complete": True})
    before = bytes_under(out)
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("conflicting policy reached metered work"))
    with pytest.raises(ValueError, match="conflicts"):
        runtime.run_arm(out, {}, protocol(), "random_full", [], {})
    assert bytes_under(out) == before


@pytest.mark.parametrize("evidence", ["checkpoint-000105/checkpoint_state.json", "grpo_stats.jsonl", "optimizer.pt"])
def test_existing_parent_only_stop_conflicting_with_partial_training_is_preserved_and_blocked(tmp_path, evidence):
    out, c = source(tmp_path)
    policy = out / "random_full/policy"
    core.atomic_json(policy / evidence, {"step": 105})
    core.atomic_json(policy / "budget_stop.json", {"use_parent_policy": True, "completed_steps": 100,
                                                  "stop_reason": "no_block_fits"})
    before = bytes_under(out)
    with pytest.raises(ValueError, match="conflicts"):
        runtime.restore_budget_stop(out, c, "random_full")
    assert bytes_under(out) == before


@pytest.mark.parametrize("evidence", ["checkpoint-000105/checkpoint_state.json", "grpo_stats.jsonl", "optimizer.pt"])
def test_small_remaining_budget_does_not_replace_incomplete_local_training_with_parent(tmp_path, monkeypatch, evidence):
    out, c = source(tmp_path)
    policy = out / "random_full/policy"
    core.atomic_json(policy / evidence, {"step": 105})
    monkeypatch.setattr(base, "verify", lambda _: c)
    monkeypatch.setattr(base, "spent", lambda _: c["budget_gpu_seconds"] - 80)
    monkeypatch.setattr(base, "meter", lambda directory, phase, gpu_type, **kw: kw["action"]())
    with pytest.raises(ValueError, match="parent-only"):
        runtime.run_arm(out, {}, protocol(), "random_full", [], {})
    assert not (policy / "budget_stop.json").exists()
    assert json.loads((policy / evidence).read_text()) == {"step": 105}
