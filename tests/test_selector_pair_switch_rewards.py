"""Checkpoint lineage, real reward provenance, queue ownership and resumption."""
import json
import shutil
from pathlib import Path

import pytest
import selector_pair_switch_rewards as switch
from fake_trainer import parse, write_policy

import evidence_downstream as ed
import selection_gate as core
import selection_gate_gpu as base
from train_policy_grpo import _latest_checkpoint, validate_policy_manifest


def checkpoint(policy, step, subset, parent, config):
    path = policy / f"checkpoint-{step:06d}"
    path.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt"):
        shutil.copy2(policy / name, path / name)
    rows = [json.loads(line) for line in (policy / "grpo_stats.jsonl").read_text().splitlines()]
    (path / "grpo_stats.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows if row["step"] <= step))
    manifest = core.read(policy / "policy_train.json")
    state = {"schema": "offpolicy-grpo-checkpoint/v2", "training_objective": "grpo",
             "base_model": config["model"], "seed": config["seed"], "world_size": 4,
             "start_step": manifest["start_step"], "target_steps": manifest["completed_steps"],
             "completed_steps": step, "prompts_sha256": base.digest(subset),
             "max_new_tokens": config["max_new_tokens"], "prompt_format": config["prompt_format"],
             "resume_adapter": str(parent.resolve()), "resume_optimizer": str((parent / "optimizer.pt").resolve()),
             "config": ed._expected_config(config),
             **{key: base.digest(path / name) for key, name in switch.FILES.items()}}
    core.atomic_json(path / "checkpoint_state.json", state)
    archive = policy / f"curve-checkpoints/step-{step}"
    archive.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "checkpoint_state.json"):
        shutil.copy2(path / name, archive / name)
    return path


@pytest.fixture
def study(tmp_path, monkeypatch):
    root, output = tmp_path / "pair", tmp_path / "switch"
    model, prompts = tmp_path / "model", root / "prompts.json"
    core.atomic_json(model / "config.json", {"model_type": "fixture"})
    core.atomic_json(prompts, {"train": [{"question": str(i), "answer": str(i)} for i in range(8)]})
    core.atomic_json(root / "pair.json", {"protocol_id": "p"})
    parent = root / "shared/policy_step_25"
    args = parse(["--model", str(model), "--prompts", str(prompts), "--output", str(parent),
                  "--target-steps", "25", "--seed", "3", "--epochs-per-batch", "1"])
    write_policy(args)
    config = {"model": str(model), "seed": 3, "drift": 25, "max_new_tokens": args.max_new_tokens,
              "prompt_format": "olmo_rlzero_math", "grpo_gradient_checkpointing": True, "temperature": 1.,
              **{field: getattr(args, flag.replace("-", "_")) for flag, field in ed.TRAIN_FLAGS.items()}}
    pools = {"on_policy": [0, 1], "cached": [2, 3], "random": [4, 5]}
    for arm in switch.ARMS[:-1]:
        directory = switch.source_arm(root, 3, arm)
        source = root / ("sr-parent-alias" if arm == "cached" else "on-parent-alias")
        source.mkdir(exist_ok=True)
        if not (source / "policy_step_25").exists():
            (source / "policy_step_25").symlink_to(parent, target_is_directory=True)
        contract = {"config": config, "source_run": str(source), "source_hashes": {"same": "prefix"},
                    "evaluation": {"val": [{"question": str(i), "answer": str(i)} for i in range(8)]},
                    "eval_k": 2, "eval_seed": 123, "scope": {"gpu_type": "fixture"}}
        core.atomic_json(directory.parent / "contract.json", contract)
        subset = directory.parent / "subsets" / f"subset-{directory.name}.json"
        core.atomic_json(subset, {"train": [core.read(prompts)["train"][i] for i in pools[arm]]})
        child = parse(["--model", str(model), "--prompts", str(subset), "--output", str(directory / "policy"),
                       "--target-steps", "150", "--start-step", "25", "--seed", "3", "--epochs-per-batch", "1",
                       "--resume-adapter", str(source / "policy_step_25"),
                       "--resume-optimizer", str(source / "policy_step_25/optimizer.pt")])
        write_policy(child)
        for step in (50, 75, 100, 125, 150):
            checkpoint(directory / "policy", step, subset, parent, config)
    core.atomic_json(root / "sr-gc/s3-t25/reference.json", {"parent": str(parent), "prompts": str(prompts)})
    initial = {"sets": pools}
    checks = [{"step": step, "d": d} for step, d in [(25, 1), (50, -1), (75, 4), (100, -2), (125, -3)]]
    monkeypatch.setattr(switch, "decision", lambda *_: (125, checks, initial))
    return root, output, config


def test_plan_uses_full_optimizer_checkpoint_and_latest_common_step(study):
    root, output, _ = study
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    plan = switch.ensure_plan(root, output, 3)
    assert plan["switch_step"] == 125 and plan["end_step"] == 150
    assert Path(plan["checkpoint"]).name == "checkpoint-000125"
    assert plan["training_estimate"]["median_hours"] == pytest.approx(25*70/3600)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert switch.ensure_plan(root, output, 3) == plan


def test_missing_optimizer_cannot_be_replaced_with_weights_only(study):
    root, _, _ = study
    full = switch.source_arm(root, 3, "on_policy") / "policy/checkpoint-000125"
    (full / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="optimizer.pt required"):
        switch.make_plan(root, 3)


def test_horizon_never_uses_rewards_or_arbitrary_100_cap():
    assert switch.common_horizon({"a": [100, 125, 315, 320], "b": [100, 125, 315],
                                   "c": [100, 125, 315, 325]}, 125) == 315
    with pytest.raises(ValueError, match="no saved common"):
        switch.common_horizon({"a": [100], "b": [125]}, 100)


def test_materialized_parent_keeps_optimizer_and_exact_lineage(study):
    root, output, _ = study
    plan = switch.ensure_plan(root, output, 3)
    parent = switch.materialize_parent(output / "s3", plan)
    manifest = validate_policy_manifest(parent, target_steps=125, world_size=4, require_complete_hashes=True)
    assert manifest["start_step"] == 25
    assert (parent / "optimizer.pt").read_bytes() == (Path(plan["checkpoint"]) / "optimizer.pt").read_bytes()
    assert switch.materialize_parent(output / "s3", plan) == parent
    command = switch.train_command(output / "s3", plan)
    assert "--nproc_per_node=4" in command
    assert command[command.index("--start-step")+1] == "125"
    assert command[command.index("--target-steps")+1] == "150"
    assert command[command.index("--prompts")+1] == plan["sr_subset"]
    assert "--wall-budget-deadline" not in command and "--target-reward" not in command
    assert command[command.index("--resume-optimizer")+1] == str(parent / "optimizer.pt")


def test_changed_frozen_source_rejected(study):
    root, output, _ = study
    plan = switch.ensure_plan(root, output, 3)
    Path(plan["sr_subset"]).write_text('{"train": []}')
    with pytest.raises(ValueError, match="frozen source changed"):
        switch.ensure_plan(root, output, 3)


def test_switch_prefix_uses_on_never_cached_control(study):
    root, output, _ = study
    plan = switch.ensure_plan(root, output, 3)
    assert switch.point_adapter(output / "s3", plan, "switch", 100) == switch.point_adapter(
        output / "s3", plan, "on_policy", 100)
    assert switch.point_adapter(output / "s3", plan, "switch", 150) is False
    requested = switch.tasks(output / "s3", plan, {})
    assert requested[0][1] == 150
    assert ("cached", 100) in requested
    assert ("switch", 125) not in requested and ("on_policy", 125) in requested


def test_partial_results_do_not_substitute_fixed_sr_reward(study, monkeypatch):
    root, output, _ = study
    plan = switch.ensure_plan(root, output, 3)
    monkeypatch.setattr(switch, "measured_point", lambda *args: None)
    row = switch.export_seed(output / "s3", plan, {"cached": [{"step": 150, "reward": .9}]})
    assert row["final_rewards"]["cached"] == .9
    assert row["final_rewards"]["switch"] is None
    assert row["executed_switch"] is False and row["complete"] is False


def test_precheckpoint_interruption_is_preserved_not_deleted(study):
    root, output, _ = study
    plan = switch.ensure_plan(root, output, 3)
    stats = output / "s3/policy/grpo_stats.jsonl"
    stats.parent.mkdir()
    raw = '{"step": 126}\n{"step":'
    stats.write_text(raw)
    switch.preserve_early_interruption(output / "s3", plan)
    saved = list((output / "s3/interrupted-attempts").glob("*/grpo_stats.jsonl"))
    assert len(saved) == 1 and saved[0].read_text() == raw
    assert not stats.parent.exists()


def test_completed_checkpoint_resume_uses_suffix_not_parent(study):
    root, output, config = study
    plan = switch.ensure_plan(root, output, 3)
    parent = switch.materialize_parent(output / "s3", plan)
    command = switch.train_command(output / "s3", plan)
    args = parse(command[command.index("--model"):])
    write_policy(args)
    cp = checkpoint(output / "s3/policy", 150, Path(plan["sr_subset"]), parent, config)
    state = core.read(cp / "checkpoint_state.json")
    contract = {key: value for key, value in state.items() if key not in (*switch.FILES, "completed_steps")}
    assert _latest_checkpoint(output / "s3/policy", 150, contract) == (cp, 150)
    assert switch.training_complete(output / "s3", plan)


def test_shared_task_lock_excludes_duplicate_node(tmp_path):
    lock = tmp_path / "switch/.train.lock"
    with base.lease(lock), pytest.raises(BlockingIOError), base.lease(lock):
        pytest.fail("duplicate training admitted")


def test_source_output_overlap_forbidden(tmp_path):
    for root, output in ((tmp_path, tmp_path), (tmp_path, tmp_path / "new"), (tmp_path / "old", tmp_path)):
        with pytest.raises(ValueError, match="separate"):
            switch.check_paths(root, output)


@pytest.mark.parametrize("values,expected", [([1, -1, 4, -2, -3, 9], 125), ([1, 1, -1, -2, 9], 100)])
def test_decision_stops_at_first_two_negatives_without_future_rewards(tmp_path, monkeypatch, values, expected):
    # Restore the real function; no study fixture patches this test.
    core.atomic_json(tmp_path / "pair.json", {})
    core.atomic_json(tmp_path / "sr-gc/s3-t25/decision.json", {})
    monkeypatch.setattr(switch, "initial_choice", lambda *a: {"d": values[0]})
    monkeypatch.setattr(switch.repeat, "inventory", lambda *a: {step: tmp_path for step in range(50, 176, 25)})
    monkeypatch.setattr(switch.repeat, "checkpoint_reference", lambda r, seed, start, step, *a: ({"step": step}, {}))
    seen = []
    def projections(directory, root, reference):
        step = reference["step"]
        seen.append(step)
        assert step <= expected, "future D must not tune the trigger"
        return {"d": values[step//25-1], "reference_sha256": "d"}
    monkeypatch.setattr(switch.repeat, "check_projections", projections)
    step, _, _ = switch.decision(tmp_path, 3)
    assert step == expected and seen[-1] == expected


def test_end_to_end_cpu_workflow_only_trains_suffix_and_reuses_shards(study, monkeypatch):
    root, output, config = study
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    import rollout
    import selector_pair_gpu
    monkeypatch.setattr(selector_pair_gpu, "environment", lambda _: {})
    monkeypatch.setattr(rollout, "load_policy", lambda model, adapter: (adapter, None))
    rollout_calls, train_calls = [], []
    def collect(model, tokenizer, prompts, k, max_tokens, temperature, path, *, idx_offset, sampling_seed_base):
        rollout_calls.append(str(path))
        path.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j,
                                         "reward": .75 if path.parent.parent.name == "switch" else .25})+"\n"
                                for i in range(idx_offset, idx_offset+len(prompts)) for j in range(k)))
    monkeypatch.setattr(rollout, "collect_rollouts", collect)
    def attempt(directory, name, plan, commands, env, seconds, lock_fd):
        assert seconds > 0
        if name == "train":
            train_calls.append(name)
            command = commands[0][0]
            assert str(root) not in command[command.index("--output")+1]
            args = parse(command[command.index("--model"):])
            write_policy(args)
            checkpoint(directory / "policy", 150, Path(plan["sr_subset"]), directory / "parent", config)
        else:
            for command, device in commands:
                switch.evaluate_shard(directory, command[command.index("--arm")+1],
                                      int(command[command.index("--step")+1]),
                                      int(command[command.index("--shard")+1]))
    monkeypatch.setattr(switch, "attempt", attempt)
    monkeypatch.setattr(switch, "plot_report", lambda *a: None)
    switch.worker(root, output, [3], list("0123"), 1, 1)
    count = len(rollout_calls)
    assert count > 0 and train_calls == ["train"]
    switch.worker(root, output, [3], list("0123"), 1, 1)
    assert len(rollout_calls) == count and train_calls == ["train"]
    data = switch.report(output, [3])
    assert data["complete"] and data["seeds"][0]["executed_switch"]
    assert data["seeds"][0]["switch_minus_on_policy"] == .5
    assert all(data["seeds"][0]["final_rewards"][arm] is not None for arm in switch.ARMS)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    shard = output / "s3/evaluations/switch/step-150/shard-1.jsonl"
    shard.write_text(shard.read_text().replace('0.75', '0.5'))
    with pytest.raises(ValueError, match="hash mismatch"):
        switch.report(output, [3])


def test_gpu_child_keeps_task_lock_after_controller_handle_closes(tmp_path):
    import subprocess
    import sys
    lock = tmp_path / "task.lock"
    process = None
    try:
        with switch.task_lease(lock) as fd, switch.inherited_task_lock(fd):
            process = base.subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                                            stdin=subprocess.PIPE)
        with pytest.raises(BlockingIOError), switch.task_lease(lock):
            pytest.fail("child lost ownership when controller released its handle")
        process.communicate(timeout=5)
        with switch.task_lease(lock):
            pass
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_extra_node_evaluates_instead_of_waiting_for_training(study, monkeypatch):
    root, output, _ = study
    import selector_pair_gpu
    monkeypatch.setattr(selector_pair_gpu, "environment", lambda _: {})
    claimed = []
    class ClaimedEvaluation(Exception):
        pass
    def attempt(directory, name, plan, commands, env, seconds, lock_fd):
        claimed.append(name)
        raise ClaimedEvaluation
    monkeypatch.setattr(switch, "attempt", attempt)
    with switch.task_lease(output / "s3/.train.lock"), pytest.raises(ClaimedEvaluation):
        switch.worker(root, output, [3], list("0123"), 1, 1)
    assert len(claimed) == 1 and claimed[0].startswith("eval-") and claimed[0].endswith("-150")


def test_initial_d_does_not_repair_or_require_old_cost_ledger(tmp_path, monkeypatch):
    directory = tmp_path / "sr-gc/s3-t25"
    parent = tmp_path / "source/policy_step_25"
    core.atomic_json(parent / "adapter_model.safetensors", {"fixture": True})
    core.atomic_json(parent.parent / "prompts.json", {"train": []})
    core.atomic_json(parent.parent / "rollouts_behavior_train.jsonl", {"fixture": True})
    core.atomic_json(tmp_path / "pair.json", {"protocol_id": "p"})
    sets = {"on_policy": [0], "cached": [1]}
    reference = {"state_id": "s3-t25", "sets": sets, "parent": str(parent),
                 "adapter_sha256": base.digest(parent / "adapter_model.safetensors"),
                 "prompts": str(parent.parent / "prompts.json"),
                 "prompts_sha256": base.digest(parent.parent / "prompts.json")}
    core.atomic_json(directory / "reference.json", reference)
    shards = {}
    for stage in switch.srgc.score.STAGES:
        for shard in range(4):
            path = directory / f"{stage}-{shard}.json"
            core.atomic_json(path, {})
            shards[path.name] = base.digest(path)
    contrast = {"d": 1., "d_a": 1., "d_b": 1., "selector": "on_policy"}
    initial = {**contrast, "seed": 3, "step": 25, "protocol_id": "p", "state_id": "s3-t25", "sets": sets,
               "reference_sha256": base.digest(directory / "reference.json"), "reference_shards": shards,
               "cache_sha256": base.digest(parent.parent / "rollouts_behavior_train.jsonl")}
    core.atomic_json(directory / "decision.json", initial)
    monkeypatch.setattr(switch.srgc, "reference_contrast", lambda *a: contrast)
    monkeypatch.setattr(base, "spent", lambda *a: pytest.fail("do not repair or depend on old costs for D"))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert switch.initial_choice(tmp_path, 3) == initial
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_plot_has_four_trajectories_and_real_switch_colors(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.axes
    calls = []
    original = matplotlib.axes.Axes.plot
    def plot(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", plot)
    rows = []
    for seed, trigger in ((3, 125), (4, 100)):
        rows.append({"seed": seed, "switch_step": trigger, "end_step": 150,
                     "curves": {arm: [{"step": step, "reward": .25+step/1000} for step in (0, 25, trigger, 150)]
                                for arm in switch.ARMS}, "final_rewards": dict.fromkeys(switch.ARMS, .4)})
    switch.plot_report(tmp_path, {"complete": True, "seeds": rows})
    assert (tmp_path / "switch-rewards.pdf").stat().st_size > 5000
    assert (tmp_path / "switch-rewards.png").stat().st_size > 10000
    assert len(calls) == 10  # three controls and two connected switch segments per seed
    assert sum(call["color"] == "#1565b0" for call in calls) == 2
    assert sum(call["color"] == "#17843c" for call in calls) == 2
    assert all(call["lw"] <= 1 for call in calls)
