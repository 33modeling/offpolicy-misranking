"""CPU-only isolation, objective, input-integrity and completion contracts."""

import json
import os
from pathlib import Path
import subprocess

import pytest

import evidence_downstream as ed
import rloo_experiment as rloo
from test_evidence_downstream import source_point


def inputs(tmp_path, *, drift=0):
    run, evaluation = source_point(tmp_path, drift=drift)
    data = ed.read(evaluation)
    data["test"] = [{"question": f"held out {i}", "answer": "2"} for i in range(300)]
    ed.atomic_json(evaluation, data)
    return run, evaluation


def fixture(tmp_path):
    run, evaluation = inputs(tmp_path)
    out = tmp_path / "rloo"
    rloo.prepare(run, out, evaluation)
    return run, out, evaluation


def test_prepare_freezes_rloo_without_changing_source(tmp_path):
    run, evaluation = inputs(tmp_path)
    before = {str(p): ed.digest(p) for p in run.rglob("*") if p.is_file()}
    out = tmp_path / "rloo"
    c = rloo.prepare(run, out, evaluation, dry=True)
    assert not out.exists()
    assert c["objective"] == "rloo" and c["steps"] == 100
    rloo.prepare(run, out, evaluation)
    assert rloo.prepare(run, out, evaluation) == c
    assert before == {str(p): ed.digest(p) for p in run.rglob("*") if p.is_file()}
    assert len(ed.read(out / "evaluation.json")["val"]) == 300


def test_native_rloo_commands_and_no_parent(tmp_path):
    _, out, _ = fixture(tmp_path)
    for arm in rloo.ARMS:
        cmd = rloo.training_command(out, arm)
        assert cmd[cmd.index("--objective") + 1] == "rloo"
        assert cmd[cmd.index("--epochs-per-batch") + 1] == "1"
        assert cmd[cmd.index("--target-steps") + 1] == "100"
        assert cmd[cmd.index("--start-step") + 1] == "0"
        assert "--resume-adapter" not in cmd and "--resume-optimizer" not in cmd


def test_d400_keeps_grpo_parent_optimizer_and_all_other_training_options(tmp_path):
    run, evaluation = inputs(tmp_path, drift=400)
    out = tmp_path / "rloo"
    rloo.prepare(run, out, evaluation)
    import sys
    expected = [sys.executable, *ed.train_args(ed.read(run / "run_config.json"), run, out, "random", 100)]
    expected[expected.index(str(ed.ROOT / "src/train_policy_grpo.py"))] = str(rloo.ROOT / "src/train_policy_rloo.py")
    ed._replace_flag(expected, "--objective", "rloo")
    assert rloo.training_command(out, "random") == expected
    assert rloo.policy(out, "before") == run / "policy_step_400"
    assert expected[expected.index("--resume-optimizer") + 1] == str(run / "policy_step_400/optimizer.pt")


@pytest.mark.parametrize("relative", ["evaluation.json", "subsets/subset-random.json"])
def test_prepared_data_tampering_rejected(tmp_path, relative):
    _, out, _ = fixture(tmp_path)
    (out / relative).write_text("{}")
    with pytest.raises(ValueError, match="prepared input changed"):
        rloo.validate(out)


def test_source_tampering_rejected(tmp_path):
    run, out, _ = fixture(tmp_path)
    (run / "scores_offpolicy.json").write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        rloo.validate(out)


def test_contract_change_rejected(tmp_path):
    run, out, evaluation = fixture(tmp_path)
    data = ed.read(evaluation)
    data["provenance"]["revision"] = "different"
    ed.atomic_json(evaluation, data)
    with pytest.raises(ValueError, match="contract changed"):
        rloo.prepare(run, out, evaluation)


def test_full_test_count_and_overlap_rejected(tmp_path):
    run, evaluation = source_point(tmp_path, drift=0)
    with pytest.raises(ValueError, match="300-question"):
        rloo.prepare(run, tmp_path / "rloo", evaluation)
    data = ed.read(evaluation)
    data["test"][0] = ed.read(run / "prompts.json")["train"][0]
    ed.atomic_json(evaluation, data)
    with pytest.raises(ValueError, match="overlaps"):
        rloo.prepare(run, tmp_path / "rloo", evaluation)


def test_outputs_cannot_contain_or_be_inside_inputs(tmp_path):
    for out, inp in [(tmp_path, tmp_path / "source"), (tmp_path / "source/new", tmp_path / "source")]:
        with pytest.raises(ValueError, match="separate"):
            rloo.disjoint(out, [inp])


def test_policy_validation_requires_rloo_not_grpo(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path)
    import train_policy_rloo as trainer
    seen = {}

    def check(path, **kwargs):
        seen.update(kwargs)
        raise ValueError("GRPO publication is not RLOO")

    monkeypatch.setattr(trainer, "validate_policy_lineage", check)
    with pytest.raises(ValueError, match="not RLOO"):
        rloo.policy(out, "random")
    assert seen["training_objective"] == "rloo"
    assert seen["expected_parent"] is None and seen["require_complete_hashes"]


def test_completion_requires_sealed_full_evaluation(tmp_path):
    _, out, _ = fixture(tmp_path)
    assert not rloo.complete(out, "before")
    target = out / "before/evaluation"
    target.mkdir(parents=True)
    for shard in range(4):
        b, _, indices = rloo.binding(out, "before", shard)
        path = target / f"shard-{shard}.jsonl"
        path.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": 1}) + "\n"
                                for i in indices for j in range(8)))
        ed.atomic_json(target / f"shard-{shard}.done.json", {"binding": b, "rollouts_sha256": ed.digest(path)})
    assert rloo.complete(out, "before")
    (target / "shard-0.jsonl").write_text("")
    with pytest.raises(ValueError, match="seal mismatch"):
        rloo.complete(out, "before")


def test_worker_meter_uses_visible_strings_and_numeric_gpu_count(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path)
    import selection_gate_gpu as gpu
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("OM_NODE_LOCK_HELD", "1")
    calls = []
    monkeypatch.setattr(gpu, "meter", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(rloo, "policy", lambda *a: None)
    monkeypatch.setattr(rloo, "complete", lambda *a: True)
    rloo.run_arm(out, "random", 60)
    assert [a[1] for a, _ in calls] == ["train", "evaluation"]
    for _, kw in calls:
        assert kw["devices"] == 4 and kw["timeout"] == 60
        assert all(isinstance(visible, str) for _, visible in kw["commands"])
        assert kw["env"]["OM_TOP_P"] == "1.0"


def test_timeout_is_not_completion(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path)
    import selection_gate_gpu as gpu
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("OM_NODE_LOCK_HELD", "1")

    def fail(*args, **kwargs):
        raise TimeoutError("phase timed out")

    monkeypatch.setattr(gpu, "meter", fail)
    with pytest.raises(TimeoutError):
        rloo.run_arm(out, "random", 1)
    assert not (out / "random/DONE").exists()


def test_runtime_is_source_bound():
    env = rloo.runtime_env({"prompt_format": "olmo_rlzero_math", "attn": "sdpa",
                            "gen_batch": 8, "lora_targets": "q_proj,v_proj", "thinking": "off"})
    assert env["OM_ATTN"] == "sdpa" and env["OM_LORA_TARGETS"] == "q_proj,v_proj"
    with pytest.raises(ValueError, match="top_p"):
        rloo.runtime_env({"top_p": 0.9})


def test_launcher_cpu_modes_no_gpu(tmp_path):
    import sys
    script = rloo.ROOT / "scripts/run_rloo.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    env = {**os.environ, "RLOO_PYTHON": sys.executable, "RLOO_ROOT": str(tmp_path / "absent")}
    result = subprocess.run(["bash", str(script), "plan"], env=env, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["training_runs"] == 18
    result = subprocess.run(["bash", str(script), "status"], env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count("not prepared") == 6
    assert not (tmp_path / "absent").exists()


def test_worker_lock_is_exclusive(tmp_path):
    with rloo.lock(tmp_path / "arm.lock"):
        with pytest.raises(BlockingIOError):
            with rloo.lock(tmp_path / "arm.lock"):
                pytest.fail("duplicate arm lease admitted")
    with rloo.lock(tmp_path / "arm.lock"):
        pass


def test_full_run_starts_all_points_without_smoke_or_timeout_option(tmp_path, monkeypatch):
    import sys
    _, out, _ = fixture(tmp_path)
    c, config = rloo.validate(out)
    monkeypatch.setattr(rloo, "validate", lambda out: (c, config))
    monkeypatch.setattr(rloo, "complete", lambda *args: False)
    calls = []
    monkeypatch.setattr(rloo, "run_arm", lambda *args: calls.append(args))
    monkeypatch.delenv("RLOO_MAX_PHASE_SECONDS", raising=False)
    monkeypatch.setattr(sys, "argv", ["rloo", "run", "--root", str(tmp_path)])
    rloo.main()
    assert len(calls) == 24
    assert sum(arm != "before" for _, arm, _ in calls) == 18
    assert all("smoke" not in str(out) and seconds == 86400 for out, _, seconds in calls)


def test_bare_launcher_automatically_prepares_before_gpu_admission(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/usr/bin/env python3\nimport sys\n"
                    "if 'ensure-prepared' in sys.argv:\n"
                    " print('automatic-preparation-reached'); sys.exit(73)\n"
                    "if sys.argv[1] == '-c': sys.exit(0)\n"
                    "sys.exit(99)\n")
    fake.chmod(0o755)
    env = {**os.environ, "RLOO_PYTHON": str(fake), "RLOO_ROOT": str(tmp_path / "out")}
    result = subprocess.run(["bash", str(rloo.ROOT / "scripts/run_rloo.sh")], env=env,
                            text=True, capture_output=True)
    assert result.returncode == 73
    assert "automatic-preparation-reached" in result.stdout


def test_handoff_only_changes_the_two_parent_objective_checks():
    import ast
    import inspect
    import train_policy_rloo as wrapper
    original = ast.parse(inspect.getsource(wrapper.ORIGINAL_TRAIN))
    adapted = wrapper.handoff_tree()
    names = [n for n in ast.walk(adapted) if isinstance(n, ast.Attribute) and n.attr == "parent_objective"]
    assert len(names) == 2
    for node in names:
        node.attr = "objective"
    assert ast.dump(adapted, include_attributes=False) == ast.dump(original, include_attributes=False)
    assert callable(wrapper.handoff_train())


def test_handoff_accepts_real_grpo_parent_without_relabeling_it():
    import ast
    from dataclasses import asdict
    from types import SimpleNamespace
    import train_policy_rloo as wrapper
    tree = wrapper.handoff_tree()
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "local_checkpoint is None and completed_steps")
    config = wrapper.canonical.GrpoConfig(epochs_per_batch=1)
    parent = {"training_objective": "grpo", "config": asdict(config)}
    calls = []

    def validate(path, **kwargs):
        calls.append(kwargs)
        return parent

    scope = {"local_checkpoint": None, "completed_steps": 400, "resume_adapter": "parent",
             "world_size": 4, "args": SimpleNamespace(objective="rloo", parent_objective="grpo"),
             "validate_policy_manifest": validate, "asdict": asdict, "config": config}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), "<test>", "exec"), scope)
    assert calls[0]["training_objective"] == "grpo"
    assert parent["training_objective"] == "grpo"
    assert scope["args"].objective == "rloo"


def test_handoff_entrypoint_checks_parent_hashes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import train_policy_rloo as wrapper
    checked, trained = [], []
    monkeypatch.setattr(wrapper.canonical, "validate_policy_manifest", lambda *a, **kw: checked.append((a, kw)))
    monkeypatch.setattr(wrapper, "handoff_train", lambda: lambda args: trained.append(args))
    args = SimpleNamespace(objective="rloo", start_step=400, resume_adapter=str(tmp_path), expected_world_size=4)
    wrapper.train(args)
    assert checked[0][1]["require_complete_hashes"]
    assert checked[0][1]["training_objective"] == "grpo"
    assert trained == [args] and args.parent_objective == "grpo"


def test_lineage_adapter_changes_only_parent_objective():
    import ast
    import inspect
    import train_policy_rloo as wrapper
    original = ast.parse(inspect.getsource(wrapper.canonical.validate_policy_lineage))
    adapted = wrapper.lineage_tree()
    calls = [n for n in ast.walk(adapted) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "validate_policy_manifest"
             and isinstance(n.args[0], ast.Name) and n.args[0].id == "parent"]
    changed = next(kw for kw in calls[0].keywords if kw.arg == "training_objective")
    assert isinstance(changed.value, ast.Constant) and changed.value.value == "grpo"
    changed.value = ast.Name(id="training_objective", ctx=ast.Load())
    assert ast.dump(adapted, include_attributes=False) == ast.dump(original, include_attributes=False)


def test_real_rloo_child_grpo_parent_lineage_and_optimizer_integrity(tmp_path):
    from test_grpo_policy import _policy_artifact
    import train_policy_rloo as wrapper
    parent, child = tmp_path / "parent", tmp_path / "child"
    _policy_artifact(parent, completed_steps=400)

    def seal(path):
        manifest = ed.read(path / "policy_train.json")
        manifest.update(optimizer_sha256=ed.digest(path / "optimizer.pt"),
                        grpo_stats_sha256=ed.digest(path / "grpo_stats.jsonl"))
        ed.atomic_json(path / "policy_train.json", manifest)

    seal(parent)
    _policy_artifact(child, objective="rloo", start_step=400, completed_steps=500, parent=parent)
    seal(child)
    kwargs = dict(target_steps=500, world_size=4, training_objective="rloo",
                  expected_start_step=400, expected_parent=parent, require_complete_hashes=True)
    with pytest.raises(ValueError, match="training_objective"):
        wrapper.canonical.validate_policy_lineage(child, **kwargs)
    result = wrapper.validate_policy_lineage(child, **kwargs)
    assert result["training_objective"] == "rloo"
    assert ed.read(parent / "policy_train.json")["training_objective"] == "grpo"
    (parent / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        wrapper.validate_policy_lineage(child, **kwargs)


def test_prepare_matrix_has_six_original_points_and_no_extra_experiment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    args = SimpleNamespace(work=tmp_path / "work", source_root=None, eval_prompts=None)
    root = tmp_path / "rloo"
    outs = [root / f"math500-d{drift}" / f"s{seed}" for drift, seed in rloo.POINTS]
    configs = {rloo.source_paths(args.work, None, seed, drift) / "run_config.json":
               {"seed": seed, "drift": drift, "model": "same-model"} for drift, seed in rloo.POINTS}
    calls = []
    monkeypatch.setattr(ed, "read", lambda path: configs[path])
    monkeypatch.setattr(rloo, "prepare", lambda *a, **kw: calls.append((a, kw)))
    rloo.prepare_matrix(args, root, outs)
    assert len(calls) == 12
    assert all(kw.get("dry") for _, kw in calls[:6])
    assert all(not kw for _, kw in calls[6:])
    for ((run, out, evaluation), _), (drift, seed) in zip(calls[6:], rloo.POINTS, strict=True):
        assert run == rloo.source_paths(args.work, None, seed, drift)
        assert out == root / f"math500-d{drift}" / f"s{seed}"
        assert evaluation == args.work / f"inputs/e5-reduced/test-math500-d{drift}.json"


def test_report_refuses_missing_baseline(tmp_path):
    _, out, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="incomplete arm: before"):
        rloo.report(out)
    assert not (out / "results.json").exists()


def test_report_has_direct_cached_comparison_and_no_h(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path)
    monkeypatch.setattr(rloo, "complete", lambda *args: True)

    def rows(out, arm, shard):
        return [{"prompt_idx": i, "rollout_idx": j,
                 "reward": {"before": 0, "random": 0.25, "passrate_beta": 0.5, "fresh_r": 0.75}[arm]}
                for i in range(shard * 75, (shard + 1) * 75) for j in range(8)]

    monkeypatch.setattr(rloo, "checked_rows", rows)
    result = rloo.report(out)
    fresh = next(r for r in result["rows"] if r["arm"] == "fresh_r")
    assert fresh["vs_passrate_beta"]["mean"] == 0.25
    assert fresh["vs_random"]["mean"] == 0.5
    assert "H" not in fresh and "total_cost" not in fresh
    assert "smoke" not in result


def test_rloo_chunk_gradients_match_full_sequence_mean():
    import torch
    from train_policy_grpo import rloo_group_advantages, rloo_loss
    advantages = rloo_group_advantages(torch.tensor([1., 0., 1., 0., 1.], dtype=torch.float64))
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    lengths = [1, 2, 3, 4, 7]
    full = rloo_loss([theta.expand(n) for n in lengths], advantages)
    expected = torch.autograd.grad(full, theta)[0]
    for start in (0, 2, 4):
        stop = min(start + 2, len(lengths))
        chunk = rloo_loss([theta.expand(n) for n in lengths[start:stop]], advantages[start:stop])
        (chunk * (stop - start) / len(lengths)).backward()
    assert torch.allclose(theta.grad, expected)
