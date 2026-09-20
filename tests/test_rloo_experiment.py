"""CPU-only isolation, objective, input-integrity and completion contracts."""

import json
import os
from pathlib import Path
import subprocess

import pytest

import evidence_downstream as ed
import rloo_experiment as rloo
from test_evidence_downstream import source_point


def fixture(tmp_path, *, smoke=False):
    run, evaluation = source_point(tmp_path, drift=0)
    data = ed.read(evaluation)
    data["test"] = [{"question": f"held out {i}", "answer": "2"} for i in range(300)]
    ed.atomic_json(evaluation, data)
    out = tmp_path / "rloo"
    rloo.prepare(run, out, evaluation, smoke=smoke)
    return run, out, evaluation


def test_prepare_freezes_rloo_without_changing_source(tmp_path):
    run, evaluation = source_point(tmp_path, drift=0)
    before = {str(p): ed.digest(p) for p in run.rglob("*") if p.is_file()}
    out = tmp_path / "rloo"
    c = rloo.prepare(run, out, evaluation, smoke=True, dry=True)
    assert not out.exists()
    assert c["objective"] == "rloo" and c["steps"] == 2
    rloo.prepare(run, out, evaluation, smoke=True)
    assert rloo.prepare(run, out, evaluation, smoke=True) == c
    assert before == {str(p): ed.digest(p) for p in run.rglob("*") if p.is_file()}
    assert len(ed.read(out / "evaluation.json")["val"]) == 4


def test_native_rloo_commands_and_no_parent(tmp_path):
    _, out, _ = fixture(tmp_path)
    for arm in rloo.ARMS:
        cmd = rloo.training_command(out, arm)
        assert cmd[cmd.index("--objective") + 1] == "rloo"
        assert cmd[cmd.index("--epochs-per-batch") + 1] == "1"
        assert cmd[cmd.index("--target-steps") + 1] == "100"
        assert cmd[cmd.index("--start-step") + 1] == "0"
        assert "--resume-adapter" not in cmd and "--resume-optimizer" not in cmd


def test_d400_rejected_before_writing(tmp_path):
    run, evaluation = source_point(tmp_path, drift=400)
    out = tmp_path / "rloo"
    with pytest.raises(ValueError, match="d0 only"):
        rloo.prepare(run, out, evaluation)
    assert not out.exists()


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
    with pytest.raises(ValueError, match="contract changed"):
        rloo.prepare(run, out, evaluation, smoke=True)


def test_full_test_count_and_overlap_rejected(tmp_path):
    run, evaluation = source_point(tmp_path, drift=0)
    with pytest.raises(ValueError, match="300-question"):
        rloo.prepare(run, tmp_path / "rloo", evaluation)
    data = ed.read(evaluation)
    data["test"][0] = ed.read(run / "prompts.json")["train"][0]
    ed.atomic_json(evaluation, data)
    with pytest.raises(ValueError, match="overlaps"):
        rloo.prepare(run, tmp_path / "rloo", evaluation, smoke=True)


def test_outputs_cannot_contain_or_be_inside_inputs(tmp_path):
    for out, inp in [(tmp_path, tmp_path / "source"), (tmp_path / "source/new", tmp_path / "source")]:
        with pytest.raises(ValueError, match="separate"):
            rloo.disjoint(out, [inp])


def test_policy_validation_requires_rloo_not_grpo(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path)
    import train_policy_grpo as trainer
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
    _, out, _ = fixture(tmp_path, smoke=True)
    assert not rloo.complete(out, "before")
    target = out / "before/evaluation"
    target.mkdir(parents=True)
    for shard in range(4):
        b, _, indices = rloo.binding(out, "before", shard)
        path = target / f"shard-{shard}.jsonl"
        path.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": 1}) + "\n"
                                for i in indices for j in range(2)))
        ed.atomic_json(target / f"shard-{shard}.done.json", {"binding": b, "rollouts_sha256": ed.digest(path)})
    assert rloo.complete(out, "before")
    (target / "shard-0.jsonl").write_text("")
    with pytest.raises(ValueError, match="seal mismatch"):
        rloo.complete(out, "before")


def test_worker_meter_uses_visible_strings_and_numeric_gpu_count(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path, smoke=True)
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
    _, out, _ = fixture(tmp_path, smoke=True)
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


def test_launcher_cpu_modes_no_gpu_and_default_status(tmp_path):
    import sys
    script = rloo.ROOT / "scripts/run_rloo.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    env = {**os.environ, "RLOO_PYTHON": sys.executable, "RLOO_ROOT": str(tmp_path / "absent")}
    result = subprocess.run(["bash", str(script), "plan"], env=env, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["training_runs"] == 9
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count("not prepared") == 4
    assert not (tmp_path / "absent").exists()


def test_worker_lock_is_exclusive(tmp_path):
    with rloo.lock(tmp_path / "arm.lock"):
        with pytest.raises(BlockingIOError):
            with rloo.lock(tmp_path / "arm.lock"):
                pytest.fail("duplicate arm lease admitted")
    with rloo.lock(tmp_path / "arm.lock"):
        pass


def test_full_run_refuses_missing_smoke_before_meter(tmp_path, monkeypatch):
    import sys
    _, out, _ = fixture(tmp_path)
    c, config = rloo.validate(out)
    monkeypatch.setattr(rloo, "validate", lambda out: (c, config))
    monkeypatch.setattr(rloo, "complete", lambda *args: False)
    monkeypatch.setattr(rloo, "run_arm", lambda *args: pytest.fail("GPU work started"))
    monkeypatch.setattr(sys, "argv", ["rloo", "run", "--root", str(tmp_path), "--max-phase-seconds", "60"])
    with pytest.raises(ValueError, match="smoke must pass"):
        rloo.main()


def test_report_refuses_missing_baseline(tmp_path):
    _, out, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="incomplete arm: before"):
        rloo.report(out)
    assert not (out / "results.json").exists()


def test_report_has_direct_cached_comparison_and_no_h(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path, smoke=True)
    monkeypatch.setattr(rloo, "complete", lambda *args: True)

    def rows(out, arm, shard):
        return [{"prompt_idx": shard, "rollout_idx": j,
                 "reward": {"before": 0, "random": 0.25, "passrate_beta": 0.5, "fresh_r": 0.75}[arm]}
                for j in range(2)]

    monkeypatch.setattr(rloo, "checked_rows", rows)
    result = rloo.report(out)
    fresh = next(r for r in result["rows"] if r["arm"] == "fresh_r")
    assert fresh["vs_passrate_beta"]["mean"] == 0.25
    assert fresh["vs_random"]["mean"] == 0.5
    assert "H" not in fresh and "total_cost" not in fresh
    assert result["smoke"]


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
