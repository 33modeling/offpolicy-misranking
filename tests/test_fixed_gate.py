"""Fixed-checkpoint gate contracts and real scorer checks on CPU toy models."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import evidence_downstream as ed
import fixed_gate as fg
import fixed_gate_worker as worker
from test_evidence_downstream import source_point
from test_gate_arm import _complete_evaluation, _train

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["0", "1", "2", "3"]
ENV = {"FIXED_GATE_GPU_TYPE": "CPU test allocation"}


def prepared(tmp_path, drift=400):
    run, evaluation = source_point(tmp_path, drift=drift)
    config = ed.read(run / "run_config.json")
    config["behavior_k"] = 8
    ed.atomic_json(run / "run_config.json", config)
    torch.save(torch.randn(100, 64), run / "val_groups.pt")
    ed.atomic_json(run / "score_protocol.json", {"test_fixture": True})
    e5 = tmp_path / "e5"
    ed.prepare(run, e5, evaluation, 100, 8, ["random", "g11"])
    rule = ed.read(ROOT / "config/fixed_gate_rule.json")
    rule["pilot_size"] = 8
    path = tmp_path / "rule.json"
    ed.atomic_json(path, rule)
    out = tmp_path / "gate"
    fg.prepare(run, out, e5, path)
    return run, e5, out, path


def fake_phase(out, name, cap, devices, env, *, calls, reliable=True):
    calls.append(name)
    c = ed.read(out / "contract.json")
    if name == "assess":
        rows, parts = fg.read_phase(out, "pilot")
        result = fg.assess(c, rows, fg.projected_remaining_cost(c, parts))
        result["record_sha256"] = fg.core.fingerprint(result)
        fg.bind(out / "assessment.json", result)
        return
    values = ed.read(Path(c["run"]) / "scores_offpolicy.json")["g11"]
    for shard in range(4):
        rows = []
        for i in fg.expected_ids(c, name, shard):
            score = values[str(i)]["score"]
            row = {"prompt_idx": i, "primary": score}
            if name == "pilot":
                row["replica"] = score if reliable else -score
            rows.append(row)
        fg.bind(out / name / f"shard-{shard}.json", {
            "contract_sha256": ed.digest(out / "contract.json"), "phase": name, "shard": shard,
            "rows": rows, "timing": {"model_load_seconds": 1., "primary_seconds": max(1, len(rows))},
            "training_updates": 0})


@pytest.mark.parametrize("drift", [0, 400])
def test_source_and_optimizer_unchanged_and_contract_frozen(tmp_path, drift):
    run, e5, out, rule = prepared(tmp_path, drift)
    before = {str(p): ed.digest(p) for p in run.rglob("*") if p.is_file()}
    c = fg.prepare(run, out, e5, rule)
    assert c["training_during_measurement"] is False and len(c["pilot_ids"]) == 8
    assert fg.verify(out) == c
    assert before == {str(p): ed.digest(p) for p in run.rglob("*") if p.is_file()}
    changed = ed.read(rule)
    changed["r_min"] = .4
    ed.atomic_json(rule, changed)
    with pytest.raises(ValueError, match="contract changed"):
        fg.prepare(run, out, e5, rule)


@pytest.mark.parametrize("reliable", [True, False])
def test_one_decision_scores_remaining_only_when_retained(tmp_path, monkeypatch, reliable):
    run, e5, out, _ = prepared(tmp_path)
    calls = []
    monkeypatch.setattr(fg, "phase", lambda *a: fake_phase(*a, calls=calls, reliable=reliable))
    decision = fg.select(out, DEVICES, ENV)
    assert decision["action"] == ("g11" if reliable else "random")
    assert ("remaining" in calls) == reliable
    assert calls.count("assess") == 1
    assert decision["policy_updated_in_pilot"] is False
    assert fg.select(out, DEVICES, ENV) == decision
    assert calls.count("pilot") == 1
    subset = ed.read(e5 / "subsets" / f"subset-{decision['action']}.json")
    assert decision["selected_idx"] == subset["selected_idx"]


def test_deadline_failure_falls_back_without_retry(tmp_path, monkeypatch):
    _, _, out, _ = prepared(tmp_path)
    calls = []
    def timeout(*args):
        calls.append(args[1])
        raise TimeoutError("deadline")
    monkeypatch.setattr(fg, "phase", timeout)
    assert fg.select(out, DEVICES, ENV)["action"] == "random"
    assert fg.select(out, DEVICES, ENV)["reason"] == "diagnostic_failed"
    assert calls == ["pilot"]


def test_interrupted_diagnostic_does_not_resample(tmp_path, monkeypatch):
    _, _, out, _ = prepared(tmp_path)
    fg.bind(out / "pilot-attempt.json", {"started_at": 1})
    monkeypatch.setattr(fg, "phase", lambda *args: pytest.fail("must not repeat pilot"))
    assert fg.select(out, DEVICES, ENV)["reason"] == "interrupted_diagnostic"


def test_remaining_timeout_is_not_a_training_failure(tmp_path, monkeypatch):
    _, _, out, _ = prepared(tmp_path)
    calls = []
    def phases(*args):
        if args[1] == "remaining":
            raise TimeoutError("scoring budget")
        fake_phase(*args, calls=calls)
    monkeypatch.setattr(fg, "phase", phases)
    decision = fg.select(out, DEVICES, ENV)
    assert decision["action"] == "random" and decision["reason"] == "remaining_scoring_failed"


def test_exact_sample_ids_and_unknown_cost_fail_closed(tmp_path):
    _, _, out, _ = prepared(tmp_path)
    c = ed.read(out / "contract.json")
    rows = {i: {"primary": i / 40., "replica": i / 40.} for i in c["pilot_ids"]}
    assert fg.assess(c, rows, None)["reason"] == "unknown_cost"
    assert fg.assess(c, rows, 1e12)["reason"] == "over_budget"
    rows.pop(next(iter(rows)))
    with pytest.raises(ValueError, match="exactly"):
        fg.assess(c, rows, 0)


def test_original_outcomes_are_reused_and_tampering_rejected(tmp_path, monkeypatch):
    _, e5, out, _ = prepared(tmp_path)
    for arm in ("random", "g11"):
        _train(e5 / "subsets" / f"train-{arm}.args", rho=.7)
    for arm, reward in (("before", .2), ("random", .6), ("g11", .8)):
        _complete_evaluation(e5, arm, reward)
    monkeypatch.setattr(fg, "phase", lambda *a: fake_phase(*a, calls=[], reliable=False))
    fg.select(out, DEVICES, ENV)
    result = fg.rewards(out)
    assert result["complete"] and result["forgone_reward"] > 0
    monkeypatch.setattr(fg.runtime, "meter", lambda *a, **k: pytest.fail("completed outcomes must not retrain"))
    fg.complete_missing(out, DEVICES, ENV)
    decision = ed.read(out / "decision.json")
    decision["action"] = "g11"
    ed.atomic_json(out / "decision.json", decision)
    with pytest.raises(ValueError, match="publication changed"):
        fg.select(out, DEVICES, ENV)


def test_mutated_shard_and_source_are_not_reused(tmp_path, monkeypatch):
    run, _, out, _ = prepared(tmp_path)
    fake_phase(out, "pilot", 10, DEVICES, ENV, calls=[])
    shard = out / "pilot/shard-0.json"
    value = ed.read(shard)
    value["rows"].append(value["rows"][0])
    ed.atomic_json(shard, value)
    with pytest.raises(ValueError, match="duplicate"):
        fg.read_phase(out, "pilot")
    (run / "val_groups.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        fg.verify(out)


def test_paired_baseline_cost_is_separate_from_deployment(tmp_path):
    out = tmp_path / "timing"
    command = [sys.executable, "-c", "pass"]
    for name, ledger in (("pilot", "deployment"), ("baseline", "research")):
        fg.runtime.meter(out, name, "CPU fake allocation", commands=[(command, "")], env={}, timeout=10, ledger=ledger)
    cost = fg.allocation(out)
    assert cost["complete"] and set(cost["phases"]) == {"pilot", "baseline"}
    assert cost["ledgers"]["deployment"]["gpu_seconds"] == cost["phases"]["pilot"]["gpu_seconds"]
    assert cost["ledgers"]["research"]["gpu_seconds"] == cost["phases"]["baseline"]["gpu_seconds"]


def test_real_g11_worker_matches_whole_group_scorer_on_cpu(tmp_path, monkeypatch):
    import artifact_contract
    from experiment import read_rollouts, split_validation_directions
    from stale_splithalf import beta_logprobs, group_score
    from test_stale_splithalf import fake_point, tiny_model
    run = fake_point(tmp_path, n=8, k=8)
    out = tmp_path / "gate"
    config = ed.read(run / "run_config.json")
    config.update(max_new_tokens=16, temperature=1., gradient_micro_batch=1, micro_batch=2)
    fg.bind(out / "contract.json", {"config": config, "rule": {"pilot_seed": 7}, "run": str(run), "n": 8, "pilot_ids": list(range(8))})
    monkeypatch.setattr(artifact_contract, "validate_generation_contract", lambda *a, **k: None)
    model = tiny_model()
    actual_score_group = worker.score_group
    def score_with_source_batch(*args):
        assert args[-1] == 1
        return actual_score_group(*args)
    monkeypatch.setattr(worker, "score_group", score_with_source_batch)
    def collect(model, tokenizer, prompts, k, tokens, temperature, path, sampling_seed_base):
        # A fixture, not real samples: worker plumbing is checked independently
        # of the production sampler, whose RNG bindings are tested separately.
        source = read_rollouts(run / "rollouts_behavior_train.jsonl")
        records = []
        for i in range(len(prompts)):
            records.extend({**r, "prompt_idx": i, "input_ids": r["input_ids"].tolist()} for r in source[i])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        assert sampling_seed_base != 0
    result = worker.compute(out, "pilot", 0, loader=lambda *a: (model, None), collector=collect)
    original = read_rollouts(run / "rollouts_behavior_train.jsonl")
    direction, _, _ = split_validation_directions(torch.load(run / "val_groups.pt", weights_only=True))
    from grads import grad_params, ProjectionSpec, sequence_logprobs_batch
    for row in result["rows"]:
        group = original[row["prompt_idx"]]
        lp = sequence_logprobs_batch(model, group, micro_batch=2)
        expected = group_score(model, grad_params(model, 1), group, lp, lp, ProjectionSpec(dim=64), direction, 10., 2)
        assert row["primary"] == pytest.approx(expected["g11"], abs=1e-6)
    assert result["training_updates"] == 0
    assert len(result["rows"]) == 2
    with pytest.raises(ValueError, match="not repeated"):
        worker.compute(out, "pilot", 0, loader=lambda *a: pytest.fail("must not load"))


def test_script_and_cli_parse(tmp_path):
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_fixed_gate.sh")], check=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}
    args = [sys.executable, str(ROOT / "src/fixed_gate.py"), "plan", "--root", str(tmp_path / "gate"),
            "--work", str(tmp_path), "--matrix", str(tmp_path / "matrix")]
    result = subprocess.run(args, env=env, text=True, capture_output=True)
    assert result.returncode == 0 and "not available" in result.stdout


def test_real_subprocess_controller_and_measured_baseline(tmp_path, monkeypatch):
    _, e5, out, _ = prepared(tmp_path)
    for arm in ("random", "g11"):
        _train(e5 / "subsets" / f"train-{arm}.args", rho=.7)
    for arm in ("before", "random", "g11"):
        _complete_evaluation(e5, arm, .6)
    def subprocess_phase(out, name, cap, devices, env):
        shards = [0] if name == "assess" else list(range(4))
        commands = [([sys.executable, str(ROOT / "tests/fake_fixed_gate_worker.py"), "--out", str(out),
                      "--phase", name, "--shard", str(s)], "") for s in shards]
        fg.runtime.meter(out, name, ENV["FIXED_GATE_GPU_TYPE"], commands=commands, env={}, timeout=30,
                         ledger="research" if name == "baseline" else "deployment")
    monkeypatch.setattr(fg, "phase", subprocess_phase)
    assert fg.select(out, DEVICES, ENV)["action"] == "g11"
    fg.measure_baseline(out, DEVICES, ENV)
    result = fg.rewards(out)
    assert result["complete"] and result["cost_comparison_complete"]
    assert result["measured_net_scoring_gpu_seconds_saved"] == pytest.approx(
        result["cost"]["phases"]["baseline"]["gpu_seconds"] - result["cost"]["ledgers"]["deployment"]["gpu_seconds"])
    assert len(result["cost"]["phases"]) == 4


def test_meter_timeout_reaps_child_processes_and_records_cost(tmp_path):
    out = tmp_path / "timeout"
    with pytest.raises(TimeoutError):
        fg.runtime.meter(out, "pilot", "CPU fake allocation",
                         commands=[([sys.executable, "-c", "import time; time.sleep(20)"], "")],
                         env={}, timeout=.05, ledger="deployment")
    cost = fg.allocation(out)
    assert cost["complete"] and cost["ledgers"]["deployment"]["failed_events"] == 1
    assert cost["phases"]["pilot"]["gpu_seconds"] > 0


def test_another_controller_cannot_claim_the_same_point(tmp_path):
    lock = tmp_path / "point.lock"
    with fg.runtime.lease(lock):
        result = subprocess.run([sys.executable, "-c", "import fcntl,sys; f=open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)", str(lock)],
                                text=True, capture_output=True)
        assert result.returncode != 0 and "BlockingIOError" in result.stderr
    with fg.runtime.lease(lock):
        pass


def test_updated_policy_shard_is_rejected(tmp_path):
    _, _, out, _ = prepared(tmp_path)
    fake_phase(out, "pilot", 10, DEVICES, ENV, calls=[])
    path = out / "pilot/shard-0.json"
    value = ed.read(path)
    value["training_updates"] = 1
    ed.atomic_json(path, value)
    with pytest.raises(ValueError, match="must not update"):
        fg.read_phase(out, "pilot")


def test_baseline_without_allocation_is_not_complete(tmp_path, monkeypatch):
    _, _, out, _ = prepared(tmp_path)
    monkeypatch.setattr(fg, "phase", lambda *a: fake_phase(*a, calls=[], reliable=False))
    fg.select(out, DEVICES, ENV)
    fg.measure_baseline(out, DEVICES, ENV)
    assert (out / "baseline.done.json").exists()
    result = fg.rewards(out)
    assert not result["cost_comparison_complete"]
    assert result["measured_net_scoring_gpu_seconds_saved"] is None


def test_results_filter_requested_drift_and_seed(tmp_path, monkeypatch):
    root = tmp_path / "results"
    for drift in (0, 400):
        fg.bind(root / f"d{drift}/s0/decision.json", {})
    called = []
    def reward(out):
        called.append(out)
        return {"complete": True, "cost_comparison_complete": True}
    monkeypatch.setattr(fg, "rewards", reward)
    fg.main(["results", "--root", str(root), "--matrix", str(tmp_path), "--work", str(tmp_path),
             "--drifts", "0", "--seeds", "0"])
    assert called == [root / "d0/s0"]
    assert not ed.read(root / "results.json")["incomplete_or_invalid"]


def test_status_shows_active_phase_even_after_result_exists(tmp_path, capsys):
    import time
    out = tmp_path / "d0/s0"
    fg.bind(out / "contract.json", {})
    fg.bind(out / "result.json", {"action": "random", "reason": "weak", "complete": True,
                                 "cost_comparison_complete": False, "forgone_reward": .01})
    fg.bind(out / "progress.json", {"phase": "baseline", "state": "running", "seconds": 12,
                                   "host": "node3", "updated": time.time()})
    fg.status(tmp_path, [0], [0])
    report = capsys.readouterr().out
    assert "baseline running" in report and "cost=pending" in report and "node3" in report


def test_decision_cannot_be_reused_with_a_changed_diagnostic(tmp_path, monkeypatch):
    _, _, out, _ = prepared(tmp_path)
    monkeypatch.setattr(fg, "phase", lambda *a: fake_phase(*a, calls=[], reliable=False))
    fg.select(out, DEVICES, ENV)
    ed.atomic_json(out / "diagnostic.json", {"action": "g11", "reason": "changed"})
    with pytest.raises(ValueError, match="different contract or diagnostic"):
        fg.select(out, DEVICES, ENV)
