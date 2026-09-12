"""CPU contracts for the executed gate arm in the E5 driver (pilot, decision, continuation)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_evidence_downstream import source_point

import evidence_downstream as ed

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "tests" / "fake_trainer.py"


def _args(path: Path) -> list[str]:
    return path.read_bytes().decode().rstrip("\0").split("\0")


def _train(args_file: Path, rho: float) -> None:
    args = _args(args_file)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "FAKE_TRAINER_RHO": str(rho),
           "OM_PROMPT_FORMAT": "olmo_rlzero_math"}
    result = subprocess.run([sys.executable, str(FAKE), *args], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


def _complete_evaluation(out: Path, arm: str, reward: float) -> None:
    contract = ed.read(out / "experiment.json")
    for shard in range(4):
        binding, _, indices = ed.eval_binding(out, arm, shard)
        target = out / arm / "evaluation"
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"shard-{shard}.jsonl"
        rows = [{"prompt_idx": i, "rollout_idx": j, "reward": reward if (i + j) % 2 else 0.0}
                for i in indices for j in range(contract["eval_k"])]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        ed.atomic_json(target / f"shard-{shard}.contract.json", binding)
        ed.atomic_json(target / f"shard-{shard}.done.json", {"binding": binding, "rollouts_sha256": ed.digest(path)})


@pytest.fixture
def rule(tmp_path, monkeypatch):
    import gate_decision as gd
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(gd.default_rule(pilot_size=40, r_min=0.25, confidence=0.90, seed=1)))
    monkeypatch.setenv("E5_GATE_RULE", str(path))
    return path


@pytest.mark.parametrize("drift", [400, 0])
def test_gate_arm_pilot_decision_and_continuation(tmp_path, rule, drift):
    run, evaluation = source_point(tmp_path, drift=drift)
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random", "passrate_beta", "gate_passrate"])
    assert ed.read(out / "gate_rule.json")["pilot_size"] == 40
    assert ed.read(out / "gate_pilot.json")["pilot_steps"] == 10
    pilot_prompts = ed.read(out / "subsets" / "subset-gate_passrate-pilot.json")
    assert len(pilot_prompts["train"]) == 40 and sorted(pilot_prompts["selected_idx"]) == list(range(40))
    pargs = _args(out / "subsets" / "train-gate_passrate-pilot.args")
    assert pargs[pargs.index("--target-steps") + 1] == str(drift + 10)
    assert pargs[pargs.index("--start-step") + 1] == str(drift) and "--reliability-log" in pargs
    assert ("--resume-adapter" in pargs) == (drift > 0)
    assert not (out / "subsets" / "train-gate_passrate.args").exists()  # written by the decision
    assert ed.arm_state(out, "gate_passrate") == "not started"

    _train(out / "subsets" / "train-gate_passrate-pilot.args", rho=0.7)
    assert ed.arm_state(out, "gate_passrate") == "pilot trained, not decided"
    decision = ed.gate_decide(out, "gate_passrate")
    assert decision["decision"] == "retain" and decision["chosen_subset"] == "passrate_beta"
    assert decision["pilot_pairs"] == 40 and decision["pilot_steps"] == 10 and decision["pilot_seconds"] == pytest.approx(750.0)
    assert ed.gate_decide(out, "gate_passrate") == decision  # idempotent
    assert ed.arm_state(out, "gate_passrate").startswith("decided retain")
    args = _args(out / "subsets" / "train-gate_passrate.args")
    assert args[args.index("--prompts") + 1].endswith("subset-passrate_beta.json")
    assert args[args.index("--start-step") + 1] == str(drift + 10)
    assert args[args.index("--target-steps") + 1] == str(drift + 100)
    assert args[args.index("--resume-adapter") + 1] == str(out / "gate_passrate" / "pilot")
    assert args[args.index("--output") + 1] == str(out / "gate_passrate" / "policy")

    _train(out / "subsets" / "train-gate_passrate.args", rho=0.7)
    assert ed.arm_policy(out, "gate_passrate") == out / "gate_passrate" / "policy"
    binding, policy, _ = ed.eval_binding(out, "gate_passrate", 0)
    assert policy == out / "gate_passrate" / "policy" and binding["adapter_sha256"] == ed.digest(policy / "adapter_model.safetensors")

    # a changed rule after the decision is refused
    frozen = ed.read(out / "gate_rule.json")
    ed.atomic_json(out / "gate_rule.json", {**frozen, "r_min": 0.5})
    with pytest.raises(ValueError, match="frozen rule"):
        ed.arm_policy(out, "gate_passrate")
    ed.atomic_json(out / "gate_rule.json", frozen)
    assert ed.arm_policy(out, "gate_passrate") == out / "gate_passrate" / "policy"


def test_weak_pilot_falls_back_to_the_random_subset(tmp_path, rule):
    run, evaluation = source_point(tmp_path, drift=400)
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random", "passrate_beta", "gate_passrate"])
    _train(out / "subsets" / "train-gate_passrate-pilot.args", rho=0.0)
    decision = ed.gate_decide(out, "gate_passrate")
    assert decision["decision"] == "random" and decision["reason"] in ("weak", "unresolved")
    args = _args(out / "subsets" / "train-gate_passrate.args")
    assert args[args.index("--prompts") + 1].endswith("subset-random.json")


def test_summary_reports_the_decision_and_forgone_reward(tmp_path, rule):
    run, evaluation = source_point(tmp_path, drift=400)
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random", "passrate_beta", "gate_passrate"])
    _train(out / "subsets" / "train-gate_passrate-pilot.args", rho=0.7)
    ed.gate_decide(out, "gate_passrate")
    for arm in ("random", "passrate_beta", "gate_passrate"):
        _train(out / "subsets" / f"train-{arm}.args", rho=0.7)
    _complete_evaluation(out, "before", 0.4)
    for arm, reward in (("random", 0.6), ("passrate_beta", 1.0), ("gate_passrate", 0.8)):
        _complete_evaluation(out, arm, reward)
    report = ed.summarize(out)
    rows = {r["selector"]: r for r in report["rows"]}
    assert report["complete"] and set(rows) == {"random", "passrate_beta", "gate_passrate"}
    gate = rows["gate_passrate"]
    assert gate["gate_decision"] == "retain" and gate["gate_pilot_steps"] == 10 and gate["gate_pilot_seconds"] == pytest.approx(750.0)
    assert gate["forgone_vs_selector"] == pytest.approx(rows["passrate_beta"]["reward_after"] - gate["reward_after"])
    assert gate["forgone_lower"] <= gate["forgone_vs_selector"] <= gate["forgone_upper"]
    assert rows["random"]["gate_decision"] is None and rows["random"]["forgone_vs_selector"] is None
    assert gate["difference_vs_random"] == pytest.approx(gate["reward_after"] - rows["random"]["reward_after"])
    csv_text = (out / "downstream_results.csv").read_text()
    assert "gate_decision" in csv_text.splitlines()[0] and "retain" in csv_text


def test_gate_arm_can_be_added_to_a_prepared_seed(tmp_path, rule):
    run, evaluation = source_point(tmp_path, drift=400)
    out = tmp_path / "out"
    first = ed.prepare(run, out, evaluation, 100, 8, ["random", "passrate_beta"])
    frozen = ed.digest(out / "experiment.json")
    ed.prepare(run, out, evaluation, 100, 8, ["gate_passrate"])
    assert ed.digest(out / "experiment.json") == frozen and first["selectors"] == ["random", "passrate_beta"]
    assert ed.arms_of(out) == ["random", "passrate_beta", "gate_passrate"]
    assert (out / "subsets" / "train-gate_passrate-pilot.args").is_file()
    with pytest.raises(ValueError, match="unknown"):
        ed.prepare(run, tmp_path / "o2", evaluation, 100, 8, ["gate_fresh"], dry=True)


def test_cli_gate_commands_and_launcher_syntax(tmp_path, rule):
    run, evaluation = source_point(tmp_path, drift=400)
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random", "passrate_beta", "gate_passrate"])
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, str(ROOT / "src/evidence_downstream.py")]
    ready = subprocess.run(cmd + ["gate-pilot-ready", "--out", str(out), "--arm", "gate_passrate"], env=env)
    assert ready.returncode == 1
    _train(out / "subsets" / "train-gate_passrate-pilot.args", rho=0.7)
    assert subprocess.run(cmd + ["gate-pilot-ready", "--out", str(out), "--arm", "gate_passrate"], env=env).returncode == 0
    decided = subprocess.run(cmd + ["gate-decide", "--out", str(out), "--arm", "gate_passrate"], capture_output=True, text=True, env=env)
    assert decided.returncode == 0 and decided.stdout.strip().startswith("decision=retain")
    status = subprocess.run(cmd + ["status", "--out", str(out)], capture_output=True, text=True, env=env)
    assert "gate_passrate  decided retain (reliable)" in status.stdout
    for name in ("scripts/run_downstream_independent.sh", "scripts/run_e5.sh"):
        subprocess.run(["bash", "-n", str(ROOT / name)], check=True)
    text = (ROOT / "scripts/run_downstream_independent.sh").read_text()
    assert "gate-pilot-ready" in text and "gate-decide" in text and "gate_passrate" in text
    assert "gate)" in (ROOT / "scripts/run_e5.sh").read_text()
