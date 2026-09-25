"""CPU checks for the common-step (275) GPU-time export of the Switch comparison."""

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair_step275_cost as sc
from test_selector_pair import allocation

SEED, TRIGGER = 3, 125


def stats(path, first, last, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"step": step, "step_seconds": seconds}) + "\n" for step in range(first, last + 1)))


def checkpoint(policy, step, event, when):
    saved = policy / f"checkpoint-{step:06d}"
    core.atomic_json(saved / "adapter_model.safetensors", {"step": step})
    state = {"completed_steps": step, "adapter_sha256": base.digest(saved / "adapter_model.safetensors")}
    core.atomic_json(saved / "checkpoint_state.json", state)
    core.atomic_json(saved / "cost-receipt.json", {"step": step, "event_id": event, "time": when,
                     "adapter_sha256": state["adapter_sha256"], "checkpoint_state_id": core.fingerprint(state)})


def control(tmp_path, arm, *, scoring):
    directory = tmp_path / "pair/branches/on_policy/states/s3-t25/points/view-25" / arm
    events = [*allocation("score", "fresh-r-candidate", 0., 100., ledger="reporting")] if scoring else []
    events += allocation("train", "train", 200., 1000.)
    for event in events:
        base.journal(directory / "cost.jsonl", event)
    stats(directory / "policy/grpo_stats.jsonl", 26, 300, 10.)
    checkpoint(directory / "policy", 275, "train", 200. + 800.)  # 800 s into a 1000 s train allocation
    return directory


def layout(tmp_path):
    controls = {"random": control(tmp_path, "random_full", scoring=False),
                "on_policy": control(tmp_path, "selection_full", scoring=True),
                "cached": control(tmp_path, "cached_full", scoring=False)}
    root = tmp_path / "pair"
    core.atomic_json(root / f"sr-gc/s{SEED}-t25/decision.json",
                     {"new_measurement_gpu_seconds": 3600., "reused_ranking_gpu_seconds": 7200.})
    for step in (50, 75, 100):  # 125 is left without a ledger: unknown, not zero
        directory = root / f"sr-gc-repeat/every-25/s{SEED}-t25/step-{step}"
        for event in allocation("measure", "srgc-measure", 0., 900.):
            base.journal(directory / "cost.jsonl", event)
    switch = tmp_path / "switch"
    plan = {"seed": SEED, "switch_step": TRIGGER, "end_step": 315, "resume_mode": "replay_on_policy_from_25",
            "controls": {arm: str(path) for arm, path in controls.items()},
            "contract": {"eval_k": 8, "eval_seed": 1, "evaluation": {"val": [{"q": i} for i in range(300)]}},
            "config": {"model": "/m", "max_new_tokens": 8, "temperature": 1.0}}
    core.atomic_json(switch / f"s{SEED}/plan.json", plan)
    stats(switch / f"s{SEED}/replay/policy/grpo_stats.jsonl", 26, TRIGGER, 12.)
    stats(switch / f"s{SEED}/policy/grpo_stats.jsonl", TRIGGER + 1, 315, 9.)
    for event in allocation("replay", "replay", 0., 1500., ledger="research"):
        base.journal(switch / f"s{SEED}/attempts/replay-abc/cost.jsonl", event)
    evaluation = tmp_path / "eval"
    core.atomic_json(evaluation / "step275-controls.json", {"rows": [
        {"seed": SEED, "arm": arm, "reward": value} for arm, value in
        (("random", .29), ("on_policy", .30), ("cached", .31))]})
    return root, switch, evaluation


def test_costs_and_rewards_are_assembled_without_inventing_unknowns(tmp_path, monkeypatch):
    root, switch, evaluation = layout(tmp_path)
    monkeypatch.setattr(sc.sw, "measured_point", lambda directory, plan, arm, step: {"step": 275, "reward": .33})
    data = sc.build(root, switch, evaluation, [SEED])
    rows = {row["arm"]: row for row in data["rows"]}
    # update timers: 250 updates x 10 s x 4 GPUs for every control
    assert rows["random"]["update_timer_gpu_seconds"] == 250 * 10 * 4
    # allocation through the receipt: 800 s x 4 GPUs of training (+ 100 s x 4 scoring for On-policy)
    assert rows["random"]["allocation_gpu_seconds"] == 800 * 4
    assert rows["on_policy"]["allocation_gpu_seconds"] == 800 * 4 + 100 * 4
    assert rows["on_policy"]["detail"]["allocation_scoring_gpu_seconds"] == 400
    # Switch: replayed prefix (100 updates x 12 s) + suffix (150 updates x 9 s), x 4 GPUs
    assert rows["switch"]["update_timer_gpu_seconds"] == (100 * 12 + 150 * 9) * 4
    assert rows["switch"]["detail"]["prefix"]["allocation_gpu_seconds"] == 1500 * 4
    assert rows["switch"]["allocation_gpu_seconds"] is None  # no suffix receipt -> unknown, not zero
    diagnosis = rows["switch"]["diagnosis"]
    assert diagnosis["known_gpu_seconds"] == 3600 + 3 * 900 * 4 and diagnosis["unknown_steps"] == [125]
    assert rows["switch"]["diagnosis_unknown"] is True
    assert rows["switch"]["reward"] == .33 and rows["cached"]["reward"] == .31
    text = sc.render(data)
    assert "3,switch,125,33.000," in text and "+unknown" in text
    table = list(csv.DictReader(io.StringIO(text.split("\n\n")[1])))
    displayed = {r["arm"]: r for r in table}
    assert displayed["cached"]["training_h"] == "2.78"
    assert displayed["cached"]["vs_on_reward_pp"] == "+1.00"
    assert displayed["switch"]["selection_h"] == displayed["on_policy"]["selection_h"] == "0.111111"
    assert displayed["switch"]["selection_training_subtotal_h"] == "2.94"
    assert displayed["switch"]["single_reference_check_h"] == "unknown"
    assert displayed["switch"]["ab_validation_h"] == "4.00+unknown"
    assert "vs_on_timer_saved_h" not in text and "timer+diag_h" not in text
    assert "CACHE_CREATION seed=3 status=unknown gpu_h=unknown" in text


def test_missing_selection_is_unknown_not_free_in_log_export(tmp_path, monkeypatch):
    root, switch, evaluation = layout(tmp_path)
    monkeypatch.setattr(sc.sw, "measured_point", lambda *args: {"step": 275, "reward": .33})
    data = sc.build(root, switch, evaluation, [SEED])
    next(r for r in data["rows"] if r["arm"] == "on_policy")["detail"]["allocation_scoring_gpu_seconds"] = None
    table = list(csv.DictReader(io.StringIO(sc.render(data).split("\n\n")[1])))
    on = next(r for r in table if r["arm"] == "on_policy")
    assert on["selection_h"] == on["selection_training_subtotal_h"] == "unknown"


def test_switch_is_charged_the_same_sr_preparation_as_the_sr_control(tmp_path, monkeypatch):
    root, switch, evaluation = layout(tmp_path)
    monkeypatch.setattr(sc.sw, "measured_point", lambda *args: {"step": 275, "reward": .33})
    data = sc.build(root, switch, evaluation, [SEED])
    cached = next(r for r in data["rows"] if r["arm"] == "cached")
    cached["detail"]["allocation_scoring_gpu_seconds"] = 1.2
    rows = {r["arm"]: r for r in csv.DictReader(io.StringIO(sc.render(data).split("\n\n")[1]))}
    assert rows["cached"]["sr_preparation_h"] == rows["switch"]["sr_preparation_h"]
    assert float(rows["switch"]["selection_h"]) == pytest.approx(401.2 / 3600, abs=1e-6)
    assert rows["cached"]["cache_creation_h"] == rows["switch"]["cache_creation_h"] == "unknown"
    assert all(r["repeated_selection_gpu_seconds"] is None for r in data["rows"]
               if r["arm"] in ("switch", "on_policy"))


def test_output_root_must_be_separate(tmp_path):
    root, switch, evaluation = layout(tmp_path)
    result = subprocess.run([sys.executable, str(sc.REPO / "scripts/selector_pair_step275_cost.py"),
                             "--root", str(root), "--switch-root", str(switch), "--eval-root", str(evaluation),
                             "--output", str(evaluation / "cost")], text=True, capture_output=True,
                            env={"PYTHONPATH": f"{sc.REPO / 'src'}:{sc.REPO / 'scripts'}", "PATH": "/usr/bin:/bin"})
    assert result.returncode != 0 and "separate" in result.stderr
    subprocess.run(["bash", "-n", str(sc.REPO / "scripts/run_selector_pair_step275_eval.sh")], check=True)


@pytest.mark.parametrize("custom_output", [False, True])
def test_cost_shell_exports_to_separate_root_without_modifying_inputs(tmp_path, custom_output):
    root, switch, evaluation = layout(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    output = tmp_path / "custom-cost" if custom_output else Path(f"{evaluation}-cost")
    env = {**os.environ, "HOME": str(home), "OM_WORK": str(tmp_path / "work"),
           "PAIR_PYTHON": sys.executable, "PAIR_ROOT": str(root),
           "PAIR_SWITCH_ROOT": str(switch), "PAIR_STEP275_ROOT": str(evaluation)}
    env.pop("PAIR_STEP275_COST_ROOT", None)
    if custom_output:
        env["PAIR_STEP275_COST_ROOT"] = str(output)
    inputs = (root, switch, evaluation)
    before = {path: base.digest(path) for directory in inputs for path in directory.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", str(sc.REPO / "scripts/run_selector_pair_step275_eval.sh"),
                             "cost", "--seed", str(SEED)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((output / "step275-cost.json").read_text())
    assert data["step"] == 275 and len(data["rows"]) == 4
    exported = home / "selector-pair-step275-cost.txt"
    assert exported.read_text() == (output / "step275-cost.txt").read_text()
    assert "[saved] " + str(exported) in result.stdout
    after = {path: base.digest(path) for directory in inputs for path in directory.rglob("*") if path.is_file()}
    assert after == before
