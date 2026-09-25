"""CPU checks for the common-step (275) GPU-time export of the Switch comparison."""

import json
from pathlib import Path
import subprocess
import sys

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


def test_costs_rewards_and_savings_are_assembled_without_inventing_unknowns(tmp_path, monkeypatch):
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
    assert "3,cached,,31.000,2.78,0.89,0.00,2.78,0.00,+1.00" in text
    assert "vs_on_timer_saved_h" in text


def test_output_root_must_be_separate(tmp_path):
    root, switch, evaluation = layout(tmp_path)
    result = subprocess.run([sys.executable, str(sc.REPO / "scripts/selector_pair_step275_cost.py"),
                             "--root", str(root), "--switch-root", str(switch), "--eval-root", str(evaluation),
                             "--output", str(evaluation / "cost")], text=True, capture_output=True,
                            env={"PYTHONPATH": f"{sc.REPO / 'src'}:{sc.REPO / 'scripts'}", "PATH": "/usr/bin:/bin"})
    assert result.returncode != 0 and "separate" in result.stderr
    subprocess.run(["bash", "-n", str(sc.REPO / "scripts/run_selector_pair_step275_eval.sh")], check=True)
