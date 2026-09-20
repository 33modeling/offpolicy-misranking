import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selection_switch as rule

ROOT = Path(__file__).resolve().parents[1]


def branch(root, state, arm, *, rewards, updates, extra_phases=(), discard=False, waiver=False, curve=False):
    seed, step = state[1:].split("-t")
    directory = root / "states" / state / "points" / f"view-{step}" / arm
    core.atomic_json(directory / "result.json", {"schema": rule.SCHEMA, "complete": True, "rewards": {str(i): r for i, r in enumerate(rewards)},
                                                 "used_gpu_seconds": 28700., "completed_steps": int(step)+updates})
    digest = hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest()
    core.atomic_json(directory / 'result.sha256.json', {'sha256': digest})
    core.atomic_json(directory / "policy/budget_stop.json", {"completed_steps": int(step)+updates, "stop_reason": "budget_exhausted"})
    core.atomic_json(directory / "decision.json", {"action": "select" if arm.startswith("selection") else "random",
                                                   "prediction": -0.004 if arm == "gated" else None})
    core.atomic_json(directory / "execution.json", {"action": "random" if arm != "selection_full" else "select"})
    for name, seconds in (("verify-inputs", 1.), ("train", 7000.), *extra_phases):
        for st in ("started", "finished"):
            row = {"event_id": name, "phase": name, "ledger": "deployment", "gpus": 4, "gpu_type": "H100", "host": "h",
                   "state": st, "time": 1.}
            if st == "finished":
                row.update(seconds=seconds, allocated_gpu_seconds=4*seconds, exit_code=0)
            base.journal(directory / "cost.jsonl", row)
    if discard:
        core.atomic_json(directory / "discards/20260916T120000Z.json", {"schema": "reset"})
    if waiver:
        core.atomic_json(directory / "waivers/train1.json", {"schema": "waiver"})
    if curve:
        core.atomic_json(directory / "curve.json", {"schema": rule.SCHEMA, "result_sha256": digest, "k": 4, "points": {str(step): {"updates": 0, "reward": .25},
                                                                        str(int(step)+50): {"updates": 50, "reward": .28},
                                                                        str(int(step)+updates): {"updates": updates, "reward": .30, "final": True}}})
    return directory


def test_results_file_lists_branches_contrasts_flags_and_rewards(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x", "selector": "difficulty", "gate": "convergence", "budget_gpu_seconds": 29040})
    core.atomic_json(tmp_path / "model.json", {"ridge": {"intercept": -0.015, "coef": [0.001, 0.002, -0.001, 0.0005]}, "features": ["a", "b", "c", "d"]})
    hi = [1., .5, .25, 0., .75, 1., .5, .5, .25, .75, 1., .5]
    lo = [.75, .5, 0., 0., .5, 1., .25, .5, .25, .5, .75, .5]
    branch(tmp_path, "s3-t25", "selection_full", rewards=lo, updates=16, extra_phases=(("fresh-r-candidate", 5000.),))
    branch(tmp_path, "s3-t25", "random_full", rewards=hi, updates=100, curve=True)
    branch(tmp_path, "s3-t25", "gated", rewards=hi, updates=100)
    branch(tmp_path, "s4-t25", "random_full", rewards=hi, updates=100)
    branch(tmp_path, "s4-t25", "gated", rewards=hi, updates=191, discard=True)
    branch(tmp_path, "s4-t25", "selection_full", rewards=lo, updates=11)
    waived = branch(tmp_path, "s4-t25", "random_reduced", rewards=lo, updates=120, waiver=True)
    (waived / "result.json").unlink()
    # A retry that resumed a waived attempt: 120 updates from one allocation that buys about 100.
    branch(tmp_path, "s3-t100", "random_reduced", rewards=hi, updates=120, waiver=True)
    branch(tmp_path, "s3-t100", "selection_reduced", rewards=lo, updates=14)
    out = tmp_path / "results.txt"
    subprocess.run([sys.executable, str(ROOT / "scripts/switch_results.py"), "--root", str(tmp_path), "--out", str(out), "--draws", "200"],
                   check=True, capture_output=True, text=True)
    text = out.read_text()
    assert "SELECTOR Difficulty  ACCOUNTING ?  GATE 비용 보정 학습 효율 기준" in text
    assert "GATE MODEL intercept=-0.01500" in text
    assert "s3/t25   selection_full     reward= 45.83 updates=  16 used= 28700 action=select" in text
    assert "'fresh-r-candidate': 20000" in text
    assert "s4/t25   gated              reward= 58.33 updates= 191" in text and "INVALID" in text
    assert "s4/t25   random_reduced     reward=  none" in text and "RERUN" in text
    assert "INVALID = more than 114 updates" in text
    assert "s3/t100  random_reduced     reward= 58.33 updates= 120" in text
    assert [l for l in text.split("\n") if l.startswith("s3/t100  random_reduced")][0].endswith("INVALID")
    assert "s3/t100  selection_reduced-random_reduced" not in text
    assert "curve k=4 points(updates:reward) 0:25.00, 50:28.00, 100:30.00" in text
    contrasts = text[text.index("CONTRASTS"):text.index("REWARDS")]
    assert "s3/t25   selection_full-random_full=-12.50 [" in contrasts and "gated-random_full=+0.00 [" in contrasts
    assert "gated-selection_full" in contrasts.split("\n")[1]
    assert "s4/t25   selection_full-random_full=" in contrasts and "gated-random_full" not in contrasts.split("s4/t25")[1]
    rewards = text[text.index("REWARDS"):]
    assert "s3/t25 random_full: 100.0 50.0 25.0 0.0 75.0" in rewards
    assert "s4/t25 gated INVALID: " in rewards
    assert "development-report.json" not in text


def result_text(root):
    return subprocess.run([sys.executable, str(ROOT / "scripts/switch_results.py"), "--root", str(root), "--draws", "20"],
                          check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize("accounting,label", [("budget", "선택비용 포함"), ("matched", "선택비용 별도")])
def test_mbpp_result_header_distinguishes_actual_accounting_without_rewriting_protocol(tmp_path, accounting, label):
    # Actual frozen settings win even when a directory has the other condition's name.
    root = tmp_path / "selection-switch-mbpp-v1"
    protocol = {"dataset": "mbpp", "selector": "fresh_r", "accounting": accounting,
                "gate": "convergence", "budget_gpu_seconds": 29040}
    core.atomic_json(root / "switch.json", protocol)
    before = (root / "switch.json").read_bytes()
    text = result_text(root)
    assert f"EXPERIMENT On-policy · {label}" in text
    assert f"SELECTOR On-policy  ACCOUNTING {label}  GATE 비용 보정 학습 효율 기준" in text
    assert "fresh_r" not in text.split("BRANCHES", 1)[0]
    assert f"ROOT {root}" in text and "BUDGET 29040  DATASET mbpp" in text
    assert "reward=" not in text and not (root / "states").exists()
    assert (root / "switch.json").read_bytes() == before


@pytest.mark.parametrize("manifest", [None, "{broken", "[]"])
def test_results_missing_or_invalid_manifest_does_not_invent_protocol_or_results(tmp_path, manifest):
    root = tmp_path / "selection-switch-mbpp-quality-v1"
    if manifest is not None:
        root.mkdir()
        (root / "switch.json").write_text(manifest)
    text = result_text(root)
    assert "MANIFEST missing, unreadable or not an object" in text
    assert "SELECTOR ?  ACCOUNTING ?  GATE ?" in text and "DATASET ?" in text
    assert "reward=" not in text and "fresh_r" not in text
    assert not (root / "states").exists()


def test_results_wrong_dataset_is_not_labeled_as_a_verified_mbpp_condition(tmp_path):
    root = tmp_path / "selection-switch-mbpp-quality-v1"
    core.atomic_json(root / "switch.json", {"dataset": "math500", "selector": "difficulty", "accounting": "budget", "gate": "final"})
    text = result_text(root)
    assert f"EXPERIMENT {root.name}" in text and "DATASET math500" in text
    assert "WARNING MBPP-named root has a different frozen dataset" in text
    assert "GATE 최종 보상 기준" in text


def cost_event(directory, event_id, phase, seconds, *, ledger="reporting", exit_code=0, open_event=False):
    for state in (("started",) if open_event else ("started", "finished")):
        row = {"event_id": event_id, "phase": phase, "ledger": ledger, "state": state}
        if state == "finished":
            row.update(allocated_gpu_seconds=4*seconds, seconds=seconds, exit_code=exit_code)
        base.journal(directory / "cost.jsonl", row)


@pytest.mark.parametrize("completed", [False, True])
def test_matched_cost_display_includes_failed_scoring_and_own_curves_not_shared_parent(tmp_path, completed):
    core.atomic_json(tmp_path / "switch.json", {"dataset": "mbpp", "selector": "fresh_r", "accounting": "matched",
                                               "gate": "convergence", "budget_gpu_seconds": 29040,
                                               "budget_source": {"kind": "explicit", "gpu_seconds": 29040}})
    directory = branch(tmp_path, "s3-t25", "selection_full", rewards=[.5, 1.], updates=100)
    core.atomic_json(directory / "decision.json", {"action": "select", "measurement_gpu_seconds": 36,
                                                   "budget_gpu_seconds": 29004})
    if not completed:
        (directory / "result.json").unlink()
        core.atomic_json(directory / "failure.json", {"error": "evaluation interrupted"})
    cost_event(directory, "val", "fresh-r-validation", 100)
    cost_event(directory, "candidate-failed", "fresh-r-candidate", 200, exit_code=1)
    cost_event(directory, "candidate-ok", "fresh-r-candidate", 300)
    cost_event(directory, "eval", "evaluate", 50)
    cost_event(directory / "curve", "curve-failed", "curve", 5, exit_code=1)
    cost_event(directory / "curve", "curve-ok", "curve", 10)
    cost_event(directory.parent / "curve-parent", "shared-parent", "curve", 1000)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    text = result_text(tmp_path)
    assert "training allocation matched, total compute not matched" in text
    assert 'budget_source={"gpu_seconds": 29040, "kind": "explicit"}' in text
    assert "scoring=2400.000 training=28000.000 evaluation=260.000 other=4.000" in text
    assert "finished_events_subtotal=30664.000 branch_total_incl_reporting=30664.000" in text
    assert "diagnostic_charge=36.000 action_total_with_diagnostic=30700.000 branch_allocation=29004.000" in text
    assert f"COST PATH {directory}" in text
    assert "count it once per point, not once per arm" in text
    assert "do not sum action totals across arms" in text
    assert "checkpoint lineage and ledger provenance not certified" in text
    assert "reward= 75.00" in text if completed else "reward=  none" in text
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("damage", ["open", "missing-start", "missing-ledger", "empty", "nonfinite", "duplicate", "torn"])
def test_incomplete_or_invalid_cost_is_unknown_not_zero(tmp_path, damage):
    core.atomic_json(tmp_path / "switch.json", {"dataset": "mbpp", "accounting": "matched"})
    directory = branch(tmp_path, "s0-t25", "selection_reduced", rewards=[.5], updates=50)
    (directory / "result.json").unlink()
    path = directory / "cost.jsonl"
    if damage == "open":
        cost_event(directory, "open", "fresh-r-candidate", 50, open_event=True)
    elif damage == "missing-start":
        base.journal(path, {"event_id": "missing-start", "phase": "train", "ledger": "deployment",
                            "state": "finished", "allocated_gpu_seconds": 30})
    elif damage == "missing-ledger":
        path.unlink()
    elif damage == "empty":
        path.write_text("")
    elif damage == "nonfinite":
        cost_event(directory, "invalid", "evaluate", 0, open_event=True)
        with path.open("a") as handle:
            handle.write(json.dumps({"event_id": "invalid", "phase": "evaluate", "ledger": "reporting",
                                     "state": "finished", "allocated_gpu_seconds": float("nan")}) + "\n")
    elif damage == "duplicate":
        cost_event(directory, "same", "curve", 30)
        cost_event(directory, "same", "curve", 30)
    else:
        with path.open("a") as handle:
            handle.write('{"torn":')
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    text = result_text(tmp_path)
    assert "reward=  none" in text
    assert "branch_total_incl_reporting=unknown" in text
    assert "diagnostic_charge=unknown action_total_with_diagnostic=unknown branch_allocation=unknown" in text
    assert "coverage=unknown:" in text
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_budget_mode_cost_display_counts_selection_without_claiming_matched_training(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"accounting": "budget", "budget_gpu_seconds": 87120})
    directory = branch(tmp_path, "s3-t25", "selection_full", rewards=[.5], updates=100,
                       extra_phases=(("fresh-r-candidate", 5000),))
    core.atomic_json(directory / "decision.json", {"action": "select", "measurement_gpu_seconds": 0,
                                                   "budget_gpu_seconds": 87120})
    text = result_text(tmp_path)
    assert "training allocation matched" not in text
    assert "scoring=20000.000 training=28000.000" in text
    assert "branch_total_incl_reporting=48004.000 diagnostic_charge=0.000 action_total_with_diagnostic=48004.000" in text
