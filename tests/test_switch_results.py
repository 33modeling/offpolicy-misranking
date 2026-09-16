import json
import subprocess
import sys
from pathlib import Path

import selection_gate as core
import selection_gate_gpu as base

ROOT = Path(__file__).resolve().parents[1]


def branch(root, state, arm, *, rewards, updates, extra_phases=(), discard=False, waiver=False, curve=False):
    seed, step = state[1:].split("-t")
    directory = root / "states" / state / "points" / f"view-{step}" / arm
    core.atomic_json(directory / "result.json", {"complete": True, "rewards": {str(i): r for i, r in enumerate(rewards)},
                                                 "used_gpu_seconds": 28700., "completed_steps": int(step)+updates})
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
        core.atomic_json(directory / "curve.json", {"k": 4, "points": {str(step): {"updates": 0, "reward": .25},
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
    out = tmp_path / "results.txt"
    subprocess.run([sys.executable, str(ROOT / "scripts/switch_results.py"), "--root", str(tmp_path), "--out", str(out), "--draws", "200"],
                   check=True, capture_output=True, text=True)
    text = out.read_text()
    assert "SELECTOR difficulty  GATE convergence" in text and "GATE MODEL intercept=-0.01500" in text
    assert "s3/t25   selection_full     reward= 45.83 updates=  16 used= 28700 action=select" in text
    assert "'fresh-r-candidate': 20000" in text
    assert "s4/t25   gated              reward= 58.33 updates= 191" in text and "INVALID" in text
    assert "s4/t25   random_reduced     reward=  none" in text and "RERUN" in text
    assert "curve k=4 points(updates:reward) 0:25.00, 50:28.00, 100:30.00" in text
    contrasts = text[text.index("CONTRASTS"):text.index("REWARDS")]
    assert "s3/t25   selection_full-random_full=-12.50 [" in contrasts and "gated-random_full=+0.00 [" in contrasts
    assert "gated-selection_full" in contrasts.split("\n")[1]
    assert "s4/t25   selection_full-random_full=" in contrasts and "gated-random_full" not in contrasts.split("s4/t25")[1]
    rewards = text[text.index("REWARDS"):]
    assert "s3/t25 random_full: 100.0 50.0 25.0 0.0 75.0" in rewards
    assert "s4/t25 gated INVALID: " in rewards
    assert "development-report.json" not in text
