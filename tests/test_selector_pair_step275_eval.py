"""CPU checks for the common-step (275) control evaluation of the Switch comparison."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import selection_gate as core
import selector_pair_step275_eval as ev

SCRIPT = ev.REPO / "scripts/run_selector_pair_step275_eval.sh"


def plan(seed, switch_step, end_step=315):
    return {"seed": seed, "switch_step": switch_step, "end_step": end_step,
            "contract": {"eval_k": 8, "eval_seed": 1, "evaluation": {"val": [{"q": i} for i in range(300)]},
                         "scope": {"gpu_type": "fixture"}},
            "config": {"model": "/m", "max_new_tokens": 8, "temperature": 1.0}, "controls": {}, "endpoints": {}}


def switch_root(tmp_path, plans):
    root = tmp_path / "switch"
    for value in plans:
        core.atomic_json(root / f"s{value['seed']}" / "plan.json", value)
    return root


def test_step_must_lie_inside_the_switch_window(tmp_path):
    root = switch_root(tmp_path, [plan(3, 125), plan(4, 300)])
    assert ev.load_plan(root, 3)["switch_step"] == 125
    with pytest.raises(ValueError, match="not inside"):
        ev.load_plan(root, 4)
    with pytest.raises(ValueError, match="no Switch plan"):
        ev.load_plan(root, 5)


def test_report_uses_fresh_control_points_and_the_existing_switch_point(tmp_path, monkeypatch):
    root = switch_root(tmp_path, [plan(3, 125), plan(4, 100)])
    output = tmp_path / "out"
    points = {(3, "random"): 0.30, (3, "on_policy"): 0.31, (3, "cached"): 0.32, (4, "random"): 0.33}
    monkeypatch.setattr(ev, "measured", lambda sr, out, p, arm: (
        {"step": 275, "reward": points[(p["seed"], arm)], "k": 8, "question_count": 300, "source": "x"}
        if (p["seed"], arm) in points else None))
    monkeypatch.setattr(ev, "switch_point", lambda sr, p: {"step": 275, "reward": 0.37 if p["seed"] == 4 else 0.33,
                                                            "k": 8, "question_count": 300, "source": "s"})
    data = ev.report(root, output, [3, 4])
    assert not data["complete"]
    assert [(r["seed"], r["arm"]) for r in data["missing"]] == [(4, "on_policy"), (4, "cached")]
    out = tmp_path / "home" / "r.txt"
    out.parent.mkdir()
    ev.write_report(root, output, [3, 4], out)
    text = out.read_text()
    assert "Overall: INCOMPLETE" in text
    assert "3,random,275,30.000,8,300" in text and "3,switch,275,33.000,8,300" in text
    assert "4,on_policy,275,missing,," in text
    saved = json.loads((output / "step275-controls.json").read_text())
    assert saved["schema"] == ev.SCHEMA and saved["step"] == 275
    assert (output / "step275-controls.txt").read_text() == text


def test_output_root_must_be_separate_and_launcher_is_valid(tmp_path):
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    root = switch_root(tmp_path, [plan(3, 125)])
    env = {**os.environ, "PYTHONPATH": f"{ev.REPO / 'src'}:{ev.REPO / 'scripts'}"}
    inside = subprocess.run([sys.executable, str(ev.REPO / "scripts/selector_pair_step275_eval.py"), "status",
                             "--switch-root", str(root), "--output", str(root / "out")],
                            env=env, text=True, capture_output=True)
    assert inside.returncode != 0 and "separate" in inside.stderr
    bad = subprocess.run(["bash", str(SCRIPT), "train"], env=env, text=True, capture_output=True)
    assert bad.returncode == 2
