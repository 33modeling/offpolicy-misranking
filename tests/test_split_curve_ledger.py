import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import selection_gate as core
import selection_gate_gpu as base

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("split_curve_ledger", ROOT / "scripts/split_curve_ledger.py")
split = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split)


def row(event_id, phase, state, *, ledger="deployment", seconds=1.0):
    r = {"event_id": event_id, "phase": phase, "ledger": ledger, "gpus": 4, "gpu_type": "H100",
         "host": "h", "state": state, "time": 1.0}
    if state == "finished":
        r.update(seconds=seconds, allocated_gpu_seconds=4*seconds, exit_code=0)
    return r


def branch(root, name, *, published=True, curve_rows=2):
    directory = root / "states/s0-t25/points/view-25" / name
    rows = [row("verify", "verify-inputs", "started"), row("verify", "verify-inputs", "finished"),
            row("train", "train", "started"), row("train", "train", "finished", seconds=7000.0),
            row("evaluate", "evaluate", "started", ledger="reporting"), row("evaluate", "evaluate", "finished", ledger="reporting", seconds=3600.0)]
    for r in rows:
        base.journal(directory / "cost.jsonl", r)
    if published:
        core.atomic_json(directory / "result.json", {"complete": True, "cost": base.cost(directory), "rewards": {"1": .5}})
    for i in range(curve_rows):
        base.journal(directory / "cost.jsonl", row(f"curve{i}", "curve", "started", ledger="reporting"))
        base.journal(directory / "cost.jsonl", row(f"curve{i}", "curve", "finished", ledger="reporting", seconds=900.0))
    return directory


def test_curve_rows_move_out_of_the_sealed_ledger_and_the_failure_clears(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = branch(tmp_path, "selection_reduced")
    core.atomic_json(directory / "failure.json", {"error": "cost ledger changed", "host": "h", "time": 1.0})
    result = core.read(directory / "result.json")
    assert base.cost(directory) != result["cost"]
    assert split.candidates(tmp_path) == [directory]
    assert "would move 4 curve row(s)" in split.split(tmp_path, directory, apply=False)
    message = split.split(tmp_path, directory, apply=True)
    assert "moved 4 curve row(s)" in message and "matches the result again" in message and "failure cleared" in message
    assert base.cost(directory) == result["cost"] and not (directory / "failure.json").exists()
    moved = [json.loads(l) for l in (directory / "curve/cost.jsonl").read_text().splitlines()]
    assert {r["event_id"] for r in moved} == {"curve0", "curve1"} and all(r["phase"] == "curve" for r in moved)
    receipt = core.read(directory / "curve/ledger-split.json")
    assert receipt["moved_event_ids"] == ["curve0", "curve1"] and receipt["rows"] == 4
    assert split.candidates(tmp_path) == []


def test_unpublished_and_held_branches_are_left_alone(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    unpublished = branch(tmp_path, "random_reduced", published=False)
    held = branch(tmp_path, "gated")
    assert split.candidates(tmp_path) == [held]
    with base.lease(held / ".task.lock"):
        assert "a worker holds this branch" in split.split(tmp_path, held, apply=True)
    assert not (held / "curve").exists() and not (unpublished / "curve").exists()
    out = subprocess.run([sys.executable, str(ROOT / "scripts/split_curve_ledger.py"), "--root", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert "would move 4 curve row(s)" in out
