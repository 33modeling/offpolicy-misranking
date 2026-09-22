import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "e5_export", Path(__file__).resolve().parents[1] / "scripts/export_e5_checkpoint_curves.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "runs"
    run = root / "math400-d400/s2"
    put(run / "experiment.json", {"drift": 400, "seed": 2, "steps": 100,
                                   "eval_k": 2, "eval_prompts": 4})
    put(run / "evaluation.json", {"val": [1, 2, 3, 4]})
    with (run / "downstream_results.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selector", "seed", "drift", "reward_before", "reward_after"])
        for arm in exporter.ARMS:
            writer.writerow([arm, 2, 400, .25, .35])
    return root, run


def arm(report, name="cached"):
    return report["experiments"][0]["arms"][name]


def test_auto_discovery_and_absolute_steps(run):
    root, directory = run
    put(directory / "passrate_beta/policy/curve-checkpoints/step-425/checkpoint_state.json",
        {"completed_steps": 425})
    put(directory / "passrate_beta/curve/step-425/eval.json",
        {"prompts": 4, "k": 2, "mean_reward": .30})
    report = exporter.export(root)
    assert report["experiments"][0]["dataset"] == "math400"
    assert [(p["updates"], p["reward"]) for p in arm(report)["points"]] == [(0, .25), (25, .3), (100, .35)]
    assert arm(report)["intermediate_points"] == 1
    assert not arm(report)["needs_evaluation"]


def test_checkpoint_only_is_not_a_reward(run):
    root, directory = run
    put(directory / "fresh_r/policy/checkpoint-000450/checkpoint_state.json", {"completed_steps": 450})
    (directory / "fresh_r/policy/checkpoint-000450/grpo_stats.jsonl").write_text('{"reward": 1, "step":450}\n')
    report = exporter.export(root)
    assert len(arm(report, "on_policy")["points"]) == 2
    assert len(arm(report, "on_policy")["needs_evaluation"]) == 1
    assert not report["training_rewards_used"]


def test_curve_summary_and_conflict(run):
    root, directory = run
    put(directory / "random/curve.json", {"k": 2, "points": {"450": {"updates": 50, "reward": .3}}})
    assert arm(exporter.export(root), "random")["intermediate_points"] == 1
    put(directory / "random/curve/step-450/eval.json", {"prompts": 4, "k": 2, "mean_reward": .9})
    result = arm(exporter.export(root), "random")
    assert result["points"] == []
    assert any("conflicting" in issue for issue in result["issues"])


def test_sharded_completed_evaluation(run):
    root, directory = run
    target = directory / "fresh_r/curve/step-450"
    for shard in range(4):
        binding = {"step": 450, "shard": shard, "k": 2}
        put(target / f"shard-{shard}.contract.json", binding)
        path = target / f"shard-{shard}.jsonl"
        path.write_text(''.join(json.dumps({"prompt_idx": shard, "reward": r})+'\n' for r in [0, 1]))
        put(target / f"shard-{shard}.done.json", {"binding": binding, "sha256": exporter.digest(path)})
    report = exporter.export(root)
    assert arm(report, "on_policy")["points"][1]["reward"] == .5
    (target / "shard-3.done.json").unlink()
    report = exporter.export(root)
    assert len(arm(report, "on_policy")["points"]) == 2
    assert report["issues"]


def test_different_k_is_not_averaged(run):
    root, directory = run
    put(directory / "random/curve.json", {"k": 1, "points": {"500": {"updates": 100, "reward": .4}}})
    points = arm(exporter.export(root), "random")["points"]
    assert {(p["eval_k"], p["reward"]) for p in points if p["updates"] == 100} == {(1, .4), (2, .35)}


def test_no_source_changes_or_overwrite(run, tmp_path):
    root, directory = run
    before = {str(p): p.read_bytes() for p in exporter.files(root)}
    report = exporter.export(root)
    out = tmp_path / "exports/curves.json"
    csv_path = exporter.write_outputs(report, out)
    assert len(list(csv.DictReader(csv_path.open()))) == 6
    assert json.loads(out.read_text())["interpolated_points"] == 0
    assert before == {str(p): p.read_bytes() for p in exporter.files(root)}
    with pytest.raises(ValueError, match="overwritten"):
        exporter.write_outputs(report, out)
    with pytest.raises(ValueError, match="outside"):
        exporter.write_outputs(report, root / "export.json")


def test_skip_quarantined_runs_and_symlinks(run, tmp_path):
    root, directory = run
    put(root / "quarantine/math400-d0/s0/experiment.json", {"drift": 0, "seed": 0})
    (root / "linked").symlink_to(directory, target_is_directory=True)
    assert len(exporter.export(root)["experiments"]) == 1


def test_no_runs_fails(tmp_path):
    with pytest.raises(ValueError, match="no .* found"):
        exporter.export(tmp_path)


def test_invalid_reward_fails_closed(run):
    root, directory = run
    put(directory / "random/curve.json", {"points": {"450": {"updates": 50, "reward": float("nan")}}})
    report = exporter.export(root)
    assert not arm(report, "random")["points"]
    assert report["issues"]


def test_adapter_only_archive_inventory(run):
    root, directory = run
    path = directory / "random/policy/curve-checkpoints/step-450/adapter_model.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not loaded by the exporter")
    result = arm(exporter.export(root), "random")
    assert result["checkpoints"][0]["adapter_present"]
    assert result["needs_evaluation"] == [str(path.parent.relative_to(root))]


def test_bash_entry_from_another_directory(run, tmp_path):
    root, directory = run
    wrapper = Path(__file__).resolve().parents[1] / "scripts/export_e5_checkpoint_curves.sh"
    out = tmp_path / "export with spaces/curves.json"
    result = subprocess.run(["bash", str(wrapper), "--root", str(root), "--out", str(out)],
                            cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "E5_EXPORT_PYTHON": sys.executable})
    assert result.returncode == 0, result.stderr
    assert "JSON:" in result.stdout
    assert len(json.loads(out.read_text())["experiments"]) == 1


def test_bash_entry_help(tmp_path):
    wrapper = Path(__file__).resolve().parents[1] / "scripts/export_e5_checkpoint_curves.sh"
    result = subprocess.run(["bash", str(wrapper), "--help"], cwd=tmp_path,
                            capture_output=True, text=True,
                            env={**os.environ, "E5_EXPORT_PYTHON": sys.executable})
    assert result.returncode == 0, result.stderr
    assert "--root" in result.stdout


def test_bash_entry_propagates_failure(tmp_path):
    wrapper = Path(__file__).resolve().parents[1] / "scripts/export_e5_checkpoint_curves.sh"
    result = subprocess.run(["bash", str(wrapper), "--root", str(tmp_path / "missing")],
                            capture_output=True, text=True,
                            env={**os.environ, "E5_EXPORT_PYTHON": sys.executable})
    assert result.returncode == 2
    assert "Export failed:" in result.stderr
