import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cached_sr_gradient_audit as audit


def inputs():
    # Ranking favors prompt 0; independent reference favors cached-SR prompt 1.
    micro = {i: torch.zeros(8, 2) for i in range(4)}
    micro[0][:2] = torch.tensor([1., 0.])
    micro[1][:2] = torch.tensor([0., 1.])
    micro[2][:2] = torch.tensor([-1., 0.])
    micro[3][:2] = torch.tensor([0., -1.])
    for i, value in enumerate([1., 3., 0., -1.]):
        micro[i][4:, 0] = value
    return micro, torch.tensor([[1., 0.]] * 8), {0: 0., 1: .5, 2: 1., 3: .25}


def write_point(tmp_path):
    point = tmp_path / "runs" / "family-math400-s0" / "math400-d25"
    point.mkdir(parents=True)
    micro, val, rates = inputs()
    torch.save(micro, point / "oracle_micro_groups.pt")
    torch.save(val, point / "val_groups.pt")
    (point / "run_config.json").write_text(json.dumps({"seed": 0, "drift": 25, "model": "fixture"}))
    (point / "prompts.json").write_text(json.dumps({"train": list(range(4)), "val": list(range(8))}))
    with (point / "rollouts_behavior_train.jsonl").open("w") as handle:
        for i, p in rates.items():
            for j in range(4):
                handle.write(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": int(j < p * 4)}) + "\n")
    return point


def test_independent_reference_not_ranking_and_no_fake_h():
    result = audit.compare(*inputs(), seed=0, frac=.25)
    assert result["selectors"]["on_policy"]["selected_ids"] == [0]
    assert result["selectors"]["cached_sr"]["selected_ids"] == [1]
    assert result["on_minus_sr_projected_dot"] == -2
    assert result["selectors"]["cached_sr"]["dot_minus_uniform_expectation"] == 2.25
    assert result["h_gpu_seconds"] is None
    assert result["decision"] == "not_estimated"


def test_reference_cannot_change_selection():
    micro, val, rates = inputs()
    before = audit.compare(micro, val, rates, seed=0, frac=.25)
    for value in micro.values():
        value[4:] *= -20
    after = audit.compare(micro, val, rates, seed=0, frac=.25)
    assert after["on_minus_sr_projected_dot"] == 40
    for method in before["selectors"]:
        assert before["selectors"][method]["selected_ids"] == after["selectors"][method]["selected_ids"]


def test_magnitude_retained_and_identical_sets_cancel():
    micro, val, rates = inputs()
    a = audit.compare(micro, val, rates, seed=0, frac=.25)
    b = audit.compare({i: x * 7 for i, x in micro.items()}, val, rates, seed=0, frac=.25)
    assert b["on_minus_sr_projected_dot"] == 7 * a["on_minus_sr_projected_dot"]
    assert b["selectors"]["on_policy"]["reference_mean_cosine"] == a["selectors"]["on_policy"]["reference_mean_cosine"]
    full = audit.compare(micro, val, rates, seed=0, frac=1)
    assert full["on_minus_sr_projected_dot"] == 0


@pytest.mark.parametrize("kind", ["nan", "groups", "val", "ids", "rates"])
def test_bad_inputs_fail(kind):
    micro, val, rates = inputs()
    if kind == "nan":
        micro[0][0, 0] = float("nan")
    elif kind == "groups":
        micro = {i: x[:4] for i, x in micro.items()}
    elif kind == "val":
        val = val[:7]
    elif kind == "ids":
        micro["0"] = micro[0]
    else:
        rates[0] = float("nan")
    with pytest.raises(ValueError):
        audit.compare(micro, val, rates, seed=0)


def test_audit_no_outcomes_and_inputs_unchanged(tmp_path):
    point = write_point(tmp_path)
    (point / "eval-after.json").write_text("not even JSON: must never be read")
    before = {p: audit.digest(p) for p in point.iterdir()}
    a = audit.audit_point(point, frac=.25)
    assert before == {p: audit.digest(p) for p in point.iterdir()}
    (point / "eval-after.json").write_text('{"mean_reward": 1}')
    b = audit.audit_point(point, frac=.25)
    assert a == b
    assert a["model"] == "fixture"
    assert a["historical_availability_certified"] is False


def test_cache_duplicate_rejected(tmp_path):
    point = write_point(tmp_path)
    path = point / "rollouts_behavior_train.jsonl"
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        audit.audit_point(point)


def test_recorded_hash_mismatch_rejected(tmp_path):
    point = write_point(tmp_path)
    (point / "oracle_protocol.json").write_text(json.dumps({
        "generation_validation": {"artifact_sha256": {"prompts.json": "wrong"}}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.audit_point(point)


def test_corrupt_tensor_exported_as_error(tmp_path):
    point = write_point(tmp_path)
    (point / "oracle_micro_groups.pt").write_bytes(b"corrupt tensor")
    out = tmp_path / "exports"
    assert audit.main([str(point), "--out", str(out)]) == 2
    doc = json.loads((out / "data.json").read_text())
    assert not doc["results"]
    assert len(doc["errors"]) == 1


def test_explicit_source_does_not_search_children_or_cycles(tmp_path):
    point = write_point(tmp_path)
    (tmp_path / "runs" / "source-link").symlink_to(point, target_is_directory=True)
    (point / "cycle").symlink_to(tmp_path / "runs", target_is_directory=True)
    result = audit.discover([point, tmp_path / "runs/source-link"])
    assert len(result) == 1
    assert result[0]["point"] == str(point.resolve())
    with pytest.raises(ValueError, match="recursive search disabled"):
        audit.discover([tmp_path / "runs"])


def test_missing_seed_not_silently_zero(tmp_path):
    point = write_point(tmp_path)
    (point / "run_config.json").write_text('{}')
    assert audit.audit_point(point)["seed"] == 0  # family path explicitly says s0
    new = tmp_path / "unknown"
    point.rename(new)
    with pytest.raises(ValueError, match="seed missing"):
        audit.audit_point(new)


def test_cli_output_protection_and_partial_status(tmp_path):
    point = write_point(tmp_path)
    with pytest.raises(SystemExit):
        audit.main([str(point), "--out", str(point / "output")])
    broken = tmp_path / "runs" / "broken"
    broken.mkdir()
    (broken / "run_config.json").write_text('{}')
    (broken / "prompts.json").write_text('{}')
    out = tmp_path / "exports" / "audit"
    assert audit.main([str(point), str(broken), "--out", str(out), "--frac", ".25"]) == 2
    doc = json.loads((out / "data.json").read_text())
    assert len(doc["results"]) == len(doc["errors"]) == 1
    assert doc["results"][0]["h_gpu_seconds"] is None
    assert (out / "comparison.csv").is_file()
    with pytest.raises(SystemExit):
        audit.main([str(point), "--out", str(out)])


def test_wrapper(tmp_path):
    point = write_point(tmp_path)
    out = tmp_path / "export"
    process = subprocess.run(["bash", str(ROOT / "scripts/run_cached_sr_gradient_audit.sh"),
                              str(point), "--out", str(out)],
                             env={**os.environ, "SR_AUDIT_PYTHON": sys.executable}, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "No H predictions generated" in process.stdout


def write_experiment(tmp_path):
    point = write_point(tmp_path)
    experiment = tmp_path / "runs/e5-reduced/math400-d25/s0"
    experiment.mkdir(parents=True)
    (experiment / "experiment.json").write_text(json.dumps({
        "schema": audit.E5_SCHEMA, "source_run": str(point), "seed": 0, "drift": 25,
        "source_hashes": {"run_config.json": audit.digest(point / "run_config.json")}}))
    for arm in audit.ARMS:
        for directory, step, name in (("", 525, "policy_train.json"),
                                      ("checkpoint-000125", 125, "checkpoint_state.json"),
                                      ("curve-checkpoints/step-425", 425, "checkpoint_state.json")):
            path = experiment / arm / "policy" / directory
            path.mkdir(parents=True, exist_ok=True)
            (path / name).write_text(json.dumps({"completed_steps": step}))
            (path / "adapter_model.safetensors").write_bytes(b"not loaded")
    return point, experiment


def test_e5_discovery_uses_contract_not_recursive_score_glob(tmp_path):
    point, experiment = write_experiment(tmp_path)
    root = tmp_path / "runs/e5-reduced"
    for unrelated in ("archive/math400-d25/s0", "quarantine/math400-d25/s0", "math400-d25-old/s0", "smoke"):
        directory = root / unrelated
        directory.mkdir(parents=True)
        (directory / "scores_splithalf.json").write_text('{}')
        (directory / "experiment.json").write_text('{"schema":"wrong"}')
    result = audit.discover([root])
    assert len(result) == 1
    assert result[0]["point"] == str(point)
    assert result[0]["experiment"] == str(experiment)
    assert len(result[0]["checkpoints"]) == 9
    assert {c["step"] for c in result[0]["checkpoints"]} == {125, 425, 525}
    assert not result[0]["missing_scoring_inputs"]


def test_default_e5_root_and_list_only_does_not_load_tensors(tmp_path, monkeypatch):
    point, experiment = write_experiment(tmp_path)
    monkeypatch.setenv("OM_WORK", str(tmp_path))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("inventory loaded tensors"))
    out = tmp_path / "exports/inventory"
    assert audit.main(["--list-only", "--out", str(out)]) == 0
    doc = json.loads((out / "data.json").read_text())
    assert doc["roots"] == [str(tmp_path / "runs/e5-reduced")]
    assert doc["targets"][0]["point"] == str(point)
    assert doc["targets"][0]["experiment"] == str(experiment)
    assert doc["results"] == []


def test_registered_missing_gradients_keeps_checkpoint_inventory(tmp_path):
    point, experiment = write_experiment(tmp_path)
    (point / "oracle_micro_groups.pt").unlink()
    out = tmp_path / "exports"
    assert audit.main([str(experiment), "--out", str(out)]) == 2
    doc = json.loads((out / "data.json").read_text())
    assert len(doc["targets"][0]["checkpoints"]) == 9
    assert doc["targets"][0]["missing_scoring_inputs"] == ["oracle_micro_groups.pt"]
    assert len(doc["errors"]) == 1


def test_registered_source_hash_mismatch_not_replaced_by_another_point(tmp_path):
    point, experiment = write_experiment(tmp_path)
    config = point / "run_config.json"
    config.write_text(config.read_text() + " ")
    target = audit.discover([experiment])[0]
    with pytest.raises(ValueError, match="experiment-bound source hash"):
        audit.check_target(target)


def test_wrong_experiment_identity_rejected(tmp_path):
    _, experiment = write_experiment(tmp_path)
    path = experiment / "experiment.json"
    value = json.loads(path.read_text())
    value["seed"] = 100
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity differs"):
        audit.discover([experiment])


def test_explicit_group_and_source_output_protection(tmp_path):
    point, experiment = write_experiment(tmp_path)
    assert audit.discover([experiment.parent])[0]["point"] == str(point)
    with pytest.raises(SystemExit):
        audit.main([str(experiment), "--out", str(point / "exports")])
