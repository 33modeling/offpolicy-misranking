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
    (point / "run_config.json").write_text(json.dumps({"seed": 0, "model": "fixture"}))
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


def test_discovery_deduplicates_symlinks_and_cycles(tmp_path):
    point = write_point(tmp_path)
    (tmp_path / "runs" / "source-link").symlink_to(point, target_is_directory=True)
    (point / "cycle").symlink_to(tmp_path / "runs", target_is_directory=True)
    assert audit.discover([tmp_path / "runs"]) == [point.resolve()]


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
    (broken / "scores_splithalf.json").write_text('{}')
    out = tmp_path / "exports" / "audit"
    assert audit.main([str(tmp_path / "runs"), "--out", str(out), "--frac", ".25"]) == 2
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
