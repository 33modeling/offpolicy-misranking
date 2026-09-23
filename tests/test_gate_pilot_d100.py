"""CPU checks for the d=100 gate pilot comparison (scripts/gate_pilot_d100.py)."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import evidence_downstream as ed

sys.path.insert(0, str(Path(ed.ROOT) / "scripts"))
import gate_pilot_d100 as pilot  # noqa: E402

SCRIPT = Path(ed.ROOT) / "scripts/run_gate_pilot_d100.sh"
CONTRACT = {"source_run": "/runs/s{seed}-d100", "source_hashes": {"run_config.json": "a"},
            "evaluation_payload_sha256": "e", "eval_seed": 1, "eval_k": 8, "eval_prompts": 300,
            "steps": 100, "drift": 100}
SUBSETS = {"random": "r", "passrate_beta": "p", "fresh_r": "f"}


def seed_dir(path: Path, seed: int, *, arms=(), **changes) -> Path:
    contract = {**CONTRACT, "source_run": CONTRACT["source_run"].format(seed=seed), "seed": seed, **changes}
    path.mkdir(parents=True, exist_ok=True)
    ed.atomic_json(path / "experiment.json", contract)
    ed.atomic_json(path / "subsets_hashes.json", SUBSETS)
    for arm in arms:
        for shard in range(4):
            done = path / arm / "evaluation" / f"shard-{shard}.done.json"
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text("{}")
    return path


def layout(tmp_path, *, new_arms=("gate_passrate",), e5_arms=("random", "passrate_beta"), **changes):
    root, work = tmp_path / "gate-pilot-d100-v1", tmp_path / "work"
    for seed in (0, 1, 2):
        seed_dir(work / f"runs/e5-reduced/math500-d100/s{seed}", seed, arms=e5_arms)
        new = seed_dir(root / f"math500-d100/s{seed}", seed, arms=new_arms, **changes)
        (new / "gate_passrate").mkdir(exist_ok=True)
        ed.atomic_json(new / "gate_passrate" / "decision.json",
                       {"decision": "retain", "chosen_subset": "passrate_beta", "r_half": 0.61})
    return root, work


def test_check_requires_complete_unchanged_e5_controls(tmp_path):
    root, work = layout(tmp_path, e5_arms=("random",))
    with pytest.raises(ValueError, match="not fully evaluated"):
        pilot.check(root, work)
    root, work = layout(tmp_path / "ok")
    assert all("inputs identical" in note for note in pilot.check(root, work))


@pytest.mark.parametrize("key,value", [("evaluation_payload_sha256", "other"), ("eval_seed", 2),
                                       ("source_hashes", {"run_config.json": "b"})])
def test_cross_root_pairing_refuses_different_inputs(tmp_path, key, value):
    root, work = layout(tmp_path, **{key: value})
    with pytest.raises(ValueError, match=f"differ in {key}"):
        pilot.check(root, work)


def test_cross_root_pairing_refuses_different_subsets(tmp_path):
    root, work = layout(tmp_path)
    ed.atomic_json(root / "math500-d100/s1/subsets_hashes.json", {**SUBSETS, "passrate_beta": "x"})
    with pytest.raises(ValueError, match="passrate_beta subset differs"):
        pilot.check(root, work)


def test_results_use_table31_signs_and_bootstrap_seeds(tmp_path, monkeypatch):
    root, work = layout(tmp_path)
    means = {"gate_passrate": 0.30, "random": 0.29, "passrate_beta": 0.31}
    monkeypatch.setattr(ed, "evaluation_means", lambda out, arm: np.full(300, means[arm]))
    seeds = []
    monkeypatch.setattr(ed, "paired_interval", lambda values, seed: (seeds.append(seed), (values.min(), values.max()))[1])
    rows, pending = pilot.results(root, work)
    assert pending == [] and len(rows) == 3
    row = rows[0]
    assert row["vs_random"] == pytest.approx(1.0) and row["vs_cached"] == pytest.approx(-1.0)
    assert row["vs_cached_ci"] == pytest.approx((-1.0, -1.0))
    assert seeds[:2] == [7, 11]
    assert "pending: none" in pilot.text(rows, pending)


def test_unfinished_seeds_are_pending_not_compared(tmp_path, monkeypatch):
    root, work = layout(tmp_path, new_arms=())
    monkeypatch.setattr(ed, "evaluation_means", lambda out, arm: pytest.fail("must not read"))
    rows, pending = pilot.results(root, work)
    assert rows == [] and pending == ["s0", "s1", "s2"]


def test_launcher_is_valid_and_rejects_extra_options():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    bad = subprocess.run(["bash", str(SCRIPT), "run", "--force"], text=True, capture_output=True)
    assert bad.returncode == 2
    bad = subprocess.run(["bash", str(SCRIPT), "train"], text=True, capture_output=True)
    assert bad.returncode == 2
