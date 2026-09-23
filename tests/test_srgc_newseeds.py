"""New-root SR-GC decisions: exact subsets, frozen before training, read-only sources."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import srgc_newseeds as newseeds

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point(tmp_path, seed=3, drift=100, n=12, groups=8, vgroups=8, dim=5):
    run = tmp_path / f"matrix/s{seed}-d{drift}"
    run.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed * 1000 + drift)
    micro = {i: torch.randn(groups, dim, generator=generator) for i in range(n)}
    validation = torch.randn(vgroups, dim, generator=generator)
    torch.save(micro, run / "oracle_micro_groups.pt")
    torch.save(validation, run / "val_groups.pt")
    (run / "prompts.json").write_text(json.dumps({"train": [f"q{i}" for i in range(n)], "val": []}))
    (run / "run_config.json").write_text(json.dumps({"seed": seed, "drift": drift}))
    return run, micro, validation


def prepared(tmp_path, run, on, cached, name="out"):
    out = tmp_path / name
    (out / "subsets").mkdir(parents=True)
    (out / "experiment.json").write_text(json.dumps({"source_run": str(run)}))
    hashes = {}
    for arm, ids in (("fresh_r", on), ("passrate_beta", cached), ("random", [0, 1])):
        path = out / "subsets" / f"subset-{arm}.json"
        path.write_text(json.dumps({"selected_idx": ids, "k": len(ids)}))
        hashes[arm] = sha(path)
    (out / "subsets_hashes.json").write_text(json.dumps(hashes))
    return out


def manual(micro, validation, on, cached):
    full = torch.stack([micro[i].double() for i in range(len(micro))])
    q, vq = full.shape[1] // 4, validation.shape[0] // 4
    halves = []
    for start, vstart in ((2 * q, 2 * vq), (3 * q, 3 * vq)):
        dots = full[:, start:start + q].mean(1) @ validation.double()[vstart:vstart + vq].mean(0)
        halves.append(float(dots[on].mean() - dots[cached].mean()))
    return halves


def test_contrast_uses_the_trained_subsets_and_the_paper_equation(tmp_path):
    run, micro, validation = point(tmp_path)
    on, cached = [0, 2, 4, 6], [1, 2, 5, 7]
    value = newseeds.contrast(run, {"fresh_r": on, "passrate_beta": cached})
    d_a, d_b = manual(micro, validation, on, cached)
    assert value["d_a"] == pytest.approx(d_a) and value["d_b"] == pytest.approx(d_b)
    assert value["d"] == pytest.approx((d_a + d_b) / 2)
    assert value["selector"] == ("cached" if value["d"] < 0 else "on_policy")


def test_zero_contrast_retains_on_policy(tmp_path):
    run, _, _ = point(tmp_path)
    same = [0, 1, 2, 3]
    assert newseeds.contrast(run, {"fresh_r": same, "passrate_beta": same})["selector"] == "on_policy"


def test_freeze_writes_once_before_training_and_verifies_afterwards(tmp_path):
    run, _, _ = point(tmp_path)
    out = prepared(tmp_path, run, [0, 2, 4, 6], [1, 3, 5, 7])
    decision = tmp_path / "root/decisions/s3-d100.json"
    first = newseeds.freeze(run, out, decision)
    assert first["prospective"] is True and first["seed"] == 3 and first["drift"] == 100
    assert first["sets"] == {"fresh_r": [0, 2, 4, 6], "passrate_beta": [1, 3, 5, 7]}
    saved = decision.read_bytes()
    (out / "fresh_r" / "policy").mkdir(parents=True)
    assert newseeds.freeze(run, out, decision) == first
    assert decision.read_bytes() == saved


def test_decision_frozen_after_training_started_is_marked_not_prospective(tmp_path):
    run, _, _ = point(tmp_path)
    out = prepared(tmp_path, run, [0, 2, 4, 6], [1, 3, 5, 7])
    (out / "passrate_beta" / "evaluation").mkdir(parents=True)
    assert newseeds.freeze(run, out, tmp_path / "d.json")["prospective"] is False


@pytest.mark.parametrize("change", ["subset", "gradients", "other_run"])
def test_changed_inputs_are_rejected(tmp_path, change):
    run, micro, validation = point(tmp_path)
    out = prepared(tmp_path, run, [0, 2, 4, 6], [1, 3, 5, 7])
    decision = tmp_path / "d.json"
    newseeds.freeze(run, out, decision)
    if change == "subset":
        (out / "subsets/subset-fresh_r.json").write_text(json.dumps({"selected_idx": [0, 2, 4, 8], "k": 4}))
        match = "prepared subset changed"
    elif change == "gradients":
        torch.save({i: value + 1 for i, value in micro.items()}, run / "oracle_micro_groups.pt")
        match = "inputs changed"
    else:
        (out / "experiment.json").write_text(json.dumps({"source_run": str(tmp_path / "elsewhere")}))
        match = "another source point"
    with pytest.raises(ValueError, match=match):
        newseeds.freeze(run, out, decision)


def test_copy_test_never_writes_the_source(tmp_path):
    source = tmp_path / "e5/test-math500-d100.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"test": [{"question": "1+1", "answer": "2"}]}))
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    dest = tmp_path / "new/inputs/test-math500-d100.json"
    newseeds.copy_test(source, dest)
    newseeds.copy_test(source, dest)
    assert dest.read_bytes() == source.read_bytes()
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match="own copy"):
        newseeds.copy_test(source, source)
    dest.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        newseeds.copy_test(source, dest)


def write_results(out, on_after, cached_after, random_after=.27, low=-.01, high=.02):
    fields = ["dataset", "seed", "drift", "selector", "reward_before", "reward_after",
              "difference_vs_fresh", "difference_lower", "difference_upper", "difference_vs_random"]
    with (out / "downstream_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for selector, after in (("random", random_after), ("passrate_beta", cached_after), ("fresh_r", on_after)):
            writer.writerow({"dataset": "math500", "seed": 3, "drift": 0, "selector": selector,
                             "reward_before": .25, "reward_after": after,
                             "difference_vs_fresh": after - on_after if selector == "passrate_beta" else "",
                             "difference_lower": low if selector == "passrate_beta" else "",
                             "difference_upper": high if selector == "passrate_beta" else "",
                             "difference_vs_random": after - random_after})


def test_results_score_frozen_decisions_against_baselines(tmp_path):
    root = tmp_path / "root"
    states = []
    for seed, drift in ((3, 0), (3, 400)):
        run, _, _ = point(tmp_path, seed=seed, drift=drift)
        out = prepared(tmp_path, run, [0, 2, 4, 6], [1, 3, 5, 7], name=f"out-{seed}-{drift}")
        states.append((out, newseeds.freeze(run, out, root / f"decisions/s{seed}-d{drift}.json")))
    (out0, d0), (out400, d400) = states
    # Make the observed ordering agree with d0's decision and disagree with d400's.
    on0, sr0 = (.30, .28) if d0["selector"] == "on_policy" else (.28, .30)
    on4, sr4 = (.30, .32) if d400["selector"] == "on_policy" else (.32, .30)
    write_results(out0, on0, sr0)
    write_results(out400, on4, sr4)
    data = newseeds.results(root)
    assert data["frozen"] == data["measured"] == 2 and data["prospective_frozen"] == 2
    by_rule = {row["rule"]: row for row in data["summary"]}
    assert by_rule["SR-GC"]["agreement"] == 1 and by_rule["SR-GC"]["decided_states"] == 2
    assert by_rule["SR-GC"]["mean_selected_reward_pct"] == pytest.approx(100 * (.30 + .30) / 2)
    stage = by_rule["stage (step 0 on-policy, later cached SR)"]
    assert stage["mean_selected_reward_pct"] == pytest.approx(100 * (on0 + sr4) / 2)
    text = newseeds.table(data)
    assert "s3-d0" in text and "s3-d400" in text and "always cached SR" in text


def test_pending_outcomes_are_reported_without_scores(tmp_path):
    run, _, _ = point(tmp_path)
    out = prepared(tmp_path, run, [0, 2, 4, 6], [1, 3, 5, 7])
    newseeds.freeze(run, out, tmp_path / "root/decisions/s3-d100.json")
    data = newseeds.results(tmp_path / "root")
    assert data["measured"] == 0 and data["states"][0]["match"] is None
    assert "pending" in newseeds.table(data)


def test_launcher_rejects_unknown_modes_and_extra_options():
    script = ROOT / "scripts/run_srgc_newseeds.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    for argv in (["bogus"], ["run", "--seeds", "5"]):
        result = subprocess.run(["bash", str(script), *argv], capture_output=True, text=True, timeout=30)
        assert result.returncode == 2, result.stdout + result.stderr
