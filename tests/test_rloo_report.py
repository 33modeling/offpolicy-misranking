"""Partial paper exports must not alter training or canonical completion."""

from pathlib import Path
import json
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rloo_report as reporting
from test_rloo_experiment import fixture


@pytest.fixture
def measured(tmp_path, monkeypatch):
    out = tmp_path / "math500-d0/s0"
    out.mkdir(parents=True)
    (out / "experiment.json").write_text("{}")
    monkeypatch.setattr(reporting.experiment, "validate", lambda out: (
        {"eval_n": 8, "source": {"seed": 0, "drift": 0}}, {}))

    def rows(out, arm, shard):
        return [{"prompt_idx": i, "reward": {"before": 0., "random": .25,
                 "passrate_beta": .5, "fresh_r": .75}[arm]}
                for i in range(shard * 2, (shard + 1) * 2)]

    monkeypatch.setattr(reporting.experiment, "checked_rows", rows)
    return out


def seal(out, arm, shards=range(4)):
    directory = out / arm / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        (directory / f"shard-{shard}.done.json").write_text("{}")


def test_missing_before_keeps_cached_comparison(measured):
    seal(measured, "passrate_beta")
    seal(measured, "fresh_r")
    result = reporting.point_report(measured)
    assert result["status"] == "incomplete"
    assert result["missing_arms"] == ["before", "random"]
    fresh = result["rows"][-1]
    assert fresh["vs_passrate_beta"]["mean"] == .25
    assert "vs_before" not in fresh and "vs_random" not in fresh
    assert fresh["missing_references"] == ["before", "random"]
    assert not (measured / "results.json").exists()


def test_partial_shards_are_explicit_and_unsealed_rows_ignored(measured):
    seal(measured, "before", [0])
    (measured / "before/evaluation/shard-1.jsonl").write_text("unfinished")
    result = reporting.point_report(measured)
    partial = result["evaluations"][0]
    assert partial["measured_prompts"] == 2
    assert partial["observed_mean_reward"] == 0.
    assert partial["missing_shards"] == [1, 2, 3]
    assert not partial["complete"] and result["rows"] == []
    assert result["evaluations"][1]["observed_mean_reward"] is None


def test_complete_matches_canonical_report(measured):
    for arm in ("before", *reporting.experiment.ARMS):
        seal(measured, arm)
    expected = reporting.experiment.report(measured)
    before = (measured / "results.json").read_bytes()
    actual = reporting.point_report(measured)
    assert actual["status"] == "complete"
    assert [{k: v for k, v in row.items() if k != "missing_references"}
            for row in actual["rows"]] == expected["rows"]
    assert (measured / "results.json").read_bytes() == before


def test_invalid_point_does_not_hide_other_points(measured, monkeypatch):
    other = measured.parent / "s1"
    seal(other, "random")
    (other / "experiment.json").write_text("{}")
    seal(measured, "fresh_r")
    original = reporting.experiment.checked_rows

    def checked(out, arm, shard):
        if out == measured:
            raise ValueError("evaluation seal mismatch")
        return original(out, arm, shard)

    monkeypatch.setattr(reporting.experiment, "checked_rows", checked)
    result = reporting.report(measured.parent.parent)
    assert result["points"][0]["status"] == "invalid"
    assert result["points"][0]["rows"] == []
    assert result["points"][1]["rows"][0]["arm"] == "random"
    assert result["points"][2]["status"] == "unprepared"
    assert not result["complete"]


def test_real_prepared_point_without_evaluation_is_exportable(tmp_path):
    _, out, _ = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    result = reporting.point_report(out)
    assert result["missing_arms"] == ["before", *reporting.experiment.ARMS]
    assert result["rows"] == []
    assert before == {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}


def test_launcher_exports_incomplete_matrix_without_gpu(tmp_path):
    import os
    root = tmp_path / "rloo"
    root.mkdir()
    process = subprocess.run(["bash", "scripts/run_rloo.sh", "report"],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True,
        env={**os.environ, "HOME": str(tmp_path), "RLOO_ROOT": str(root), "RLOO_PYTHON": sys.executable})
    assert process.returncode == 0, process.stderr
    exports = list(tmp_path.glob("rloo-results.txt"))
    assert len(exports) == 1
    assert "unprepared" in exports[0].read_text()
    assert "TXT saved:" in process.stdout


def test_missing_root_is_not_created(tmp_path):
    with pytest.raises(ValueError, match="no RLOO root"):
        reporting.report(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def old_display_contract(out):
    ed = reporting.experiment.ed
    contract = ed.read(out / "experiment.json")
    contract["code_hashes"]["src/matrix_status.py"] = reporting.REPORT_DISPLAY_UPGRADES[
        "src/matrix_status.py"][0]
    ed.atomic_json(out / "experiment.json", contract)
    return contract


def measured_policy(out, arm, reward):
    from test_grpo_policy import _policy_artifact
    ed = reporting.experiment.ed
    contract, config = reporting.experiment.validate(out)
    policy = out / arm / "policy"
    _policy_artifact(policy, objective="rloo", completed_steps=100)
    manifest = ed.read(policy / "policy_train.json")
    manifest.update(
        optimizer_sha256=ed.digest(policy / "optimizer.pt"),
        grpo_stats_sha256=ed.digest(policy / "grpo_stats.jsonl"),
        base_model=str(Path(config["model"]).resolve()), seed=config["seed"],
        max_new_tokens=config["max_new_tokens"], prompt_format=config["prompt_format"],
        config=ed._expected_config(config), samples_per_step=32,
        prompts_sha256=ed.digest(out / "subsets" / f"subset-{arm}.json"))
    ed.atomic_json(policy / "policy_train.json", manifest)
    target = out / arm / "evaluation"
    target.mkdir()
    for shard in range(4):
        binding, _, indices = reporting.experiment.binding(out, arm, shard)
        path = target / f"shard-{shard}.jsonl"
        path.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j,
            "reward": reward}) + "\n" for i in indices for j in range(contract["eval_k"])))
        ed.atomic_json(target / f"shard-{shard}.done.json",
                       {"binding": binding, "rollouts_sha256": ed.digest(path)})


def test_reviewed_status_drift_exports_real_sealed_comparison_read_only(tmp_path):
    _, out, _ = fixture(tmp_path)
    old_display_contract(out)
    with pytest.raises(ValueError, match="code changed since preparation: src/matrix_status.py"):
        reporting.frozen_experiment.validate(out)
    measured_policy(out, "passrate_beta", .5)
    measured_policy(out, "fresh_r", .75)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = reporting.point_report(out)
    assert result["status"] == "incomplete"
    assert result["missing_arms"] == ["before", "random"]
    fresh = result["rows"][-1]
    assert fresh["mean_reward"] == .75
    assert fresh["vs_passrate_beta"] == {"mean": .25, "lower": .25, "upper": .25}
    assert result["report_display_code_changes"]["src/matrix_status.py"]["frozen_sha256"] == (
        reporting.REPORT_DISPLAY_UPGRADES["src/matrix_status.py"][0])
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="code changed since preparation"):
        reporting.frozen_experiment.training_command(out, "fresh_r")


def test_results_launcher_exports_reviewed_drift_to_one_txt(tmp_path):
    import os
    _, prepared, _ = fixture(tmp_path)
    out = tmp_path / "matrix/math500-d0/s0"
    out.parent.mkdir(parents=True)
    prepared.rename(out)
    old_display_contract(out)
    measured_policy(out, "fresh_r", .75)
    process = subprocess.run(["bash", "scripts/run_rloo.sh", "results"],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True,
        env={**os.environ, "HOME": str(tmp_path), "RLOO_ROOT": str(out.parent.parent),
             "RLOO_PYTHON": sys.executable})
    assert process.returncode == 0, process.stderr
    exports = list(tmp_path.glob("rloo-results*.txt"))
    assert len(exports) == 1
    text = exports[0].read_text()
    assert "0\t0\tincomplete\tfresh_r\t0.75\t" in text
    assert '"report_display_code_changes":{"src/matrix_status.py"' in text
    assert not (out / "results.json").exists()


@pytest.mark.parametrize("name", ["src/train_policy_rloo.py", "src/train_policy_grpo.py",
                                 "src/evidence_downstream.py", "src/matrix_status.py"])
def test_report_rejects_unreviewed_code_even_with_display_upgrade(tmp_path, name):
    _, out, _ = fixture(tmp_path)
    contract = old_display_contract(out)
    contract["code_hashes"][name] = "0" * 64
    reporting.experiment.ed.atomic_json(out / "experiment.json", contract)
    with pytest.raises(ValueError, match=f"code changed since preparation: {name}"):
        reporting.point_report(out)


def test_report_rejects_unknown_runtime_display_version(tmp_path, monkeypatch):
    _, out, _ = fixture(tmp_path)
    old_display_contract(out)
    ed = reporting.experiment.ed
    digest = ed.digest
    monkeypatch.setattr(ed, "digest", lambda p: "0" * 64
                        if p == reporting.experiment.ROOT / "src/matrix_status.py" else digest(p))
    with pytest.raises(ValueError, match="code changed since preparation: src/matrix_status.py"):
        reporting.point_report(out)


@pytest.mark.parametrize("artifact,reason", [
    ("subsets/subset-fresh_r.json", "prepared input changed"),
    ("fresh_r/evaluation/shard-0.jsonl", "evaluation seal mismatch"),
    ("fresh_r/policy/optimizer.pt", "hash"),
])
def test_display_upgrade_keeps_data_policy_and_seal_validation(tmp_path, artifact, reason):
    _, out, _ = fixture(tmp_path)
    old_display_contract(out)
    measured_policy(out, "fresh_r", .75)
    (out / artifact).write_text("tampered")
    with pytest.raises(ValueError, match=reason):
        reporting.point_report(out)
