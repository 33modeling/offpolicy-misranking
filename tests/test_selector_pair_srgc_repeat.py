import json
import sys

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair_results as results
import selector_pair_srgc as srgc
import selector_pair_srgc_repeat as repeat
import export_selector_pair_srgc_inventory as inventory_export
import selector_pair_srgc_score as score
from test_selector_pair_gpu import fake_study
from test_selector_pair_parallel import study
from test_selector_pair_srgc import current_policy, srgc_study


@pytest.fixture
def saved_path(tmp_path, srgc_study):
    protocol, _, _, states = srgc_study
    srgc.activate(tmp_path, protocol)
    identity, entries = states(tmp_path, 3, 50)
    initial = srgc.measure(tmp_path / "sr-gc/s3-t50", identity, entries["on_policy"], list("0123"), protocol)
    out = entries["on_policy"][1]
    reference = core.read(tmp_path / "sr-gc/s3-t50/reference.json")
    prompts = core.read(reference["prompts"])
    subset = out / "subsets/subset-selection_full.json"
    core.atomic_json(subset, {"train": [prompts["train"][i] for i in initial["sets"]["on_policy"]]})

    def checkpoint(step, *, projected_d=None):
        path = out / f"selection_full/policy/curve-checkpoints/step-{step}"
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(f"checkpoint-{step}".encode())
        core.atomic_json(path / "checkpoint_state.json", {
            "schema": "offpolicy-grpo-checkpoint/v2", "seed": 3, "start_step": 50,
            "completed_steps": step, "training_objective": "grpo", "world_size": 4,
            "resume_adapter": reference["parent"], "prompts_sha256": base.digest(subset),
            "adapter_sha256": base.digest(path / "adapter_model.safetensors")})
        if projected_d is not None:
            expected, _ = repeat.checkpoint_reference(tmp_path, 3, 50, step, path, initial, 25)
            directory = repeat.output_dir(tmp_path, 3, 50, 25) / f"step-{step}"
            core.atomic_json(directory / "reference.json", expected)
            for stage in score.STAGES:
                ids = score.indices(expected, prompts, stage)
                for shard in range(4):
                    values = {str(i): ([1., 0.] if stage.startswith("validation") else
                              [projected_d if i in initial["sets"]["on_policy"] else 0., 0.])
                              for i in ids[len(ids)*shard//4:len(ids)*(shard+1)//4]}
                    payload = directory / f"{stage}-{shard}.json"
                    core.atomic_json(payload, values)
                    core.atomic_json(directory / f"{stage}-{shard}.done.json", {
                        "reference_sha256": base.digest(directory / "reference.json"),
                        "stage": stage, "shard": shard, "sha256": base.digest(payload)})
        return path
    return initial, checkpoint, protocol


def test_rechecks_on_then_locks_sr_and_never_reads_later_d(tmp_path, saved_path, monkeypatch):
    initial, checkpoint, _ = saved_path
    checkpoint(75, projected_d=2.)
    checkpoint(100, projected_d=-3.)
    later = checkpoint(125, projected_d=9.)
    (later / "checkpoint_state.json").write_text("corrupt later data must not be read")
    monkeypatch.setattr(repeat, "measure_point", lambda *a: pytest.fail("export cannot launch GPU work"))
    result = repeat.scan_state(tmp_path, initial, 25)
    assert [(d["step"], d["d"]) for d in result["decisions"]] == [(50, 1.), (75, 2.), (100, -3.)]
    assert result["first_sr_step"] == 100 and result["first_sr_updates"] == 50
    assert result["status"] == "sr_locked" and result["next_check_step"] is None
    assert result["executed_switch"] is False and result["switched_policy_rewards"] is None


def test_zero_retains_on_and_initial_sr_never_scans(tmp_path, saved_path, monkeypatch):
    initial, checkpoint, _ = saved_path
    checkpoint(75, projected_d=0.)
    row = repeat.scan_state(tmp_path, initial, 25)
    assert row["first_sr_step"] is None and row["on_through_step"] == 75
    assert row["next_check_step"] == 100
    negative = {**initial, **srgc.choose(-1., -1.)}
    monkeypatch.setattr(repeat, "inventory", lambda *a: pytest.fail("SR is absorbing"))
    assert repeat.scan_state(tmp_path, negative, 25)["first_sr_step"] == 50


def test_targeted_recovery_stops_at_requested_checkpoint(tmp_path, saved_path):
    initial, checkpoint, _ = saved_path
    checkpoint(75, projected_d=2.)
    later = checkpoint(100, projected_d=-3.)
    (later / "checkpoint_state.json").write_text("must not inspect later checkpoint")
    row = repeat.scan_state(tmp_path, initial, 25, through_step=75)
    assert [decision["step"] for decision in row["decisions"]] == [50, 75]
    assert row["status"] == "on_through_checked_step"
    assert row["next_check_step"] == 100


def test_missing_checkpoint_or_gradient_is_not_on_and_not_skipped(tmp_path, saved_path):
    initial, checkpoint, _ = saved_path
    checkpoint(100, projected_d=-3.)
    row = repeat.scan_state(tmp_path, initial, 25)
    assert row["status"] == "awaiting_checkpoint" and row["first_sr_step"] is None
    assert row["pending"][0]["step"] == 75
    checkpoint(75)
    row = repeat.scan_state(tmp_path, initial, 25)
    assert row["status"] == "awaiting_projections" and len(row["decisions"]) == 1
    assert row["pending"][0]["step"] == 75


def test_inventory_scans_saved_d_after_missing_step_without_writing(tmp_path, saved_path, monkeypatch):
    initial, checkpoint, _ = saved_path
    checkpoint(75)
    checkpoint(100, projected_d=-3.)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(repeat, "measure_point", lambda *a: pytest.fail("inventory must not measure"))
    row = inventory_export.scan_state(tmp_path, initial)
    assert [(p["step"], p["status"]) for p in row["points"]] == [
        (50, "measured"), (75, "projection_missing"), (100, "measured")]
    assert "validation-b-0.json" in row["points"][1]["missing_files"]
    assert row["points"][2]["d"] == -3.
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_no_outcome_inputs_no_source_writes_and_no_other_t_join(tmp_path, saved_path):
    initial, checkpoint, _ = saved_path
    checkpoint(75, projected_d=2.)
    checkpoint(100, projected_d=-3.)
    other = repeat.trajectory(tmp_path, 3, 25) / "selection_full/policy/curve-checkpoints/step-75"
    other.mkdir(parents=True)
    (other / "checkpoint_state.json").write_text("must not read another trajectory")
    (other / "adapter_model.safetensors").write_bytes(b"wrong branch")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    row = repeat.scan_state(tmp_path, initial, 25)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    core.atomic_json(tmp_path / "report.json", {"future_reward": -1e99})
    core.atomic_json(repeat.trajectory(tmp_path, 3, 50) / "selection_full/result.json", {"reward": 1e99})
    assert repeat.scan_state(tmp_path, initial, 25) == row


@pytest.mark.parametrize("damage", ["adapter", "parent", "subset", "shard", "reference"])
def test_wrong_checkpoint_or_projection_is_rejected(tmp_path, saved_path, damage):
    initial, checkpoint, _ = saved_path
    path = checkpoint(75, projected_d=-3.)
    directory = repeat.output_dir(tmp_path, 3, 50, 25) / "step-75"
    if damage == "adapter":
        (path / "adapter_model.safetensors").write_bytes(b"other weights")
    elif damage in {"parent", "subset"}:
        value = core.read(path / "checkpoint_state.json")
        value["resume_adapter" if damage == "parent" else "prompts_sha256"] = "wrong"
        core.atomic_json(path / "checkpoint_state.json", value)
    elif damage == "shard":
        core.atomic_json(directory / "candidate-a-0.json", {"0": [9., 0.]})
    else:
        value = core.read(directory / "reference.json")
        value["sampling_seed"] += 1
        core.atomic_json(directory / "reference.json", value)
    row = repeat.scan_state(tmp_path, initial, 25)
    assert row["status"] == "invalid" and row["first_sr_step"] is None and row["errors"]


def test_measure_is_explicit_resumes_existing_shards_and_no_training(tmp_path, saved_path, monkeypatch):
    initial, checkpoint, protocol = saved_path
    path = checkpoint(75, projected_d=-3.)
    directory = repeat.output_dir(tmp_path, 3, 50, 25) / "step-75"
    expected, contract = repeat.checkpoint_reference(tmp_path, 3, 50, 75, path, initial, 25)
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("completed projection shards must be reused"))
    repeat.measure_point(directory, expected, contract, protocol, list("0123"), 14400.)
    with pytest.raises(ValueError, match="distinct"):
        repeat.measure_point(directory, expected, contract, protocol, ["0"] * 4, 14400.)


def test_missing_shards_launch_only_score_worker_with_separate_cost(tmp_path, saved_path, monkeypatch):
    initial, checkpoint, protocol = saved_path
    path = checkpoint(75)
    directory = repeat.output_dir(tmp_path, 3, 50, 25) / "step-75"
    expected, contract = repeat.checkpoint_reference(tmp_path, 3, 50, 75, path, initial, 25)
    calls = []
    monkeypatch.setattr(base, "meter", lambda *a, **k: calls.append((a, k)))
    repeat.measure_point(directory, expected, contract, protocol, list("0123"), 14400.)
    assert len(calls) == 4
    for args, kwargs in calls:
        assert args[0] == directory and kwargs["ledger"] == "research"
        assert kwargs["timeout"] == 3600.
        assert len(kwargs["commands"]) == 4
        for command, device in kwargs["commands"]:
            assert command[1].endswith("selector_pair_srgc_score.py")
            assert "train" not in command[1] and "--root" in command


def test_unknown_measurement_cost_does_not_erase_d_or_repair_files(tmp_path, saved_path, monkeypatch):
    initial, checkpoint, _ = saved_path
    checkpoint(75, projected_d=-3.)
    directory = repeat.output_dir(tmp_path, 3, 50, 25) / "step-75"
    event = {"event_id": "interrupted", "state": "started", "phase": "sr-gc-repeat-candidate-b",
             "ledger": "research", "gpus": 4, "gpu_type": "H100", "time": 1.}
    (directory / "cost.jsonl").write_text(json.dumps(event) + "\n")
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(base, "recover_cost_receipts", lambda *a: pytest.fail("read-only export"))
    row = repeat.scan_state(tmp_path, initial, 25)
    assert row["status"] == "sr_locked" and row["first_sr_step"] == 75
    assert row["decisions"][-1]["measurement_gpu_seconds"] is None
    assert row["decisions"][-1]["measurement_cost_complete"] is False
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_export_includes_recomputed_history_without_running_gpu(tmp_path, saved_path, monkeypatch):
    _, checkpoint, _ = saved_path
    checkpoint(75, projected_d=-3.)
    monkeypatch.setattr(results, "run_report", lambda *a: pytest.fail("no complete paired outcomes"))
    monkeypatch.setattr(repeat, "measure_point", lambda *a: pytest.fail("results must be read-only"))
    original_collect = repeat.collect
    monkeypatch.setattr(repeat, "collect", lambda root, initial, interval:
                        original_collect(root, initial, interval, start_step=50))
    output = tmp_path / "results.txt"
    monkeypatch.setattr(sys, "argv", ["results", "--root", str(tmp_path), "--out", str(output)])
    results.main()
    text = output.read_text()
    data = json.loads(text.split("DATA_JSON\n")[1])
    row = data["srgc_repeated"]["trajectories"][0]
    assert row["first_sr_step"] == 75 and len(row["decisions"]) == 2
    assert "SR-GC REPEATED CHECKS: t=50, every 25 updates; SR is absorbing" in text


def test_default_collect_scans_only_t25(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(repeat, "scan_state", lambda root, value, interval, **kwargs:
                        checked.append(value["step"]) or {"decisions": [], "errors": [],
                        "first_sr_step": None})
    report = repeat.collect(tmp_path, {"status": "validated", "decisions": [
        {"seed": 4, "step": 100}, {"seed": 3, "step": 50},
        {"seed": 4, "step": 25}, {"seed": 3, "step": 25}]})
    assert checked == [25, 25]
    assert report["start_step"] == 25 and report["status"] == "recorded"
    assert repeat.collect(tmp_path, {"status": "validated", "decisions": [
        {"seed": 3, "step": 50}]})["status"] == "initial_decisions_unavailable"


def test_targeted_collect_filters_seed_and_through_step(tmp_path, monkeypatch):
    checked = []
    def scan(root, value, interval, **kwargs):
        checked.append((value["seed"], kwargs["through_step"]))
        return {"decisions": [], "errors": [], "first_sr_step": None}
    monkeypatch.setattr(repeat, "scan_state", scan)
    initial = {"status": "validated", "decisions": [
        {"seed": 3, "step": 25}, {"seed": 4, "step": 25}, {"seed": 4, "step": 50}]}
    report = repeat.collect(tmp_path, initial, seed=4, through_step=75)
    assert checked == [(4, 75)]
    assert report["seed_filter"] == 4 and report["through_step"] == 75
    with pytest.raises(ValueError, match="scheduled check"):
        repeat.collect(tmp_path, initial, through_step=76)


@pytest.mark.parametrize("interval", [0, -1, True, 2.5])
def test_invalid_interval_rejected(tmp_path, interval):
    with pytest.raises(ValueError):
        repeat.collect(tmp_path, {"status": "not_recorded"}, interval)
