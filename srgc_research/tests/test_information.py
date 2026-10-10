"""Real OLMo/LoRA/AdamW observations plus failure and evidence validation."""

import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.torch_backend import TorchBackend
from srgc_research.information import (
    PHASES,
    PROTOCOL,
    InformationStudy,
    checkpoint_state,
    inspect_update,
    probe_gradient,
    source_identity,
)
from srgc_research.information_cli import collect, copy_verified, main, prepare, status
from srgc_research.information_report import digest, read_measurement, write_report
from srgc_research.tests.test_research import equal_tree, study_backend


def setup_measurement(folder, backend, *, stage=0):
    folder.mkdir(parents=True, exist_ok=True)
    records = {f"p{i}": {"prompt": f"problem {i}", "question": f"Question {i}", "answer": "3"}
               for i in range(7)}
    records["p0"]["question"] = '<script>alert("unsafe")</script>'
    backend.records = records
    data = {"records": records, "candidate_ids": [f"p{i}" for i in range(4)],
            "ranking_validation_ids": ["p4"], "validation_pool_ids": ["p4", "p5"],
            "evaluation_ids": ["p6"], "cached_rewards": {f"p{i}": [0, 1] * 4 for i in range(4)}}
    config = Config(seed=5, scoring_prompts=4, training_prompts=2, projection_dim=16)
    atomic_json(folder / "inputs.json", data)
    atomic_json(folder / "plan.json", asdict(config))
    identity = {"protocol": PROTOCOL, "dataset": "math", "seed": 5, "stage": stage,
                "input_sha256": digest(folder / "inputs.json"), "plan_sha256": digest(folder / "plan.json"),
                "source_checkpoint_sha256": None, "measurement_sha256": "tiny-test-code"}
    atomic_json(folder / "manifest.json", {"identity": identity, "input_sha256": identity["input_sha256"]})
    return data, config, identity


@pytest.fixture
def backend():
    return study_backend()


@pytest.fixture
def measured(tmp_path, backend):
    folder = tmp_path / "measurement"
    data, config, identity = setup_measurement(folder, backend)
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        before = backend.state_dict()
        InformationStudy(backend, data, config, 0, folder, identity, probe_prompts=1).run()
        equal_tree(before, backend.state_dict())
    return folder


def collect_batch(backend):
    records = backend.collect(["p0", "p1"], 8, 29)
    probe = backend.frozen_probe(["p2"], responses=8, seed=31)
    return records, probe, probe_gradient(backend, probe)


def test_shared_queue_audits_real_measurement_and_rejects_changed_tensors(measured):
    from srgc_research.dispatch import information_queue as queue
    endpoint, _ = read_measurement(measured)
    task = SimpleNamespace(output=measured, receipt=measured.parent / "queue.json",
        identity=endpoint["identity"], key="math.seed-5.t0", plan=measured / "plan.json",
        inputs=measured / "inputs.json", plan_sha256=digest(measured / "plan.json"),
        input_sha256=digest(measured / "inputs.json"))
    counts, code = queue.sweep([task], available=lambda: pytest.fail("completed measurement admitted GPUs"))
    assert code is None and counts["complete"] == 1
    assert queue.sweep([task])[0]["complete"] == 1
    phase = json.loads((measured / "score-A.json").read_text())
    tensor = measured / phase["artifacts"][0]["file"]
    tensor.write_bytes(b"damaged measurement evidence")
    counts, code = queue.sweep([task], runner=lambda *_: pytest.fail("corrupt tensors overwritten"))
    assert code is None and counts["failed"] == 1


def test_update_equals_direct_grpo_adam_step_and_restores_moments(backend):
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        backend.train(["p0", "p1"], responses=8, objective="grpo", seed=19)
        records, probe, gradient = collect_batch(backend)
        initial = backend.state_dict()
        result, tensors = inspect_update(backend, records, seed=29, probe=probe, probe_loss_gradient=gradient)
        equal_tree(initial, backend.state_dict())
        with backend.replaying(records):
            direct = backend.train(list(records), responses=8, objective="grpo", seed=29)
        equal_tree(tensors["backend_after"], backend.state_dict())
        assert result["metrics"]["gradient_before_clip_norm"] == pytest.approx(direct["gradient_norm"], rel=1e-6)
        vectors = tensors["vectors"]
        torch.testing.assert_close(vectors["parameter_after"] - vectors["parameter_before"], vectors["update"], rtol=0, atol=0)
        reconstructed = torch.stack(list(tensors["problem_loss_gradients"].values())).mean(0)
        torch.testing.assert_close(reconstructed, vectors["gradient_before_clip"], rtol=1e-4, atol=1e-7)
        assert result["metrics"]["updates"] == 1
        assert result["metrics"]["update_norm"] > 0
        assert result["metrics"]["zero_gradient_update_norm"] > 0
        for row in result["responses"].values():
            assert row["successes"] == 4 and row["mixed"]
            for sample in row["samples"]:
                assert sample["mean_logp_change"] == pytest.approx(np.mean(sample["logps_after"]) - np.mean(sample["logps_before"]))
                assert len(sample["logps_before"]) == sample["completion_tokens"]


@pytest.mark.parametrize("reward", [0., 1.])
def test_equal_reward_gradient_zero_but_adam_history_moves_weights(backend, reward):
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        backend.train(["p0"], responses=8, objective="grpo", seed=19)
        records, probe, gradient = collect_batch(backend)
        records = {i: (r[0], [reward] * 8, r[2]) for i, r in records.items()}
        initial = backend.state_dict()
        result, tensors = inspect_update(backend, records, seed=29, probe=probe, probe_loss_gradient=gradient)
        assert result["metrics"]["gradient_before_clip_norm"] == 0
        assert result["metrics"]["batch_incremental_update_norm"] == 0
        assert result["metrics"]["update_norm"] > 0
        torch.testing.assert_close(tensors["vectors"]["update"], tensors["vectors"]["zero_gradient_update"], rtol=0, atol=0)
        equal_tree(initial, backend.state_dict())


@pytest.mark.parametrize("fault", ["train", "post-logps", "probe-loss"])
def test_update_failure_restores_weights_optimizer_and_clip_function(backend, fault):
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        backend.train(["p0"], responses=8, objective="grpo", seed=19)
        records, probe, gradient = collect_batch(backend)
        initial = backend.state_dict()
        original_clip = torch.nn.utils.clip_grad_norm_
        if fault == "train":
            original = backend.train

            def broken(*a, **kw):
                original(*a, **kw)
                raise RuntimeError("injected after a real update")

            context = patch.object(backend, "train", side_effect=broken)
        elif fault == "probe-loss":
            context = patch.object(backend, "probe_loss", side_effect=RuntimeError("injected probe failure"))
        else:
            from srgc_research import information
            original = information.log_probabilities
            calls = []

            def broken(*a, **kw):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("injected post-logps failure")
                return original(*a, **kw)

            context = patch.object(information, "log_probabilities", side_effect=broken)
        with context, pytest.raises(RuntimeError, match="injected"):
            inspect_update(backend, records, seed=29, probe=probe, probe_loss_gradient=gradient)
        equal_tree(initial, backend.state_dict())
        assert torch.nn.utils.clip_grad_norm_ is original_clip
        assert backend.replay is None


def test_probe_gradient_matches_actual_grpo_and_preserves_existing_gradients(backend):
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        _records, probe, _ = collect_batch(backend)
        initial = backend.state_dict()
        for _, p in backend.train_parameters:
            p.grad = torch.full_like(p, 2.)
        gradient = probe_gradient(backend, probe)
        assert all(torch.equal(p.grad, torch.full_like(p, 2.)) for _, p in backend.train_parameters)
        equal_tree(initial, backend.state_dict())
        _, tensors = inspect_update(backend, {"p2": probe["p2"][:3]}, seed=31,
                                    probe=probe, probe_loss_gradient=gradient)
        torch.testing.assert_close(gradient, tensors["vectors"]["gradient_before_clip"], rtol=1e-4, atol=1e-7)
        # Independent central finite difference of the frozen clipped objective.
        direction = gradient / gradient.norm()
        base = backend.parameter_vector()
        losses = []
        for sign in (-1, 1):
            offset = 0
            with torch.no_grad():
                for _, p in backend.train_parameters:
                    n = p.numel()
                    p.copy_((base[offset:offset+n] + sign * .002 * direction[offset:offset+n]).view_as(p))
                    offset += n
            losses.append(backend.probe_loss(probe))
        backend.load_state_dict(initial)
        assert (losses[1] - losses[0]) / .004 == pytest.approx(float(gradient.dot(direction)), rel=.03, abs=1e-5)


def test_real_gradient_clipping_is_captured(backend):
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        records, probe, gradient = collect_batch(backend)
        hooks = [p.register_hook(lambda g: g * 10000) for _, p in backend.train_parameters]
        try:
            result, _ = inspect_update(backend, records, seed=29, probe=probe, probe_loss_gradient=gradient)
        finally:
            for hook in hooks:
                hook.remove()
        assert result["metrics"]["gradient_before_clip_norm"] > 1
        assert result["metrics"]["gradient_after_clip_norm"] == pytest.approx(1., abs=1e-6)


def test_bfloat16_deltas_are_measured_from_actual_quantized_weights(backend):
    with torch.no_grad():
        for _, p in backend.train_parameters:
            p.data = p.data.to(torch.bfloat16)
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        records, probe, gradient = collect_batch(backend)
        initial = backend.state_dict()
        result, tensors = inspect_update(backend, records, seed=29, probe=probe, probe_loss_gradient=gradient)
        equal_tree(initial, backend.state_dict())
        with backend.replaying(records):
            backend.train(list(records), responses=8, objective="grpo", seed=29)
        equal_tree(tensors["backend_after"], backend.state_dict())
        assert result["metrics"]["update_norm"] > 0


def test_actual_generation_works_without_rollout_replacement(tmp_path, backend):
    data, config, identity = setup_measurement(tmp_path, backend)
    initial = backend.state_dict()
    value = InformationStudy(backend, data, config, 0, tmp_path, identity, probe_prompts=1)
    value.run()
    equal_tree(initial, backend.state_dict())
    read_measurement(tmp_path)


def test_partial_receipt_cannot_change_selection_without_hash_failure(tmp_path, backend):
    data, config, identity = setup_measurement(tmp_path, backend)
    value = InformationStudy(backend, data, config, 0, tmp_path, identity, probe_prompts=1)
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        result, tensors = value.acquire("A")
        value.publish("score-A", result, tensors)
    receipt = json.loads((tmp_path / "score-A.json").read_text())
    receipt["result"]["scores"][value.candidates[0]] += .1
    atomic_json(tmp_path / "score-A.json", receipt)
    with pytest.raises(ValueError, match="hash"):
        value.run()


@pytest.mark.parametrize("phase", PHASES)
def test_resume_after_each_phase_never_repeats_saved_work(tmp_path, backend, phase):
    folder = tmp_path / "measurement"
    data, config, identity = setup_measurement(folder, backend)
    value = InformationStudy(backend, data, config, 0, folder, identity, probe_prompts=1)
    initial = backend.state_dict()
    publish = value.publish

    def interrupt(name, *args):
        result = publish(name, *args)
        if name == phase:
            raise RuntimeError("injected after receipt publication")
        return result

    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout), \
            patch.object(value, "publish", side_effect=interrupt), pytest.raises(RuntimeError, match="injected"):
        value.run()
    equal_tree(initial, backend.state_dict())
    saved = {p: digest(folder / f"{p}.pt") for p in PHASES if (folder / f"{p}.pt").exists()}
    resumed = InformationStudy(backend, data, config, 0, folder, identity, probe_prompts=1)
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        resumed.run()
    assert all(digest(folder / f"{p}.pt") == sha for p, sha in saved.items())
    equal_tree(initial, backend.state_dict())
    with patch.object(TorchBackend, "_rollout", side_effect=AssertionError("completed measurement regenerated")), \
            patch.object(backend, "train", side_effect=AssertionError("completed measurement retrained")):
        resumed.run()
    read_measurement(folder)


def test_report_full_evidence_and_html_escaping(measured, tmp_path):
    output = tmp_path / "report"
    report = write_report(output, folders=[measured, measured])
    assert len(report["selected_problems"]) == 4
    assert len(report["candidate_information"]) == 4
    assert len(report["batch_updates"]) == 2
    assert len(report["information_summary"]) == 2
    assert all(r["weight_update_available"] and r["training_updates"] == 1 for r in report["selected_problems"])
    text = (output / "report.html").read_text()
    assert '<script>' not in text
    # p0 may not be selected, so check escaping with an explicit selected text.
    report["selected_problems"][0]["question"] = '<script>alert("unsafe")</script>'
    from srgc_research.information_report import render_html
    render_html(output / "report.html", report)
    assert "&lt;script&gt;" in (output / "report.html").read_text()
    assert status(measured)["status"] == "complete"
    assert write_report(tmp_path / "empty", folders=[measured], dataset="mbpp")["selected_problems"] == []


@pytest.mark.parametrize("damage", ["tensor", "source", "identity", "rewards", "tokens", "norm", "escape", "missing-phase"])
def test_corrupt_evidence_never_reports_complete(measured, damage):
    if damage == "tensor":
        with (measured / "sr.pt").open("ab") as handle:
            handle.write(b"corrupted")
    elif damage == "source":
        with (measured / "inputs.json").open("a") as handle:
            handle.write(" ")
    elif damage == "missing-phase":
        (measured / "probe.json").unlink()
    elif damage == "escape":
        v = json.loads((measured / "endpoint.json").read_text())
        v["phases"]["sr"] = "../outside.json"
        atomic_json(measured / "endpoint.json", v)
    else:
        path = measured / "sr.json"
        v = json.loads(path.read_text())
        if damage == "identity":
            v["identity"]["seed"] += 1
        elif damage == "rewards":
            next(iter(v["result"]["responses"].values()))["successes"] += 1
        elif damage == "tokens":
            next(iter(v["result"]["responses"].values()))["samples"][0]["completion_tokens"] += 1
        else:
            v["result"]["metrics"]["update_norm"] += 1
        atomic_json(path, v)
    with pytest.raises((ValueError, OSError)):
        read_measurement(measured)
    with pytest.raises((ValueError, OSError)):
        status(measured)


def test_source_checkpoint_uses_anchor_not_live_counterfactual(backend, tmp_path):
    data, config, _ = setup_measurement(tmp_path, backend)
    engine = Engine(backend, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"], config=config)
    anchor = engine.state_dict()
    saved = {"anchor": anchor, "backend": {"wrong": "branch weights"}}
    assert checkpoint_state(saved, data, config, 0) is anchor["backend"]
    with pytest.raises(ValueError, match="stage"):
        checkpoint_state(saved, data, config, 100)
    source_identity({"input_sha256": "a", "plan_sha256": "b", "seed": 5}, input_hash="a", plan_hash="b", seed=5)
    with pytest.raises(ValueError, match="identity"):
        source_identity({"input_sha256": "wrong"}, input_hash="a", plan_hash="b", seed=5)
    with pytest.raises(ValueError, match="no saved anchor"):
        checkpoint_state({"anchor": None}, data, config, 0)


def test_report_refuses_source_overwrite(measured):
    with pytest.raises(ValueError, match="separate"):
        write_report(measured, folders=[measured])
    with pytest.raises(ValueError, match="separate"):
        write_report(measured.parent, folders=[measured])


def test_verified_copy_rejects_source_mutation(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_text("original")
    import shutil
    original = shutil.copyfile

    def changed(a, b):
        original(a, b)
        source.write_text("changed")

    with patch("srgc_research.information_cli.shutil.copyfile", side_effect=changed), pytest.raises(ValueError, match="changed"):
        copy_verified(source, target)
    assert not target.exists() and not target.with_suffix(".tmp").exists()


def legacy_bundle():
    ids = [f"q{i}" for i in range(40)]
    m = {"stage": 0, "candidate_ids": ids, "selected": {"on_policy": ids[:4], "sr": ids[4:8]},
         "cached_success_rates": dict.fromkeys(ids, .5)}
    for name in ("A", "B"):
        m[name] = {"cosines": [i / 100 for i in range(40)], "rewards": {i: [0, 1] * 4 for i in ids}}
    training = [{"stage": 0, "mode": mode, "update": update, "train_ids": selected,
                 "rewards": {i: [0, 1] * 4 for i in selected}, "metrics": {"gradient_norm": .3}}
                for mode, selected in m["selected"].items() for update in range(1, 26)]
    endpoint = {"arm": "stage_mechanism", "protocol": "grpo-stage-interventions-v1", "horizon": 25,
                "input_sha256": "missing-input", "implementation_sha256": "test-implementation",
                "measurements": [m], "training_records": training}
    return {"rows": [{"dataset": "math", "seed": 7, "result": endpoint},
                     {"dataset": "mbpp", "seed": 5, "result": None}], "errors": []}


def test_legacy_dedup_and_missing_weights_are_explicit(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    atomic_json(a, legacy_bundle())
    b.write_bytes(a.read_bytes())
    report = write_report(tmp_path / "report", legacy=[a, b])
    assert report["unique_legacy_files"] == 1
    assert len(report["selected_problems"]) == 8
    assert len(report["candidate_information"]) == 40
    assert report["pending"] == [{"dataset": "mbpp", "seed": 5}]
    assert all(r["updates"] == 25 and r["update_norm"] is None for r in report["batch_updates"])
    assert all(not r["weight_update_available"] and r["question"] is None for r in report["selected_problems"])


@pytest.mark.parametrize("fault", ["duplicate-update", "conflict", "nonbinary", "nan-score"])
def test_invalid_legacy_rejected(tmp_path, fault):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    bundle = legacy_bundle()
    atomic_json(a, bundle)
    value = bundle["rows"][0]["result"]
    if fault == "duplicate-update":
        value["training_records"][0]["update"] = 2
    elif fault == "conflict":
        value["training_records"][0]["metrics"]["gradient_norm"] = .9
    elif fault == "nonbinary":
        value["measurements"][0]["A"]["rewards"]["q0"][0] = .2
    else:
        value["measurements"][0]["A"]["cosines"][0] = float("nan")
    b.write_text(json.dumps(bundle))
    with pytest.raises(ValueError):
        write_report(tmp_path / "report", legacy=[a, b] if fault == "conflict" else [b])


def test_cli_errors_and_no_implicit_training(tmp_path):
    assert main(["all", "status", "--output", str(tmp_path)]) == 0
    args = SimpleNamespace(dataset="all", stage=0, checkpoint=None)
    with pytest.raises(ValueError, match="one explicit dataset"):
        prepare(args)
    args.dataset, args.stage = "math", 100
    with pytest.raises(ValueError, match="existing"):
        prepare(args)


@pytest.fixture
def collect_args(tmp_path, monkeypatch):
    from srgc_rebuttal.tests.test_cluster import write_inputs
    plan = write_inputs(tmp_path)
    group = tmp_path / "group"
    group.mkdir()
    monkeypatch.setenv("GROUP_VOLUME", str(group))
    monkeypatch.setenv("OM_WORK", str(group / "work"))
    monkeypatch.setenv("SRGC_STORAGE_ROOT", str(group / "work/srgc-rebuttal"))
    return SimpleNamespace(dataset="math", stage=0, checkpoint=None, seed=5,
                           inputs=tmp_path / "inputs-5.json", plan=plan,
                           output=group / "information/t0", probe_prompts=8, attention="sdpa")


def test_prepare_freezes_sources_and_rejects_changed_identity(collect_args):
    a = collect_args
    first = prepare(a)
    assert prepare(a) == first
    assert (a.output / "inputs.json").read_bytes() == a.inputs.read_bytes()
    assert (a.output / "plan.json").read_bytes() == a.plan.read_bytes()
    with (a.inputs).open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="changed"):
        prepare(a)


@pytest.mark.parametrize("fault", ["dataset", "probe-count", "overlap"])
def test_prepare_rejects_wrong_dataset_probe_or_source_directory(collect_args, fault):
    a = collect_args
    if fault == "dataset":
        a.dataset = "mbpp"
    elif fault == "probe-count":
        a.probe_prompts = 0
    else:
        a.inputs = a.output / "inputs.json"
    with pytest.raises(ValueError):
        prepare(a)


def test_launcher_resume_has_unique_admission_and_requires_valid_endpoint(collect_args):
    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    admissions, commands = [], []

    def admit(path, *a, **kw):
        path.mkdir(parents=True, exist_ok=False)
        admissions.append(path)

    def child(command, log, env, **kw):
        commands.append(command)
        assert "srgc_research/information_rank.py" in command[-3]
        assert env["PYTHONPATH"].split(":")[0] != str(collect_args.output)
        assert "--max_restarts=0" in command
        return 2 if len(commands) == 1 else 0

    with patch.object(guard, "process_guard", return_value=nullcontext()), \
            patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", ("a", "b", "c", "d"))), \
            patch.object(cluster, "device_leases", side_effect=lambda *a: nullcontext(())), \
            patch.object(cluster, "admit", side_effect=admit), \
            patch.object(cluster, "run_child", side_effect=child), \
            patch("srgc_rebuttal.existing_runtime.python_path", return_value=sys.executable):
        assert collect(collect_args) == 2
        with pytest.raises(FileNotFoundError):
            collect(collect_args)
    assert len(admissions) == 2 and admissions[0] != admissions[1]


def test_incomplete_costs_do_not_turn_unknown_gpu_time_into_zero(measured):
    endpoint = json.loads((measured / "endpoint.json").read_text())
    endpoint["cost_receipts"] = {"complete": False, "known_gpu_seconds": {
        "selection_gpu_seconds": 5., "training_gpu_seconds": 3.},
        "unfinished_phases": [{"id": "failed", "phase": "diagnostic"}], "total_gpu_seconds": None}
    atomic_json(measured / "endpoint.json", endpoint)
    read_measurement(measured)
    endpoint["cost_receipts"]["total_gpu_seconds"] = 8.
    atomic_json(measured / "endpoint.json", endpoint)
    with pytest.raises(ValueError, match="incomplete cost"):
        read_measurement(measured)


@pytest.mark.parametrize("text", ["null", "[]", "1", "{"])
def test_malformed_json_is_reported_as_an_error(tmp_path, text):
    (tmp_path / "manifest.json").write_text(text)
    assert main(["all", "status", "--output", str(tmp_path)]) == 1


def test_invalid_output_creates_no_lock_or_directory(collect_args):
    a = collect_args
    a.output = a.inputs.parent / "outside-group"
    with pytest.raises(ValueError, match="inside group"):
        collect(a)
    assert not a.output.exists()
