"""N12/N13 recover measured data without retraining or inventing missing signals."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.timing import torch_meter
from srgc_research.followup import checkpoints, n12, n13
from srgc_research.followup.__main__ import main
from srgc_research.followup.common import (
    copy_bound,
    file_hash,
    fingerprint,
    pool,
    read,
    write,
)
from srgc_research.followup.measure import (
    captured_backend,
    collect,
    n12_acquire,
    n13_acquire,
    profile_name,
)
from srgc_research.stage_rollout_analysis import Analysis
from srgc_research.tests.test_stage_rollout_analysis import group, trace


def bundle(size=400):
    candidates = [f"p{i}" for i in range(size)]
    reference = [f"v{i}" for i in range(50)]
    return {"candidate_ids": candidates, "ranking_validation_ids": reference, "validation_pool_ids": reference,
            "evaluation_ids": ["held"], "records": {i: {"prompt": i, "answer": "3"} for i in candidates + reference + ["held"]},
            "cached_rewards": {p: [1] * (j % 9) + [0] * (8 - j % 9) for j, p in enumerate(candidates)}}


class Backend:
    """Small deterministic rollout backend; genuine parameter/Adam/RNG state."""
    gpu_count, rank, world, projection_dim, projection_seed = 0, 0, 1, 2, 0

    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.device = torch.device("cpu")
        self.train_parameters = list(self.model.named_parameters())
        self.score_parameters = self.train_parameters
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.cost_meter = torch_meter(cuda=False)
        self.calls = []

    def state_dict(self):
        return {"trainable": {n: p.detach().clone() for n, p in self.train_parameters},
                "optimizer": copy.deepcopy(self.optimizer.state_dict()), "projection_dim": 2, "projection_seed": 0,
                "score_names": [n for n, _ in self.score_parameters]}

    def _rollout(self, prompt_id, responses, seed):
        self.calls.append((prompt_id, responses, seed))
        return [torch.tensor([1, 2, 3, i + 4]) for i in range(responses)], np.array([0., 1.] * 4), 2

    def score_gradients(self, prompts, *, responses, group_size, seed):
        for prompt in prompts:
            self._rollout(prompt, responses, seed)
        return {i: np.array([1. + j % 3, 2.]) for j, i in enumerate(prompts)}

    def _gather(self, value):
        return value


@pytest.fixture
def source(tmp_path):
    data = bundle()
    backend = Backend()
    config = Config(seed=5, projection_dim=2)
    state = Engine(backend, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"],
                   arm="on_policy", config=config, step=125).state_dict()
    state["used_training_ids"] = data["candidate_ids"][:70]
    plan = {**asdict(config), "scoring_prompts_per_set": 40, "projection_seed": 0,
            "model": "allenai/OLMo-3", "model_revision": "a" * 40, "world_size": 4}
    bundle_path, plan_path = tmp_path / "inputs.json", tmp_path / "plan.json"
    write(bundle_path, data)
    write(plan_path, plan)
    marker = {"seed": 5, "input_sha256": file_hash(bundle_path), "plan_sha256": file_hash(plan_path),
              "implementation_sha256": "c" * 64}
    write(tmp_path / "run.json", marker)
    checkpoint = tmp_path / "on_policy-latest.pt"
    torch.save(state, checkpoint)
    identity = {"seed": 5, "input_sha256": marker["input_sha256"], "plan_sha256": marker["plan_sha256"],
                "source_implementation_sha256": "c" * 64, "policy_sha256": "d" * 64, "measurement_sha256": "e" * 64}
    return SimpleNamespace(data=data, backend=backend, state=state, plan=plan, marker=marker,
                           bundle=bundle_path, plan_path=plan_path, checkpoint=checkpoint, identity=identity)


@pytest.mark.parametrize("size", [400, 512])
def test_pool_size_and_panel_are_source_bound_and_sampled_once(size):
    data = bundle(size)
    assert len(pool(data)[0]) == size
    fixed = n12.panel(data, "a" * 64, 5)
    assert fixed == n12.panel(data, "a" * 64, 5)
    assert fixed != n12.panel(data, "a" * 64, 6)
    assert len(set(fixed["prompt_ids"])) == 20
    assert set(fixed["prompt_ids"]) <= set(data["candidate_ids"])
    assert not set(fixed["prompt_ids"]) & set(data["evaluation_ids"])


def measured_groups(values):
    analysis = Analysis()
    analysis.document(trace([group(n, prompt=p) for p, n in values]), {"path": "fixture", "sha256": "e" * 64})
    return list(analysis.groups.values())


def test_fixed_mixture_removes_composition_effect_and_keeps_exposures():
    rows = measured_groups([("p0", 0), ("p1", 0), ("p2", 4)])
    counts = {"p0": 0, "p1": 0, "p2": 4, "unselected1": 4, "unselected2": 4, "unselected3": 4}
    result = n12.standardized_window(rows, counts)
    assert result["original"]["mixed_fraction"] == pytest.approx(1 / 3)
    assert result["standardized"]["mixed_fraction"] == pytest.approx(2 / 3)
    assert sum(result["standardized"]["success_histogram_fraction"]) == pytest.approx(1)
    rows.append({**rows[0], "update": 27})
    result = n12.standardized_window(rows, counts)
    assert result["groups"] == 4 and result["unique_prompts"] == 3 and result["repeat_exposures"] == 1


def test_unsupported_positive_stratum_and_missing_tokens_stay_null():
    rows = measured_groups([("p0", 4)])
    result = n12.standardized_window(rows, {"p0": 4, "p1": 8})
    assert result["unsupported_positive_strata"] == ["8/8"]
    assert all(value is None for value in result["standardized"].values())
    rows[0]["response_token_lengths"] = None
    result = n12.standardized_window(rows, {"p0": 4})
    assert result["standardized"]["mixed_fraction"] == 1
    assert result["standardized"]["mean_completion_tokens"] is None


@pytest.mark.parametrize("damage", ["cache7", "cache_bool", "incomplete", "leak"])
def test_invalid_cache_and_split_fail_before_sampling(damage):
    data = bundle()
    if damage == "cache7":
        data["cached_rewards"]["p0"].pop()
    elif damage == "cache_bool":
        data["cached_rewards"]["p0"][0] = True
    elif damage == "incomplete":
        del data["cached_rewards"]["p0"]
    else:
        data["evaluation_ids"] = ["p0"]
    with pytest.raises(ValueError):
        n12.panel(data, "a" * 64, 5)


def test_n12_report_exports_applied_groups_and_separate_rewards(source, tmp_path):
    g = {**group(4, prompt="p4"), "selected_for_training": True}
    document = trace([g])
    document["identity"]["input_sha256"] = file_hash(source.bundle)
    document["history_record"] = {"checkpoint": 25, "completed_updates": 26, "train_ids": ["p4", "p3", "p2", "p1"],
                                  "metrics": {"sample_reward": .5}}
    document["reward"] = .8
    path = tmp_path / "trace.json"
    write(path, document)
    output = tmp_path / "report"
    before = file_hash(path)
    main(["n12-report", str(path), "--bundle", str(source.bundle), "--output-dir", str(output)])
    result = read(output / "report.json")
    assert result["windows_25"][0]["groups"] == 1
    assert not result["windows_25"][0]["complete_training_window"]
    assert result["training_rewards"][0]["reward"] == .5
    assert result["held_out_rewards"][0]["reward"] == .8
    assert (output / "training_rewards.csv").exists() and (output / "held_out_rewards.csv").exists()
    assert file_hash(path) == before
    with pytest.raises(SystemExit):
        main(["n12-report", str(path), "--bundle", str(source.bundle), "--output-dir", str(output)])
    document["history_record"]["train_ids"] = ["other"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="actually applied"):
        n12.report(source.bundle, [path])


def test_scalar_only_does_not_reconstruct_d4_or_generate():
    result = n13.compare({"d": -.3, "ranking_scores": [.4] * 40, "on_top4_dot": .7})
    assert result["status"] == "unmeasured" and result["d4_A"] is None


def contrast_document():
    c = [f"c{i}" for i in range(40)]
    s = [f"s{i}" for i in range(40)]
    reference = [f"v{i}" for i in range(50)]
    # ON picks high cosine with small norm; cached SR picks a lower cosine
    # with a large dot. Candidate mean still beats S40: opposite judgments.
    gradients = {i: [1., 0.] if j < 4 else [10., 10.] for j, i in enumerate(c)}
    gradients.update({i: [5., 0.] for i in s})
    return {"schema": "srgc-n13-vectors-v1", "identity": {k: "a" * 64 for k in
            ("input_sha256", "plan_sha256", "policy_sha256", "measurement_sha256", "source_implementation_sha256")},
            "seed": 5, "stage": 125, "projection_dim": 2, "candidate_ids": c, "sr_comparison_ids": s,
            "reference_ids": reference, "cached_sr_ranked_ids": s + c[4:] + c[:4],
            "blocks": {"A": {"gradients": gradients, "reference_gradients": {i: [1., 0.] for i in reference}}}}


def test_d40_and_d4_can_disagree_and_sr4_is_restricted_to_c40():
    result = n13.compare(contrast_document())
    assert result["d40_A"] > 0 > result["d4_A"]
    assert result["sr4_ids"] == ["c4", "c5", "c6", "c7"]
    assert set(result["on4_ids"]) == {"c0", "c1", "c2", "c3"}
    assert not result["sign_match_A"] and result["d4_B"] is None


def test_b_remeasures_a_selection_without_selecting_again():
    document = contrast_document()
    block = copy.deepcopy(document["blocks"]["A"])
    block["gradients"].update({i: [-2., 0.] for i in document["candidate_ids"][:4]})
    document["blocks"]["B"] = block
    result = n13.compare(document)
    assert set(result["on4_ids"]) == {"c0", "c1", "c2", "c3"}
    assert result["d4_B"] == -12
    assert result["independent_B_uses_A_selection"]


@pytest.mark.parametrize("damage", ["identity", "dimension", "nan", "missing_vector", "wrong_on4", "reference"])
def test_n13_rejects_unbound_or_invalid_vectors(damage):
    document = contrast_document()
    if damage == "identity":
        del document["identity"]["policy_sha256"]
    elif damage == "dimension":
        document["projection_dim"] = 3
    elif damage == "nan":
        document["blocks"]["A"]["gradients"]["c0"] = [float("nan"), 0]
    elif damage == "missing_vector":
        del document["blocks"]["A"]["gradients"]["s0"]
    elif damage == "wrong_on4":
        document["selected_on_ids_A"] = document["candidate_ids"][4:8]
    else:
        document["reference_ids"] = document["reference_ids"][:25]
    with pytest.raises(ValueError):
        n13.compare(document)


def test_overlap_uses_exactly_one_vector_and_cancels():
    document = contrast_document()
    document["sr_comparison_ids"] = document["candidate_ids"]
    document["blocks"]["A"]["gradients"] = {i: document["blocks"]["A"]["gradients"][i] for i in document["candidate_ids"]}
    result = n13.compare(document)
    assert result["d40_A"] == 0 and result["candidate_sr40_overlap"] == 40


def test_weights_only_archives_reuse_policy_without_optimizer(source, tmp_path):
    archive = tmp_path / "reward-curves" / "on_policy" / "step-125.pt"
    archive.parent.mkdir(parents=True)
    state = copy.deepcopy(source.state)
    del state["backend"]["optimizer"]
    state["evaluation_only"] = True
    protocol = {**source.marker, "arm": "on_policy"}
    state["reward_curve_binding"] = {"protocol_sha256": fingerprint(protocol), "step": 125}
    write(archive.parent / "protocol.json", protocol)
    torch.save(state, archive)
    write(archive.with_suffix(".json"), {"checkpoint_sha256": file_hash(archive), "step": 125})
    before = file_hash(archive)
    _, _, _, info = checkpoints.load_source(archive, source.bundle, source.plan_path, 5)
    assert info["n12_panel_weights_usable"] and info["n13_rescoring_weights_and_sets_usable"]
    assert not info["optimizer_state_present"] and not info["training_resume_readiness_verified"]
    optimizer_before = copy.deepcopy(source.backend.optimizer.state_dict())
    checkpoints.load_weights(source.backend, state["backend"])
    assert source.backend.optimizer.state_dict() == optimizer_before
    assert file_hash(archive) == before
    rows = checkpoints.inventory([archive], source.bundle, source.plan_path, 5)
    assert rows["rows"][0]["status"] == "usable_for_diagnostics"
    assert any(not r["present"] for r in rows["requested_stage_coverage"])


@pytest.mark.parametrize("damage", ["seed", "cache", "plan", "marker", "projection", "used", "point_sha"])
def test_checkpoint_reuse_rejects_mismatched_sources(source, damage):
    if damage == "seed":
        source.state["config"]["seed"] = 6
    elif damage == "cache":
        source.state["sr_ranked_ids"] = list(reversed(source.state["sr_ranked_ids"]))
    elif damage == "plan":
        source.state["config"]["selection_interval"] = 50
    elif damage == "marker":
        (source.checkpoint.parent / "run.json").unlink()
    elif damage == "projection":
        source.state["backend"]["projection_seed"] = 99
    elif damage == "used":
        source.state["used_training_ids"] = ["held"]
    else:
        source.state["evaluation_only"] = True
        write(source.checkpoint.with_suffix(".json"), {"checkpoint_sha256": "wrong", "step": 125})
    torch.save(source.state, source.checkpoint)
    with pytest.raises(ValueError):
        checkpoints.load_source(source.checkpoint, source.bundle, source.plan_path, 5)


def test_loading_invalid_tensor_does_not_partially_mutate_policy(source):
    before = source.backend.state_dict()
    weights = copy.deepcopy(before)
    first, last = list(weights["trainable"])
    weights["trainable"][first].add_(1)
    weights["trainable"][last] = weights["trainable"][last].double()
    with pytest.raises(ValueError):
        checkpoints.load_weights(source.backend, weights)
    for key, value in before["trainable"].items():
        torch.testing.assert_close(value, source.backend.state_dict()["trainable"][key], rtol=0, atol=0)


def diagnostic_backend(tmp_path):
    backend = captured_backend(Backend)()
    backend.rollout_cache_root = tmp_path / "responses"
    return backend


def test_acquisitions_preserve_state_record_tokens_and_resume_without_regeneration(source, tmp_path):
    backend = diagnostic_backend(tmp_path)
    state_before = copy.deepcopy(source.state)
    weights = backend.state_dict()
    result = n13_acquire(backend, source.state, source.data, source.identity, tmp_path)
    calls = len(backend.calls)
    assert result["d4_B"] is not None and calls > 0
    records = read(tmp_path / "vectors.json.gz")
    assert len(records["reference_ids"]) == 50
    assert records["used_training_ids"] == source.state["used_training_ids"]
    assert set(records["sr_comparison_ids"]).isdisjoint(source.state["used_training_ids"])
    assert all(len(g["responses"]) == 8 for g in records["blocks"]["A"]["rollouts"].values())
    assert n13_acquire(backend, source.state, source.data, source.identity, tmp_path) == result
    assert len(backend.calls) == calls
    for key, value in weights["trainable"].items():
        torch.testing.assert_close(value, backend.state_dict()["trainable"][key], rtol=0, atol=0)
        torch.testing.assert_close(state_before["backend"]["trainable"][key], source.state["backend"]["trainable"][key], rtol=0, atol=0)
    assert state_before["used_training_ids"] == source.state["used_training_ids"]
    with pytest.raises(RuntimeError, match="cannot train"):
        backend.train(["p0"])


def test_fixed_panel_runs_once_per_prompt_and_reuses_cached_rollouts(source, tmp_path):
    backend = diagnostic_backend(tmp_path)
    fixed = n12.panel(source.data, source.identity["input_sha256"], 5)
    rng = torch.get_rng_state().clone()
    result = n12_acquire(backend, source.data, fixed, source.identity)
    assert result["response_count"] == 160 and len(backend.calls) == 20
    assert not result["is_fixed_held_out_evaluation"] and not result["used_for_training_or_selection"]
    assert n12_acquire(backend, source.data, fixed, source.identity) == result and len(backend.calls) == 20
    assert torch.equal(rng, torch.get_rng_state())
    result["source"] = {"stage": 25, "arm": "on_policy"}
    first, second = tmp_path / "panel25.json.gz", tmp_path / "panel125.json.gz"
    write(first, result)
    later = copy.deepcopy(result)
    later["source"]["stage"] = 125
    later["identity"]["policy_sha256"] = "f" * 64
    write(second, later)
    report = n12.panel_report(source.bundle, [first, second, first], 5)
    assert report["distinct_measured_policies"] == 2 and report["response_count"] == 320
    assert report["duplicate_policy_records_removed"] == 1
    assert all(row["success_rate_change_from_t25"] == 0 for row in report["rows"])


def test_completed_collect_verifies_evidence_before_gpu_or_base_load(source, tmp_path):
    args = SimpleNamespace(checkpoint=source.checkpoint, bundle=source.bundle, plan=source.plan_path,
                           seed=5, experiment="n12", output_root=tmp_path / "diagnostics", a_only=False)
    _, _, _, actual = checkpoints.load_source(source.checkpoint, source.bundle, source.plan_path, 5)
    identity = {k: actual[k] for k in ("input_sha256", "plan_sha256", "policy_sha256", "seed", "source_implementation_sha256", "attention")}
    identity.update(measurement_sha256="e" * 64, experiment="n12", independent_B=True)
    output = args.output_root / "n12" / fingerprint(identity)
    write(output / "evidence.json", {"measured": True})
    write(output / "endpoint.json", {"identity": identity, "artifact_sha256": {"evidence.json": file_hash(output / "evidence.json")}})
    with patch("srgc_research.followup.measure.measurement_hash", return_value="e" * 64), \
            patch("srgc_rebuttal.distributed.initialize", side_effect=AssertionError("loaded GPU")), \
            pytest.raises(ValueError, match="required artifacts"):
        collect(args)


@pytest.mark.parametrize(("extension", "expected"), [("qwen35-9b-v2", "qwen35"), ("gemma4-12b-pt-v1", "gemma4"),
                                                   ("gemma4-12b-text-v2", "gemma4_text"),
                                                   ("gemma4-12b-text-v3", "gemma4_text"),
                                                   ("llama31-8b-v1", "llama31"), (None, "olmo3")])
def test_profiles_match_existing_pinned_adapters(extension, expected):
    assert profile_name({"extension": extension}) == expected


def test_n13_cli_exports_valid_count_and_null_missing_record(tmp_path):
    full, scalar = tmp_path / "vectors.json", tmp_path / "scalar.json"
    write(full, contrast_document())
    write(scalar, {"d": -.5})
    main(["n13-report", str(full), str(scalar), "--output-dir", str(tmp_path / "report")])
    report = read(tmp_path / "report" / "report.json")
    assert report["coverage"]["valid_A"] == 1 and report["coverage"]["unmeasured"] == 1
    assert report["rows"][1]["d4_A"] is None


def test_interrupted_scoring_recovers_completed_prompt_rollouts(source, tmp_path):
    backend = diagnostic_backend(tmp_path)
    original = Backend._rollout

    def interrupted(self, *args, **kwargs):
        if len(self.calls) == 3:
            raise RuntimeError("node preempted")
        return original(self, *args, **kwargs)

    with patch.object(Backend, "_rollout", interrupted), pytest.raises(RuntimeError, match="preempted"):
        n13_acquire(backend, source.state, source.data, source.identity, tmp_path, independent=False)
    assert len(backend.calls) == 3 and not (tmp_path / "score-A.json.gz").exists()
    resumed = diagnostic_backend(tmp_path)
    n13_acquire(resumed, source.state, source.data, source.identity, tmp_path, independent=False)
    vectors = read(tmp_path / "vectors.json.gz")
    required = len(set(vectors["candidate_ids"] + vectors["sr_comparison_ids"])) + 50
    assert len(backend.calls) + len(resumed.calls) == required
    assert not {i for i, _, _ in backend.calls} & {i for i, _, _ in resumed.calls}


def test_real_lora_projected_scoring_preserves_weights_and_adam(source, tmp_path):
    from srgc_rebuttal.tests.test_torch_backend import ModelBackendTests
    from srgc_rebuttal.torch_backend import TorchBackend
    from srgc_research.tests.test_research import equal_tree

    fixture = ModelBackendTests()
    fixture.setUp()
    original = fixture.backend
    backend = captured_backend(TorchBackend)(original.model, original.tokenizer, source.data["records"], original.verifier,
                                            projection_dim=16, max_new_tokens=2)
    backend.rollout_cache_root = tmp_path / "real-responses"
    state = copy.deepcopy(source.state)
    state["config"]["projection_dim"] = 16
    state["backend"] = backend.state_dict()
    before, rng = backend.state_dict(), torch.get_rng_state().clone()
    with patch.object(TorchBackend, "_rollout", side_effect=fixture.deterministic_rollout):
        result = n13_acquire(backend, state, source.data, source.identity, tmp_path, independent=False)
    assert result["status"] == "measured" and np.isfinite(result["d4_A"])
    equal_tree(before, backend.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert not any(p.grad is not None for _, p in backend.train_parameters)


def test_compact_archive_coverage_counts_checks_instead_of_inventing_vectors(tmp_path):
    archive = {"source_results": {"seed-5/switch-endpoint.json": {"seed": 5, "arm": "switch",
        "checks": [{"step": 25, "d": -.2}, {"step": 50, "d": -.3}]},
        "replicates/repeat/seed-5/switch-endpoint.json": {"seed": 5, "arm": "switch", "checks": [{"step": 25, "d": -1}]}}}
    path = tmp_path / "archive.json"
    write(path, archive)
    main(["n13-report", str(path), "--output-dir", str(tmp_path / "coverage")])
    report = read(tmp_path / "coverage" / "report.json")
    assert report["coverage"]["unmeasured"] == 2 and report["coverage"]["valid_A"] == 0
    assert [row["recorded_historical_d"] for row in report["rows"]] == [-.2, -.3]


def test_backup_receipt_reuses_original_policy_and_checks_checksum(source, tmp_path):
    target = tmp_path / "backups" / file_hash(source.checkpoint) / "checkpoint.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.checkpoint.read_bytes())
    write(target.parent / "receipt.json", {"schema": "srgc-checkpoint-backup-v1", "sha256": file_hash(target),
                                          "identity": source.marker, "source": str(source.checkpoint)})
    _, _, _, info = checkpoints.load_source(target, source.bundle, source.plan_path, 5)
    assert info["identity_binding"] == "backup_receipt_checksum_and_engine_contract"
    assert info["optimizer_state_present"]
    target.write_bytes(b"broken archive")
    report = checkpoints.inventory([target], source.bundle, source.plan_path, 5)
    assert report["rows"][0]["status"] == "invalid_or_other_run"


def test_n12_retains_missing_windows_and_only_real_metadata(source, tmp_path):
    source.data["records"]["p4"].update(subject="algebra", level=3)
    source.bundle.write_text(json.dumps(source.data))
    document = trace([{**group(4, prompt="p4"), "selected_for_training": True}])
    document["identity"]["input_sha256"] = file_hash(source.bundle)
    path = tmp_path / "record.json"
    write(path, document)
    report = n12.report(source.bundle, [path])
    first = report["windows_25"][0]
    assert first["metadata_breakdown"][0]["label"] == "algebra"
    assert first["metadata_breakdown"][0]["pool_size"] == 1
    assert first["metadata_breakdown"][0]["metadata_pool_coverage"] == 1 / 400
    assert len(report["windows_25"]) == 10
    assert all(r["groups"] == 0 and r["standardized"]["mixed_fraction"] is None for r in report["windows_25"][1:])


def test_nearby_evaluation_uses_held_out_source_matches_only():
    analysis = Analysis()
    identity = {"input_sha256": "a" * 64, "plan_sha256": "b" * 64, "implementation_sha256": "c" * 64}
    context = analysis.context({"model": "fixture", "dataset": "math", "seed": 5},
                               {**identity, "seed": 5, "arm": "on_policy"}, {"path": "source", "sha256": "f" * 64})
    analysis.curve(context, 125, .4, "held_out", "source")
    analysis.curve(context, 150, .6, "held_out", "source")
    analysis.curve(context, 150, 1., "training", "source")
    row = {"stage": 125, "seed": 5, "identity": {**identity, "source_implementation_sha256": "c" * 64, "arm": "on_policy"}}
    result = n13.nearby_evaluation(row, analysis)
    assert result["held_out_gain_next_25"] == pytest.approx(.2)
    assert all(p["kind"] == "held_out" for p in result["nearby_held_out_points"])
    row["identity"]["input_sha256"] = "d" * 64
    result = n13.nearby_evaluation(row, analysis)
    assert result["held_out_gain_next_25"] is None and not result["nearby_held_out_points"]


def test_n13_rejects_duplicate_acquisition_under_another_serialization(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    document = contrast_document()
    write(first, document)
    second.write_text(json.dumps(document, separators=(",", ":")))
    with pytest.raises(SystemExit):
        main(["n13-report", str(first), str(second), "--output-dir", str(tmp_path / "report")])
    assert not (tmp_path / "report").exists()


def test_snapshot_refuses_changed_source_and_preserves_frozen_bytes(tmp_path):
    source, target = tmp_path / "latest.pt", tmp_path / "frozen.pt"
    source.write_bytes(b"original-policy")
    expected = file_hash(source)
    source.write_bytes(b"later-policy")
    with pytest.raises(ValueError, match="changed since"):
        copy_bound(source, target, expected)
    assert not target.exists()
    source.write_bytes(b"original-policy")
    copy_bound(source, target, expected)
    source.write_bytes(b"later-policy")
    copy_bound(source, target, expected)
    assert target.read_bytes() == b"original-policy"


@pytest.mark.parametrize("experiment", ["n12", "n13"])
def test_collect_freezes_source_and_can_resume_from_the_frozen_checkpoint(source, tmp_path, experiment):
    # A simulated rank with zero allocated GPUs exercises receipts on CPU.
    source.plan["world_size"] = 0
    source.plan_path.write_text(json.dumps(source.plan))
    source.marker["plan_sha256"] = file_hash(source.plan_path)
    (source.checkpoint.parent / "run.json").write_text(json.dumps(source.marker))
    args = SimpleNamespace(checkpoint=source.checkpoint, bundle=source.bundle, plan=source.plan_path,
                           seed=5, experiment=experiment, output_root=tmp_path / "diagnostics", a_only=False)
    backend = diagnostic_backend(tmp_path)
    before = file_hash(source.checkpoint)

    def make_backend(*arguments):
        backend.cost_meter = arguments[-1]
        return backend

    with patch("srgc_research.followup.measure.measurement_hash", return_value="e" * 64), \
            patch("srgc_rebuttal.distributed.initialize", return_value=(0, 0)), \
            patch("torch.distributed.destroy_process_group"), \
            patch("scripts.srgc_verifier_fallback.install"), \
            patch("srgc_research.followup.measure.make_backend", side_effect=make_backend) as make:
        collect(args)
        endpoint_path = next(args.output_root.rglob("endpoint.json"))
        endpoint = read(endpoint_path)
        frozen = endpoint_path.parent / "source-checkpoint.pt"
        assert file_hash(frozen) == before and endpoint["optimizer_updates"] == 0
        assert endpoint["artifact_sha256"]["source-checkpoint.pt"] == before
        if experiment == "n12":
            expected_calls = 20
        else:
            vectors = read(endpoint_path.parent / "vectors.json.gz")
            expected_calls = 2 * (len(set(vectors["candidate_ids"] + vectors["sr_comparison_ids"])) + 50)
            assert set(endpoint["costs"]["diagnostic_phases"]) >= {"n13_scoring_A", "n13_scoring_B"}
        assert len(backend.calls) == expected_calls and make.call_count == 1
        source.checkpoint.unlink()
        args.checkpoint = frozen
        collect(args)
        assert make.call_count == 1 and len(backend.calls) == expected_calls
        bad_result = copy.deepcopy(endpoint)
        if experiment == "n13":
            bad_result["result"]["d4_A"] += 1
        else:
            bad_result["result"]["response_count"] = 0
        endpoint_path.write_text(json.dumps(bad_result))
        with pytest.raises(ValueError, match="result differs"):
            collect(args)
        missing_evidence = copy.deepcopy(endpoint)
        del missing_evidence["artifact_sha256"]["source-checkpoint.pt"]
        endpoint_path.write_text(json.dumps(missing_evidence))
        with pytest.raises(ValueError, match="required artifacts"):
            collect(args)
        assert make.call_count == 1


def test_new_phase_labels_reconcile_with_original_cost_receipts(tmp_path):
    from srgc_research.followup.costs import DiagnosticLedger

    ledger = DiagnosticLedger(tmp_path / "costs")
    meter = torch_meter(ledger.record, cuda=False)
    with meter.phase("n13_scoring_A", 125, 0):
        meter.count("responses", 8)
    with meter.phase("n13_scoring_B", 125, 0):
        meter.count("responses", 8)
    totals = ledger.totals()
    assert totals["complete"] and totals["total_gpu_seconds"] == 0
    assert set(totals["diagnostic_phases"]) == {"n13_scoring_A", "n13_scoring_B"}
    assert totals["counts"]["selection.responses"] == 16
    assert not totals["diagnostic_phases_additive_to_phase_totals"]


@pytest.mark.parametrize("experiment", ["n12", "n13"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_kill_after_cost_snapshot_can_finish_without_new_rollouts(source, tmp_path, experiment, cleanup_failure):
    from srgc_research.followup import measure

    source.plan["world_size"] = 0
    source.plan_path.write_text(json.dumps(source.plan))
    source.marker["plan_sha256"] = file_hash(source.plan_path)
    (source.checkpoint.parent / "run.json").write_text(json.dumps(source.marker))
    args = SimpleNamespace(checkpoint=source.checkpoint, bundle=source.bundle, plan=source.plan_path,
                           seed=5, experiment=experiment, output_root=tmp_path / "diagnostics", a_only=False)
    backend = diagnostic_backend(tmp_path)
    original_write = measure.write

    def interrupted(path, value):
        if path.name == "endpoint.json" and (path.parent / "costs.json").exists():
            raise RuntimeError("node killed after cost snapshot")
        original_write(path, value)

    def make(*arguments):
        backend.cost_meter = arguments[-1]
        return backend

    with patch("srgc_research.followup.measure.measurement_hash", return_value="e" * 64), \
            patch("srgc_rebuttal.distributed.initialize", return_value=(0, 0)), \
            patch("torch.distributed.destroy_process_group", side_effect=[RuntimeError("NCCL cleanup failure"), None] if cleanup_failure else None), \
            patch("scripts.srgc_verifier_fallback.install"), \
            patch("srgc_research.followup.measure.make_backend", side_effect=make):
        with patch.object(measure, "write", side_effect=interrupted), pytest.raises(RuntimeError, match="node killed") as failure:
            collect(args)
        if cleanup_failure:
            assert any("cleanup also failed" in note for note in failure.value.__notes__)
        calls = len(backend.calls)
        assert calls > 0 and not list(args.output_root.rglob("endpoint.json"))
        collect(args)
        assert len(backend.calls) == calls
        endpoint = read(next(args.output_root.rglob("endpoint.json")))
        assert endpoint["optimizer_updates"] == 0 and endpoint["costs"]["recorded_phases"] >= 4


def test_training_draw_duplicate_is_not_a_four_prompt_batch(source, tmp_path):
    groups = []
    for prompt, draw in (("p1", 0), ("p1", 1), ("p2", 0), ("p3", 0)):
        groups.append({**group(4, prompt=prompt), "draw": draw, "cached_success_rate": int(prompt[1:]) / 8})
    document = trace(groups)
    document["identity"]["input_sha256"] = file_hash(source.bundle)
    document["history_record"] = {"checkpoint": 25, "completed_updates": 26, "train_ids": ["p1", "p2", "p3", "p4"]}
    path = tmp_path / "duplicate-draw.json"
    write(path, document)
    with pytest.raises(ValueError, match="training.*(draw|duplicate)"):
        n12.report(source.bundle, [path])


def test_same_policy_preserves_each_arm_stage_without_generating_again(source, tmp_path):
    source.plan["world_size"] = 0
    source.plan_path.write_text(json.dumps(source.plan))
    source.marker["plan_sha256"] = file_hash(source.plan_path)
    (source.checkpoint.parent / "run.json").write_text(json.dumps(source.marker))
    args = SimpleNamespace(checkpoint=source.checkpoint, bundle=source.bundle, plan=source.plan_path,
                           seed=5, experiment="n12", output_root=tmp_path / "diagnostics", a_only=False)
    backend = diagnostic_backend(tmp_path)

    def make(*arguments):
        backend.cost_meter = arguments[-1]
        return backend

    with patch("srgc_research.followup.measure.measurement_hash", return_value="e" * 64), \
            patch("srgc_rebuttal.distributed.initialize", return_value=(0, 0)), \
            patch("torch.distributed.destroy_process_group"), \
            patch("scripts.srgc_verifier_fallback.install"), \
            patch("srgc_research.followup.measure.make_backend", side_effect=make) as mocked:
        collect(args)
        source.state.update(step=250, arm="sr")
        torch.save(source.state, source.checkpoint)
        collect(args)
        assert mocked.call_count == 1 and len(backend.calls) == 20
        endpoints = list(args.output_root.rglob("endpoint.json"))
        assert len(endpoints) == 1
        report = n12.panel_report(source.bundle, endpoints, 5)
        assert {(r["stage"], r["arm"]) for r in report["rows"]} == {(125, "on_policy"), (250, "sr")}
        assert len(report["rows"]) == 40 and report["distinct_measured_policies"] == 1
        assert report["response_count"] == 160
        source.state["checkpoint_policy"] = {"attention": "sdpa"}
        torch.save(source.state, source.checkpoint)
        collect(args)
        assert mocked.call_count == 2 and len(backend.calls) == 40
        report = n12.panel_report(source.bundle, list(args.output_root.rglob("endpoint.json")), 5)
        assert report["response_count"] == 320 and {r["attention"] for r in report["rows"]} == {"eager", "sdpa"}


def test_historical_progress_endpoint_overlap_is_counted_once(tmp_path):
    identity = {k: "a" * 64 for k in ("input_sha256", "plan_sha256", "implementation_sha256")}
    endpoint = {**identity, "seed": 5, "arm": "switch", "checks": [{"step": 25, "d": -.4}]}
    archive = {"source_results": {"seed-5/switch-endpoint.json": endpoint, "seed-5/switch-progress.json": endpoint}}
    path = tmp_path / "overlap.json"
    write(path, archive)
    main(["n13-report", str(path), "--output-dir", str(tmp_path / "report")])
    report = read(tmp_path / "report" / "report.json")
    assert report["coverage"]["unmeasured"] == 1 and report["duplicate_historical_checks_removed"] == 1
    archive["source_results"]["seed-5/switch-progress.json"] = {**endpoint, "checks": [{"step": 25, "d": .2}]}
    path.write_text(json.dumps(archive))
    with pytest.raises(SystemExit):
        main(["n13-report", str(path), "--output-dir", str(tmp_path / "conflicting")])


@pytest.mark.parametrize("damage", ["attention_none", "attention_unknown", "float_config", "string_score_names"])
def test_malformed_checkpoint_is_rejected_in_inventory(source, damage):
    if damage == "attention_none":
        source.state["checkpoint_policy"] = None
    elif damage == "attention_unknown":
        source.state["checkpoint_policy"] = {"attention": "unknown-kernel"}
    elif damage == "float_config":
        source.state["config"]["candidate_group_size"] = 4.
    else:
        source.state["backend"]["score_names"] = "weight"
    torch.save(source.state, source.checkpoint)
    report = checkpoints.inventory([source.checkpoint], source.bundle, source.plan_path, 5)
    assert report["rows"][0]["status"] == "invalid_or_other_run"


def test_equal_rewards_cannot_join_different_prefix_trajectories():
    analysis = Analysis()
    identity = {"input_sha256": "a" * 64, "plan_sha256": "b" * 64, "implementation_sha256": "c" * 64}
    for prefix in ("d" * 64, "e" * 64):
        context = analysis.context({"model": "fixture", "dataset": "math", "seed": 5},
            {**identity, "prefix_checkpoint_sha256": prefix, "seed": 5, "arm": "on_policy"}, {"path": "source", "sha256": "f" * 64})
        analysis.curve(context, 125, .4, "held_out", "source")
        analysis.curve(context, 150, .6, "held_out", "source")
    row = {"stage": 125, "seed": 5, "identity": {**identity, "source_implementation_sha256": "c" * 64, "arm": "on_policy"}}
    result = n13.nearby_evaluation(row, analysis)
    assert result["held_out_gain_next_25"] is None and result["evaluation_link_issue"]
    row["identity"]["prefix_checkpoint_sha256"] = "d" * 64
    assert n13.nearby_evaluation(row, analysis)["held_out_gain_next_25"] == pytest.approx(.2)


@pytest.mark.parametrize("value", [True, "1"])
def test_projected_vectors_do_not_coerce_booleans_or_strings(value):
    document = contrast_document()
    document["blocks"]["A"]["gradients"]["c0"] = [value, value]
    with pytest.raises(ValueError, match="numeric values"):
        n13.compare(document)


@pytest.mark.parametrize("case", ["complete", "root_error", "rank_error"])
def test_four_worker_cpu_startup_agrees_before_cuda(tmp_path, case):
    output = tmp_path / case
    output.mkdir()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "--max_restarts=0",
               "--module", "srgc_research.tests.followup_startup_smoke", "--case", case, "--output", str(output)]
    process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90, check=False)
    assert process.returncode == 0, process.stderr[-4000:]
    rows = [read(output / f"rank-{rank}.json") for rank in range(4)]
    assert all(not r["cuda_initialized"] and not r["group_initialized"] for r in rows)
    if case == "complete":
        assert all(r["result"]["complete"] is True and r["workers"] == 4 for r in rows)
        assert read(output / "root-called.json")["rank"] == 0
    else:
        assert len({r["error"] for r in rows}) == 1
        assert ("root checkpoint corrupted" if case == "root_error" else "rank 2 source checksum differs") in rows[0]["error"]


@pytest.mark.parametrize("damage", ["missing_records", "invalid_reference"])
def test_checkpoint_source_requires_usable_generation_records(source, damage):
    if damage == "missing_records":
        del source.data["records"]
    else:
        source.data["records"]["v0"] = None
    source.bundle.write_text(json.dumps(source.data))
    source.marker["input_sha256"] = file_hash(source.bundle)
    (source.checkpoint.parent / "run.json").write_text(json.dumps(source.marker))
    with pytest.raises((ValueError, TypeError), match="prompt/verifier records"):
        checkpoints.load_source(source.checkpoint, source.bundle, source.plan_path, 5)


def test_gemma_disables_optional_media_before_importing_peft(tmp_path):
    import builtins

    from srgc_research.followup.measure import make_backend

    class StopBeforeModelLoad(Exception):
        pass

    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "peft":
            assert disabled.call_count == 1
            raise StopBeforeModelLoad
        return original(name, *args, **kwargs)

    with patch("srgc_research.dispatch.gemma4.runtime.disable_optional_media") as disabled, \
            patch("builtins.__import__", side_effect=importing), pytest.raises(StopBeforeModelLoad):
        make_backend(tmp_path / "plan.json", {"extension": "gemma4-12b-pt-v1"}, {}, {}, torch.device("cpu"), None)


@pytest.mark.parametrize("publisher", ["sr", "random", "switch"])
def test_shared_prefix_archive_published_by_a_continuation_arm_is_usable(source, tmp_path, publisher):
    # reward_curves.capture_boundary lets the first continuation arm publish
    # reward-curves/prefix/step-25.pt when the prefix task did not; the saved
    # Engine arm is then that fork although the weights are the shared prefix.
    archive = tmp_path / "reward-curves" / "prefix" / "step-25.pt"
    archive.parent.mkdir(parents=True)
    state = copy.deepcopy(source.state)
    state.update(step=25, arm=publisher, evaluation_only=True)
    del state["backend"]["optimizer"]
    protocol = {**source.marker, "arm": "prefix"}
    state["reward_curve_binding"] = {"protocol_sha256": fingerprint(protocol), "step": 25}
    write(archive.parent / "protocol.json", protocol)
    torch.save(state, archive)
    _, _, _, info = checkpoints.load_source(archive, source.bundle, source.plan_path, 5)
    assert info["shared_prefix"] and info["stage"] == 25
    report = checkpoints.inventory([archive], source.bundle, source.plan_path, 5)
    assert all(r["present"] for r in report["requested_stage_coverage"] if r["stage"] == 25)
    # A prefix protocol never vouches for a later continuation state.
    state["step"] = 125
    state["reward_curve_binding"] = {"protocol_sha256": fingerprint(protocol), "step": 125}
    torch.save(state, archive)
    with pytest.raises(ValueError, match="arm differs"):
        checkpoints.load_source(archive, source.bundle, source.plan_path, 5)


def test_shared_prefix_contexts_from_different_publishers_are_one_panel_row(source, tmp_path):
    backend = diagnostic_backend(tmp_path)
    fixed = n12.panel(source.data, source.identity["input_sha256"], 5)
    output = tmp_path / "measurement"
    result = {**n12_acquire(backend, source.data, fixed, source.identity), "source": {"stage": 25, "arm": "on_policy"}}
    write(output / "panel-measurement.json.gz", result)
    for arm, checkpoint in (("on_policy", "a" * 64), ("sr", "b" * 64)):
        n12.record_context(output, source.identity, {**source.identity, "stage": 25, "arm": arm, "attention": None,
                                                    "checkpoint_sha256": checkpoint, "shared_prefix": True})
    report = n12.panel_report(source.bundle, [output / "panel-measurement.json.gz"], 5)
    assert len(report["rows"]) == 20 and {r["arm"] for r in report["rows"]} == {"prefix"}
    assert report["arm_stage_contexts"] == 1 and report["duplicate_policy_records_removed"] == 1


def test_backup_of_shared_prefix_keeps_its_prefix_identity(source, tmp_path):
    source.state["step"] = 25
    prefix = source.checkpoint.with_name("prefix.pt")
    torch.save(source.state, prefix)
    target = tmp_path / "checkpoint-backups" / "seed-5" / "prefix" / file_hash(prefix) / "checkpoint.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(prefix.read_bytes())
    write(target.parent / "receipt.json", {"schema": "srgc-checkpoint-backup-v1", "sha256": file_hash(target),
                                          "identity": source.marker, "source": str(prefix)})
    _, _, _, native = checkpoints.load_source(prefix, source.bundle, source.plan_path, 5)
    _, _, _, backup = checkpoints.load_source(target, source.bundle, source.plan_path, 5)
    assert native["shared_prefix"] and backup["shared_prefix"]
    report = checkpoints.inventory([target], source.bundle, source.plan_path, 5)
    assert all(r["present"] for r in report["requested_stage_coverage"] if r["stage"] == 25)
    # The latest file of a continuation arm is never relabelled as the prefix.
    latest = target.parent.parent.parent / "on_policy-latest" / file_hash(prefix) / "checkpoint.pt"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(prefix.read_bytes())
    write(latest.parent / "receipt.json", {"schema": "srgc-checkpoint-backup-v1", "sha256": file_hash(latest),
                                          "identity": source.marker, "source": str(source.checkpoint)})
    assert not checkpoints.load_source(latest, source.bundle, source.plan_path, 5)[3]["shared_prefix"]


def test_fixed_panel_records_its_generation_seed(source, tmp_path):
    backend = diagnostic_backend(tmp_path)
    fixed = n12.panel(source.data, source.identity["input_sha256"], 5)
    result = n12_acquire(backend, source.data, fixed, source.identity)
    assert {seed for _, _, seed in backend.calls} == {result["generation_seed"]}
    assert result["generation_world_size"] == backend.world
