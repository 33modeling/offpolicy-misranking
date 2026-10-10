"""One-file export of audited observations from both datasets and all stages."""

import json
import shutil
from unittest.mock import patch

import pytest

from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.torch_backend import TorchBackend
from srgc_research.dispatch import information_results as export
from srgc_research.information import InformationStudy
from srgc_research.tests import test_information as fixtures
from srgc_research.tests import test_information_wrapper as wrappers

backend, measured, wrapper = fixtures.backend, fixtures.measured, wrappers.wrapper


def test_both_datasets_and_essential_updates_fit_one_file(tmp_path, measured, backend):
    root = tmp_path / "selection-information"
    shutil.copytree(measured, root / "math/seed-5/t0")
    mbpp = root / "mbpp/seed-5/t0"
    data, config, identity = fixtures.setup_measurement(mbpp, backend)
    identity["dataset"] = "mbpp"
    atomic_json(mbpp / "manifest.json", {"identity": identity, "input_sha256": identity["input_sha256"]})
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        InformationStudy(backend, data, config, 0, mbpp, identity, probe_prompts=1).run()
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert export.export(work=tmp_path) == 0
    files = list((tmp_path / "results").iterdir())
    assert [p.name for p in files] == ["information-results.json"]
    result = json.loads(files[0].read_text())
    assert result["coverage"]["completed_initial_measurements"] == 2
    assert set(result["datasets"]) == {"math", "mbpp"} and not result["complete"]
    assert len(result["measurements"]) == 2 and len(result["pending"]) == 8
    assert {r["dataset"] for r in result["selected_problems"]} == {"math", "mbpp"}
    assert result["format"] == "compact"
    assert all("phases" not in measurement and "endpoint" not in measurement for measurement in result["measurements"])
    assert all(row["response_updates"] for row in result["selected_problems"])
    assert not {"sequence_ids", "logps_before", "logps_after", "raw_samples", "parameter_updates"} & set(files[0].read_text().split('"'))
    assert files[0].stat().st_size < 150_000
    assert before == {p: p.read_bytes() for p in before}
    assert not list(tmp_path.rglob("*.html")) and not list(tmp_path.rglob("*.csv"))


def test_damaged_tensor_is_excluded_with_visible_error(tmp_path, measured):
    folder = tmp_path / "selection-information/math/seed-5/t0"
    shutil.copytree(measured, folder)
    with (folder / "sr.pt").open("ab") as handle:
        handle.write(b"corrupt")
    assert export.export(work=tmp_path) == 1
    result = json.loads((tmp_path / "results/information-results.json").read_text())
    assert not result["complete"] and result["measurements"] == []
    assert result["errors"][0]["seed"] == 5


def test_additional_saved_stage_is_included_without_changing_t0_coverage(tmp_path, measured, backend):
    folder = tmp_path / "selection-information/math/seed-5/t100"
    data, config, identity = fixtures.setup_measurement(folder, backend, stage=100)
    with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
        InformationStudy(backend, data, config, 100, folder, identity, probe_prompts=1).run()
    assert export.export(work=tmp_path) == 0
    result = json.loads((tmp_path / "results/information-results.json").read_text())
    assert result["coverage"]["completed_initial_measurements"] == 0
    assert result["measurements"][0]["stage"] == 100
    assert result["datasets"]["math"]["stages"] == [0, 100]


def test_wrong_directory_identity_is_rejected(tmp_path, measured):
    shutil.copytree(measured, tmp_path / "selection-information/mbpp/seed-5/t0")
    assert export.export(work=tmp_path) == 1
    result = json.loads((tmp_path / "results/information-results.json").read_text())
    assert result["measurements"] == [] and "identity differs" in result["errors"][0]["error"]


def test_public_one_line_command_needs_no_input_flags_or_gpu_environment(wrapper):
    run, work, _ = wrapper
    result, calls = run("all", "results", overrides={"PAIR_PYTHON": "/missing", "SWITCH_PYTHON": "/missing"})
    assert result.returncode == 0, result.stderr
    assert calls == []
    destination = work / "results/information-results.json"
    assert f"RESULT: {destination}" in result.stdout
    assert len(json.loads(destination.read_text())["pending"]) == 10
    assert list((work / "results").iterdir()) == [destination]


def test_rerun_updates_the_same_file_without_touching_measurements(tmp_path, measured):
    source = tmp_path / "selection-information/math/seed-5/t0"
    shutil.copytree(measured, source)
    for _ in range(2):
        assert export.export("math", work=tmp_path) == 0
    assert [p.name for p in (tmp_path / "results").iterdir()] == ["information-results.json"]


def test_result_directory_cannot_redirect_output_over_sources(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    elsewhere = tmp_path / "sources"
    elsewhere.mkdir()
    (work / "results").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        export.export(work=work)
    assert list(elsewhere.iterdir()) == []


def test_large_response_arrays_and_duplicate_text_are_removed_without_changing_metrics():
    identity = {"dataset": "math", "seed": 7, "stage": 0, "input_sha256": "a" * 64}
    sample = {"response": 0, "reward": 1, "completion_tokens": 20_000, "mean_logp_change": .01,
              "sequence_ids": list(range(20_000)), "logps_before": [-2.0] * 20_000,
              "logps_after": [-1.99] * 20_000, "text": "response " * 20_000}
    row = {**identity, "id": "p1", "question": "chosen problem", "prompt": "duplicate problem",
           "answer": "2", "training_success_rate": .5, "loss_gradient_norm": .25, "raw_samples": [sample] * 8}
    original = {"measurements": [{"dataset": "math", "seed": 7, "stage": 0,
        "endpoint": {"identity": identity, "configuration": {"responses": 8}},
        "phases": {"score-A": {"responses": {"p1": [sample] * 100}}}}],
        "selected_problems": [row], "candidate_information": [row],
        "batch_updates": [{**identity, "update_norm": .003}],
        "parameter_updates": [{"name": "layer0", "unused_detail": sample}]}
    result = export.compact(original)
    assert len(json.dumps(result)) < 4_000
    selected = result["selected_problems"][0]
    assert selected["training_success_rate"] == .5 and selected["loss_gradient_norm"] == .25
    assert selected["question"] == "chosen problem" and selected["answer"] == "2"
    assert result["batch_updates"][0]["update_norm"] == .003
    assert result["candidate_information"][0]["id"] == "p1"
    assert "question" not in result["candidate_information"][0]
    assert "input_sha256" not in selected
    assert result["measurements"][0]["identity"]["input_sha256"] == "a" * 64
    assert original["selected_problems"][0]["raw_samples"][0]["sequence_ids"]
    assert export.compact(result) == result
