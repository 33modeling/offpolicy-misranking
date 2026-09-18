"""Random-only inventory must distinguish roots and preserve every saved byte."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/random_storage_audit.py"


@pytest.fixture
def auditor(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("random_storage_audit_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def make_root(work, name, manifest="switch.json"):
    root = work / "runs" / name
    write_json(root / manifest, {"dataset": "mbpp"})
    return root


def branch(root, arm="random_full", step=25, seed=0):
    path = root / f"states/s{seed}-t{step}/points/view-{step}" / arm
    path.mkdir(parents=True, exist_ok=True)
    return path


def publish(path, step=35):
    write_json(path / "result.json", {"complete": True, "completed_steps": step})
    write_json(path / "result.sha256.json", {
        "sha256": hashlib.sha256((path / "result.json").read_bytes()).hexdigest(),
    })


def test_all_roots_and_random_arms_are_counted_without_21_nonrandom_results(auditor, tmp_path):
    roots = [make_root(tmp_path, name) for name in ("fresh", "quality", "math")]
    for root in roots:
        publish(branch(root, "random_full"))
        publish(branch(root, "random_reduced"), 45)
    for seed in range(21):
        publish(branch(roots[0], "selection_full", seed=seed))
    reports, problems = auditor.inventory(tmp_path)
    assert not problems
    assert len(reports) == 3
    assert sum(len(report["rows"]) for report in reports) == 6
    for report in reports:
        assert {row["arm"] for row in report["rows"]} == {"RF", "RR"}
        assert all(row["code"] == "D" for row in report["rows"])
    output = auditor.render(tmp_path, reports, problems, "test")
    assert output.count("RF:D=1 RR:D=1") == 3
    assert "selection_full" not in output


def test_mopps_random_online_uses_its_actual_direct_state_layout(auditor, tmp_path):
    root = make_root(tmp_path, "mopps-comparison-v1", "mopps.json")
    path = root / "states/s3-t100/random_online"
    publish(path, 140)
    reports, problems = auditor.inventory(tmp_path)
    assert not problems
    assert len(reports) == 1
    assert len(reports[0]["rows"]) == 1
    assert reports[0]["rows"][0]["code"] == "D"
    assert reports[0]["rows"][0]["step"] == 140
    assert reports[0]["rows"][0]["arm"] not in {"RF", "RR"}


def test_top_level_points_random_layout_is_discovered(auditor, tmp_path):
    root = make_root(tmp_path, "point-root")
    publish(root / "points/view-25/random_reduced")
    reports, _ = auditor.inventory(tmp_path)
    assert reports[0]["rows"][0]["arm"] == "RR"
    assert reports[0]["rows"][0]["code"] == "D"


@pytest.mark.parametrize("kind,flag", [
    ("missing-stop", "FINAL_STOP_MISSING"),
    ("parent-stop", "PARENT_STOP_WITH_SAVED_POLICY"),
    ("missing-result", "RESULT_MISSING_SEAL_REMAINS"),
])
def test_random_saved_state_inconsistencies_are_explicit(auditor, tmp_path, kind, flag):
    directory = branch(make_root(tmp_path, "fresh"))
    if kind == "missing-result":
        write_json(directory / "result.sha256.json", {"sha256": "f" * 64})
    else:
        write_json(directory / "policy/policy_train.json", {"completed_steps": 35})
        if kind == "parent-stop":
            write_json(directory / "policy/budget_stop.json", {"use_parent_policy": True})
    reports, _ = auditor.inventory(tmp_path)
    assert flag in reports[0]["rows"][0]["flags"]


def test_active_and_archived_result_coexistence_is_not_hidden(auditor, tmp_path):
    directory = branch(make_root(tmp_path, "fresh"))
    publish(directory)
    publish(directory / "discarded/old-attempt", 30)
    reports, _ = auditor.inventory(tmp_path)
    row = reports[0]["rows"][0]
    assert row["code"] == "D"
    assert "ARCHIVE=old-attempt" in row["flags"]
    output = auditor.render(tmp_path, reports, [], "test")
    assert "ARCHIVE=old-attempt" in output


def test_invalid_result_seal_is_never_counted_as_done(auditor, tmp_path):
    directory = branch(make_root(tmp_path, "fresh"))
    publish(directory)
    write_json(directory / "result.sha256.json", {"sha256": "bad"})
    reports, _ = auditor.inventory(tmp_path)
    assert reports[0]["rows"][0]["code"] == "U"


def test_inventory_does_not_read_payloads_or_modify_storage(auditor, tmp_path, monkeypatch):
    directory = branch(make_root(tmp_path, "fresh"))
    publish(directory)
    write_json(directory / "policy/policy_train.json", {"completed_steps": 35})
    write_json(directory / "policy/budget_stop.json", {"use_parent_policy": False})
    for name in ("adapter_model.safetensors", "optimizer.pt", "rollouts.jsonl"):
        (directory / "policy" / name).write_bytes(b"private tensor or rollout payload")
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    original_open = Path.open

    def forbid_payload_read(path, *args, **kwargs):
        assert path.name not in {"adapter_model.safetensors", "optimizer.pt", "rollouts.jsonl"}
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", forbid_payload_read)
        reports, problems = auditor.inventory(tmp_path)
        assert reports[0]["rows"][0]["code"] == "D"
        assert not problems
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_render_is_at_most_four_kib_with_many_failures(auditor, tmp_path):
    root = make_root(tmp_path, "fresh")
    for seed in range(70):
        directory = branch(root, seed=seed)
        write_json(directory / "failure.json", {"error": "failure " + "한글" * 80})
    reports, problems = auditor.inventory(tmp_path)
    output = auditor.render(tmp_path, reports, problems, "test")
    assert len(output.encode()) <= 4096
    assert "RF:N=70" in output
    assert "omitted" in output


@pytest.mark.parametrize("wrapper", [False, True])
def test_cli_saves_tiny_report_in_isolated_home_even_when_volume_missing(tmp_path, wrapper):
    report_home = tmp_path / "isolated-home"
    report_home.mkdir()
    missing = tmp_path / "missing-group-volume"
    env = {**os.environ, "HOME": str(report_home), "OM_WORK": str(missing),
           "SWITCH_PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}
    command = (["bash", "scripts/check_random_storage.sh"] if wrapper
               else [sys.executable, str(SCRIPT), "--work", str(missing)])
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True,
                            timeout=20, check=False)
    assert result.returncode == 2, result.stdout + result.stderr
    reports = list(report_home.glob("random-storage-*.txt"))
    assert len(reports) == 1
    assert 0 < reports[0].stat().st_size <= 4096
    assert "STORAGE_UNAVAILABLE" in reports[0].read_text()
    assert str(reports[0]) in result.stdout
    assert not missing.exists()
