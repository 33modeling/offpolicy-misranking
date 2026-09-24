"""CPU checks for the separate RLOO d=100 control (scripts/rloo_d100.py)."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import evidence_downstream as ed
import rloo_experiment as rloo
from test_rloo_experiment import inputs

sys.path.insert(0, str(rloo.ROOT / "scripts"))
import rloo_d100  # noqa: E402

SCRIPT = rloo.ROOT / "scripts/run_rloo_d100.sh"


@pytest.fixture(autouse=True)
def original_points():
    # The wrapper sets the module's point list; every other test keeps d0/d400.
    saved = rloo.POINTS
    yield
    rloo.POINTS = saved


def test_prepare_uses_only_d100_sources_and_outputs(tmp_path, monkeypatch):
    work, root = tmp_path / "work", tmp_path / "rloo-d100"
    configs = {rloo.source_paths(work, None, seed, 100) / "run_config.json":
               {"seed": seed, "drift": 100, "model": "same-model"} for seed in (0, 1, 2)}
    calls = []
    monkeypatch.setattr(ed, "read", lambda path: configs[path])
    monkeypatch.setattr(rloo, "prepare", lambda *a, **kw: calls.append((a, kw)))
    rloo_d100.prepare(work, root, None, None)
    assert len(calls) == 6
    assert all(kw.get("dry") for _, kw in calls[:3]) and all(not kw for _, kw in calls[3:])
    for ((run, out, evaluation), _), seed in zip(calls[3:], (0, 1, 2), strict=True):
        assert run == rloo.source_paths(work, None, seed, 100)
        assert out == root / "math500-d100" / f"s{seed}"
        assert evaluation == work / "inputs/e5-reduced/test-math500-d100.json"
    assert rloo.POINTS == ((100, 0), (100, 1), (100, 2))


def test_prepare_rejects_a_source_from_another_checkpoint(tmp_path, monkeypatch):
    work = tmp_path / "work"
    configs = {rloo.source_paths(work, None, seed, 100) / "run_config.json":
               {"seed": seed, "drift": 400 if seed == 1 else 100, "model": "m"} for seed in (0, 1, 2)}
    monkeypatch.setattr(ed, "read", lambda path: configs[path])
    monkeypatch.setattr(rloo, "prepare", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="not seed 1 at d=100"):
        rloo_d100.prepare(work, tmp_path / "rloo-d100", None, None)


@pytest.mark.parametrize("make", [
    lambda root: root.with_name("rloo-selector-v2"),
    lambda root: root.with_name("rloo-selector-v2") / "d100",
    lambda root: (root / "math500-d0").mkdir(parents=True) or root,
    lambda root: (root / "math500-d400").mkdir(parents=True) or root,
])
def test_original_root_is_never_reused(tmp_path, make):
    with pytest.raises(ValueError, match="never reused"):
        rloo_d100.guard_root(make(tmp_path / "new"))


def test_real_single_point_prepare_at_d100_and_grpo_subset_comparison(tmp_path, monkeypatch):
    run, evaluation = inputs(tmp_path, drift=100)
    out = tmp_path / "rloo-d100" / "math500-d100" / "s0"
    rloo.prepare(run, out, evaluation)
    contract = ed.read(out / "experiment.json")
    assert contract["source"]["drift"] == 100 and contract["objective"] == "rloo"
    monkeypatch.setattr(rloo_d100, "SEEDS", (0,))
    work = tmp_path / "work"
    notes = rloo_d100.matching_grpo_subsets(work, tmp_path / "rloo-d100")
    assert all("not compared" in note for note in notes)
    grpo = work / "runs/e5-reduced/math500-d100/s0/subsets"
    grpo.mkdir(parents=True)
    for arm in rloo.ARMS:
        (grpo / f"subset-{arm}.json").write_bytes((out / "subsets" / f"subset-{arm}.json").read_bytes())
    assert all("identical" in note for note in rloo_d100.matching_grpo_subsets(work, tmp_path / "rloo-d100"))
    changed = ed.read(grpo / "subset-fresh_r.json")
    changed["selected_idx"] = changed["selected_idx"][::-1]
    (grpo / "subset-fresh_r.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from the GRPO"):
        rloo_d100.matching_grpo_subsets(work, tmp_path / "rloo-d100")


def test_queue_and_report_see_only_d100_points():
    import queue_rloo
    import rloo_report
    rloo_d100.use_d100(queue_rloo.experiment)
    rloo_d100.use_d100(rloo_report.experiment)
    assert queue_rloo.experiment.POINTS == rloo_report.experiment.POINTS == rloo_d100.POINTS


def test_launcher_ignores_an_exported_original_root_and_needs_no_gpu(tmp_path):
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    env = {**os.environ, "RLOO_PYTHON": sys.executable, "OM_WORK": str(tmp_path),
           "RLOO_ROOT": str(tmp_path / "runs/rloo-selector-v2")}
    env.pop("RLOO_D100_ROOT", None)
    result = subprocess.run(["bash", str(SCRIPT), "plan"], env=env, text=True, capture_output=True, check=True)
    assert str(tmp_path / "runs/rloo-selector-d100") in result.stdout
    assert "9 continuations" in result.stdout
    result = subprocess.run(["bash", str(SCRIPT), "status"], env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count("not prepared") == 3
    assert not (tmp_path / "runs").exists()
    bad = subprocess.run(["bash", str(SCRIPT), "run", "--extra"], env=env, text=True, capture_output=True)
    assert bad.returncode == 2


def test_results_export_lists_only_d100_points_and_never_the_default_file(tmp_path):
    out = tmp_path / "rloo-d100-results.txt"
    (tmp_path / "rloo-selector-d100").mkdir()
    env = {**os.environ, "OM_WORK": str(tmp_path), "HOME": str(tmp_path / "home")}
    subprocess.run([sys.executable, str(rloo.ROOT / "scripts/rloo_d100.py"), "results",
                    "--root", str(tmp_path / "rloo-selector-d100"), "--out", str(out)],
                   env=env, text=True, capture_output=True, check=True)
    body = out.read_text()
    assert body.startswith("RLOO PAPER DATA")
    data = json.loads(body.split("DATA_JSON\n", 1)[1])
    assert [(p["drift"], p["seed"], p["status"]) for p in data["points"]] == [
        (100, 0, "unprepared"), (100, 1, "unprepared"), (100, 2, "unprepared")]
    assert not (tmp_path / "home" / "rloo-results.txt").exists()
