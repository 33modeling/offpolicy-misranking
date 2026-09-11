import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_additional_experiments import _write_run
from test_evidence_downstream import source_point

import method_choice as mc


def test_ties_share_seed_hash_order_and_keep_all_maxima():
    left = mc.choose({"g00": 1., "g10": 1., "g11": 0.}, 3)
    right = mc.choose({"g10": .2, "g00": .2, "g11": .1}, 3)
    assert left["method"] == right["method"]
    assert left["tied_maxima"] == ["g00", "g10"]


def test_decisions_match_existing_subset_writer(tmp_path):
    run, _ = source_point(tmp_path, drift=100)
    choice = mc.decisions(run)
    written = mc.ed.write_subsets(run, tmp_path / "subsets", .1, 0)
    for method, path in written.items():
        assert mc.ed.read(path)["selected_idx"] == choice["subsets"][method]
    assert set(choice["alignment"]["values"]) == set(mc.ESTIMATORS)
    assert set(choice["half_sensitivity"]) == {"a", "b"}


def test_lease_is_nonblocking_and_released(tmp_path):
    path = tmp_path / "arm.lock"
    with mc.lease(path) as first:
        assert first
        with mc.lease(path) as second:
            assert not second
    with mc.lease(path) as third:
        assert third


def test_controller_never_accepts_cpu_execution_as_gpu_work(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "verify", lambda _: {"seeds": []})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="four allocated"):
        mc.work(tmp_path)


def test_phase_records_failed_attempts_and_does_not_lose_exit_status(tmp_path):
    out = tmp_path / "seed"
    out.mkdir()
    mc.ed.atomic_json(out / "experiment.json", {"seed": 0})
    code = mc.phase(tmp_path, out, "g00", "training", [[sys.executable, "-c", "raise SystemExit(7)"]], ["0"], os.environ)
    events = [json.loads(line) for line in (tmp_path / "cost.jsonl").read_text().splitlines()]
    assert code == 7
    assert events[0]["state"] == "started" and events[1]["state"] == "finished"
    assert events[1]["exit_code"] == 7 and events[1]["allocated_gpu_seconds"] > 0


def test_summary_keeps_equal_choices_and_negative_seeds(tmp_path, monkeypatch):
    suite = {"seeds": []}
    for seed in range(5):
        out = tmp_path / f"s{seed}"
        out.mkdir()
        suite["seeds"].append({"seed": seed, "path": str(out)})
        mc.ed.atomic_json(out / "decision.json", {"alignment": {"method": "g00"},
                                                  "overlap": {"method": "g00" if seed == 0 else "g11"}})
        for arm in ("g00", "g11"):
            directory = out / arm / "evaluation"
            directory.mkdir(parents=True)
            for shard in range(4):
                (directory / f"shard-{shard}.done.json").write_text("{}")
    mc.ed.atomic_json(tmp_path / "suite.json", suite)
    monkeypatch.setattr(mc, "verify", lambda _: suite)
    monkeypatch.setattr(mc.ed, "evaluation_means", lambda out, arm: np.full(8, .4 if arm == "g00" else .6))
    result = mc.summarize(tmp_path)
    assert result["complete_primary_contrast"]
    assert result["rows"][0]["difference"] == 0
    assert result["mean_difference"] == pytest.approx(-.16)
    assert not result["cost"]["end_to_end_complete"]
    assert result["rows"][0]["full_pool_reward"] is None
    (tmp_path / "s4/g11/evaluation/shard-0.done.json").unlink()
    incomplete = mc.summarize(tmp_path)
    assert incomplete["missing_seed_contrasts"] == [4]
    assert incomplete["mean_difference"] is None


def test_scripts_parse_and_plan_does_not_create_suite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("run_method_choice.sh", "run_measurement_audit.sh"):
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)
    env = dict(os.environ, OM_WORK=str(tmp_path / "work"), METHOD_CHOICE_ROOT=str(tmp_path / "suite"))
    result = subprocess.run(["bash", "scripts/run_method_choice.sh", "plan"], cwd=root, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "8000 updates" in result.stdout
    assert not (tmp_path / "suite").exists()


def suite_inputs(tmp_path):
    matrix = tmp_path / "matrix"
    for seed in range(5):
        old, _ = source_point(tmp_path / f"fixture{seed}", drift=100)
        run = _write_run(matrix / f"family-math500-s{seed}", n=400, seed=seed, drift=100)
        config = mc.ed.read(old / "run_config.json")
        config.update(seed=seed, n_train=400)
        mc.ed.atomic_json(run / "run_config.json", config)
        shutil.copytree(old / "policy_step_100", run / "policy_step_100")
        policy = mc.ed.read(run / "policy_step_100/policy_train.json")
        policy["seed"] = seed
        mc.ed.atomic_json(run / "policy_step_100/policy_train.json", policy)
        (run / "DONE").write_text("complete")
    data = tmp_path / "data"
    data.mkdir()
    pool = data / "pool.jsonl"
    pool.write_text("".join(json.dumps({"problem": f"test question {i}", "answer": "2"})+"\n" for i in range(520)))
    manifest = data / "manifest.json"
    mc.ed.atomic_json(manifest, {"source_revision": "test-frozen"})
    return matrix, pool, manifest


def test_full_suite_freeze_resume_and_tamper_rejection(tmp_path):
    matrix, pool, manifest = suite_inputs(tmp_path)
    original = {str(p): mc.ed.digest(p) for p in matrix.rglob("*") if p.is_file()}
    root = tmp_path / "suite"
    first = mc.prepare(root, matrix, pool, manifest)
    assert first == mc.prepare(root, matrix, pool, manifest)
    assert first == mc.verify(root)
    assert original == {str(p): mc.ed.digest(p) for p in matrix.rglob("*") if p.is_file()}
    assert len(first["seeds"]) == 5
    for entry in first["seeds"]:
        out = Path(entry["path"])
        assert len(mc.ed.read(out / "subsets/subset-full_pool.json")["train"]) == 400
        assert len(mc.ed.read(out / "subsets/subset-g11.json")["train"]) == 40
        assert len(mc.ed.read(out / "evaluation.json")["val"]) == 500
        assert mc.ed.digest(out / "evaluation.json") == mc.ed.read(out / "experiment.json")["evaluation_payload_sha256"]
    changed = root / "seeds/s0/decision.json"
    changed.write_text("{}")
    with pytest.raises(ValueError, match="frozen file changed"):
        mc.verify(root)


def test_source_failure_cannot_freeze_partial_seed_set(tmp_path):
    matrix, pool, manifest = suite_inputs(tmp_path)
    next((matrix / "family-math500-s4").glob("*/DONE")).unlink()
    root = tmp_path / "suite"
    with pytest.raises(ValueError, match="not complete"):
        mc.prepare(root, matrix, pool, manifest)
    assert not (root / "suite.json").exists() and not (root / "seeds").exists()


def test_work_skips_busy_arms_and_continues_after_training_failure(tmp_path, monkeypatch):
    out = tmp_path / "seed"
    out.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    mc.ed.atomic_json(source / "run_config.json", {"seed": 0})
    mc.ed.atomic_json(out / "experiment.json", {"seed": 0, "source_run": str(source),
                                               "selectors": ["g00", "g10", "full_pool"], "steps": 200})
    monkeypatch.setattr(mc, "verify", lambda _: {"seeds": [{"seed": 0, "path": str(out)}]})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("OM_NODE_LOCK_HELD", "1")
    monkeypatch.setattr(mc.signal, "signal", lambda *args: None)
    monkeypatch.setattr(mc.ed, "arm_policy", lambda *args: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(mc.ed, "train_args", lambda *args: ["mock-train"])
    calls = []
    def fake_phase(root, output, arm, phase, commands, gpus, env):
        calls.append((arm, phase))
        return int(arm == "g00" and phase == "training")
    monkeypatch.setattr(mc, "phase", fake_phase)
    with mc.lease(out / ".g10.lock"):
        assert mc.work(tmp_path) == 1
    assert ("g00", "training") in calls
    assert not any(arm == "g10" for arm, phase in calls)
    assert ("full_pool", "training") in calls and ("full_pool", "test_evaluation") in calls


def test_interrupted_preparation_resumes_but_not_after_training(tmp_path, monkeypatch):
    matrix, pool, manifest = suite_inputs(tmp_path)
    root = tmp_path / "suite"
    original = mc.ed.atomic_json
    def interrupted(path, value):
        if path.name == "suite.json":
            raise OSError("interrupted before final publication")
        return original(path, value)
    monkeypatch.setattr(mc.ed, "atomic_json", interrupted)
    with pytest.raises(OSError):
        mc.prepare(root, matrix, pool, manifest)
    assert (root / "preparation.json").exists() and not (root / "suite.json").exists()
    monkeypatch.setattr(mc.ed, "atomic_json", original)
    mc.prepare(root, matrix, pool, manifest)
    assert mc.verify(root)["prospective"]
    (root / "suite.json").unlink()
    (root / "seeds/s0/g00/policy").mkdir(parents=True)
    with pytest.raises(ValueError, match="predates"):
        mc.prepare(root, matrix, pool, manifest)


def test_cleanup_includes_late_export_controller_but_preserves_other_suite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    legacy = scripts / "run_method_choice.sh"
    legacy.write_text('export OUT_ROOT="$TARGET"\nsleep 120 &\ntouch "$READY"\nwait\n')
    ready = tmp_path / "ready"
    target = tmp_path / "suite"
    env = dict(os.environ, TARGET=str(target), READY=str(ready))
    env.pop("OUT_ROOT", None)
    old = subprocess.Popen(["bash", str(legacy)], env=env, start_new_session=True)
    unrelated = subprocess.Popen(["sleep", "120"], env=dict(env, OUT_ROOT=str(tmp_path / "reduced-e5")), start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.02)
        result = subprocess.run([sys.executable, str(root / "src/cleanup_run_processes.py"),
                                 "--run-prefix", str(target), "--command-pattern", str(target),
                                 "--command-pattern", "scripts/run_method_choice.sh",
                                 "--require-environment", f"OUT_ROOT={target}",
                                 "--launcher-environment-from-child", "--timeout", "2", "--compact"],
                                capture_output=True, text=True, timeout=8, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert old.wait(timeout=5) != 0
        assert unrelated.poll() is None
    finally:
        for proc in (old, unrelated):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
