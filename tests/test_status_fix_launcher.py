"""MBPP read-only views follow the verified repair that the default queue runs.

`run` and `restart` switch the default quality queue to a verified
selection-switch-mbpp-quality-repair-v1, and `status` shows that root too. `progress`,
`why` and `saved` still read the original quality root. So progress said DONE 37 RUN 0
while status said DONE 39 RUN 1 for the same files (audit findings mbpp-04, mbpp-06).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "selection-switch-mbpp-quality-v1"
REPAIR = "selection-switch-mbpp-quality-repair-v1"
FRESH = "selection-switch-mbpp-v1"
CLEARED = ("SWITCH_MBPP_ROOT", "SWITCH_MBPP_QUALITY_ROOT", "SWITCH_MBPP_DIFFICULTY_ROOT", "SWITCH_MBPP_LONG_ROOT",
           "MBPP_REPAIR_ROOT", "EXPERIMENTS_MBPP_SUITE", "SWITCH_ROOT", "OUT_ROOT", "MODE")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def repaired_work(work):
    """Original quality root, its verified repair and the legacy fresh root (identity files only)."""
    runs = work / "runs"
    manifest = {"schema": "selection-switch/v1", "dataset": "mbpp", "selector": "fresh_r",
                "accounting": "matched", "gate": "convergence"}
    write_json(runs / ORIGINAL / "switch.json", manifest)
    write_json(runs / FRESH / "switch.json", {**manifest, "accounting": "budget", "gate": "final"})
    source = (runs / ORIGINAL / "switch.json").read_bytes()
    (runs / REPAIR).mkdir(parents=True)
    (runs / REPAIR / "switch.json").write_bytes(source)
    write_json(runs / REPAIR / "repair.json", {
        "schema": "mbpp-repair/v1", "source_root": str(runs / ORIGINAL), "root": str(runs / REPAIR),
        "source_switch_sha256": hashlib.sha256(source).hexdigest()})
    return runs


@pytest.fixture
def sandbox(tmp_path):
    """The three MBPP launcher files as shipped; the two viewers only record their roots."""
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_mbpp_experiments.sh", "_mbpp_experiments.sh", "run_experiments.sh"):
        shutil.copy(ROOT / "scripts" / name, scripts)
    recorder = ('import json, os, sys\nassert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
                'with open(os.environ["CALLS"], "a") as f:\n'
                '    f.write(json.dumps({"script": os.path.basename(sys.argv[0]), "argv": sys.argv[1:]}) + "\\n")\n'
                'print("[viewer]", " ".join(sys.argv[1:]))\n')
    for name in ("experiments_progress.py", "mbpp_failure_summary.py"):
        (scripts / name).write_text(recorder)
    # Repair verification is the real one that run/restart use.
    (scripts / "mbpp_queue_readiness.py").write_text(
        f"import runpy, sys\nsys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        f"runpy.run_path({str(ROOT / 'scripts/mbpp_queue_readiness.py')!r}, run_name='__main__')\n")
    work = tmp_path / "work"
    runs = repaired_work(work)
    env = {k: v for k, v in os.environ.items() if k not in CLEARED}
    env.update(OM_WORK=str(work), SWITCH_PYTHON=sys.executable, CALLS=str(tmp_path / "calls.jsonl"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(
                   [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]))

    def run(script, *args, **overrides):
        calls = Path(env["CALLS"])
        calls.unlink(missing_ok=True)
        result = subprocess.run(["bash", f"scripts/{script}", *args], cwd=scripts.parent,
                                env={**env, **overrides}, capture_output=True, text=True, timeout=120)
        recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, recorded

    return run, runs


def roots(call):
    argv = call["argv"]
    return [Path(argv[i + 1]).name for i, arg in enumerate(argv) if arg == "--root"]


@pytest.mark.parametrize("entry", ["wrapper", "node-launcher"])
def test_progress_follows_the_verified_repair_that_run_and_status_use(sandbox, entry):
    run, runs = sandbox
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in runs.rglob("*") if p.is_file()}
    if entry == "wrapper":
        result, calls = run("run_mbpp_experiments.sh", "progress")
    else:
        result, calls = run("run_experiments.sh", "progress", EXPERIMENTS_MBPP_SUITE="all")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call["script"] for call in calls] == ["experiments_progress.py"]
    assert roots(calls[0]) == [REPAIR, FRESH]
    assert "[mbpp-route] default queue uses prepared repair=" + str(runs / REPAIR) in result.stdout
    assert "original preserved=" + str(runs / ORIGINAL) in result.stdout
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in runs.rglob("*") if p.is_file()}


@pytest.mark.parametrize("mode", ["why", "saved"])
def test_why_and_saved_report_the_repair_and_keep_the_original_as_history(sandbox, mode):
    run, runs = sandbox
    result, calls = run("run_mbpp_experiments.sh", mode)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call["script"] for call in calls] == ["mbpp_failure_summary.py"]
    assert roots(calls[0]) == [REPAIR, ORIGINAL, FRESH]
    assert ("--storage" in calls[0]["argv"]) == (mode == "saved")
    route = "original preserved=" + str(runs / ORIGINAL)
    # saved promises at most 4 KiB on stdout; the route line goes to stderr there.
    assert route in (result.stderr if mode == "saved" else result.stdout)
    assert "[mbpp-route]" not in (result.stdout if mode == "saved" else result.stderr)


@pytest.mark.parametrize("mode", ["why", "saved"])
def test_history_root_never_exceeds_the_four_root_summary_limit(sandbox, mode):
    run, runs = sandbox
    for name in ("selection-switch-mbpp-difficulty-v1", "selection-switch-mbpp-long-v1"):
        (runs / name).mkdir()
    result, calls = run("run_mbpp_experiments.sh", mode)
    assert result.returncode == 0, result.stdout + result.stderr
    assert roots(calls[0]) == [REPAIR, FRESH, "selection-switch-mbpp-difficulty-v1", "selection-switch-mbpp-long-v1"]


@pytest.mark.parametrize("mode", ["progress", "why", "saved"])
@pytest.mark.parametrize("case", ["unverified", "explicit-quality", "custom-root"])
def test_read_only_views_keep_operator_roots_without_a_verified_default_repair(sandbox, mode, case):
    run, runs = sandbox
    args, overrides, expected = [mode], {}, [ORIGINAL, FRESH]
    if case == "unverified":
        receipt = json.loads((runs / REPAIR / "repair.json").read_text())
        write_json(runs / REPAIR / "repair.json", {**receipt, "source_switch_sha256": "0" * 64})
    elif case == "explicit-quality":
        args, expected = [mode, "quality"], [ORIGINAL]
    else:
        shutil.copytree(runs / ORIGINAL, runs / "custom-quality")
        overrides, expected = {"SWITCH_MBPP_QUALITY_ROOT": str(runs / "custom-quality")}, ["custom-quality", FRESH]
    result, calls = run("run_mbpp_experiments.sh", *args, **overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    assert roots(calls[0]) == expected
    assert "[mbpp-route]" not in result.stdout + result.stderr


def audit_fixture(work):
    """The audited state: original 37 DONE + 5 BUDGET; repair 39 DONE + 1 training heartbeat."""
    import selection_gate as core
    import selection_switch as rule

    def publish(directory):
        core.atomic_json(directory / "result.json", {"schema": rule.SCHEMA, "complete": True})
        digest = hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
        core.atomic_json(directory / "result.sha256.json", {"sha256": digest})
        core.atomic_json(directory / "curve.json", {"schema": rule.SCHEMA, "result_sha256": digest,
                                                    "points": {"25": {"reward": .1}}})

    def point(root, seed, step):
        return root / f"states/s{seed}-t{step}/points/view-{step}"

    now = time.time()
    original = work / "runs" / ORIGINAL
    core.atomic_json(original / "switch.json", {"schema": rule.SCHEMA, "dataset": "mbpp", "selector": "fresh_r",
                                                "accounting": "matched", "gate": "convergence"})
    for seed in range(5):
        for step in (25, 50, 100):
            core.atomic_json(original / f"prefixes/seed-{seed}/prefix-{step}.json", {"schema": rule.SCHEMA})
    failed = {(2, 50, "selection_reduced"), (3, 25, "selection_full"), (4, 25, "random_full"),
              (4, 50, "selection_reduced"), (4, 100, "random_reduced")}
    for seed in range(5):
        for step in (25, 50, 100):
            for arm in rule.DEV_ARMS if seed < 3 else rule.TEST_ARMS:
                if arm == "gated":
                    continue
                if (seed, step, arm) in failed:
                    core.atomic_json(point(original, seed, step) / arm / "failure.json", {
                        "error": "branch allocation exhausted before further GPU work", "time": now - 90000})
                else:
                    publish(point(original, seed, step) / arm)
    repair = work / "runs" / REPAIR
    shutil.copytree(original, repair)
    for seed, step, arm in failed:
        (point(repair, seed, step) / arm / "failure.json").unlink()
    core.atomic_json(repair / "repair.json", {
        "schema": "mbpp-repair/v1", "source_root": str(original), "root": str(repair),
        "source_switch_sha256": hashlib.sha256((original / "switch.json").read_bytes()).hexdigest()})
    publish(point(repair, 3, 25) / "selection_full")
    publish(point(repair, 4, 25) / "random_full")
    core.atomic_json(point(repair, 2, 50) / "selection_reduced/progress.json", {
        "state": "running", "phase": "train", "host": "run284373-wts-16-g8c1d", "pid": 4242, "updated": now - 2,
        "seconds": 600, "timeout": 7000, "event_id": "ev-rep", "ledger": "research", "gpus": 4, "gpu_type": "H100"})


def test_progress_and_status_count_the_same_repaired_queue(tmp_path):
    work = tmp_path / "work"
    audit_fixture(work)
    env = {k: v for k, v in os.environ.items() if k not in CLEARED}
    env.update(OM_WORK=str(work), SWITCH_PYTHON=sys.executable, COLUMNS="200", PYTHONDONTWRITEBYTECODE="1")

    def launch(mode):
        result = subprocess.run(["bash", "scripts/run_mbpp_experiments.sh", mode], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    status = launch("status")
    assert "완료 확인 39개" in status and "분기 RUN 1개" in status
    progress = [line for line in launch("progress").splitlines()
                if line.startswith("MBPP On-policy · 선택비용 별도:")]
    assert len(progress) == 1, progress
    assert "DONE 39/48" in progress[0] and "RUN 1" in progress[0], progress
