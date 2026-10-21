"""Exercise public text-v2 startup, phase boundaries and real stop receipts."""

import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import srgc_process_guard
from srgc_rebuttal import cluster, cluster_queue
from srgc_rebuttal.runtime import atomic_json
from srgc_research.dispatch import gemma_text_run as run
from srgc_research.dispatch import model_results
from srgc_research.dispatch.gemma4 import worker
from srgc_research.dispatch.gemma4_text import adapter, cli, entry, model, runtime
from srgc_research.tests.test_model_results import make_cohort

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_environment():
    # setup_storage intentionally rewrites cache variables in the real CLI.
    with patch.dict(os.environ):
        yield


@pytest.fixture
def phase_start(tmp_path, monkeypatch):
    # cli has installed the historical scripts import path at this point.
    import srgc_checkpoint_backup
    import srgc_log_format

    cohort = make_cohort("gemma4_text", tmp_path)
    monkeypatch.setenv("GROUP_VOLUME", cohort.env["GROUP_VOLUME"])
    monkeypatch.setenv("OM_WORK", cohort.env["OM_WORK"])
    monkeypatch.setenv("SRGC_GEMMA_TEXT_ROOT", str(cohort.root))
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    monkeypatch.delenv("SRGC_MODEL_CHAIN_CHILD", raising=False)
    # Only the MATH tasks are complete; MBPP must remain pending. This fixture
    # tests scheduling, never reports these injected completions as measurements.
    monkeypatch.setattr(cluster_queue.TaskQueue, "complete",
                        lambda q, task: q.plan["dataset"] == "math_train")
    monkeypatch.setattr(cli, "prepare_missing", lambda *a, **kw: list(cohort.plans.values()))
    monkeypatch.setattr(model, "find_snapshot", lambda *a: tmp_path)
    monkeypatch.setattr(runtime, "activate", lambda *a: None)
    monkeypatch.setattr(srgc_checkpoint_backup, "automatic_backup", lambda *a: nullcontext())
    monkeypatch.setattr(srgc_log_format, "uniform_log", nullcontext)
    monkeypatch.setattr(srgc_process_guard, "process_guard", lambda *a: nullcontext())
    monkeypatch.setattr(model_results, "publish", lambda *a: 0)
    monkeypatch.setattr(cluster, "gpu_identity", lambda: pytest.fail("completed MATH requested GPUs"))
    monkeypatch.setattr(worker, "runtime_signature",
                        lambda: (_ for _ in ()).throw(AssertionError("unfiltered startup reached GPU runtime")))
    return cohort


def test_frozen_entry_reproduces_the_startup_binding_bug(phase_start):
    with pytest.raises(pytest.fail.Exception, match="completed MATH requested GPUs"):
        entry.main(["all"])


@pytest.mark.parametrize("chain_child", ["0", "1"])
def test_public_operational_entry_finishes_math_without_admitting_mbpp(phase_start, monkeypatch, chain_child):
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", chain_child)
    before = adapter.adapter_digest()
    assert run.main(["all"]) == 0
    assert adapter.adapter_digest() == before
    rows = list((phase_start.root / "runs/math/.queue/workers").glob("*.json"))
    assert len(rows) == 1
    assert json.loads(rows[0].read_text())["status"] == "complete"
    # Worker completion describes the exited launcher, not MBPP task results.
    assert not list((phase_start.root / "runs/mbpp/.queue/receipts").glob("*.json"))


@pytest.mark.parametrize("phase", ["math", "mbpp"])
def test_admission_checks_only_the_current_dataset(tmp_path, monkeypatch, phase):
    cohort = make_cohort("gemma4_text", tmp_path)
    monkeypatch.setenv("SRGC_DATASET_PHASE", phase)
    environment = {**cohort.env, "SRGC_DATASET_PHASE": phase,
                   "SRGC_GEMMA_PLAN": str(cohort.plans["math"]),
                   "SRGC_GEMMA_PLANS": json.dumps([str(p) for p in cohort.plans.values()])}
    original = dict(environment)
    stages = []
    def child(command, log, env, **kwargs):
        stages.append((command[command.index("--stage") + 1], command[command.index("--plan") + 1]))
        return 0
    def base(root, env, run_child, **kwargs):
        env["NCCL_P2P_DISABLE"] = "1"
        env["SRGC_LOCAL_MODEL"] = "/verified-local-model"
        return {"allocated_gpu_seconds": 0}
    run.scoped_admission(cli.admit_with_smoke)(base,
        tmp_path / "admission", environment, child)
    assert stages == [("smoke", str(cohort.plans[phase])), ("preflight", str(cohort.plans[phase]))]
    assert environment == {**original, "NCCL_P2P_DISABLE": "1", "SRGC_LOCAL_MODEL": "/verified-local-model"}


def test_explicit_preflight_honors_immediate_stop_and_records_cancellation(tmp_path):
    cohort = make_cohort("gemma4_text", tmp_path)
    environment = {**cohort.env, "SRGC_GEMMA_PLAN": str(cohort.plans["math"])}
    admission = tmp_path / "admission"
    calls = []
    def child(command, log, env, **kwargs):
        stage = command[command.index("--stage") + 1]
        calls.append(stage)
        if stage == "preflight":
            atomic_json(cohort.root / "runs/math/.queue/stop.json", {"immediate": True})
            assert kwargs["should_stop"]()
            raise KeyboardInterrupt
        return 0
    with pytest.raises(KeyboardInterrupt):
        run.scoped_admission(cli.admit_with_smoke)(lambda *a, **kw: {"allocated_gpu_seconds": 0},
            admission, environment, child)
    assert calls == ["smoke", "preflight"]
    report = json.loads((admission / "gemma-admission.json").read_text())
    assert report["gemma_model_smoke"] == "passed"
    assert report["gemma_initial_preflight"] == "stopped"
    assert report["stop_requested"]


def test_existing_stop_prevents_any_preflight_child(tmp_path):
    cohort = make_cohort("gemma4_text", tmp_path)
    atomic_json(cohort.root / "runs/math/.queue/stop.json", {"immediate": False})
    with pytest.raises(KeyboardInterrupt):
        run.scoped_admission(cli.admit_with_smoke)(lambda *a, **kw: pytest.fail("admission ran"),
            tmp_path / "admission", {**cohort.env, "SRGC_GEMMA_PLAN": str(cohort.plans["math"])},
            lambda *a, **kw: pytest.fail("child launched"))


def test_status_and_stop_work_without_model_python_and_without_resetting_data(tmp_path):
    cohort = make_cohort("gemma4_text", tmp_path)
    before = {p: p.read_bytes() for p in cohort.root.rglob("*.json")}
    env = {**cohort.env, "GEMMA_PYTHON": "/missing-model-python",
           "SRGC_GEMMA_LEGACY_PROTOCOL": "0", "SRGC_GEMMA_TEXT_PROTOCOL": "1"}
    for action in (["status", "--json"], ["all", "stop", "--now"], ["stop", "--now"]):
        result = subprocess.run(["sh", str(REPO / "scripts/run_srgc_gemma4.sh"), *action],
                                cwd="/tmp", env=env, capture_output=True, text=True,
                                timeout=30, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
    assert all(p.read_bytes() == content for p, content in before.items())
    for dataset in ("math", "mbpp"):
        row = json.loads((cohort.root / f"runs/{dataset}/.queue/stop.json").read_text())
        assert row["immediate"]


def test_wrapper_restores_live_hooks_and_preserves_frozen_adapter():
    before = adapter.adapter_digest()
    old_worker, old_child, old_owner = worker.worker, cluster.run_child, srgc_process_guard.OWNER_MARKERS
    with patch.object(entry, "main", return_value=0):
        assert run.main(["all"]) == 0
    assert worker.worker is old_worker and cluster.run_child is old_child
    assert srgc_process_guard.OWNER_MARKERS == old_owner
    assert adapter.adapter_digest() == before


def test_delayed_gpu_release_and_live_owner_protection_are_restored(capsys):
    from srgc_rebuttal.runtime import Busy
    from srgc_research.dispatch import model_launch

    old_owner = srgc_process_guard.OWNER_MARKERS
    def start(args):
        assert cluster.gpu_identity() == ("0,1,2,3", ("GPU-a",))
        table = {
            90001: (1, os.getuid(), "python -m srgc_research.dispatch.gemma_text_run all"),
            90002: (90001, os.getuid(), "python -m srgc_research.dispatch.gemma4.rank --plan live"),
            90003: (1, os.getuid(), "python -m srgc_research.dispatch.gemma4.rank --plan orphan"),
        }
        assert srgc_process_guard.orphan_pids(table=table) == [90003]
        return 0
    with patch.object(entry, "main", side_effect=start), \
            patch.object(srgc_process_guard, "TARGET_MARKERS", ("srgc_research.dispatch.gemma4.rank",)), \
            patch.object(cluster, "gpu_identity", side_effect=[Busy("CUDA teardown pending"),
                ("0,1,2,3", ("GPU-a",))]), patch.object(model_launch.time, "sleep") as sleep, \
            patch.dict(os.environ, {"SRGC_MODEL_CHAIN_CHILD": "0"}):
        assert run.main(["all"]) == 0
    assert sleep.call_count == 1
    assert srgc_process_guard.OWNER_MARKERS == old_owner
    assert "GEMMA GPU memory release complete" in capsys.readouterr().out


def test_failure_footer_is_printed_once_after_backup_cleanup(tmp_path, capsys):
    from srgc_research.dispatch import model_resume

    def fail(args):
        model_resume.failure_footer(tmp_path, [], "GEMMA TEXT V2")
        print("BACKUP final=true")
        raise RuntimeError("injected admission failure")
    def footer(*args):
        print("GEMMA FINAL FAILURE")
    with patch.object(entry, "main", side_effect=fail), \
            patch.object(model_resume, "failure_footer", side_effect=footer), \
            pytest.raises(RuntimeError, match="injected admission failure"):
        assert run.main(["all", "run", "--root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert output.count("GEMMA FINAL FAILURE") == 1
    assert output.index("GEMMA FINAL FAILURE") > output.index("BACKUP final=true")
