"""Exercise public text-v3 startup, phase boundaries and real stop receipts."""

import json
import os
import subprocess
import sys
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
from srgc_research.tests.test_gemma4_text import write_panel
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


@pytest.mark.parametrize("prepared", ["math", "mbpp"])
def test_public_stop_handles_a_partially_prepared_cohort(tmp_path, prepared):
    cohort = make_cohort("gemma4_text", tmp_path)
    missing = "mbpp" if prepared == "math" else "math"
    cohort.plans[missing].unlink()
    # Stopping must not bind/rewrite a historical or mismatching queue identity.
    identity = cohort.root / f"runs/{prepared}/.queue/protocol.json"
    atomic_json(identity, {"implementation_sha256": "older-running-code"})
    before = {p: p.read_bytes() for p in cohort.root.rglob("*") if p.is_file()}
    result = subprocess.run(["sh", str(REPO / "scripts/run_srgc_gemma4.sh"), "stop", "--now"],
                            cwd="/tmp", env={**cohort.env, "GEMMA_PYTHON": "/missing-model-python",
                                "SRGC_GEMMA_LEGACY_PROTOCOL": "0", "SRGC_GEMMA_TEXT_PROTOCOL": "1"},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(p.read_bytes() == value for p, value in before.items())
    assert json.loads((identity.parent / "stop.json").read_text())["immediate"]


@pytest.mark.parametrize("args, expected", [
    (["all", "--root", "/a path"], ["all", "run", "--root", "/a path"]),
    (["math", "--retry-failed"], ["math", "run", "--retry-failed"]),
    (["--root", "/a path"], ["all", "run", "--root", "/a path"]),
])
@pytest.mark.parametrize("model_name", ["qwen35", "llama31", "gemma4"])
def test_implicit_run_with_options_keeps_model_chaining(tmp_path, args, expected, model_name):
    driver = tmp_path / "python"
    driver.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    driver.chmod(0o755)
    name = {"qwen35": "QWEN", "llama31": "LLAMA", "gemma4": "GEMMA"}[model_name]
    env = {**os.environ, name + "_PYTHON": str(driver), "SRGC_MODEL_CHAIN": "1",
           "SRGC_MODEL_CHAIN_CHILD": "0", "SRGC_GEMMA_LEGACY_PROTOCOL": "0"}
    result = subprocess.run(["sh", str(REPO / f"scripts/run_srgc_{model_name}.sh"), *args],
                            env=env, capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(result.stdout) == ["-m", "srgc_research.dispatch.model_chain", "--start", model_name,
                                        *expected]


@pytest.mark.parametrize("profile", ["gemma4_text", "gemma4_it"])
def test_initial_preparation_observes_stop_and_tears_down_its_real_child(tmp_path, monkeypatch, profile):
    import importlib
    module = "srgc_research.dispatch."
    adapter = importlib.import_module(module + profile + ".adapter")
    cli = importlib.import_module(module + profile + ".cli")
    entry = importlib.import_module(module + profile + ".entry")
    run = importlib.import_module(module + ("gemma_it_run" if profile == "gemma4_it" else "gemma_text_run"))
    cohort = make_cohort(profile, tmp_path)
    cohort.plans["math"].unlink()
    for key in ("GROUP_VOLUME", "OM_WORK", "SRGC_GEMMA_TEXT_ROOT", "SRGC_GEMMA_IT_ROOT",
                "SRGC_GEMMA_TEXT_PROTOCOL", "SRGC_GEMMA_IT_PROTOCOL"):
        if key in cohort.env:
            monkeypatch.setenv(key, cohort.env[key])
    monkeypatch.setattr(cli, "resolve_snapshot", lambda env: (tmp_path, "a" * 64))
    monkeypatch.setattr(adapter, "runtime_packages", dict)
    real_child, started = cluster.run_child, []

    def child(command, log, env, **kwargs):
        assert "should_stop" in kwargs, "input preparation has no cancellation callback"
        def stop(pid):
            started.append(pid)
            result = subprocess.run(["sh", str(REPO / "scripts/run_srgc_gemma4.sh"), "math", "stop", "--now"],
                                    env=env, capture_output=True, text=True, timeout=30, check=False)
            assert result.returncode == 0, result.stdout + result.stderr
        return real_child([sys.executable, "-c", "import time; time.sleep(60)"], log, env,
                          **kwargs, interval=.01, heartbeat=stop)

    monkeypatch.setattr(cluster, "run_child", child)
    monkeypatch.setattr(entry, "main", lambda args: cli.prepare_missing(["math"], cohort.root, os.environ))
    with pytest.raises(KeyboardInterrupt):
        run.main(["math", "--root", str(cohort.root)])
    assert len(started) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(started[0], 0)
    assert not cohort.plans["math"].exists()
    assert not (cohort.root / "runs/math").exists()


def test_stop_before_any_plan_blocks_startup_and_explicit_resume_can_prepare(tmp_path, monkeypatch):
    root = tmp_path / "work/empty experiment"
    monkeypatch.setenv("GROUP_VOLUME", str(tmp_path))
    monkeypatch.setenv("OM_WORK", str(tmp_path / "work"))
    monkeypatch.delenv("SRGC_DATASET_PHASE", raising=False)
    assert run.main(["stop", "--now", "--root", str(root)]) == 0
    assert not (root / "runs").exists()
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    # The public shell defaults to the IT protocol; select this text-v3 root's protocol.
    result = subprocess.run(["sh", str(REPO / "scripts/run_srgc_gemma4.sh"), "status", "--json",
                             "--root", str(root)], env={**os.environ, "GEMMA_PYTHON": "/missing",
                                                        "SRGC_GEMMA_IT_PROTOCOL": "0",
                                                        "SRGC_GEMMA_TEXT_PROTOCOL": "1"},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 1  # Plans remain unprepared, not completed.
    reports = json.loads(result.stdout)["datasets"]
    assert all(row["stop_requested"] and not row["prepared"] for row in reports)
    assert {path: path.read_bytes() for path in before} == before
    with patch.object(entry, "main", side_effect=AssertionError("stopped startup reached model loading")), \
            pytest.raises(KeyboardInterrupt):
        run.main(["all", "--root", str(root)])
    def resume(args):
        assert not list(root.rglob("stop.json"))
        return 0
    with patch.object(entry, "main", side_effect=resume):
        assert run.main(["resume", "--root", str(root)]) == 0
    assert not (root / "runs").exists()


def test_stop_can_replace_a_damaged_marker_and_cannot_downgrade_an_immediate_stop(tmp_path, monkeypatch):
    from srgc_research.dispatch.gemma_text_control import RunControl

    root = tmp_path / "work/experiment"
    monkeypatch.setenv("GROUP_VOLUME", str(tmp_path))
    monkeypatch.setenv("OM_WORK", str(tmp_path / "work"))
    control = RunControl(cli.parse_args(["math", "stop", "--root", str(root)]))
    marker = root / ".startup-control/math/stop.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("broken JSON")
    assert control.stop(immediate=False) == 0
    assert not json.loads(marker.read_text())["immediate"]
    control.stop(immediate=True)
    control.stop(immediate=False)
    assert json.loads(marker.read_text())["immediate"]


def test_resume_of_incompatible_plan_does_not_remove_any_stop_receipts(tmp_path, monkeypatch):
    cohort = make_cohort("gemma4_text", tmp_path)
    for key in ("GROUP_VOLUME", "OM_WORK"):
        monkeypatch.setenv(key, cohort.env[key])
    assert run.main(["all", "stop", "--now", "--root", str(cohort.root)]) == 0
    plan = json.loads(cohort.plans["mbpp"].read_text())
    plan["adapter_sha256"] = "b" * 64
    atomic_json(cohort.plans["mbpp"], plan)
    before = {p: p.read_bytes() for p in cohort.root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError), patch.object(entry, "main", side_effect=AssertionError("model loading")):
        run.main(["all", "resume", "--root", str(cohort.root)])
    assert all(p.read_bytes() == value for p, value in before.items())


def test_future_dataset_stop_does_not_block_the_current_chain_phase(tmp_path, monkeypatch):
    cohort = make_cohort("gemma4_text", tmp_path)
    atomic_json(cohort.root / "runs/mbpp/.queue/stop.json", {"immediate": True})
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    with patch.object(entry, "main", return_value=0):
        assert run.main(["all", "--root", str(cohort.root)]) == 0
    assert json.loads((cohort.root / "runs/mbpp/.queue/stop.json").read_text())["immediate"]


def test_graceful_preparation_stop_preserves_finished_plan_and_skips_the_next_stage(tmp_path, monkeypatch):
    cohort = make_cohort("gemma4_text", tmp_path)
    saved_plans = {dataset: path.read_bytes() for dataset, path in cohort.plans.items()}
    for path in cohort.plans.values():
        path.unlink()
    for key in ("GROUP_VOLUME", "OM_WORK"):
        monkeypatch.setenv(key, cohort.env[key])
    monkeypatch.delenv("SRGC_DATASET_PHASE", raising=False)
    monkeypatch.setattr(cli, "resolve_snapshot", lambda env: (tmp_path, "a" * 64))
    monkeypatch.setattr(adapter, "runtime_packages", dict)
    calls = []
    def child(command, log, env, **kwargs):
        calls.append(command)
        cohort.plans["math"].write_bytes(saved_plans["math"])
        assert run.main(["all", "stop", "--root", str(cohort.root)]) == 0
        assert not kwargs["should_stop"](), "graceful stop interrupted the current stage"
        return 0
    monkeypatch.setattr(cluster, "run_child", child)
    monkeypatch.setattr(entry, "main", lambda args: cli.prepare_missing(["math", "mbpp"], cohort.root, os.environ))
    with pytest.raises(KeyboardInterrupt):
        run.main(["all", "--root", str(cohort.root)])
    assert len(calls) == 1
    assert cohort.plans["math"].read_bytes() == saved_plans["math"]
    assert not cohort.plans["mbpp"].exists()


@pytest.mark.parametrize("args", [["--help"], ["all", "--help"], ["math", "run", "--help"]])
def test_public_help_does_not_require_the_model_python(args):
    result = subprocess.run(["sh", str(REPO / "scripts/run_srgc_gemma4.sh"), *args],
                            env={**os.environ, "GEMMA_PYTHON": "/missing-model-python",
                                 "SRGC_GEMMA_LEGACY_PROTOCOL": "0", "SRGC_GEMMA_TEXT_PROTOCOL": "1"},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("model_name, name", [("qwen35", "QWEN"), ("llama31", "LLAMA")])
@pytest.mark.parametrize("args", [["--help"], ["all", "--help"], ["all", "run", "--help"]])
def test_help_never_enters_the_training_chain(tmp_path, model_name, name, args):
    driver = tmp_path / "python"
    driver.write_text("#!" + sys.executable + "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    driver.chmod(0o755)
    result = subprocess.run(["sh", str(REPO / f"scripts/run_srgc_{model_name}.sh"), *args],
                            env={**os.environ, name + "_PYTHON": str(driver),
                                 "SRGC_MODEL_CHAIN": "1", "SRGC_MODEL_CHAIN_CHILD": "0"},
                            capture_output=True, text=True, check=True, timeout=30)
    module = "qwen_run" if model_name == "qwen35" else "llama_run"
    assert json.loads(result.stdout) == ["-m", f"srgc_research.dispatch.{module}",
                                        "all", "run", "--help"]


def test_wrapper_restores_live_hooks_and_preserves_frozen_adapter():
    before = adapter.adapter_digest()
    old_worker, old_child, old_owner = worker.worker, cluster.run_child, srgc_process_guard.OWNER_MARKERS
    with patch.object(entry, "main", return_value=0):
        assert run.main(["all"]) == 0
    assert worker.worker is old_worker and cluster.run_child is old_child
    assert srgc_process_guard.OWNER_MARKERS == old_owner
    assert adapter.adapter_digest() == before


def test_failed_saved_panel_stops_before_model_import_or_gpu_admission(tmp_path, monkeypatch, capsys):
    cohort = make_cohort("gemma4_text", tmp_path)
    for key, value in cohort.env.items():
        if key in {"GROUP_VOLUME", "OM_WORK"}:
            monkeypatch.setenv(key, value)
    directory = write_panel(cohort.plans["math"], success=False)
    before = {p: p.read_bytes() for p in cohort.root.rglob("*") if p.is_file()}
    with patch.object(entry, "main", side_effect=AssertionError("GPU/model entry must not run")), \
            pytest.raises(ValueError, match="no GPU diagnostics launched"):
        run.main(["math", "preflight", "--root", str(cohort.root)])
    assert {p: p.read_bytes() for p in before} == before
    text = capsys.readouterr().out
    assert "0/64 correct" in text and str(directory) in text
    assert "Answer: 6" in text


def test_passed_partial_and_phase_excluded_panels_do_not_block_dispatch(tmp_path, monkeypatch):
    cohort = make_cohort("gemma4_text", tmp_path)
    for key in ("GROUP_VOLUME", "OM_WORK"):
        monkeypatch.setenv(key, cohort.env[key])
    options = cli.parse_args(["all", "preflight", "--root", str(cohort.root)])
    write_panel(cohort.plans["math"], success=True)
    run.check_saved_preflight(options)
    directory = write_panel(cohort.plans["mbpp"], success=False)
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    run.check_saved_preflight(options)
    monkeypatch.delenv("SRGC_DATASET_PHASE")
    receipt = next(p for p in directory.glob("*.json") if p.name != "summary.json")
    receipt.unlink()
    run.check_saved_preflight(options)


def test_saved_panel_guard_rejects_an_altered_summary_instead_of_trusting_the_zero_flag(tmp_path, monkeypatch):
    cohort = make_cohort("gemma4_text", tmp_path)
    for key in ("GROUP_VOLUME", "OM_WORK"):
        monkeypatch.setenv(key, cohort.env[key])
    directory = write_panel(cohort.plans["math"], success=True)
    summary = directory / "summary.json"
    value = json.loads(summary.read_text())
    value["passed"] = False
    atomic_json(summary, value)
    with pytest.raises(ValueError, match="summary differs"):
        run.check_saved_preflight(cli.parse_args(["math", "preflight", "--root", str(cohort.root)]))


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
        model_resume.failure_footer(tmp_path, [], "GEMMA TEXT V3")
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
