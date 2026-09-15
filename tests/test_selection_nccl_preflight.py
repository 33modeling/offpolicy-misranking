import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import selection_gate as core
import selection_gate_gpu as base


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("nccl_preflight", ROOT / "scripts/selection_nccl_preflight.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def reports(version=(2, 26, 2), count=4):
    return [{"rank": rank, "nccl": list(version), "gpu_uuid": f"GPU-{rank}", "state": "passed"}
            for rank in range(count)]


@pytest.mark.parametrize("version,expected", [((2, 23, 4), False), ((2, 24, 0), True),
    ((2, 26, 2), True), ((2, 26, 4), True), ((2, 26, 5), False), ((2, 27, 0), False)])
def test_host_fallback_is_limited_to_affected_nccl_versions(version, expected):
    assert check.host_allocation_fallback(reports(version), "ncclUnhandledCudaError", {}, 4) is expected


@pytest.mark.parametrize("error", ["ChildFailedError", "timeout", "CUDA allocation mismatch",
    "ncclUnhandledCudaError: out of memory", "ncclUnhandledCudaError: illegal memory access",
    "ncclUnhandledCudaError: duplicate GPU", "ncclUnhandledCudaError: driver version is insufficient",
    "ncclUnhandledCudaError: no kernel image",
    "ncclUnhandledCudaError: Cuda failure 802 'system not yet initialized'",
    "ncclUnhandledCudaError: CUDA_ERROR_SYSTEM_NOT_READY",
    "ncclUnhandledCudaError: CUDA error: 802"])
def test_unrelated_or_nonrecoverable_failures_do_not_change_transport_settings(error):
    assert not check.host_allocation_fallback(reports(), error, {}, 4)


@pytest.mark.parametrize("value", ["0", "1", ""])
def test_explicit_host_allocation_setting_is_never_overridden(value):
    assert not check.host_allocation_fallback(reports(), "ncclUnhandledCudaError", {"NCCL_CUMEM_HOST_ENABLE": value}, 4)


def test_missing_or_mixed_rank_versions_do_not_guess_a_compatibility_fix():
    assert not check.host_allocation_fallback(reports(count=3), "ncclUnhandledCudaError", {}, 4)
    rows = reports()
    rows[0]["nccl"] = [2, 27, 0]
    assert not check.host_allocation_fallback(rows, "ncclUnhandledCudaError", {}, 4)


def fake_attempt(monkeypatch, outcomes):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("NCCL_CUMEM_HOST_ENABLE", raising=False)
    def attempt(directory, visible, env, world_size, timeout):
        calls.append(dict(env))
        error = outcomes[len(calls)-1]
        rows = reports(count=world_size)
        for row in rows:
            core.atomic_json(directory / f"rank-{row['rank']}.json", {**row, "state": "failed" if error else "passed"})
        def action():
            if error:
                raise RuntimeError(error)
        base.meter(directory, "nccl-check", "fixture", action=action, ledger="research", devices=world_size)
        return rows
    monkeypatch.setattr(check, "check_attempt", attempt)
    return calls


def admission(root):
    paths = list(root.glob("node-preflight/*/admission.json"))
    assert len(paths) == 1
    return core.read(paths[0])


def test_successful_probe_does_not_change_configuration_or_claim_training(tmp_path, monkeypatch):
    calls = fake_attempt(monkeypatch, [None])
    assert check.preflight(tmp_path) == {}
    assert len(calls) == 1
    value = admission(tmp_path)
    assert value["state"] == "passed" and value["overrides"] == {}
    assert value["allocated_gpu_seconds"] >= 0
    assert value["attempts"][0]["cost"]["complete"]
    assert set(path.name for path in tmp_path.iterdir()) == {"node-preflight"}


def test_failed_probe_rechecks_legacy_host_allocation_before_exporting_it(tmp_path, monkeypatch):
    calls = fake_attempt(monkeypatch, ["ncclUnhandledCudaError: Call to CUDA function failed", None])
    assert check.preflight(tmp_path) == {"NCCL_CUMEM_HOST_ENABLE": "0"}
    assert len(calls) == 2
    assert "NCCL_CUMEM_HOST_ENABLE" not in calls[0]
    assert calls[1]["NCCL_CUMEM_HOST_ENABLE"] == "0"
    assert "NCCL_CUMEM_HOST_ENABLE" not in os.environ
    value = admission(tmp_path)
    assert value["state"] == "passed"
    assert value["attempts"][0]["error"] and value["attempts"][1]["error"] is None
    assert all(row["cost"]["complete"] for row in value["attempts"])
    assert all(row["cost"]["ledgers"]["research"]["gpu_seconds"] >= 0 for row in value["attempts"])
    assert all(not key.startswith(("NCCL_P2P", "NCCL_IB", "NCCL_NVLS")) for key in value["overrides"])


@pytest.mark.parametrize("failures", [["ChildFailedError"], ["ncclUnhandledCudaError", "ncclUnhandledCudaError"]])
def test_unresolved_probe_exits_without_repeated_training_attempts(tmp_path, monkeypatch, failures):
    calls = fake_attempt(monkeypatch, failures)
    with pytest.raises(RuntimeError, match="no training task claimed"):
        check.preflight(tmp_path)
    assert len(calls) == len(failures)
    value = admission(tmp_path)
    assert value["state"] == "failed"
    assert all(row["cost"]["complete"] for row in value["attempts"])
    assert not (tmp_path / "states").exists()


@pytest.mark.parametrize("rank_only", [False, True])
def test_cluster_802_is_diagnosed_without_host_allocation_retry(tmp_path, monkeypatch, capsys, rank_only):
    error = ("ncclUnhandledCudaError: Call to CUDA function failed.\n"
             "Last error:\nCuda failure 802 'system not yet initialized'")
    calls = fake_attempt(monkeypatch, ["ChildFailedError" if rank_only else error])
    if rank_only:
        original = check.rank_reports
        monkeypatch.setattr(check, "rank_reports", lambda *args: [
            {**row, "error": error} for row in original(*args)])
    with pytest.raises(RuntimeError, match="CUDA 802.*no training task claimed"):
        check.preflight(tmp_path)
    assert len(calls) == 1
    value = admission(tmp_path)
    assert value["failure_kind"] == "cuda_system_not_ready"
    assert value["state"] == "failed" and "cluster administrator" in value["diagnosis"]
    assert value["attempts"][0]["cost"]["complete"]
    assert value["attempts"][0]["overrides"] == {}
    assert not (tmp_path / "states").exists()
    assert "retrying only the tiny probe" not in capsys.readouterr().out


@pytest.mark.parametrize("error", ["ncclUnhandledCudaError at /tmp/job-802/rank.log",
    "CUDA failure 803 'system has unsupported display driver / cuda driver combination'",
    "ncclUnhandledCudaError: failed with code 1802"])
def test_cuda_802_classifier_does_not_match_other_numbers(error):
    assert not check.cuda_system_not_ready(error)


def test_interruption_does_not_trigger_fallback_or_claim_work(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    def stop(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(check, "check_attempt", stop)
    with pytest.raises(KeyboardInterrupt):
        check.preflight(tmp_path)
    assert admission(tmp_path)["state"] == "interrupted"
    assert len(admission(tmp_path)["attempts"]) == 1
    assert not (tmp_path / "states").exists()


def test_probe_timeout_reaps_children_and_never_executes_controller(tmp_path):
    marker = tmp_path / "must-not-run"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/selection_nccl_preflight.py"),
        "--root", str(tmp_path), "--world-size", "1", "--timeout", ".05", "--", sys.executable, "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 78, result.stdout + result.stderr
    assert not marker.exists()
    value = admission(tmp_path)
    assert value["state"] == "failed" and len(value["attempts"]) == 1
    attempt = value["attempts"][0]
    assert "allocation limit" in attempt["error"]
    assert attempt["cost"]["complete"]
    for path in Path(attempt["directory"], "cost-events").glob("*.json"):
        assert not base.owned_worker_groups(core.read(path)["event_id"])


def test_preflight_sigterm_closes_attempt_and_does_not_start_training(tmp_path):
    script = '''
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('probe', sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
def attempt(directory, visible, env, world_size, timeout):
    probe.base.meter(directory, 'nccl-check', 'CPU fixture', devices=0, timeout=60,
        commands=[([sys.executable, '-c', 'import time; time.sleep(60)'], '')])
probe.check_attempt = attempt
sys.argv = [sys.argv[1], '--root', sys.argv[2], '--world-size', '1', '--',
            sys.executable, '-c', 'raise RuntimeError("controller must not start")']
raise SystemExit(probe.main())
'''
    process = subprocess.Popen([sys.executable, "-c", script,
        str(ROOT / "scripts/selection_nccl_preflight.py"), str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not list(tmp_path.glob("node-preflight/*/baseline/nccl-check-0.log")):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.01)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 130, stdout + stderr
        value = admission(tmp_path)
        assert value["state"] == "interrupted" and len(value["attempts"]) == 1
        attempt = value["attempts"][0]
        assert attempt["cost"]["complete"]
        for path in Path(attempt["directory"], "cost-events").glob("*.json"):
            assert not base.owned_worker_groups(core.read(path)["event_id"])
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        process.communicate(timeout=15)


@pytest.mark.parametrize("setting", [None, "", "expandable_segments:False,max_split_size_mb:128"])
def test_setup_preserves_explicit_cuda_allocator_configuration(tmp_path, setting):
    env = {**os.environ, "OM_REPO": str(tmp_path / "repo"), "OM_WORK": str(tmp_path), "GROUP_VOLUME": str(tmp_path / "missing")}
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if setting is not None:
        env["PYTORCH_CUDA_ALLOC_CONF"] = setting
    result = subprocess.run(["bash", "-c", 'source scripts/setup_env.sh >/dev/null; printf "%s" "$PYTORCH_CUDA_ALLOC_CONF"'],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ("expandable_segments:True" if setting is None else setting)


@pytest.mark.skipif("SWITCH_TEST_CUDA" not in os.environ, reason="explicit SWITCH_TEST_CUDA device required")
@pytest.mark.parametrize("host_allocation", [None, "0"])
def test_real_cuda_nccl_admission_runs_ddp_then_executes_worker(tmp_path, host_allocation):
    visible = os.environ["SWITCH_TEST_CUDA"]
    marker = tmp_path / "admitted"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible}
    env.pop("NCCL_CUMEM_HOST_ENABLE", None)
    if host_allocation is not None:
        env["NCCL_CUMEM_HOST_ENABLE"] = host_allocation
    result = subprocess.run([sys.executable, str(ROOT / "scripts/selection_nccl_preflight.py"),
        "--root", str(tmp_path), "--world-size", "1", "--", sys.executable, "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('admitted')", str(marker)],
        env=env, capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_text() == "admitted"
    value = admission(tmp_path)
    assert value["state"] == "passed"
    ranks = value["attempts"][-1]["ranks"]
    assert len(ranks) == 1 and ranks[0]["state"] == "passed" and ranks[0]["gpu_uuid"]
    assert ranks[0]["cumem_host"] == host_allocation
    assert "DDP forward/backward and collectives passed" in Path(value["attempts"][-1]["directory"], "nccl-check-0.log").read_text()


@pytest.mark.skipif("SWITCH_TEST_CUDA" not in os.environ, reason="explicit SWITCH_TEST_CUDA device required")
def test_real_cuda_rank_allocation_error_blocks_worker_before_model_load(tmp_path):
    visible = os.environ["SWITCH_TEST_CUDA"]
    marker = tmp_path / "must-not-run"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/selection_nccl_preflight.py"),
        "--root", str(tmp_path), "--world-size", "2", "--", sys.executable, "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": f"{visible},999999"},
        capture_output=True, text=True, timeout=150)
    assert result.returncode == 78, result.stdout + result.stderr
    assert not marker.exists()
    value = admission(tmp_path)
    assert value["state"] == "failed" and len(value["attempts"]) == 1
    assert value["attempts"][0]["cost"]["complete"]
    assert "GPU allocation mismatch" in result.stdout
