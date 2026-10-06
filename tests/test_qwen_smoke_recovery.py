import json
from pathlib import Path

import pytest

from scripts.srgc_qwen35_admission import FAILURE_PREFIX, RANK_ENTRY, SMOKE_ENTRY, with_smoke_recovery


CUDA802 = "transport/nvls.cc:254 NCCL WARN Cuda failure 802 'system not yet initialized'"


def fake_admission(outcomes):
    calls = []

    def admit(original, root, environment, run_child, **kwargs):
        calls.append((root, dict(environment), original, run_child, kwargs))
        root.mkdir(parents=True)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        (root / "qwen-smoke.log").write_text(outcome or "passed")
        report = {"allocated_gpu_seconds": 10 * len(calls), "qwen_smoke_gpu_seconds": 7 * len(calls),
                  "qwen_model_smoke": "failed" if outcome else "passed",
                  "qwen_smoke_log": str(root / "qwen-smoke.log"), "runtime_overrides": {"TINY": "yes"}}
        (root / "qwen-admission.json").write_text(json.dumps(report))
        if outcome:
            raise RuntimeError(FAILURE_PREFIX + str(root / "qwen-smoke.log"))
        return report

    return with_smoke_recovery(admit), calls


def test_normal_admission_is_unchanged(tmp_path):
    admit, calls = fake_admission([None])
    environment = {"EXPLICIT": "setting"}
    result = admit("original", tmp_path / "admission", environment, "child")
    assert len(calls) == 1
    assert result["allocated_gpu_seconds"] == 10
    assert "smoke_recovery" not in result
    assert environment == {"EXPLICIT": "setting"}
    assert not (tmp_path / "admission/qwen-admission.initial.json").exists()


@pytest.mark.parametrize("stage", ["smoke", "train", "cache"])
def test_only_smoke_uses_lightweight_entry_without_changing_launch_settings(tmp_path, stage):
    command = ["python", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "--max_restarts=0", str(RANK_ENTRY), "--stage", stage, "--plan", "/saved/plan.json"]
    original_command = list(command)
    environment = {"SRGC_QWEN_PLAN": "/saved/plan.json", "NCCL_NVLS_ENABLE": "0"}
    seen = []

    def child(args, log, env, **kwargs):
        seen.append((args, log, env, kwargs))
        return 0

    def original(original_admit, root, env, run_child, **kwargs):
        return run_child(command, root / "qwen-smoke.log", env, **kwargs)

    heartbeat = lambda pid: None
    stop = lambda: False
    options = {"pass_fds": (3, 4), "heartbeat": heartbeat, "should_stop": stop, "timeout": 3600}
    result = with_smoke_recovery(original)(None, tmp_path, environment, child, **options)
    expected = [str(SMOKE_ENTRY) if arg == str(RANK_ENTRY) and stage == "smoke" else arg
                for arg in original_command]
    assert result == 0
    assert seen[0][0] == expected
    assert seen[0][3] == options
    assert seen[0][2]["SRGC_QWEN_PLAN"] == "/saved/plan.json"
    assert command == original_command
    assert "NCCL_DEBUG" not in environment


@pytest.mark.parametrize("log", ["ChildFailedError", "CUDA error: 802", "ncclUnhandledCudaError",
    "NCCL WARN Cuda failure 1 'invalid argument'", CUDA802 + "\nCUDA out of memory",
    "ChildFailedError: worker exited 802"])
def test_non_fabric_failures_never_retry(tmp_path, log):
    admit, calls = fake_admission([log])
    with pytest.raises(RuntimeError, match=FAILURE_PREFIX):
        admit(None, tmp_path / "admission", {}, None)
    assert len(calls) == 1


def test_successful_recovery_preserves_arguments_and_sums_costs_once(tmp_path):
    admit, calls = fake_admission([CUDA802, None])
    environment = {"NCCL_DEBUG": "WARN", "TOKEN": "private"}
    heartbeat = lambda pid: None
    stop = lambda: False
    result = admit("original", tmp_path / "admission", environment, "child",
                   pass_fds=(4, 5), heartbeat=heartbeat, should_stop=stop, plan={"id": 7})
    assert len(calls) == 2
    assert calls[1][0] == tmp_path / "admission/recovery-01"
    assert calls[1][1] == {**calls[0][1], "NCCL_NVLS_ENABLE": "0"}
    assert calls[1][2:] == calls[0][2:]
    assert environment == calls[1][1]
    assert result["runtime_overrides"] == {"TINY": "yes", "NCCL_NVLS_ENABLE": "0"}
    assert result["allocated_gpu_seconds"] == 30
    assert result["qwen_smoke_gpu_seconds"] == 21
    assert result["qwen_smoke_log"] == str(calls[1][0] / "qwen-smoke.log")
    attempts = result["smoke_recovery"]["attempts"]
    assert [row["status"] for row in attempts] == ["failed", "passed"]
    assert "TOKEN" not in attempts[0]["settings"]
    assert json.loads(Path(attempts[0]["receipt"]).read_text())["allocated_gpu_seconds"] == 10
    assert json.loads(Path(attempts[1]["receipt"]).read_text())["allocated_gpu_seconds"] == 20
    assert json.loads((tmp_path / "admission/qwen-admission.json").read_text()) == result


@pytest.mark.parametrize("value", ["0", "1", ""])
def test_explicit_settings_are_preserved(tmp_path, value):
    admit, calls = fake_admission([CUDA802, None])
    environment = {"NCCL_NVLS_ENABLE": value, "NCCL_CUMEM_HOST_ENABLE": "1"}
    admit(None, tmp_path / "admission", environment, None)
    assert calls[1][1] == {**calls[0][1], "NCCL_CUMEM_ENABLE": "0"}
    assert environment["NCCL_NVLS_ENABLE"] == value


def test_exhaustion_is_bounded_and_keeps_all_receipts(tmp_path):
    admit, calls = fake_admission([CUDA802] * 4)
    environment = {}
    with pytest.raises(RuntimeError) as caught:
        admit(None, tmp_path / "admission", environment, None)
    assert len(calls) == 4
    assert environment == {}
    assert calls[-1][1] == {"NCCL_NVLS_ENABLE": "0", "NCCL_CUMEM_ENABLE": "0", "NCCL_P2P_DISABLE": "1"}
    assert "fabric readiness" in " ".join(caught.value.__notes__)
    result = json.loads((tmp_path / "admission/qwen-admission.json").read_text())
    assert result["smoke_recovery"]["state"] == "failed"
    assert result["allocated_gpu_seconds"] == 100
    assert result["qwen_smoke_gpu_seconds"] == 70
    assert len({row["receipt"] for row in result["smoke_recovery"]["attempts"]}) == 4


def test_explicit_complete_ladder_does_not_retry(tmp_path):
    admit, calls = fake_admission([CUDA802])
    environment = {"NCCL_NVLS_ENABLE": "1", "NCCL_CUMEM_ENABLE": "1", "NCCL_P2P_DISABLE": "0"}
    with pytest.raises(RuntimeError):
        admit(None, tmp_path / "admission", environment, None)
    assert len(calls) == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("unrelated failure")])
def test_interrupts_and_unknown_errors_propagate_without_retry(tmp_path, error):
    admit, calls = fake_admission([error])
    with pytest.raises(type(error)) as caught:
        admit(None, tmp_path / "admission", {}, None)
    assert caught.value is error
    assert len(calls) == 1


def test_stop_callback_prevents_recovery(tmp_path):
    admit, calls = fake_admission([CUDA802])
    with pytest.raises(RuntimeError):
        admit(None, tmp_path / "admission", {}, None, should_stop=lambda: True)
    assert len(calls) == 1


def test_original_error_survives_large_shutdown_tail(tmp_path):
    admit, calls = fake_admission([CUDA802 + "\n" + "shutdown noise\n" * 50000, None])
    result = admit(None, tmp_path / "admission", {}, None)
    assert len(calls) == 2
    assert result["qwen_model_smoke"] == "passed"


@pytest.mark.parametrize("diagnostic", [
    "torch.distributed.DistBackendError: NCCL error in: NCCLUtils.cpp:77, unhandled cuda error",
    "ncclUnhandledCudaError: Call to CUDA function failed.",
])
def test_original_nccl_exception_and_last_error_qualify(tmp_path, diagnostic):
    admit, calls = fake_admission([diagnostic + "\nLast error:\nCuda failure 802 'system not yet initialized'", None])
    assert admit(None, tmp_path / "admission", {}, None)["qwen_model_smoke"] == "passed"
    assert len(calls) == 2


def test_intermediate_tiny_overrides_remain_in_successful_runtime_receipt(tmp_path):
    wrapped, calls = fake_admission([CUDA802, CUDA802, None])
    original = wrapped.__wrapped__

    def admit_with_tiny_override(original_admit, root, environment, run_child, **kwargs):
        if len(calls) == 1:
            environment["NCCL_CUMEM_ENABLE"] = "0"
        try:
            return original(original_admit, root, environment, run_child, **kwargs)
        finally:
            if len(calls) == 2:
                path = root / "qwen-admission.json"
                report = json.loads(path.read_text())
                report["runtime_overrides"]["NCCL_CUMEM_ENABLE"] = "0"
                path.write_text(json.dumps(report))

    environment = {}
    result = with_smoke_recovery(admit_with_tiny_override)(None, tmp_path / "admission", environment, None)
    assert environment == {"NCCL_NVLS_ENABLE": "0", "NCCL_CUMEM_ENABLE": "0", "NCCL_P2P_DISABLE": "1"}
    assert result["runtime_overrides"] == {"TINY": "yes", **environment}
    assert len(calls) == 3


def test_failed_retry_tiny_probe_is_accounted_and_never_passes(tmp_path):
    wrapped, calls = fake_admission([CUDA802])
    original = wrapped.__wrapped__

    def fail_tiny(original_admit, root, environment, run_child, **kwargs):
        if not calls:
            return original(original_admit, root, environment, run_child, **kwargs)
        receipt = root / "node-preflight/node/admission.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({"allocated_gpu_seconds": 6, "state": "failed"}))
        raise RuntimeError("four-GPU admission failed")

    with pytest.raises(RuntimeError, match="four-GPU admission failed"):
        with_smoke_recovery(fail_tiny)(None, tmp_path / "admission", {}, None)
    result = json.loads((tmp_path / "admission/qwen-admission.json").read_text())
    assert result["allocated_gpu_seconds"] == 16
    assert result["qwen_smoke_gpu_seconds"] == 7
    assert result["qwen_model_smoke"] == "failed"
    assert result["cost_accounting_complete"] is True
    assert result["qwen_smoke_log"] == str(tmp_path / "admission/qwen-smoke.log")
    assert len(result["smoke_recovery"]["attempts"]) == 2


def test_interrupted_retry_propagates_and_marks_missing_cost_receipt(tmp_path):
    interruption = KeyboardInterrupt()
    admit, calls = fake_admission([CUDA802, interruption])
    with pytest.raises(KeyboardInterrupt) as caught:
        admit(None, tmp_path / "admission", {}, None)
    assert caught.value is interruption
    assert len(calls) == 2
    result = json.loads((tmp_path / "admission/qwen-admission.json").read_text())
    assert result["smoke_recovery"]["state"] == "interrupted"
    assert result["cost_accounting_complete"] is False


@pytest.mark.parametrize("explicit", [{}, {"NCCL_DEBUG": "WARN", "NCCL_DEBUG_SUBSYS": "INIT", "NCCL_DEBUG_FILE": "/tmp/nccl.log"}])
def test_smoke_diagnostics_are_child_only_and_preserve_explicit_settings(tmp_path, explicit):
    seen = []
    environment = {"OTHER": "unchanged", **explicit}

    def child(command, log_path, child_environment, **kwargs):
        seen.append((command, child_environment, kwargs))
        return 0

    def original(original_admit, root, supplied_environment, run_child, **kwargs):
        assert supplied_environment is environment
        run_child(["tiny"], root / "preflight.log", supplied_environment, **kwargs)
        run_child(["rank", "--stage", "smoke"], root / "qwen-smoke.log", supplied_environment, **kwargs)
        return "unchanged result"

    result = with_smoke_recovery(original)(None, tmp_path, environment, child, pass_fds=(3,))
    assert result == "unchanged result"
    assert seen[0][1] is environment
    assert seen[1][1] == {"OTHER": "unchanged", "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "ALL", **explicit}
    assert seen[0][2] == seen[1][2] == {"pass_fds": (3,)}
    assert environment == {"OTHER": "unchanged", **explicit}
