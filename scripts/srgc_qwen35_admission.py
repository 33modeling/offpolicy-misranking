"""Bounded transport recovery for the real Qwen generation/backward admission."""

from functools import wraps
import json
from pathlib import Path
import re
import shutil
import sys


FAILURE_PREFIX = "Qwen generation/backward admission failed: "
LOG_WINDOW_BYTES = 256 * 1024
RANK_ENTRY = Path(__file__).resolve().with_name("srgc_qwen35_rank.py")
SMOKE_ENTRY = RANK_ENTRY.with_name("srgc_qwen35_smoke.py")
RUNTIME_ENTRY = RANK_ENTRY.with_name("srgc_qwen35_rank_runtime.py")


def with_rank_runtime(original_task_command):
    @wraps(original_task_command)
    def task_command(*args, **kwargs):
        command = original_task_command(*args, **kwargs)
        if any(flag == "--stage" and value in {"cache", "train"}
               for flag, value in zip(command, command[1:])):
            return [str(RUNTIME_ENTRY) if str(arg) == str(RANK_ENTRY) else arg for arg in command]
        return command

    return task_command


def _fabric_failure(root, error):
    if not isinstance(error, RuntimeError) or not str(error).startswith(FAILURE_PREFIX):
        return False
    path = Path(root) / "qwen-smoke.log"
    if Path(str(error)[len(FAILURE_PREFIX):]) != path:
        return False
    try:
        with path.open("rb") as handle:
            head = handle.read(LOG_WINDOW_BYTES)
            size = handle.seek(0, 2)
            handle.seek(max(len(head), size - LOG_WINDOW_BYTES))
            tail = handle.read(LOG_WINDOW_BYTES)
    except OSError:
        return False
    log = (head + b"\n" + tail).decode("utf-8", errors="replace")
    lower = log.lower()
    if any(message in lower for message in (
            "out of memory", "illegal memory access", "duplicate gpu", "insufficient driver",
            "driver version is insufficient", "no kernel image", "allocation mismatch")):
        return False
    if not re.search(r"\bNCCL WARN\b|\bncclUnhandledCudaError\b|\bNCCL error\b|\bNCCLUtils\.cpp:\d+",
                     log, re.IGNORECASE):
        return False
    try:
        from selection_nccl_preflight import cuda_system_not_ready
    except ModuleNotFoundError as exc:
        if exc.name != "selection_nccl_preflight":
            raise
        from scripts.selection_nccl_preflight import cuda_system_not_ready
    return cuda_system_not_ready(log)


def _ladder():
    try:
        from selection_nccl_preflight import FABRIC_LADDER
    except ModuleNotFoundError as exc:
        if exc.name != "selection_nccl_preflight":
            raise
        from scripts.selection_nccl_preflight import FABRIC_LADDER
    return FABRIC_LADDER


def _receipt(root):
    path = root / "qwen-admission.json"
    if path.is_file():
        return path, json.loads(path.read_text())
    # A retry can fail in the tiny probe before the Qwen receipt is written.
    paths = list((root / "node-preflight").glob("*/admission.json"))
    if len(paths) == 1:
        return paths[0], json.loads(paths[0].read_text())
    return None, {}


def _attempt(root, receipt, report, environment, smoke_settings, status, error=None):
    return {
        "directory": str(root), "receipt": str(receipt) if receipt else None,
        "status": status, "error": str(error) if error else None,
        "settings": {key: value for key, value in environment.items() if key.startswith("NCCL_")},
        "smoke_settings": smoke_settings.get(str(root / "qwen-smoke.log"), {}),
        "allocated_gpu_seconds": report.get("allocated_gpu_seconds", 0),
        "qwen_smoke_gpu_seconds": report.get("qwen_smoke_gpu_seconds", 0),
        "cost_receipt_available": receipt is not None,
    }


def _aggregate(root, latest, attempts, overrides, status):
    from srgc_rebuttal.runtime import atomic_json
    result = {
        **latest, "qwen_model_smoke": "passed" if status == "passed" else "failed",
        "runtime_overrides": {**overrides, **latest.get("runtime_overrides", {})},
        "allocated_gpu_seconds": sum(row["allocated_gpu_seconds"] for row in attempts),
        "qwen_smoke_gpu_seconds": sum(row["qwen_smoke_gpu_seconds"] for row in attempts),
        "cost_accounting_complete": all(row["cost_receipt_available"] for row in attempts),
        "smoke_recovery": {"state": status, "attempts": attempts},
    }
    atomic_json(root / "qwen-admission.json", result)
    return result


def _readiness_note(error):
    note = (
        "Qwen admission still reports CUDA 802 after the available bounded transport fallbacks. "
        "Check this node's driver/CUDA library and NVSwitch fabric readiness, including Fabric "
        "Manager where applicable. These logs do not identify which component is unhealthy. "
        "Explicit NCCL settings were preserved; no training task was claimed."
    )
    if hasattr(error, "add_note"):
        error.add_note(note)
    else:
        print(note, file=sys.stderr)


def with_smoke_recovery(original_admit_with_smoke):
    @wraps(original_admit_with_smoke)
    def admit(original, root, environment, run_child, **kwargs):
        smoke_settings = {}

        def diagnostic_child(command, log_path, child_environment, **child_kwargs):
            if any(flag == "--stage" and value == "smoke" for flag, value in zip(command, command[1:])):
                # Replace only the operational probe, not the pinned training entry.
                command = [str(SMOKE_ENTRY) if str(arg) == str(RANK_ENTRY) else arg for arg in command]
                child_environment = dict(child_environment)
                child_environment.setdefault("NCCL_DEBUG", "INFO")
                child_environment.setdefault("NCCL_DEBUG_SUBSYS", "ALL")
                smoke_settings[str(log_path)] = {
                    key: value for key, value in child_environment.items() if key.startswith("NCCL_")}
            return run_child(command, log_path, child_environment, **child_kwargs)

        try:
            return original_admit_with_smoke(original, root, environment, diagnostic_child, **kwargs)
        except RuntimeError as initial_error:
            if kwargs.get("should_stop", lambda: False)() or not _fabric_failure(root, initial_error):
                raise
            root = Path(root)
            initial_path, latest = _receipt(root)
            if initial_path != root / "qwen-admission.json":
                raise
            backup = root / "qwen-admission.initial.json"
            with initial_path.open("rb") as source, backup.open("xb") as target:
                shutil.copyfileobj(source, target)
            candidate = dict(environment)
            overrides = dict(latest.get("runtime_overrides", {}))
            attempts = [_attempt(root, backup, latest, candidate, smoke_settings, "failed", initial_error)]
            _aggregate(root, latest, attempts, overrides, "failed")
            error = initial_error
            for index in range(1, 4):
                if kwargs.get("should_stop", lambda: False)():
                    raise error
                following = next(((name, key, value) for name, key, value in _ladder()
                                  if key not in candidate), None)
                if following is None:
                    _readiness_note(error)
                    raise error
                name, key, value = following
                candidate = {**candidate, key: value}
                overrides[key] = value
                attempt_root = root / f"recovery-{index:02d}"
                print(f"[qwen-admission] {name}: repeating tiny and Qwen smoke admission with "
                      f"{key}={value}; {attempt_root}", flush=True)
                try:
                    result = original_admit_with_smoke(
                        original, attempt_root, candidate, diagnostic_child, **kwargs)
                except BaseException as retry_error:
                    path, report = _receipt(attempt_root)
                    status = "interrupted" if not isinstance(retry_error, Exception) else "failed"
                    attempts.append(_attempt(attempt_root, path, report, candidate,
                                             smoke_settings, status, retry_error))
                    overrides.update(report.get("runtime_overrides", {}))
                    latest = {**latest, **report}
                    if (attempt_root / "qwen-smoke.log").is_file():
                        latest["qwen_smoke_log"] = str(attempt_root / "qwen-smoke.log")
                    _aggregate(root, latest, attempts, overrides, status)
                    if (kwargs.get("should_stop", lambda: False)()
                            or not _fabric_failure(attempt_root, retry_error)):
                        raise
                    error = retry_error
                else:
                    path, report = _receipt(attempt_root)
                    attempts.append(_attempt(attempt_root, path, report, candidate, smoke_settings, "passed"))
                    environment.update(candidate)
                    return _aggregate(root, result, attempts, overrides, "passed")
            _readiness_note(error)
            raise error

    return admit
