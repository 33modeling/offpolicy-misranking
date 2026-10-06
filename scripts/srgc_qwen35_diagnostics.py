#!/usr/bin/env python3
"""Diagnose and recover Qwen admission without changing the pinned runtime."""

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys

try:
    from srgc_log_tail import tail_lines
    from srgc_qwen35_storage import default_root
except ModuleNotFoundError:
    from scripts.srgc_log_tail import tail_lines
    from scripts.srgc_qwen35_storage import default_root


FAILURE_PREFIX = "Qwen generation/backward admission failed: "
EXCEPTION_LINE = re.compile(
    r"^(?:\[rank\d+\]:\s*)?(?:[\w.]*?(?:Error|Exception)|KeyboardInterrupt|SystemExit):"
)
PREFLIGHT_FAILURE = re.compile(r"^four-GPU admission failed \(exit=[^)]*\); inspect (.+/preflight\.log)$")
TRACE_FRAME = re.compile(
    r'File "[^"]*/(?:srgc_qwen35_smoke|srgc_qwen35|distributed|distributed_c10d|selection_nccl_preflight'
    r'|pathlib|triton/(?:compiler/compiler|runtime/autotuner|runtime/cache))\.py", line \d+, in '
)


def stage_details(lines):
    protocols, latest = {}, {}
    for line in lines:
        marker = "[qwen-smoke] "
        if marker not in line:
            continue
        try:
            record = json.loads(line.split(marker, 1)[1])
        except (ValueError, TypeError):
            continue
        if not isinstance(record, dict) or not isinstance(record.get("stage"), str):
            continue
        rank = str(record.get("rank"))
        latest[rank] = line
        if record.get("protocol"):
            protocols[rank] = line
    return list(dict.fromkeys([*list(protocols.values())[-4:], *list(latest.values())[-4:]]))


def failure_details(path):
    """Preserve stage and original CUDA evidence around torchrun's summary."""
    path = Path(path)
    log_label = "Four-GPU admission log" if path.name == "preflight.log" else "Qwen smoke log"
    try:
        lines = tail_lines(path, lines=2000)
        # Protocol markers precede model startup and can fall outside the tail.
        with path.open("rb") as handle:
            head = handle.read(64 * 1024).decode("utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"{log_label}: {path}\nUnable to read log: {exc}"
    sections = []
    stages = stage_details([*head, *lines])
    if stages:
        sections.extend(["Recorded smoke protocol and latest stages:", *stages])
    frames = list(dict.fromkeys(
        re.sub(r"^\[rank\d+\]:\s*", "", line.strip()) for line in lines if TRACE_FRAME.search(line)
    ))
    if frames:
        sections.extend(["Relevant traceback frames:", *frames[-8:]])
    warnings = list(dict.fromkeys(line for line in lines if "nccl warn" in line.lower()))
    cuda_errors = list(dict.fromkeys(line for line in lines if re.search(
        r"Last error:|Cuda failure|CUDA_ERROR_SYSTEM_NOT_READY|system not yet initialized", line, re.IGNORECASE)))
    evidence = list(dict.fromkeys([*warnings[-8:], *cuda_errors[-12:]]))
    if evidence:
        sections.extend(["Original NCCL/CUDA details:", *evidence])
    exceptions = list(dict.fromkeys(
        line for line in lines
        if EXCEPTION_LINE.match(line.strip()) and "ChildFailedError" not in line
    ))
    if exceptions:
        label, selected = "Recorded exception lines:", exceptions[-12:]
    else:
        label = "No Python exception found in the bounded log tail. Last recorded lines:"
        selected = [line for line in lines if "ChildFailedError" not in line][-25:]
    return "\n".join([f"{log_label}: {path}", *sections, label, *selected])


def run_with_diagnostics(entrypoint):
    try:
        return entrypoint()
    except RuntimeError as exc:
        message = str(exc)
        preflight = PREFLIGHT_FAILURE.fullmatch(message)
        if message.startswith(FAILURE_PREFIX):
            detail = failure_details(message[len(FAILURE_PREFIX):])
        elif preflight:
            detail = failure_details(preflight.group(1))
        else:
            raise
        if detail:
            if hasattr(exc, "add_note"):
                exc.add_note(detail)
            else:
                print(detail, file=sys.stderr)
        raise


def show_errors(dataset, root):
    datasets = ("math", "mbpp") if dataset == "all" else (dataset,)
    found = False
    for name in datasets:
        directory = Path(root) / "runs" / name / ".queue" / "admission"
        logs = list(directory.glob("*/qwen-smoke.log"))
        if not logs:
            print(f"{name}: no Qwen smoke log found below {directory}")
            continue
        try:
            latest = max(logs, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        except OSError as exc:
            print(f"{name}: unable to inspect smoke logs: {exc}")
            continue
        receipt = latest.parent / "qwen-admission.json"
        if receipt.is_file():
            try:
                recorded = json.loads(receipt.read_text()).get("qwen_smoke_log")
                candidate = Path(recorded).resolve() if recorded else latest
                if candidate.is_relative_to(latest.parent.resolve()) and candidate.is_file():
                    latest = candidate
            except (OSError, ValueError, TypeError):
                pass
        print(failure_details(latest))
        found = True
    return found


def main():
    if len(sys.argv) > 2 and sys.argv[2] == "error":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("dataset", choices=("math", "mbpp", "all"))
        parser.add_argument("action", choices=("error",))
        parser.add_argument("--root", type=Path, default=default_root(os.environ))
        args = parser.parse_args()
        show_errors(args.dataset, args.root)
        return
    try:
        launcher = importlib.import_module("run_srgc_qwen35")
    except ModuleNotFoundError as exc:
        if exc.name != "run_srgc_qwen35":
            raise
        launcher = importlib.import_module("scripts.run_srgc_qwen35")
    if len(sys.argv) > 2 and sys.argv[2] == "run":
        from unittest.mock import patch
        try:
            from srgc_qwen35_admission import with_smoke_recovery
        except ModuleNotFoundError as exc:
            if exc.name != "srgc_qwen35_admission":
                raise
            from scripts.srgc_qwen35_admission import with_smoke_recovery
        with patch.object(launcher, "admit_with_smoke", with_smoke_recovery(launcher.admit_with_smoke)):
            return run_with_diagnostics(launcher.main)
    return run_with_diagnostics(launcher.main)


if __name__ == "__main__":
    main()
