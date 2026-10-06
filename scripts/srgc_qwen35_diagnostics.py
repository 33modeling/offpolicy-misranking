#!/usr/bin/env python3
"""Read Qwen admission failures without changing the pinned experiment runtime."""

import argparse
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


def failure_details(path):
    """Prefer recorded Python exceptions to torchrun's secondary failure report."""
    path = Path(path)
    try:
        lines = tail_lines(path, lines=2000)
    except OSError as exc:
        return f"Qwen smoke log: {path}\nUnable to read log: {exc}"
    exceptions = list(dict.fromkeys(
        line for line in lines
        if EXCEPTION_LINE.match(line.strip()) and "ChildFailedError" not in line
    ))
    if exceptions:
        label, selected = "Recorded exception lines:", exceptions[-12:]
    else:
        label = "No Python exception found in the bounded log tail. Last recorded lines:"
        selected = [line for line in lines if "ChildFailedError" not in line][-25:]
    return "\n".join([f"Qwen smoke log: {path}", label, *selected])


def run_with_diagnostics(entrypoint):
    try:
        return entrypoint()
    except RuntimeError as exc:
        message = str(exc)
        if message.startswith(FAILURE_PREFIX):
            detail = failure_details(message[len(FAILURE_PREFIX):])
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
        from run_srgc_qwen35 import main as run_main
    except ModuleNotFoundError as exc:
        if exc.name != "run_srgc_qwen35":
            raise
        from scripts.run_srgc_qwen35 import main as run_main
    return run_with_diagnostics(run_main)


if __name__ == "__main__":
    main()
