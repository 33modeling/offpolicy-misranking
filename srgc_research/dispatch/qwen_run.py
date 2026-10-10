"""Attach the shared resume-first policy to the existing pinned Qwen launcher."""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

from .qwen_resume import resume_first_worker

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv) or ["all"]
    sys.path.insert(0, str(REPO / "scripts"))
    if args[0] not in {"all", "math", "mbpp"}:
        raise ValueError("dataset must be math, mbpp or all")
    if len(args) == 1:
        start = importlib.import_module("srgc_qwen35_start")
        original_exec = os.execv

        def redirect(python, command):
            if len(command) > 1 and Path(command[1]).name == "srgc_qwen35_diagnostics.py":
                return original_exec(python, [python, "-m", "srgc_research.dispatch.qwen_run", *command[2:]])
            return original_exec(python, command)

        with patch.object(sys, "argv", [str(REPO / "scripts/srgc_qwen35_start.py"), *args]), \
                patch.object(start.os, "execv", redirect):
            return start.main()
    if args[1] != "run":
        raise ValueError("resume-first launcher accepts run only")
    diagnostics = importlib.import_module("srgc_qwen35_diagnostics")
    worker = importlib.import_module("srgc_qwen35_worker")
    from .qwen_resume import ResumeFirst
    from .resume_drain import resume_worker
    with patch.object(sys, "argv", [str(REPO / "scripts/srgc_qwen35_diagnostics.py"), *args]), \
            resume_first_worker(), \
            resume_worker(worker, ResumeFirst, pattern="qwen35-9b-{dataset}.json",
                          env_key="SRGC_QWEN_PLANS", label="QWEN"):
        return diagnostics.main()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"QWEN resume refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
