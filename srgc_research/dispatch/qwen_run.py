"""Attach the shared resume-first policy to the existing pinned Qwen launcher."""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

from .model_launch import released_gpu_identity
from .qwen_resume import resume_first_worker

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv) or ["all"]
    sys.path.insert(0, str(REPO / "scripts"))
    if args[0] in {"status", "results"}:
        args = ["all", *args]
    if args[0] not in {"all", "math", "mbpp"}:
        raise ValueError("dataset must be math, mbpp or all")
    if len(args) > 1 and args[1] == "status":
        from .model_status import main as status

        return status(["qwen35", *args])
    if len(args) > 1 and args[1] == "results":
        from .model_results import main as results

        return results(["qwen35", *args])
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
    adapter = importlib.import_module("srgc_qwen35")
    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster

    from .model_chain import enabled
    from .ready_dispatch import ResumeFirst
    from .resume_drain import resume_worker
    from .reward_curves import recording_command
    identity = cluster.gpu_identity
    original_task_command = adapter.task_command
    with patch.object(sys, "argv", [str(REPO / "scripts/srgc_qwen35_diagnostics.py"), *args]), \
            patch.object(guard, "TARGET_MARKERS", (*guard.TARGET_MARKERS, "srgc_qwen35_reward_curve_rank.py")), \
            patch.object(adapter, "task_command", lambda *a, **kw: recording_command(original_task_command(*a, **kw))), \
            patch.object(cluster, "gpu_identity", lambda: identity() if enabled() else released_gpu_identity(identity, label="QWEN")), \
            resume_first_worker(), \
            resume_worker(worker, ResumeFirst, pattern="qwen35-9b-{dataset}.json",
                          env_key="SRGC_QWEN_PLANS", label="QWEN", result_model="qwen35"):
        return diagnostics.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"QWEN resume refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
