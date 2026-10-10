"""Protect Gemma task transitions without changing its frozen model adapter."""

import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from srgc_research.dispatch.model_launch import (
    admission_failure_footer,
    released_gpu_identity,
)


def main(argv=None):
    from srgc_research.dispatch.gemma4 import cli, diagnostics, entry

    args = list(sys.argv[1:] if argv is None else argv) or ["all"]
    if args[0] == "status":
        args = ["all", "status", *args[1:]]
    options = cli.parse_args(args)
    if options.action not in {"run", "resume"}:
        return entry.main(args)

    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    from srgc_research.dispatch.gemma4 import worker
    from srgc_research.dispatch.model_resume import GemmaResumeFirst, failure_footer
    from srgc_research.dispatch.resume_drain import resume_worker

    identity = cluster.gpu_identity
    with ExitStack() as stack:
        stack.enter_context(patch.object(cli, "resume_first_worker", lambda: resume_worker(
            worker, GemmaResumeFirst, pattern="gemma4-12b-pt-{dataset}.json",
            env_key="SRGC_GEMMA_PLANS", label="GEMMA")))
        stack.enter_context(patch.object(guard, "OWNER_MARKERS", (
            *guard.OWNER_MARKERS, "srgc_research.dispatch.gemma_run", "srgc_research.dispatch.gemma4.entry",
        )))
        stack.enter_context(patch.object(cluster, "gpu_identity", lambda: released_gpu_identity(identity, label="GEMMA")))
        # Print once after the entry's backup contexts have closed, including
        # failures in the other dataset's resume backlog.
        stack.enter_context(patch.object(diagnostics, "failure_footer", lambda *args: None))
        try:
            return entry.main(args)
        except (subprocess.CalledProcessError, OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
            plans = [Path(options.root) / "experiments" / f"gemma4-12b-pt-{name}.json" for name in ("math", "mbpp")]
            failure_footer(options.root, plans, "GEMMA")
            admission_failure_footer(error, options.root, label="GEMMA", model_name="Gemma")
            raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode) from None
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        print(f"GEMMA refused: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None
