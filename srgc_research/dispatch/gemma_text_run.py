"""Operational fixes around the immutable Gemma text-v3 experiment protocol."""

import hashlib
import json
import os
import subprocess
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from srgc_rebuttal.plan import load_plan
from srgc_rebuttal.runtime import atomic_json, run_root

from .model_launch import admission_failure_footer, released_gpu_identity


def check_saved_preflight(options):
    """Do not repeat CUDA diagnostics for an unchanged, fully recorded zero panel."""
    from srgc_rebuttal.plan import input_path

    from .gemma4_text import adapter, preflight

    datasets = ("math", "mbpp") if options.dataset == "all" else (options.dataset,)
    phase = os.environ.get("SRGC_DATASET_PHASE")
    if phase in {"math", "mbpp"}:
        datasets = tuple(name for name in datasets if name == phase)
    for dataset in datasets:
        path = options.root / "experiments" / f"gemma4-12b-text-{dataset}.json"
        if not path.is_file():
            continue
        plan = adapter.validate_extension(path)
        directory = preflight.folder(path, plan)
        if not (directory / "summary.json").is_file():
            continue
        bundle = json.loads(input_path(path, plan, plan["seeds"][0]).read_text())
        adapter.validate_bundle_model(bundle, plan, plan["seeds"][0])
        protocol = preflight.identity(path, plan, bundle)
        rows = [preflight.read_receipt(directory / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json",
                                       protocol, candidate)
                for candidate in protocol["candidate_ids"]]
        if any(row is None for row in rows):
            continue
        report = preflight.summarize(protocol, rows)
        if json.loads((directory / "summary.json").read_text()) != report:
            raise ValueError(f"Gemma preflight summary differs from its raw receipts: {directory}")
        if report["passed"]:
            continue
        print(f"GEMMA PREFLIGHT {dataset}: {report['correct']:g}/{report['responses']} correct, "
              f"empty={report['empty_responses']}; saved results verified on CPU", flush=True)
        print(f"Raw responses: {directory}", flush=True)
        for row in rows[:2]:
            print(f"  {row['id']}: " + json.dumps(row["responses"][:2], ensure_ascii=False)[:700], flush=True)
        raise ValueError(f"Gemma preflight has zero correct responses in the saved {dataset} panel; "
                         "no GPU diagnostics launched and no full cache/training launched. "
                         "Unchanged results are not rerun; raw responses and the original experiment are preserved.")


def scoped_admission(original):
    """Check only the active dataset and honor stop receipts during diagnostics."""
    def admit(base, root, environment, run_child, **kwargs):
        from srgc_rebuttal import cluster

        plans = json.loads(environment.get("SRGC_GEMMA_PLANS", "null")) or [environment["SRGC_GEMMA_PLAN"]]
        phase = environment.get("SRGC_DATASET_PHASE")
        if phase is not None:
            if phase not in {"math", "mbpp"}:
                raise ValueError("SRGC_DATASET_PHASE must be math or mbpp")
            selected = "math_train" if phase == "math" else "mbpp"
            plans = [path for path in plans if load_plan(Path(path))["dataset"] == selected]
            if not plans:
                raise ValueError(f"no {phase} plan for Gemma admission")
        scoped = {"SRGC_GEMMA_PLAN": plans[0], "SRGC_GEMMA_PLANS": json.dumps(plans)}
        previous = {key: environment.get(key) for key in scoped}
        queues = [SimpleNamespace(directory=run_root(Path(path), load_plan(Path(path))) / ".queue")
                  for path in plans]
        previous_stop = kwargs.get("should_stop", lambda: False)

        def stopping(*, boundary=False):
            return previous_stop() or any(cluster.stop_requested(queue, immediate=not boundary)
                                          for queue in queues)

        def child(command, log, env, **options):
            # A graceful stop is observed between diagnostic stages; --now is
            # also polled inside the existing process-group teardown loop.
            if stopping(boundary=True):
                raise KeyboardInterrupt
            options["should_stop"] = lambda: stopping()
            return run_child(command, log, env, **options)

        try:
            # Admission mutates this mapping with validated model/NCCL options;
            # keep those available to the actual training child.
            environment.update(scoped)
            if stopping(boundary=True):
                raise KeyboardInterrupt
            return original(base, root, environment, child,
                            **{**kwargs, "should_stop": lambda: stopping()})
        except KeyboardInterrupt:
            # The frozen helper's finally block records timings. Preserve those
            # and successful stages, but distinguish cancellation from failure.
            receipt = Path(root) / "gemma-admission.json"
            if receipt.exists():
                report = json.loads(receipt.read_text())
                for field in ("gemma_model_smoke", "gemma_initial_preflight"):
                    if report.get(field) == "failed":
                        report[field] = "stopped"
                report["stop_requested"] = True
                atomic_json(receipt, report)
            raise
        finally:
            # Restore only diagnostic scope, not admission's runtime overrides.
            for key, value in previous.items():
                if value is None:
                    environment.pop(key, None)
                else:
                    environment[key] = value

    return admit


def main(argv=None):
    from .gemma4_text import cli, entry

    args = list(sys.argv[1:] if argv is None else argv) or ["all"]
    if args[0] in {"status", "results", "stop", "resume", "doctor", "prepare", "preflight"}:
        args = ["all", *args]
    options = cli.parse_args(args)
    if options.action in {"status", "results", "stop", "prepare", "doctor"}:
        return entry.main(args)

    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster

    from .gemma4 import worker
    from .gemma4_text.resume import GemmaTextResumeFirst
    from .model_chain import enabled
    from .model_resume import failure_footer
    from .resume_drain import resume_worker

    identity = cluster.gpu_identity
    with ExitStack() as stack:
        # The outer handler prints once after the entry's backup scopes close.
        stack.enter_context(patch("srgc_research.dispatch.model_resume.failure_footer", lambda *a: None))
        stack.enter_context(patch.object(cli, "admit_with_smoke", scoped_admission(cli.admit_with_smoke)))
        if options.action in {"run", "resume"}:
            # Install before cli.main binds `from ...worker import worker`.
            # Installing only the inner drain hook leaves the old startup and
            # its unfiltered MATH/MBPP completion check active.
            stack.enter_context(resume_worker(
                worker, GemmaTextResumeFirst, pattern="gemma4-12b-text-{dataset}.json",
                env_key="SRGC_GEMMA_PLANS", label="GEMMA TEXT V3", result_model="gemma4_text"))
            stack.enter_context(patch.object(cli, "resume_first_worker", nullcontext))
            stack.enter_context(patch.object(guard, "OWNER_MARKERS", (
                *guard.OWNER_MARKERS, "srgc_research.dispatch.gemma_text_run")))
            stack.enter_context(patch.object(cluster, "gpu_identity", lambda: identity()
                if enabled() else released_gpu_identity(identity, label="GEMMA")))
        try:
            check_saved_preflight(options)
            return entry.main(args)
        except (subprocess.CalledProcessError, OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
            plans = [Path(options.root) / "experiments" / f"gemma4-12b-text-{name}.json"
                     for name in ("math", "mbpp")]
            failure_footer(options.root, plans, "GEMMA TEXT V3")
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
