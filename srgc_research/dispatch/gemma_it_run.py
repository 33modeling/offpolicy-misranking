"""Gemma IT operation with fresh-cache initialization and isolated experiment data."""

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

from .gemma4_it.resume import GemmaITResumeFirst
from .model_launch import admission_failure_footer, released_gpu_identity
from .resume_drain import active_dataset_phase


def saved_preflight(path):
    """A saved fixed panel verified against its raw receipts on CPU; None if not run or partial."""
    from srgc_rebuttal.plan import input_path

    from .gemma4_it import adapter, preflight

    plan = adapter.validate_extension(path)
    directory = preflight.folder(path, plan)
    if not (directory / "summary.json").is_file():
        return None
    bundle = json.loads(input_path(path, plan, plan["seeds"][0]).read_text())
    adapter.validate_bundle_model(bundle, plan, plan["seeds"][0])
    protocol = preflight.identity(path, plan, bundle)
    rows = [preflight.read_receipt(directory / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json",
                                   protocol, candidate)
            for candidate in protocol["candidate_ids"]]
    if any(row is None for row in rows):
        return None
    report = preflight.summarize(protocol, rows)
    if json.loads((directory / "summary.json").read_text()) != report:
        raise ValueError(f"Gemma preflight summary differs from its raw receipts: {directory}")
    return SimpleNamespace(report=report, rows=rows, directory=directory, protocol=protocol)


PANEL_STATES = {}


def preflight_state(path):
    """Saved panel state for dispatch: "passed", "paused" or None (not run or partial)."""
    from .gemma4_it import preflight

    path = Path(path)
    summary = preflight.folder(path, load_plan(path)) / "summary.json"
    try:
        stat = summary.stat()
    except FileNotFoundError:
        return None
    key = (str(path.resolve()), stat.st_ino, stat.st_mtime_ns, stat.st_size)
    if key not in PANEL_STATES:
        try:
            saved = saved_preflight(path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            # Keep MATH running; the MBPP phase reports this panel again.
            print(f"GEMMA MBPP early start closed: {error}", flush=True)
            PANEL_STATES[key] = "paused"
        else:
            PANEL_STATES[key] = None if saved is None else "passed" if saved.report["passed"] else "paused"
    return PANEL_STATES[key]


# Training ranks run as rank_runtime -> reward_curve_rank, so their command line
# never names gemma4_it.rank, the pinned orphan reaper's only Gemma IT marker.
RANK_WRAPPERS = ("srgc_research.dispatch.rank_runtime", "srgc_research.dispatch.reward_curve_rank")


def wrapped_rank_target(original):
    """Also reap orphaned Gemma IT ranks launched through those wrappers (2026-10-31)."""
    def is_target(cmdline, plan=None):
        if original(cmdline, plan):
            return True
        from scripts import srgc_process_guard as guard

        argv0 = os.path.basename(cmdline.split(" ", 1)[0]) if cmdline else ""
        return (argv0.startswith("python") and (plan is None or plan in cmdline)
                and any(marker in cmdline for marker in RANK_WRAPPERS)
                and "--model gemma4_it" in cmdline
                and not any(marker in cmdline for marker in guard.OWNER_MARKERS))
    return is_target


def early_enabled(environment=None):
    return (os.environ if environment is None else environment).get("SRGC_GEMMA_MBPP_EARLY", "1") != "0"


class GemmaITEarlyMBPP(GemmaITResumeFirst):
    """MBPP cache and prefix may start while MATH work remains (2026-10-29).

    Ready MBPP arms follow this worker's MATH claims on spare allocations.
    Every task keeps its own prefix dependency and verified MBPP preflight."""

    @property
    def early_arms(self):
        return ("cache", "prefix") if early_enabled() else ()

    @property
    def late_mbpp(self):
        return early_enabled()

    def early_state(self, queue):
        return preflight_state(queue.plan_path)


def early_preflight_plan(plans, environment):
    """MBPP plan whose panel a MATH-training admission also runs, or None."""
    from srgc_rebuttal import cluster

    if (not early_enabled(environment) or active_dataset_phase(environment) != "math"
            or environment.get("SRGC_CACHE_ONLY") == "1"):
        return None
    for path in plans:
        plan = load_plan(Path(path))
        if plan["dataset"] != "mbpp":
            continue
        if cluster.stop_requested(SimpleNamespace(directory=run_root(Path(path), plan) / ".queue")):
            return None
        return path if preflight_state(path) is None else None
    return None


def check_saved_preflight(options):
    """Do not repeat CUDA diagnostics for an unchanged failed baseline panel."""
    datasets = ("math", "mbpp") if options.dataset == "all" else (options.dataset,)
    phase = active_dataset_phase()
    if phase is not None:
        datasets = tuple(name for name in datasets if name == phase)
    for dataset in datasets:
        path = options.root / "experiments" / f"gemma4-12b-it-{dataset}.json"
        if not path.is_file():
            continue
        saved = saved_preflight(path)
        if saved is None or saved.report["passed"]:
            continue
        report, rows, directory, protocol = saved.report, saved.rows, saved.directory, saved.protocol
        print(f"GEMMA PREFLIGHT {dataset}: {report['correct']:g}/{report['responses']} correct, "
              f"empty={report['empty_responses']}; saved results verified on CPU", flush=True)
        print(f"Raw responses: {directory}", flush=True)
        for row in rows[:2]:
            print(f"  {row['id']}: " + json.dumps(row["responses"][:2], ensure_ascii=False)[:700], flush=True)
        raise ValueError(f"Gemma preflight failed in the saved {dataset} panel "
                         f"({report['success_percent']:.2f}%, minimum={protocol['minimum_success_percent']:g}%, "
                         f"empty={report['empty_responses']}); "
                         "no GPU diagnostics launched and no full cache/training launched. "
                         "Unchanged results are not rerun; raw responses and the original experiment are preserved.")


def scoped_admission(original):
    """Check only the active dataset and honor stop receipts during diagnostics."""
    def admit(base, root, environment, run_child, **kwargs):
        from srgc_rebuttal import cluster

        plans = json.loads(environment.get("SRGC_GEMMA_PLANS", "null")) or [environment["SRGC_GEMMA_PLAN"]]
        early = early_preflight_plan(plans, environment)
        phase = active_dataset_phase(environment)
        if phase is not None:
            selected = "math_train" if phase == "math" else "mbpp"
            plans = [path for path in plans if load_plan(Path(path))["dataset"] == selected]
            if not plans:
                raise ValueError(f"no {phase} plan for Gemma admission")
        # MBPP cache/prefix may start in this MATH worker, so its unrun fixed
        # panel is checked here too. Stop receipts stay those of the MATH scope.
        scoped = {"SRGC_GEMMA_PLAN": plans[0], "SRGC_GEMMA_PLANS": json.dumps(plans + ([early] if early else []))}
        previous = {key: environment.get(key) for key in scoped}
        queues = [SimpleNamespace(directory=run_root(Path(path), load_plan(Path(path))) / ".queue")
                  for path in plans]
        previous_stop = kwargs.get("should_stop", lambda: False)
        stage = {"early": False}

        def stopping(*, boundary=False):
            return previous_stop() or any(cluster.stop_requested(queue, immediate=not boundary)
                                          for queue in queues)

        def child(command, log, env, **options):
            # A graceful stop is observed between diagnostic stages; --now is
            # also polled inside the existing process-group teardown loop.
            if stopping(boundary=True):
                raise KeyboardInterrupt
            options["should_stop"] = lambda: stopping()
            stage["early"] = early is not None and Path(log).name == f"gemma-preflight-{Path(early).stem}.log"
            return run_child(command, log, env, **options)

        try:
            # Admission mutates this mapping with validated model/NCCL options;
            # keep those available to the actual training child.
            environment.update(scoped)
            if stopping(boundary=True):
                raise KeyboardInterrupt
            try:
                return original(base, root, environment, child,
                                **{**kwargs, "should_stop": lambda: stopping()})
            except Exception as error:
                if not stage["early"]:
                    raise
                # MATH passed its smoke and panel. A failed or interrupted MBPP
                # panel keeps MBPP behind its preflight; MATH is admitted alone.
                print(f"GEMMA MBPP early preflight did not pass ({type(error).__name__}: {error}); "
                      "MATH continues and MBPP cache/prefix stay closed", flush=True)
                stage["early"] = False
                environment.update({"SRGC_GEMMA_PLANS": json.dumps(plans)})
                if stopping(boundary=True):
                    raise KeyboardInterrupt from None
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


def run(argv=None):
    from .gemma4_it import cli, entry

    args = list(sys.argv[1:] if argv is None else argv) or ["all"]
    if args[0] in {"status", "results", "stop", "resume", "doctor", "prepare", "preflight"}:
        args = ["all", *args]
    options = cli.parse_args(args)
    if options.action in {"status", "results", "doctor"}:
        return entry.main(args)

    if (options.action in {"run", "resume"} and options.dataset == "all"
            and os.environ.get("SRGC_DATASET_PHASE") is None
            and os.environ.get("SRGC_CACHE_ONLY") != "1"):
        # A direct/reset launch has no outer model-chain phase. Give it the
        # same cache -> MATH -> MBPP order without a cross-dataset cache barrier.
        for phase in ("math", "mbpp"):
            with patch.dict(os.environ, {"SRGC_DATASET_PHASE": phase}):
                code = run(args)
            if code:
                return code
            if options.action == "resume":
                args[1] = "run"  # Clear old stops once; respect later requests.
        return 0

    from .gemma_it_control import RunControl

    control = RunControl(options)
    if options.action == "stop":
        return control.stop(immediate=options.now)
    if options.action == "resume":
        control.resume()
    else:
        control.check()
    if options.action == "prepare":
        return entry.main(args)

    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster

    from .gemma4 import worker
    from .model_chain import YIELD, enabled
    from .model_resume import failure_footer
    from .resume_drain import resume_worker

    identity = cluster.gpu_identity
    with ExitStack() as stack:
        # The outer handler prints once after the entry's backup scopes close.
        stack.enter_context(patch("srgc_research.dispatch.model_resume.failure_footer", lambda *a: None))
        stack.enter_context(patch.object(cli, "prepare_missing", control.preparing(cli.prepare_missing)))
        stack.enter_context(patch.object(cli, "admit_with_smoke", scoped_admission(cli.admit_with_smoke)))
        if options.action in {"run", "resume"}:
            # Install before cli.main binds `from ...worker import worker`.
            # Installing only the inner drain hook leaves the old startup and
            # its unfiltered MATH/MBPP completion check active.
            stack.enter_context(resume_worker(
                worker, GemmaITEarlyMBPP, pattern="gemma4-12b-it-{dataset}.json",
                env_key="SRGC_GEMMA_PLANS", label="GEMMA IT", result_model="gemma4_it"))
            stack.enter_context(patch.object(cli, "resume_first_worker", nullcontext))
            stack.enter_context(patch.object(guard, "OWNER_MARKERS", (
                *guard.OWNER_MARKERS, "srgc_research.dispatch.gemma_it_run")))
            stack.enter_context(patch.object(guard, "is_target", wrapped_rank_target(guard.is_target)))
            stack.enter_context(patch.object(cluster, "gpu_identity", lambda: identity()
                if enabled() else released_gpu_identity(identity, label="GEMMA")))
        try:
            check_saved_preflight(options)
            if (options.action in {"run", "resume"} and options.dataset == "all"
                    and os.environ.get("SRGC_GEMMA_CACHE_FIRST") == "1"
                    and os.environ.get("SRGC_CACHE_ONLY") != "1"):
                # Cache only the active dataset. Missing/owned MBPP caches
                # must never hold a MATH prefix or learner behind a barrier.
                # Explicit resume was already applied above; never clear a new
                # stop request again between cache and training.
                if options.action == "resume":
                    args[1] = "run"
                phase = os.environ.get("SRGC_DATASET_PHASE", "math")
                print(f"GEMMA preparing {phase.upper()} initial caches before {phase.upper()} training", flush=True)
                with patch.dict(os.environ, {"SRGC_CACHE_ONLY": "1", "SRGC_CACHE_DATASET": phase}):
                    check_saved_preflight(options)
                    try:
                        code = entry.main(args)
                    except SystemExit as error:
                        if error.code != YIELD or phase != "math" or not early_enabled():
                            raise
                        code = YIELD
                    if code and not (code == YIELD and phase == "math" and early_enabled()):
                        return code
                if code == YIELD:
                    # Peer-owned MATH caches do not block independent MBPP
                    # preparation. Resume-only keeps fresh MATH work closed.
                    print("GEMMA MATH caches waiting on peers; checking available MBPP work", flush=True)
                    with patch.dict(os.environ, {"SRGC_CACHE_ONLY": "0", "SRGC_RESUME_ONLY": "1"}):
                        code = entry.main(args)
                    # Finishing this fallback never declares MATH complete.
                    return code or YIELD
                print(f"GEMMA {phase.upper()} initial caches complete; starting {phase.upper()} work", flush=True)
            return entry.main(args)
        except (subprocess.CalledProcessError, OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
            plans = [Path(options.root) / "experiments" / f"gemma4-12b-it-{name}.json"
                     for name in ("math", "mbpp")]
            failure_footer(options.root, plans, "GEMMA IT")
            admission_failure_footer(error, options.root, label="GEMMA", model_name="Gemma")
            raise


def main(argv=None):
    from .gemma4_it import cli
    from .gemma4_it.reset import session

    args = list(sys.argv[1:] if argv is None else argv) or ["all"]
    if args[0] in {"status", "results", "stop", "resume", "doctor", "prepare", "preflight", "reset"}:
        args = ["all", *args]
    options = cli.parse_args(args)
    if options.action in {"status", "results", "stop", "doctor"}:
        return run(args)
    from .gemma_it_bootstrap import discard_previous_pt_path

    discard_previous_pt_path(os.environ)
    from . import startup
    from .gemma4_it import entry

    with session(options.root, reset=options.action == "reset"), \
            patch.object(entry, "main", startup.wrap_entry(entry.main, options.root)):
        if options.action == "reset":
            args[1] = "run"
            # An explicit fresh start must not inherit a chain's resume-only or
            # single-dataset phase. Each dataset's cache precedes its training.
            with patch.dict(os.environ, {"SRGC_CACHE_ONLY": "0", "SRGC_RESUME_ONLY": "0",
                                         "SRGC_GEMMA_CACHE_FIRST": "1"}):
                phase = os.environ.pop("SRGC_DATASET_PHASE", None)
                try:
                    return run(args)
                finally:
                    if phase is not None:
                        os.environ["SRGC_DATASET_PHASE"] = phase
        return run(args)


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
