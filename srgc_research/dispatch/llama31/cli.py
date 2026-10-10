"""Run the Llama SRGC comparison using already downloaded weights."""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from . import adapter
from .model import resolve_snapshot
from .resume import resume_first_worker
from .storage import default_root, setup_storage


def validate_saved(path, identity):
    from srgc_rebuttal.plan import input_path

    plan = adapter.validate_extension(path)
    if plan["model_snapshot_sha256"] != identity:
        raise ValueError("Saved Llama plan uses different local weights")
    for seed in plan["seeds"]:
        adapter.validate_bundle_model(
            json.loads(input_path(path, plan, seed).read_text()), plan, seed
        )


def prepare_missing(datasets, root, environment, source_plan=None):
    from srgc_rebuttal import cluster
    from srgc_rebuttal.runtime import lease

    paths = [root / "experiments" / f"llama31-8b-{name}.json" for name in datasets]
    with lease(root / ".start-prepare.lock", wait=True) as guard:
        # Refuse missing plans before model work or any replacement inputs.
        for dataset, path in zip(datasets, paths):
            run = root / "runs" / dataset
            if (
                not path.exists()
                and run.exists()
                and (not run.is_dir() or any(run.iterdir()))
            ):
                raise ValueError(
                    f"Missing plan {path} with existing run; restore the original plan"
                )
        _, identity = resolve_snapshot(environment)
        environment["SRGC_LLAMA_MODEL_SHA256"] = identity
        adapter.runtime_packages()
        for dataset, path in zip(datasets, paths):
            if not path.exists():
                command = [
                    sys.executable,
                    "-m",
                    "srgc_research.dispatch.llama31.cli",
                    dataset,
                    "prepare",
                    "--root",
                    str(root),
                ]
                if source_plan:
                    command += ["--source-plan", str(source_plan)]
                log = root / "startup-logs" / f"{uuid.uuid4().hex}.log"
                print(f"LLAMA preparing {dataset}; log={log}", flush=True)
                code = cluster.run_child(
                    command, log, dict(environment), pass_fds=(guard.fileno(),)
                )
                if code:
                    raise subprocess.CalledProcessError(code, command)
            validate_saved(path, identity)
    return paths


def admit_with_smoke(original, root, environment, run_child, **kwargs):
    from srgc_rebuttal.runtime import atomic_json

    report = original(root, environment, run_child, **kwargs)
    log = root / "llama-smoke.log"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "--max_restarts=0",
        "--module",
        "srgc_research.dispatch.llama31.rank",
        "--stage",
        "smoke",
        "--plan",
        environment["SRGC_LLAMA_PLAN"],
    ]
    started, state = time.perf_counter(), "failed"
    try:
        code = run_child(
            command,
            log,
            environment,
            pass_fds=kwargs.get("pass_fds", ()),
            heartbeat=kwargs.get("heartbeat", lambda pid: None),
            should_stop=kwargs.get("should_stop", lambda: False),
            timeout=3600,
        )
        if code:
            raise RuntimeError(f"Llama generation/backward admission failed: {log}")
        state = "passed"
    finally:
        cost = 4 * (time.perf_counter() - started)
        result = {
            **report,
            "llama_model_smoke": state,
            "llama_smoke_log": str(log),
            "llama_smoke_gpu_seconds": cost,
            "allocated_gpu_seconds": report["allocated_gpu_seconds"] + cost,
        }
        atomic_json(root / "llama-admission.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", choices=("math", "mbpp", "all"), nargs="?", default="all"
    )
    parser.add_argument(
        "action",
        choices=("prepare", "run", "status", "results", "stop", "resume", "doctor"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--root", type=Path, default=default_root(os.environ))
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--now", action="store_true")
    parser.add_argument("--json", action="store_true", help="status JSON")
    args = parser.parse_args(argv)
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    datasets = ("math", "mbpp") if args.dataset == "all" else (args.dataset,)
    if args.source_plan and len(datasets) != 1:
        parser.error("--source-plan requires math or mbpp")
    if args.action == "status":
        from .status import show

        return show(args.dataset, args.root, os.environ, as_json=args.json)
    group, common = setup_storage(args.root, os.environ)
    args.root = args.root.resolve()
    if args.action == "doctor":
        path, identity = resolve_snapshot(os.environ)
        print(
            json.dumps(
                {
                    "model": adapter.MODEL,
                    "model_path": str(path),
                    "snapshot_sha256": identity,
                    "packages": adapter.runtime_packages(),
                },
                indent=2,
            )
        )
        return 0
    if args.action == "prepare":
        from srgc_pair_inputs import default_plan
        from srgc_shared_storage import route_plan
        from transformers import AutoTokenizer

        path, identity = resolve_snapshot(os.environ)
        expected = os.environ.get("SRGC_LLAMA_MODEL_SHA256")
        if expected and expected != identity:
            raise ValueError("Llama model differs from the parent preparation")
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        for dataset in datasets:
            source = args.source_plan or route_plan(
                default_plan(REPO, dataset, os.environ, writing=False), writing=False
            )
            target = adapter.prepare(
                dataset, source, args.root, tokenizer, snapshot_sha256=identity
            )
            print(
                f"LLAMA prepared/verified: {target}; no training launched", flush=True
            )
        return 0
    if args.action in {"run", "resume"}:
        plans = prepare_missing(datasets, args.root, os.environ, args.source_plan)
    else:
        plans = [
            args.root / "experiments" / f"llama31-8b-{name}.json" for name in datasets
        ]
        for path in plans:
            adapter.validate_extension(path, read_only=args.action == "results")
    with adapter.runtime_adapter():
        from srgc_rebuttal import cluster

        if args.action in {"stop", "resume"}:
            for path in plans:
                command = [sys.argv[0], args.action, "--plan", str(path)]
                if args.action == "stop" and args.now:
                    command.append("--now")
                with patch.object(sys, "argv", command):
                    cluster.main()
            if args.action == "stop":
                return 0
        if args.action == "results":
            from srgc_rebuttal import reports

            failed = False
            for path in plans:
                report = reports.snapshot(path, include_costs=True)
                report.update(model=adapter.MODEL, model_revision=adapter.REVISION)
                reports.export(
                    report, args.root / "reports" / f"{path.stem}-results.txt"
                )
                print(
                    f"Model: {adapter.MODEL}\n" + reports.render(report, results=True)
                )
                failed |= bool(report["errors"])
            return int(failed)
        from srgc_checkpoint_backup import automatic_backup
        from srgc_log_format import uniform_log

        from scripts import srgc_process_guard as guard

        from .worker import worker

        args.poll_seconds, args.heartbeat_seconds = 10, 5
        args.retry_delay, args.stall_seconds = 60, 1800
        with ExitStack() as stack:
            stack.enter_context(uniform_log())
            for path in plans:
                stack.enter_context(automatic_backup(path))
            # Reap only this model's orphan ranks, preserving other experiments.
            stack.enter_context(
                patch.object(
                    guard, "TARGET_MARKERS", ("srgc_research.dispatch.llama31.rank",)
                )
            )
            stack.enter_context(
                patch.object(
                    guard,
                    "OWNER_MARKERS",
                    (*guard.OWNER_MARKERS, "srgc_research.dispatch.llama31.cli"),
                )
            )
            stack.enter_context(guard.process_guard(plans[0]))
            stack.enter_context(resume_first_worker())
            worker(plans, args, group, common, admit_with_smoke)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            exc.returncode if exc.returncode > 0 else 128 - exc.returncode
        ) from None
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"LLAMA refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
