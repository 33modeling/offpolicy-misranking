"""Research-only node queue; original experiment queues remain untouched."""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

from srgc_rebuttal.runtime import Busy, atomic_json, lease

from .design import SCOPES, Condition, tasks
from .storage import (
    REPO,
    complete,
    configure_cache,
    freeze,
    prepare,
    read_manifest,
    root,
    verify_inputs,
    verify_runtime,
)


def dependencies(folder, condition, manifest):
    if condition.needs_cache and not complete(folder, manifest, Condition("cache", "cache", updates=0)):
        return False
    return not (condition.kind == "diagnostic" and
                not (folder / "anchors" / f"step-{condition.stage}.json").exists())


def launch_current(folder, condition, manifest, task_lock):
    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    from srgc_rebuttal.existing_runtime import python_path
    runtime = Path(manifest["runtime"])
    verify_runtime(runtime, manifest["implementation_sha256"])
    verify_inputs(folder, manifest)
    out = folder / condition.key
    atomic_json(out / "condition.json", condition.record())
    configure_cache()
    # Only this launcher knows the new children; old launchers are not changed.
    guard.TARGET_MARKERS = tuple(dict.fromkeys((*guard.TARGET_MARKERS, "srgc_research/rank.py")))
    guard.OWNER_MARKERS = tuple(dict.fromkeys((*guard.OWNER_MARKERS, "srgc_research.cli", "srgc_research/launch.py")))
    with guard.process_guard(Path(manifest["source_plan"])):
        devices, uuids = cluster.gpu_identity()
        with cluster.device_leases(root() / "gpu-locks", uuids) as gpu_fds:
            env = cluster.child_environment()
            env["CUDA_VISIBLE_DEVICES"] = devices
            env["PYTHONPATH"] = os.pathsep.join((str(runtime), str(runtime / "src"), env.get("PYTHONPATH", "")))
            python = python_path(manifest["dataset"], env)
            fds = (*gpu_fds, task_lock.fileno())
            # Admission uses the same already-installed interpreter as the rank workers.
            if os.path.abspath(python) != os.path.abspath(sys.executable):
                raise ValueError(f"launcher Python must match {manifest['dataset']} runtime: {python}")
            cluster.admit(out / "admission", env, cluster.run_child, pass_fds=fds, plan=manifest["plan"])
            progress = out / "rank-progress"
            env["SRGC_PROGRESS_DIR"] = str(progress)
            command = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "--max_restarts=0",
                       str(runtime / "srgc_research/rank.py"), "--folder", str(folder),
                       "--condition", str(out / "condition.json")]
            print(f"RUN {manifest['dataset']} seed={manifest['seed']} task={condition.key} log={out / 'task.log'}", flush=True)
            return cluster.run_child(command, out / "task.log", env, pass_fds=fds,
                heartbeat=lambda pid: atomic_json(out / "worker.json", {"pid": pid, "host": socket.gethostname(),
                    "heartbeat": time.time(), "status": "running"}), progress=lambda: cluster.progress_signature(progress))


def launch(folder, condition, manifest, task_lock):
    from srgc_rebuttal import cluster
    from srgc_rebuttal.existing_runtime import python_path
    python = python_path(manifest["dataset"], os.environ)
    if os.path.abspath(python) == os.path.abspath(sys.executable):
        return launch_current(folder, condition, manifest, task_lock)
    # MATH and MBPP may use different existing environments on the same queue.
    # Run admission and ranks in that environment, without installing packages.
    runtime = Path(manifest["runtime"])
    verify_runtime(runtime, manifest["implementation_sha256"])
    out = folder / condition.key
    atomic_json(out / "condition.json", condition.record())
    command = [python, str(runtime / "srgc_research/launch.py"), str(folder),
               str(out / "condition.json"), str(task_lock.fileno())]
    from scripts import srgc_process_guard as guard
    guard.OWNER_MARKERS = tuple(dict.fromkeys((*guard.OWNER_MARKERS, "srgc_research.cli", "srgc_research/launch.py")))
    return guard.guarded_run_child(cluster.run_child, Path(manifest["source_plan"]))(command,
        out / "launcher.log", cluster.child_environment(), pass_fds=(task_lock.fileno(),))


def prepare_datasets(root_path, datasets):
    from scripts.srgc_child_tuning import configured_attention
    from scripts.srgc_extra_plan import select_plan
    from scripts.srgc_pair_inputs import default_plan
    from scripts.srgc_shared_storage import route_plan
    runtime, implementation = freeze(root_path)
    folders = []
    for dataset in datasets:
        source = default_plan(REPO, dataset, os.environ, writing=False)
        active = route_plan(source, writing=False)
        for seed in range(5, 10):
            plan = select_plan(active, seed)
            folder = root_path / dataset / f"seed-{seed}"
            manifest = prepare(folder, dataset, seed, plan, runtime, implementation, configured_attention() or "eager")
            folders.append((folder, manifest))
    return folders


def run_queue(folders, scope, *, poll=30, max_attempts=3, retry_delay=120):
    while True:
        waiting, unfinished, failed = False, 0, []
        for condition in tasks(scope):
            for folder, manifest in folders:
                if complete(folder, manifest, condition):
                    continue
                unfinished += 1
                if not dependencies(folder, condition, manifest):
                    continue
                out = folder / condition.key
                try:
                    with lease(out / ".dispatch.lock") as handle:
                        with lease(out / ".execution.lock"):
                            pass
                        if complete(folder, manifest, condition):
                            continue
                        receipt = out / "queue.json"
                        previous = json.loads(receipt.read_text()) if receipt.exists() else {}
                        attempts = previous.get("attempts", 0)
                        if type(attempts) is not int or attempts < 0:
                            raise ValueError(f"invalid retry receipt: {receipt}")
                        if attempts >= max_attempts:
                            failed.append(str(out))
                            continue
                        if previous.get("status") == "failed" and time.time() < previous.get("finished", 0) + retry_delay:
                            waiting = True
                            continue
                        atomic_json(receipt, {"status": "running", "attempts": attempts, "started": time.time()})
                        error = None
                        try:
                            code = launch(folder, condition, manifest, handle)
                        except Busy:
                            code = 75
                        except TimeoutError as exc:
                            # run_child already reaped its process group and released GPU leases.
                            code, error = 124, str(exc)
                            print(f"TIMEOUT {condition.key}: {error}", file=sys.stderr, flush=True)
                        except KeyboardInterrupt:
                            atomic_json(receipt, {"status": "interrupted", "attempts": attempts})
                            raise
                        if code == 0 and not complete(folder, manifest, condition):
                            raise ValueError("worker exited without a validated endpoint")
                        atomic_json(receipt, {"status": "complete" if code == 0 else "busy" if code == 75 else "failed",
                            "attempts": attempts + (code not in (0, 75)), "finished": time.time(), "exit_code": code,
                            "error": error})
                        if code == 75:
                            print("BUSY: this node is occupied; existing work was not stopped", flush=True)
                            return 75
                        waiting = True
                except Busy:
                    waiting = True
        if not unfinished:
            return 0
        if failed and not waiting:
            print("FAILED tasks: " + ", ".join(failed), file=sys.stderr)
            return 2
        print(f"QUEUE pending={unfinished} failed={len(failed)}; waiting {poll}s", flush=True)
        time.sleep(poll)


def status(root_path, datasets, scope):
    rows = []
    for dataset in datasets:
        for seed in range(5, 10):
            folder = root_path / dataset / f"seed-{seed}"
            marker = folder / "manifest.json"
            manifest, manifest_error = None, None
            try:
                if marker.exists():
                    manifest = read_manifest(marker, dataset, seed)
            except (OSError, ValueError, TypeError) as exc:
                manifest_error = str(exc)
            for condition in tasks(scope):
                out, state = folder / condition.key, "not-started"
                details = {}
                try:
                    if manifest_error:
                        raise ValueError(manifest_error)
                    if manifest and complete(folder, manifest, condition):
                        state = "complete"
                    elif manifest:
                        if (out / "progress.json").exists():
                            details = json.loads((out / "progress.json").read_text())
                            if not isinstance(details, dict):
                                raise ValueError(f"invalid progress record: {out / 'progress.json'}")
                            state = "checkpointed"
                        if not dependencies(folder, condition, manifest):
                            state = "waiting-prerequisite"
                        try:
                            with lease(out / ".dispatch.lock"):
                                pass
                        except Busy:
                            state = "running"
                        receipt = out / "queue.json"
                        if state != "running" and receipt.exists():
                            receipt = json.loads(receipt.read_text())
                            if not isinstance(receipt, dict):
                                raise ValueError(f"invalid queue record: {out / 'queue.json'}")
                            if receipt.get("status") == "failed":
                                state = "failed"
                                details["error"] = receipt.get("error")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    state, details = "invalid", {"error": str(exc)}
                rows.append({"dataset": dataset, "seed": seed, "task": condition.key, "status": state,
                             "step": details.get("step"), "phase": details.get("phase"),
                             "error": details.get("error"), "result": str(out / "endpoint.json")})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument("action", nargs="?", default="run", choices=("run", "status", "results", "json"))
    parser.add_argument("archive", nargs="?", type=Path, help="N08: existing results JSON bundle")
    args = parser.parse_args(argv)
    if args.scope == "n08":
        if args.archive is None:
            parser.error("N08 needs the existing results JSON: math n08 results /path/results.json")
        if args.action not in {"results", "json"}:
            parser.error("N08 is an analysis action: use results or json, not run/status")
        from .report import analyze_archive
        bundle = json.loads(args.archive.read_text())
        observed = {bundle.get("dataset")} | {r.get("dataset") for r in bundle.get("source_results", {}).values()
                                             if isinstance(r, dict)}
        allowed = {args.dataset, "math_train" if args.dataset == "math" else "mbpp", None}
        if args.dataset != "all" and observed - allowed:
            parser.error("archive dataset differs from the requested dataset")
        report = analyze_archive(bundle)
        report["source"] = {"path": str(args.archive.resolve())}
        from srgc_rebuttal.plan import digest
        report["source"]["sha256"] = digest(args.archive)
        target = args.archive.with_name(args.archive.stem + "-n08.json")
        report["result_path"] = str(target)
        atomic_json(target, report)
        print(json.dumps(report, indent=2) if args.action == "json" else
              f"RESULT {target}\npaired comparisons={len(report['comparisons'])} errors={len(report['errors'])}")
        for error in report["errors"]:
            print(error, file=sys.stderr)
        return int(bool(report["errors"]))
    if args.archive is not None:
        parser.error("archive path is only used by N08")
    datasets = ("math", "mbpp") if args.dataset == "all" else (args.dataset,)
    root_path = root()
    if args.action == "status":
        rows = status(root_path, datasets, args.scope)
        print(f"RESULT ROOT {root_path}")
        for row in rows:
            print(f"{row['dataset']} seed={row['seed']} {row['task']} {row['status']} "
                  f"step={row['step']} phase={row['phase']} result={row['result']}" +
                  (f" error={row['error']}" if row['error'] else ""))
        return int(any(r["status"] == "invalid" for r in rows))
    if args.action in {"results", "json"}:
        from .report import collect
        report = collect(root_path, datasets, args.scope)
        print(json.dumps(report, indent=2) if args.action == "json" else
              f"RESULT {report['result_path']}\ncomplete={len(report['rows'])} pending={len(report['pending'])} errors={len(report['errors'])}")
        for error in report["errors"]:
            print(error, file=sys.stderr)
        return int(bool(report["errors"]))
    return run_queue(prepare_datasets(root_path, datasets), args.scope)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"RESEARCH ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
