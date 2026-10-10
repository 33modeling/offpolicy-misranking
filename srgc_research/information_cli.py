"""Explicit checkpoint measurements and offline reports; no carrier training."""

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from srgc_rebuttal.plan import load_plan, validate_inputs
from srgc_rebuttal.runtime import Busy, atomic_json, lease

from .information_report import (
    PHASES,
    PROTOCOL,
    digest,
    read_measurement,
    read_object,
    result_digest,
    write_report,
)
from .storage import configure_cache, freeze, verify_runtime


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    expected = digest(source)
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError("source differs from the already frozen measurement input")
        return expected
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        if digest(temporary) != expected or digest(source) != expected:
            raise ValueError("source changed while being copied; retry after its atomic checkpoint save")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def measurement_location(args):
    """Reject source/output collisions before creating even a lock file."""
    if args.dataset == "all":
        raise ValueError("collect takes one explicit dataset/checkpoint")
    if args.stage != 0 and args.checkpoint is None:
        raise ValueError("nonzero stages require an existing model+optimizer checkpoint")
    if args.seed < 0 or args.stage < 0:
        raise ValueError("seed and stage must be nonnegative")
    from scripts.srgc_shared_storage import storage_root
    group, _ = storage_root(os.environ)
    output = args.output.resolve()
    if not output.is_relative_to(group) or output == group:
        raise ValueError("GPU measurements must use a separate output directory inside group storage")
    for source in (args.inputs, args.plan, args.checkpoint):
        if source is not None and source.resolve().is_relative_to(output):
            raise ValueError("measurement output must not contain or overwrite its source files")
    if output.exists() and not (output / "manifest.json").exists():
        allowed = {".dispatch.lock", ".execution.lock", "measurement-code", "inputs.json", "plan.json", "source-checkpoint.pt"}
        if any(p.name not in allowed for p in output.iterdir()):
            raise ValueError("output contains unrelated files; use a separate measurement directory")
    return output


def prepare(args):
    """Freeze explicit inputs, source checkpoint and measurement code separately."""
    output = measurement_location(args)
    plan = load_plan(args.plan)
    if plan["objective"] != "grpo":
        raise ValueError("information collection currently supports the actual GRPO backend")
    data = read_object(args.inputs)
    validate_inputs(data, recorded_rewards=True)
    expected_dataset = "mbpp" if "code_reward" in plan["verifier"] else "math"
    if args.dataset != expected_dataset:
        raise ValueError("dataset label differs from the plan's verifier")
    if not 1 <= args.probe_prompts <= len(set(data["validation_pool_ids"]) - set(data["ranking_validation_ids"])):
        raise ValueError("invalid independent probe size")
    # read once here and copy with hashes below; frozen rank validation reads it again.
    runtime, implementation = freeze(output / "measurement-code")
    identity = {"protocol": PROTOCOL, "dataset": args.dataset, "seed": args.seed, "stage": args.stage,
                "input_sha256": digest(args.inputs), "plan_sha256": digest(args.plan),
                "source_checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
                "measurement_sha256": implementation, "probe_prompts": args.probe_prompts,
                "requested_attention": args.attention,
                "model": plan["model"], "model_revision": plan["model_revision"]}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_object(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError("measurement inputs, checkpoint, settings or code changed; use a new output directory")
        verify_runtime(Path(manifest["runtime"]), implementation)
    else:
        copy_verified(args.inputs, output / "inputs.json")
        copy_verified(args.plan, output / "plan.json")
        if args.checkpoint:
            copy_verified(args.checkpoint, output / "source-checkpoint.pt")
        manifest = {"identity": identity, "input_sha256": identity["input_sha256"],
                    "plan_sha256": identity["plan_sha256"], "runtime": str(runtime),
                    "source_inputs": str(args.inputs.resolve()), "source_plan": str(args.plan.resolve()),
                    "source_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                    "measurement_kind": "same-checkpoint-two-selector-one-update",
                    "world_size": plan["world_size"]}
        atomic_json(manifest_path, manifest)
    for name, expected in (("inputs.json", identity["input_sha256"]), ("plan.json", identity["plan_sha256"]),
                           ("source-checkpoint.pt", identity["source_checkpoint_sha256"])):
        if expected is not None and digest(output / name) != expected:
            raise ValueError(f"frozen source differs: {name}")
    return manifest


def collect(args):
    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    from srgc_rebuttal.existing_runtime import python_path
    output = measurement_location(args)
    with lease(output / ".dispatch.lock") as task_lock:
        with lease(output / ".execution.lock"):
            pass
        manifest = prepare(args)
        if (output / "endpoint.json").exists():
            read_measurement(output)
            print(f"COMPLETE: {output / 'endpoint.json'}")
            return 0
        runtime = Path(manifest["runtime"])
        configure_cache()
        guard.TARGET_MARKERS = (*guard.TARGET_MARKERS, "srgc_research/information_rank.py")
        guard.OWNER_MARKERS = (*guard.OWNER_MARKERS, "srgc_research.information_cli")
        with guard.process_guard(args.plan):
            devices, uuids = cluster.gpu_identity()
            with cluster.device_leases(output.parent / "information-gpu-locks", uuids) as gpu_fds:
                env = cluster.child_environment()
                env["CUDA_VISIBLE_DEVICES"] = devices
                env["PYTHONPATH"] = os.pathsep.join((str(runtime), str(runtime / "src"), env.get("PYTHONPATH", "")))
                python = python_path(args.dataset, env)
                if os.path.abspath(python) != os.path.abspath(sys.executable):
                    raise ValueError("run the wrapper with the dataset's existing Python environment")
                fds = (*gpu_fds, task_lock.fileno())
                cluster.admit(output / "admission" / uuid.uuid4().hex, env, cluster.run_child, pass_fds=fds,
                              plan=load_plan(output / "plan.json"))
                progress = output / "rank-progress"
                env["SRGC_PROGRESS_DIR"] = str(progress)
                command = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                    "--max_restarts=0", str(runtime / "srgc_research/information_rank.py"), "--output", str(output)]
                code = cluster.run_child(command, output / "task.log", env, pass_fds=fds,
                    heartbeat=lambda pid: atomic_json(output / "worker.json", {"pid": pid, "heartbeat": time.time()}),
                    progress=lambda: cluster.progress_signature(progress))
                if code == 0:
                    read_measurement(output)
                return code


def status(output):
    output = Path(output)
    if not (output / "manifest.json").exists():
        return {"status": "not-started", "phases": {p: "missing" for p in PHASES}}
    manifest = read_object(output / "manifest.json")
    for name, expected in (("inputs.json", manifest["identity"]["input_sha256"]),
                           ("plan.json", manifest["identity"]["plan_sha256"]),
                           ("source-checkpoint.pt", manifest["identity"]["source_checkpoint_sha256"])):
        if expected is not None and digest(output / name) != expected:
            raise ValueError("frozen source hash differs")
    phases = {}
    for name in PHASES:
        path = output / f"{name}.json"
        if not path.exists():
            phases[name] = "pending"
            continue
        receipt = read_object(path)
        if receipt.get("identity") != manifest["identity"] or receipt.get("phase") != name:
            raise ValueError("phase identity differs from measurement")
        if not receipt["artifacts"] or result_digest(receipt["result"]) != receipt["result_sha256"]:
            raise ValueError("phase result hash differs")
        for artifact in receipt["artifacts"]:
            target = (output / artifact["file"]).resolve()
            if not target.is_relative_to(output.resolve()) or digest(target) != artifact["sha256"]:
                raise ValueError("phase artifact hash differs")
        phases[name] = "saved"
    complete = (output / "endpoint.json").exists()
    if complete:
        read_measurement(output)
    return {"identity": manifest["identity"], "status": "complete" if complete else "partial",
            "phases": phases, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("collect", help="inspect an existing checkpoint; no new learning trajectory")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--stage", type=int, required=True)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--attention", choices=("eager", "sdpa", "flash_attention_2"))
    run.add_argument("--probe-prompts", type=int, default=8)
    run.add_argument("--output", type=Path, required=True)
    show = actions.add_parser("status")
    show.add_argument("--output", type=Path, required=True)
    report = actions.add_parser("report", help="CPU-only HTML/CSV/JSON export")
    report.add_argument("--measurement", action="append", type=Path, default=[])
    report.add_argument("--legacy", action="append", type=Path, default=[])
    report.add_argument("--inputs", action="append", type=Path, default=[])
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "collect":
            return collect(args)
        if args.action == "status":
            print(json.dumps(status(args.output), indent=2, ensure_ascii=False))
        else:
            if not args.measurement and not args.legacy:
                parser.error("report requires --measurement or --legacy")
            result = write_report(args.output, folders=args.measurement, legacy=args.legacy, inputs=args.inputs,
                                  dataset=None if args.dataset == "all" else args.dataset)
            print(f"REPORT: {args.output / 'report.html'}")
            print(f"SELECTED: {len(result['selected_problems'])} records; UPDATES: {len(result['batch_updates'])} batches")
        return 0
    except Busy:
        print("BUSY: measurement or node is already in use", file=sys.stderr)
        return 75
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
