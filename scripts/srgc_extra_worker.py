#!/usr/bin/env python3
"""Launch one extra arm with the same device admission and teardown as P0."""

import argparse
import faulthandler
import os
from pathlib import Path
import socket
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

if __name__ == "__main__":
    from scripts.srgc_saved_runtime import bootstrap
    bootstrap(sys.argv[1:])

from srgc_rebuttal import cluster  # noqa: E402
from srgc_rebuttal.plan import load_plan  # noqa: E402
from srgc_rebuttal.runtime import Busy, atomic_json, lease, run_root  # noqa: E402
from scripts.srgc_process_guard import process_guard  # noqa: E402
from scripts.srgc_sr_refresh import arm_name, extra_arm, extra_complete, prepare_run_storage, replicate_of, SCOPES  # noqa: E402


def launch(args):
    print(f"START seed-{args.seed}.{args.arm or arm_name(args.scope)}: verifying saved prefix",
          flush=True)
    # A stalled shared-volume read must identify its exact blocking operation.
    faulthandler.dump_traceback_later(60, repeat=True, file=sys.__stderr__)
    try:
        prepare_run_storage(args)
    finally:
        faulthandler.cancel_dump_traceback_later()
    plan = load_plan(args.plan)
    name = args.arm or arm_name(args.scope)
    replicate = replicate_of(name)
    arm = replicate[1] if replicate else name
    if arm.startswith("switch_fixed"):
        from scripts.srgc_switch_fixed import fixed_step_of
        step = fixed_step_of(arm)
        if not plan["shared_prefix_updates"] <= step < plan["total_updates"] or step % plan["selection_interval"]:
            raise ValueError("fixed step must be a refresh boundary from the shared prefix to before the endpoint")
    root = run_root(args.plan, plan)
    folder = root / f"seed-{args.seed}"
    out = folder / f"replicate-{replicate[0]}" if replicate else folder
    if extra_complete(args.plan, plan, args.seed, name):
        print(f"PASS: {name} already complete for seed {args.seed}", flush=True)
        return 0

    # Serialize this tuple across nodes before allocating GPUs. The rank keeps
    # the existing execution lock; never remove or replace that lock file.
    print("PREFLIGHT checking task ownership and GPU processes", flush=True)
    with process_guard(args.plan), lease(out / f".{arm}.launch.lock") as launch_lock:
        with lease(out / f".{arm}.execution.lock"):
            pass  # Also recognize a live job started by the old raw-torchrun launcher.
        if extra_complete(args.plan, plan, args.seed, name):
            return 0
        environment = cluster.child_environment()
        environment.setdefault("NCCL_DEBUG", "WARN")
        devices, uuids = cluster.gpu_identity()
        environment["CUDA_VISIBLE_DEVICES"] = devices
        session = out / "launches" / arm / uuid.uuid4().hex
        record = {"arm": name, "seed": args.seed, "host": socket.gethostname(),
                  "pid": os.getpid(), "started": time.time(), "status": "preflight"}

        def update(state, **details):
            record.update(status=state, heartbeat=time.time(), **details)
            atomic_json(session / "worker.json", record)

        with cluster.device_leases(root.parent / "gpu-node-locks", uuids) as gpu_fds:
            fds = (*gpu_fds, launch_lock.fileno())
            try:
                print(f"ADMISSION checking four GPUs and NCCL log={session / 'admission'}", flush=True)
                admission = cluster.admit(session / "admission", environment, cluster.run_child,
                    pass_fds=fds, plan=plan, heartbeat=lambda pid: update("preflight", child_pid=pid))
                progress = session / "progress"
                environment["SRGC_PROGRESS_DIR"] = str(progress)
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node=4", "--max_restarts=0", str(REPO / "scripts/srgc_sr_refresh.py"),
                    "run", "--plan", str(args.plan), "--seed", str(args.seed), "--arm", name]
                log = session / "task.log"
                update("running", admission=admission, log=str(log))
                print(f"RUN seed-{args.seed}.{name} log={log}", flush=True)
                code = cluster.run_child(command, log, environment, pass_fds=fds,
                    heartbeat=lambda pid: update("running", child_pid=pid),
                    progress=lambda: cluster.progress_signature(progress))
                if code == 0 and not extra_complete(args.plan, plan, args.seed, name):
                    code = 2
                    record["error"] = "child exited successfully without a verified endpoint"
                update("complete" if code == 0 else "failed", exit_code=code, finished=time.time())
                return code
            except TimeoutError as exc:
                update("failed", exit_code=124, error=str(exc), finished=time.time())
                print(f"TIMEOUT: {exc}", file=sys.stderr, flush=True)
                return 124
            except BaseException as exc:
                update("stopped" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error=f"{type(exc).__name__}: {exc}", finished=time.time())
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scope", choices=SCOPES, default="candidates")
    parser.add_argument("--arm", type=extra_arm)
    args = parser.parse_args()
    try:
        return launch(args)
    except Busy as exc:
        print(f"BUSY: existing task or GPU owner holds {exc}; no process stopped, no lock removed", file=sys.stderr)
        return 75
    except (ValueError, KeyError, TypeError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
