#!/usr/bin/env python3
"""Bounded node admission before Switch/MoPPS claims any training task."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import selection_gate as core
import selection_gate_gpu as base


def worker(directory, world_size):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    evidence = {
        "rank": rank, "local_rank": local_rank, "host": socket.gethostname(), "pid": os.getpid(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()), "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cumem_host": os.environ.get("NCCL_CUMEM_HOST_ENABLE"),
    }
    path = directory / f"rank-{rank}.json"
    core.atomic_json(path, {**evidence, "state": "starting"})
    try:
        if int(os.environ["WORLD_SIZE"]) != world_size or int(os.environ["LOCAL_WORLD_SIZE"]) != world_size:
            raise ValueError("NCCL admission requires a single node with the requested rank count")
        count = torch.cuda.device_count()
        if count != world_size or not 0 <= local_rank < count:
            raise ValueError(f"GPU allocation mismatch: visible CUDA devices={count}, ranks={world_size}, local_rank={local_rank}")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(device)
        evidence.update(gpu=properties.name, gpu_uuid=str(properties.uuid))
        core.atomic_json(path, {**evidence, "state": "initializing"})
        print("[nccl-rank] " + json.dumps(evidence, sort_keys=True), flush=True)
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=30))
        value = torch.tensor([rank + 1.], device=device)
        dist.all_reduce(value)
        if value.item() != world_size * (world_size + 1) / 2:
            raise RuntimeError("NCCL all-reduce returned an incorrect result")
        torch.manual_seed(1701)
        model = DistributedDataParallel(torch.nn.Linear(8, 4).to(device), device_ids=[local_rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        model(torch.ones(2, 8, device=device) * (rank + 1)).sum().backward()
        optimizer.step()
        parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = [torch.empty_like(parameters) for _ in range(world_size)]
        dist.all_gather(gathered, parameters)
        if not torch.isfinite(parameters).all().item() or any(not torch.equal(parameters, p) for p in gathered):
            raise RuntimeError("DDP update differs across ranks or is nonfinite")
        torch.cuda.synchronize(device)
        dist.destroy_process_group()
        core.atomic_json(path, {**evidence, "state": "passed"})
        print(f"[nccl-rank] rank={rank} DDP forward/backward and collectives passed", flush=True)
    except BaseException as exc:
        core.atomic_json(path, {**evidence, "state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def host_allocation_fallback(reports, error, env, world_size):
    """Only the pre-2.26.5 host-allocation workaround, after an observed failure."""
    if "NCCL_CUMEM_HOST_ENABLE" in env or len(reports) != world_size:
        return False
    versions = {tuple(row.get("nccl", ())) for row in reports}
    if len(versions) != 1 or not (2, 24, 0) <= next(iter(versions)) < (2, 26, 5):
        return False
    lower = error.lower()
    if any(message in lower for message in (
            "out of memory", "illegal memory access", "duplicate gpu", "insufficient driver",
            "driver version is insufficient", "no kernel image", "allocation mismatch")):
        return False
    return "ncclunhandledcudaerror" in lower or ("nccl warn" in lower and "cuda failure" in lower)


def rank_reports(directory, world_size):
    return [core.read(directory / f"rank-{rank}.json") for rank in range(world_size)
            if (directory / f"rank-{rank}.json").is_file()]


def check_attempt(directory, visible, env, world_size, timeout):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={world_size}", "--max_restarts=0", str(Path(__file__).resolve()),
               "--worker", "--root", str(directory), "--world-size", str(world_size)]
    debug_env = {**env, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "ALL"}
    # meter merges the current environment; the rank clears this masked override.
    debug_env["NCCL_DEBUG_FILE"] = ""
    base.meter(directory, "nccl-check", "node admission (hardware in rank reports)",
               commands=[(command, visible)], env=debug_env, timeout=timeout,
               ledger="research", devices=world_size)
    reports = rank_reports(directory, world_size)
    if (len(reports) != world_size or any(row.get("state") != "passed" for row in reports)
            or len({row.get("gpu_uuid") for row in reports}) != world_size):
        raise RuntimeError("NCCL admission lacks successful evidence from distinct GPUs on every rank")
    return reports


def preflight(root, *, world_size=4, timeout=90.):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",")
    if len(devices) != world_size or len(set(devices)) != world_size or not all(devices):
        raise ValueError(f"NCCL admission requires {world_size} distinct allocated CUDA_VISIBLE_DEVICES")
    directory = root / "node-preflight" / f"{socket.gethostname()}-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": "selection-nccl-admission/v1", "host": socket.gethostname(),
              "pid": os.getpid(), "world_size": world_size, "visible": visible,
              "runtime_commit": os.environ.get("SWITCH_RUNTIME_COMMIT"),
              "probe_sha256": base.digest(Path(__file__)), "attempts": [], "state": "running",
              "cost_scope": "shared node-admission research cost; no task claimed or deployment cost waived"}
    overrides = {}
    try:
        for name in ("baseline", "legacy-host-allocation"):
            attempt = directory / name
            env = {**os.environ, **overrides}
            print(f"[nccl-preflight] {name}: {world_size} ranks; timeout={timeout:.0f}s; {attempt}", flush=True)
            error = None
            reports = []
            try:
                reports = check_attempt(attempt, visible, env, world_size, timeout)
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                reports = rank_reports(attempt, world_size)
                if not isinstance(exc, Exception):
                    raise
            finally:
                report["attempts"].append({"name": name, "directory": str(attempt), "ranks": reports,
                                           "overrides": dict(overrides), "error": error, "cost": base.cost(attempt)})
            if error is None:
                report.update(state="passed", overrides=dict(overrides))
                print(f"[nccl-preflight] passed; overrides={json.dumps(overrides)}", flush=True)
                return overrides
            print(f"[nccl-preflight] failed: {error}", flush=True)
            rank_errors = "\n".join(f"rank {row['rank']}: {row['error']}" for row in reports if row.get("error"))
            if rank_errors:
                print(f"[nccl-preflight] original rank errors:\n{rank_errors}", flush=True)
            if name == "baseline" and host_allocation_fallback(reports, error + "\n" + rank_errors, env, world_size):
                overrides = {"NCCL_CUMEM_HOST_ENABLE": "0"}
                print("[nccl-preflight] retrying only the tiny probe with legacy host allocation; "
                      "no policy training was started", flush=True)
            else:
                raise RuntimeError(f"NCCL node admission failed; no training task claimed. "
                                   f"Rank errors and original logs: {directory}")
        raise RuntimeError("NCCL node admission did not pass")
    except BaseException as exc:
        report.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["allocated_gpu_seconds"] = world_size * report["wall_seconds"]
        with base.defer_stop_signals():
            core.atomic_json(directory / "admission.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.world_size < 1 or not 0 < args.timeout <= 300:
        parser.error("world size must be positive; timeout must be in (0, 300] seconds")
    if args.worker:
        os.environ.pop("NCCL_DEBUG_FILE", None)
        from torch.distributed.elastic.multiprocessing.errors import record
        record(worker)(args.root, args.world_size)
        return 0
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    try:
        overrides = preflight(args.root, world_size=args.world_size, timeout=args.timeout)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[blocked] {exc}", file=sys.stderr, flush=True)
        return 78
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command:
        env = {**os.environ, **overrides}
        env.setdefault("NCCL_DEBUG", "WARN")
        os.execvpe(command[0], command, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
