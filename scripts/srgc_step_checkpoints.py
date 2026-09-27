"""Add per-update checkpoint writes without changing the experiment's computations."""

import argparse
import os
from pathlib import Path
import sys


def save_checkpoint(engine, folder, task):
    import torch
    from srgc_rebuttal.distributed import primary
    path = folder / ("prefix-latest.pt" if task == "prefix" else f"{task}-latest.pt")
    meter = engine.backend.cost_meter
    with meter.phase("checkpoint_save", engine.step, engine.backend.gpu_count):
        with meter.stage("state_snapshot"):
            state = engine.state_dict()
        def write():
            with meter.stage("write"):
                temporary = path.with_suffix(".tmp")
                try:
                    with temporary.open("wb") as handle:
                        torch.save(state, handle)
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        primary(write)
    primary(lambda: print(f"CHECKPOINT saved seed={engine.config.seed} task={task} "
                          f"step={engine.step} file={path}", flush=True))


def checkpoint_engine(base, folder, task, policy, *, total_updates):
    legacy_interval = 5 if task == "prefix" else 25
    phase = "shared-prefix" if task == "prefix" else "continuation"

    class StepCheckpointEngine(base):
        def log_step(self, status):
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                step = self.step + 1 if status == "running" else self.step
                print(f"TRAIN seed={self.config.seed} phase={phase} arm={self.arm} "
                      f"step={step}/{total_updates} status={status} "
                      f"completed={self.step}/{total_updates}", flush=True)

        def state_dict(self):
            return {**super().state_dict(), "checkpoint_policy": dict(policy)}

        def update(self):
            self.log_step("running")
            result = super().update()
            # The unchanged runner saves its original boundaries immediately after update().
            if self.step % legacy_interval:
                save_checkpoint(self, folder, task)
            self.log_step("completed")
            return result

    return StepCheckpointEngine


def checkpoint_command(original, queue, task):
    command = original(queue, task)
    if task.arm == "cache":
        return command
    module = command.index("srgc_rebuttal.run_experiment")
    if command[module - 1] != "-m":
        raise ValueError("unexpected training launch command")
    return [*command[:module - 1], str(Path(__file__).resolve()), *command[module + 1:]]


def worker_main():
    from srgc_rebuttal import cluster
    original = cluster.task_command
    cluster.task_command = lambda queue, task: checkpoint_command(original, queue, task)
    try:
        cluster.main()
    finally:
        cluster.task_command = original


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from srgc_rebuttal import run_experiment
    from srgc_rebuttal.plan import DEFAULT_PLAN, digest, load_plan
    from srgc_rebuttal.runtime import run_root
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", choices=("prefix", "random", "sr", "on_policy", "switch"), required=True)
    args, _ = parser.parse_known_args()
    plan = load_plan(args.plan)
    folder = run_root(args.plan, plan) / f"seed-{args.seed}"
    policy = {"interval_updates": 1, "storage_adapter_sha256": digest(Path(__file__)),
              "legacy_boundary_saves_retained": True}
    original = run_experiment.Engine
    total = plan["shared_prefix_updates"] if args.task == "prefix" else plan["total_updates"]
    run_experiment.Engine = checkpoint_engine(original, folder, args.task, policy, total_updates=total)
    try:
        run_experiment.main()
    finally:
        run_experiment.Engine = original


if __name__ == "__main__":
    main()
