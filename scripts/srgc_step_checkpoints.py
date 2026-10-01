"""Add per-update checkpoint writes without changing the experiment's computations."""

import argparse
import os
from pathlib import Path
import sys


def save_checkpoint(engine, folder, task, *, metadata=None):
    import torch
    from srgc_rebuttal.distributed import primary
    path = folder / ("prefix-latest.pt" if task == "prefix" else f"{task}-latest.pt")
    meter = engine.backend.cost_meter
    with meter.phase("checkpoint_save", engine.step, engine.backend.gpu_count):
        with meter.stage("state_snapshot"):
            state = engine.state_dict()
            state.update(metadata or {})
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
    try:
        from srgc_direction_records import DirectionRecordMixin
    except ImportError:
        from scripts.srgc_direction_records import DirectionRecordMixin

    class StepCheckpointEngine(DirectionRecordMixin, base):
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
    from scripts.srgc_worker_status import run_with_status
    original = cluster.task_command
    original_worker = cluster.run_worker
    cluster.task_command = lambda queue, task: checkpoint_command(original, queue, task)
    cluster.run_worker = lambda *args, **kwargs: run_with_status(original_worker, *args, **kwargs)
    try:
        cluster.main()
    finally:
        cluster.task_command = original
        cluster.run_worker = original_worker


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from srgc_verifier_fallback import install as install_tolerant_verifier
    from srgc_child_tuning import apply_attention, count_progress
    from srgc_rebuttal import run_experiment
    install_tolerant_verifier()
    count_progress()
    attention = apply_attention()
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
              "legacy_boundary_saves_retained": True, "attention": attention,
              "rollout_cache": "per-prompt rollouts persisted during scoring/evaluation; removed when the block completes"}
    from srgc_resumable_rollouts import install as install_resumable_rollouts
    install_resumable_rollouts(folder / "rollout-cache" / args.task)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"ATTENTION {attention} (SRGC_ATTENTION selects sdpa|eager|flash_attention_2)", flush=True)
    original = run_experiment.Engine
    total = plan["shared_prefix_updates"] if args.task == "prefix" else plan["total_updates"]
    run_experiment.Engine = checkpoint_engine(original, folder, args.task, policy, total_updates=total)
    try:
        run_experiment.main()
    finally:
        run_experiment.Engine = original


if __name__ == "__main__":
    main()
