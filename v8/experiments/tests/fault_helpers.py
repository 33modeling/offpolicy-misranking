"""Subprocess / torchrun entry points for test_runtime_faults (kills, torn writes, rank faults, stops).

    python fault_helpers.py <mode> <out_dir> [crash|resume]
    torchrun --standalone --nproc_per_node=2 fault_helpers.py <gloo-mode> <out_dir>

Every mode uses the real switching.runner on the code repository's toy backend (CPU only).
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("V8_FSYNC", "0")


def toy(out, horizon):
    from switching import backends, coderepo, matrix as mx, spec
    from switching.engine import make_engine_class
    from switching.smoke import Locator, make_prefix, toy_ctx
    srgc = coderepo.srgc()
    PE = make_engine_class(srgc)
    feat, ans, data = backends.toy_problem(0)
    cfg = spec.resolve("olmo_math", role="smoke", extra={"cohort_id": "smoke_toy", "seeds": {"training": [5]},
                       "model": {"repository": "toy", "revision": "x"}})
    cfg["training"]["matched_update_endpoint"] = horizon
    ctx = toy_ctx(out, 5, srgc, PE, feat, ans, data, horizon)
    make_prefix(ctx, 5, 25, srgc)
    task = mx.build_stage("bank-trajectory", cfg, Locator(out, cfg, 25), out, role="rebuttal_bank", seeds=[5])[0][0]
    return ctx, task, cfg, data


def crash_index(out, phase):
    """Kill between a retained snapshot file and its snapshots.jsonl line (RT-02)."""
    from switching import common
    from switching.runner import execute_train
    ctx, task, _, _ = toy(out, 100)
    if phase == "crash":
        original = common.JsonlLog.append

        def append(self, row):
            if self.path.name == "snapshots.jsonl" and row.get("completed_update") == 50:
                os._exit(137)
            return original(self, row)
        common.JsonlLog.append = append
    print("RESULT", execute_train(task, ctx)["status"], task["run_id"], flush=True)


def torn_line(out, phase):
    """Kill in the middle of an events.jsonl append (RT-08)."""
    from switching import common
    from switching.runner import execute_train
    ctx, task, _, _ = toy(out, 75)
    if phase == "crash":
        original = common.JsonlLog.append

        def append(self, row):
            if self.path.name == "events.jsonl" and row.get("completed_update") == 40:
                line = common.canonical_json(row)
                with open(self.path, "a") as handle:
                    handle.write(line[: len(line) // 2])
                os._exit(137)
            return original(self, row)
        common.JsonlLog.append = append
    print("RESULT", execute_train(task, ctx)["status"], task["run_id"], flush=True)


def killed_cost(out, phase):
    """Kill inside the training phase of update 31 (RT-04)."""
    from switching.runner import execute_train
    ctx, task, _, _ = toy(out, 50)
    if phase == "crash":
        base, step = ctx.engine_class, {"now": None}

        class Probe(base):
            def update(self):
                step["now"] = self.step
                return super().update()
        original = ctx.backend.train

        def train(*args, **kwargs):
            if step["now"] == 30:
                os._exit(137)
            return original(*args, **kwargs)
        ctx.engine_class, ctx.backend.train = Probe, train
    print("RESULT", execute_train(task, ctx)["status"], task["run_id"], flush=True)


def _gloo_context(out):
    import datetime
    import torch.distributed as dist
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=float(os.environ.get("GLOO_TIMEOUT", "60"))))
    rank = dist.get_rank()
    from helpers import toy_setup
    from srgc_rebuttal.distributed import primary  # the cluster child's primary
    from switching.runner import Context
    srgc, PE, data, cfg, backend, _ = toy_setup()
    task = {"run_id": "r1", "study_id": "s", "cohort_id": "toy", "method_id": "on_continued", "training_seed": 5,
            "role": "rebuttal_bank", "config_hash": "h", "horizon": 8, "retain_updates": [5],
            "method": {"policy": {"policy": "never"}, "family": "on_continued", "initial_selector": "on_policy"},
            "parent": {"kind": "initial", "completed_update": 0}}
    ctx = Context(srgc=srgc, engine_class=PE, backend=backend(), data=data, config=cfg, out_root=out,
                  is_primary=rank == 0, primary=primary, broadcast=lambda v: primary(lambda: v))
    return dist, rank, PE, ctx, task


def gloo_fault(out):
    """RT-01: a rank-local exception inside an update (FAULT_RANK) must end every rank promptly."""
    dist, rank, PE, ctx, task = _gloo_context(out)
    fault_rank = int(os.environ.get("FAULT_RANK", "1"))

    class Gathering(PE):
        def update(self):
            if rank == fault_rank and self.step == 3:
                raise ValueError(f"rank {rank}: model produced no response tokens (simulated rank-local fault)")
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, {"rank": rank, "step": self.step})  # like TorchBackend._gather
            return super().update()
    ctx.engine_class = Gathering
    from switching.runner import execute_train
    execute_train(task, ctx)


def gloo_stop(out):
    """RT-03: SIGTERM inside a long update (both ranks in Python, or rank 0 blocked in a collective)."""
    dist, rank, PE, ctx, task = _gloo_context(out)
    import signal
    from switching.runner import execute_train, install_stop_handlers, run_child_task, start_stop_watchdog
    blocked = os.environ.get("STOP_MODE") == "blocked"
    ready = Path(out) / "slow-update-started"

    class Slow(PE):
        def update(self):
            if self.step == 3:
                if rank == 0:
                    ready.write_text(str(time.time()))
                if blocked and rank == 1:
                    # a rank that cannot react (stuck in C): rank 0 waits in the collective below
                    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
                    time.sleep(12)
                    os._exit(0)
                if not blocked:
                    deadline = time.time() + 60
                    while time.time() < deadline:  # a long generation loop (Python-level)
                        time.sleep(0.1)
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, self.step)
            return super().update()
    ctx.engine_class = Slow
    install_stop_handlers()
    start_stop_watchdog(is_primary=rank == 0)
    run_child_task(lambda: execute_train(task, ctx), is_primary=rank == 0, task_id=task["run_id"])


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    globals()[mode.replace("-", "_")](out, *sys.argv[3:])
