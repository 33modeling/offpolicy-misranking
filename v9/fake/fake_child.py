"""CPU stand-in for the GPU child (local software checks only; V9_ALLOW_FAKE=1 and a fake locator).

Builds the same runner Context as cluster_child.run() but with the code repository's ToyBackend
(categorical policy, fresh sampling, AdamW) or a tiny random OLMo-3 + LoRA TorchBackend, then runs
the identical V8/V9 task functions (execute_train / execute_eval / execute_measure) with the V8
exit contract (75 = busy/interrupted). Never produces paper numbers.
"""

from __future__ import annotations

import os
from pathlib import Path


def build_ctx(cohort_root, seed, seed_loc, data):
    from switching import backends, coderepo
    from switching.cache import CacheStore
    from switching.engine import make_engine_class
    from switching.runner import Context
    srgc = coderepo.srgc()
    PE = make_engine_class(srgc)
    fake = seed_loc["fake"]
    if fake["backend"] == "toy":
        feat, ans, _ = backends.toy_problem(fake.get("problem_seed", 0))
        ctx = Context(srgc=srgc, engine_class=PE, backend=None, data=data,
                      config=srgc.Config(seed=seed, projection_dim=64), out_root=cohort_root,
                      measurement_status="simulated_cpu", gpu_type="virtual-4gpu-cpu",
                      cache_store=CacheStore(Path(cohort_root) / "caches"),
                      cache_context={"prompt_hash": "toy", "generation_spec": "toy-categorical", "verifier_hash": "toy"})
        ctx.backend = backends.toy_backend(feat, ans, seed=seed, record=ctx.on_meter_event)
        ctx.meter = ctx.backend.cost_meter
        from srgc_rebuttal.toy_backend import ToyBackend
        beta = ToyBackend(feat, ans, seed=fake.get("problem_seed", 0)).weights  # make_problem's cache policy
        return ctx, beta
    if fake["backend"] == "tiny":
        records = {i: data["records"][i] for i in data["records"]}
        ctx = Context(srgc=srgc, engine_class=PE, backend=None, data=data, config=tiny_config(srgc, seed),
                      out_root=cohort_root, measurement_status="measured_cpu",
                      cache_store=CacheStore(Path(cohort_root) / "caches"),
                      cache_context={"prompt_hash": "tiny", "generation_spec": "tiny", "verifier_hash": "tiny"})
        ctx.backend = backends.tiny_backend(records, record=ctx.on_meter_event)
        ctx.meter = ctx.backend.cost_meter
        return ctx, None
    raise ValueError(f"unknown fake backend {fake['backend']}")


def tiny_config(srgc, seed):
    return srgc.Config(seed=seed, scoring_prompts=40, training_prompts=4, responses=8, candidate_group_size=4,
                       projection_dim=16)


def main(args, task, seed_loc):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from switching import runner
    from switching.cluster import load_bundle
    from v9lib.measure import execute_measure
    runner.install_stop_handlers()
    task_id = task.get("task_id") or task["run_id"]

    def run():
        root = Path(args.root) / args.cohort
        data = load_bundle(seed_loc["input_bundle"])
        ctx, beta = build_ctx(root, task["training_seed"], seed_loc, data)
        if os.environ.get("V9_FAKE_STOP_AFTER_UPDATE") and task["kind"] == "train":
            task["debug_stop_after_update"] = int(os.environ["V9_FAKE_STOP_AFTER_UPDATE"])
        if task["kind"] == "evaluate":
            return runner.execute_eval(task, ctx)
        if task["kind"] == "measure":
            return execute_measure(task, ctx, beta=beta)
        if task["kind"] == "bench":
            from v9lib.bench import execute_bench
            return execute_bench(task, ctx)
        return runner.execute_train(task, ctx)
    runner.run_child_task(run, is_primary=True, task_id=task_id)
