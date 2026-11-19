"""Llama child: v8/experiments/cluster_child.run() with the original Llama adapter contexts.

Differences from cluster_child.run (everything else is the same code path and order):
  - identity: sha256(srgc_rebuttal digest + Llama adapter digest) must equal run.json (llama.verify_child),
    instead of switching.coderepo.source_runtime, which has no 'llama' profile;
  - storage: srgc_research.dispatch.llama31.storage.setup_storage (group-volume caches) before Torch/HF;
  - runtime: Llama adapter runtime_packages + validate_extension(read_only), runtime_adapter() and
    training_adapter() entered before the model is built (Llama load_model, LoRA target check,
    LlamaBackend replaces TorchBackend);
  - attention: eager (the Llama plan's fixed setting; validate_extension checks it).
GPU only; not executable on CPU (no Llama weights).
"""

from __future__ import annotations

from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sys


def main(args, task_id):
    import cluster_child as cc
    from switching.runner import install_stop_handlers, run_child_task, start_stop_watchdog
    install_stop_handlers()
    primary_rank = os.environ.get("RANK", "0") == "0"
    start_stop_watchdog(is_primary=primary_rank)
    run_child_task(lambda: run(args, task_id), is_primary=primary_rank, task_id=task_id)


def run(args, task_id):
    import cluster_child as cc
    from switching import coderepo, spec
    from switching.cluster import load_bundle
    from switching.common import read_json
    from . import llama
    code_root = coderepo.activate()
    root = Path(args.root) / args.cohort
    task = json.loads(Path(args.task_file).read_text())["tasks"][0]
    loc = read_json(root / "locator.json")
    seed_loc = loc["seeds"][str(task["training_seed"])]
    llama.verify_child(seed_loc)
    from srgc_research.dispatch.llama31 import adapter
    from srgc_research.dispatch.llama31.storage import setup_storage
    plan_path = Path(seed_loc["plan_path"])
    setup_storage(plan_path.absolute().parent.parent, os.environ, scan_tree=False)
    cfg = spec.resolve(args.cohort, role=task.get("role", "rebuttal_bank"))

    import torch
    import torch.distributed as dist
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.plan import validate_inputs
    from srgc_rebuttal.srgc import Config
    from srgc_rebuttal import srgc
    from srgc_rebuttal.timing import torch_meter
    from switching.engine import make_engine_class
    from switching.runner import Context, execute_eval, execute_train
    from switching.cache import CacheStore

    plan = adapter.validate_extension(plan_path, read_only=True)
    world = plan["world_size"]
    rank, local_rank = initialize(world)
    cc.install_diagnostics(root, task_id, rank)
    with ExitStack() as stack:
        stack.push(lambda exc_type, *_: dist.destroy_process_group() if exc_type is None else None)
        stack.enter_context(adapter.runtime_adapter())
        stack.enter_context(adapter.training_adapter())

        def runtime():
            from scripts.srgc_sr_refresh import prepare_verifier_runtime
            prepare_verifier_runtime()
            return adapter.runtime_packages()
        cc.all_ranks("Llama runtime (verifier bundle + package pins)", runtime)
        data = primary(lambda: load_bundle(seed_loc["input_bundle"]))
        validate_inputs(data)
        attention = plan.get("attention", "eager")
        ctx_holder = {}

        def on_meter_event(event):
            ctx_holder["ctx"].on_meter_event(event)
        meter = torch_meter(on_meter_event)
        import peft
        from srgc_rebuttal import torch_backend  # TorchBackend is LlamaBackend inside training_adapter()
        torch.manual_seed(task["training_seed"])
        fallback = cc.install_verifier_fallback(code_root, rank)
        cc.install_progress(rank)
        model, tokenizer = adapter.load_model(plan["model"], plan["model_revision"], torch.device("cuda", local_rank),
                                              attention=attention)
        model = peft.get_peft_model(model, peft.LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                                           lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        module, function = plan["verifier"].split(":", 1)
        Backend = cc.protected_backend_class(torch_backend.TorchBackend)
        backend = Backend(model, tokenizer, data["records"], getattr(importlib.import_module(module), function),
                          projection_dim=plan["projection_dim"], projection_seed=plan["projection_seed"],
                          max_new_tokens=plan["max_new_tokens"], logprob_micro_batch=plan.get("logprob_micro_batch", 2),
                          logit_chunk_tokens=plan.get("logit_chunk_tokens", 512), cost_meter=meter)
        config = Config(seed=task["training_seed"], objective=plan["objective"],
                        selection_interval=plan["selection_interval"], check_interval=plan["check_interval"],
                        first_check=plan["first_check"], scoring_prompts=plan["scoring_prompts_per_set"],
                        training_prompts=plan["training_prompts"], responses=plan["responses"],
                        projection_dim=plan["projection_dim"])
        identity = cc.verifier_identity(plan, fallback)
        ctx = Context(srgc=srgc, engine_class=make_engine_class(srgc), backend=backend, data=data, config=config,
                      out_root=root, meter=meter, measurement_status="measured",
                      gpu_type=os.environ.get("V8_GPU_TYPE", cfg["runtime"]["gpu_type"]), is_primary=rank == 0,
                      primary=primary, broadcast=lambda v: primary(lambda: v), cache_store=CacheStore(root / "caches"),
                      cache_context={"prompt_hash": plan.get("prompt_format") or seed_loc["plan_sha256"],
                                     "generation_spec": f"max_new_tokens={plan['max_new_tokens']},T=1,top_p=1",
                                     "verifier_hash": plan["verifier"]},
                      verifier_identity=identity)
        ctx_holder["ctx"] = ctx
        return execute_eval(task, ctx) if task["kind"] == "evaluate" else execute_train(task, ctx)


_ = (sys, json)
