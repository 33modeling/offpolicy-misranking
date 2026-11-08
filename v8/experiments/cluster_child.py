#!/usr/bin/env python3
"""torchrun child for one v8 task on a 4xH100 node (one process per GPU).

    torchrun --standalone --nproc_per_node=4 cluster_child.py --cohort olmo_math \
        --matrix $V8_ROOT/olmo_math/matrix/bank-branches.rebuttal_bank.json --task <run_id|task_id>

Model loading, LoRA, TorchBackend/QwenBackend, verifier runtime and plan resolution are
the code repository's own functions (V8_CODE_ROOT = the frozen checkout that produced
the seed runs; implementation_sha256 must match run.json). Exit 75 = task busy elsewhere.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from switching import coderepo  # noqa: E402
from switching.cluster import load_bundle, v8_root  # noqa: E402
from switching.common import read_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--matrix", required=True, type=Path)
    ap.add_argument("--task", required=True)
    ap.add_argument("--v8-root", default=None)
    args = ap.parse_args()
    code_root = coderepo.activate()
    root = v8_root(args) / args.cohort
    m = read_json(args.matrix)
    task = next(t for t in m["tasks"] if (t.get("task_id") or t["run_id"]) == args.task)
    loc = read_json(root / "locator.json")
    seed_loc = loc["seeds"][str(task["training_seed"])]
    from switching import spec
    cfg = spec.resolve(args.cohort, role=task.get("role", "rebuttal_bank"))
    profile = cfg["runtime"]["profile"]
    if not seed_loc.get("implementation_matches_code_root"):
        raise SystemExit("V8_CODE_ROOT srgc_rebuttal digest differs from the seed run's implementation_sha256")
    verified=coderepo.source_runtime(code_root,profile,seed_loc['run_implementation_sha256'])
    if seed_loc.get('runtime_selection')!=verified:
        raise SystemExit('Verified source runtime selection changed before child startup')
    if verified['mode']=='saved':
        coderepo.activate_source_runtime(code_root,seed_loc['plan_path'],task['training_seed'],verified)

    import torch
    import torch.distributed as dist
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.plan import load_plan, validate_inputs
    from srgc_rebuttal.srgc import Config
    from srgc_rebuttal import srgc
    from srgc_rebuttal.timing import torch_meter
    from switching.engine import make_engine_class
    from switching.runner import Context, Interrupted, execute_eval, execute_train
    from switching.cache import CacheStore

    plan = load_plan(Path(seed_loc["plan_path"]))
    world = plan["world_size"]
    rank, local_rank = initialize(world)
    with ExitStack() as stack:
        stack.push(lambda exc_type, *_: dist.destroy_process_group() if exc_type is None else None)
        if profile == "qwen":
            sys.path.insert(0, str(code_root / "scripts"))
            import srgc_qwen35
            stack.enter_context(srgc_qwen35.training_adapter())
            load_model = srgc_qwen35.load_model
        else:
            from srgc_rebuttal.existing_runtime import load_model
        try:  # identical verifier runtime to the code repo's extra arms
            from scripts.srgc_sr_refresh import prepare_verifier_runtime
            prepare_verifier_runtime()
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print(f"WARN verifier runtime preparation: {exc}", flush=True)
        data = primary(lambda: load_bundle(seed_loc["input_bundle"]))
        validate_inputs(data)
        attention = os.environ.get("V8_ATTENTION", "eager")
        if seed_loc.get("prefix_path"):
            prefix_state = torch.load(seed_loc["prefix_path"], map_location="cpu", weights_only=False)
            try:
                from scripts.srgc_step_checkpoints import saved_attention
                attention = saved_attention(prefix_state)  # every branch inherits its prefix's kernel
            except Exception:  # noqa: BLE001
                pass
            del prefix_state
        ctx_holder = {}
        meter = torch_meter(lambda e: ctx_holder["ctx"].on_meter_event(e))
        from peft import LoraConfig, get_peft_model
        from srgc_rebuttal import torch_backend
        torch.manual_seed(task["training_seed"])  # LoRA initialization protocol of run_experiment.py
        model, tokenizer = load_model(plan["model"], plan["model_revision"], torch.device("cuda", local_rank),
                                      attention=attention)
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        module, function = plan["verifier"].split(":", 1)
        backend = torch_backend.TorchBackend(
            model, tokenizer, data["records"], getattr(importlib.import_module(module), function),
            projection_dim=plan["projection_dim"], projection_seed=plan["projection_seed"],
            max_new_tokens=plan["max_new_tokens"], logprob_micro_batch=plan.get("logprob_micro_batch", 2),
            logit_chunk_tokens=plan.get("logit_chunk_tokens", 512), cost_meter=meter)
        config = Config(seed=task["training_seed"], objective=plan["objective"],
                        selection_interval=plan["selection_interval"], check_interval=plan["check_interval"],
                        first_check=plan["first_check"], scoring_prompts=plan["scoring_prompts_per_set"],
                        training_prompts=plan["training_prompts"], responses=plan["responses"],
                        projection_dim=plan["projection_dim"])
        ctx = Context(srgc=srgc, engine_class=make_engine_class(srgc), backend=backend, data=data, config=config,
                      out_root=root, meter=meter, measurement_status="measured",
                      gpu_type=os.environ.get("V8_GPU_TYPE", cfg["runtime"]["gpu_type"]), is_primary=rank == 0,
                      primary=primary, broadcast=lambda v: primary(lambda: v),
                      cache_store=CacheStore(root / "caches"),
                      cache_context={"prompt_hash": plan.get("prompt_format") or seed_loc["plan_sha256"],
                                     "generation_spec": f"max_new_tokens={plan['max_new_tokens']},T=1,top_p=1",
                                     "verifier_hash": plan["verifier"]})
        ctx_holder["ctx"] = ctx
        try:
            res = execute_eval(task, ctx) if task["kind"] == "evaluate" else execute_train(task, ctx)
        except Interrupted as exc:
            if rank == 0:
                print(f"INTERRUPTED {exc}", flush=True)
            raise SystemExit(75)
        if rank == 0:
            print("RESULT " + json.dumps(res, default=str), flush=True)
        if res.get("status") == "busy":
            raise SystemExit(75)


if __name__ == "__main__":
    main()
