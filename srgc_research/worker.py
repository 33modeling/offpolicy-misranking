"""Four-rank worker using the existing pinned model, verifier and LoRA runtime."""

import argparse
import importlib
import json
import os
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

from .design import STAGES, Condition
from .storage import (
    atomic_torch,
    complete,
    configure_cache,
    identity,
    publish_anchor,
    verify_anchor,
    verify_inputs,
    verify_runtime,
)


def execute(folder, condition, manifest, data, backend):
    """Shared production/CPU integration path; every transition is recoverable."""
    import torch

    from srgc_rebuttal.cost_ledger import PhaseLedger
    from srgc_rebuttal.distributed import primary
    from srgc_rebuttal.plan import digest
    from srgc_rebuttal.runtime import atomic_json
    from srgc_rebuttal.srgc import Config, stream_seed

    from .study import Diagnostic, NestedEngine, Trajectory, feature_comparison
    out = folder / condition.key
    out.mkdir(parents=True, exist_ok=True)
    plan, seed, meter, world = manifest["plan"], manifest["seed"], backend.cost_meter, backend.gpu_count
    config = Config(seed=seed, projection_dim=plan["projection_dim"])
    expected, ledger = identity(manifest, condition), PhaseLedger(out / "cost-receipts")

    def save(state, step, phase):
        with meter.phase("checkpoint_save", step, world), meter.stage("write"):
            primary(lambda: atomic_torch(out / "state-latest.pt", {**expected, "state": state}))
        primary(lambda: atomic_json(out / "progress.json", {**expected, "step": step, "phase": phase,
                                                           "updated": time.time()}))
        primary(lambda: shutil.rmtree(out / "rollouts", ignore_errors=True))
        primary(lambda: print(f"RESEARCH seed={seed} task={condition.key} step={step} phase={phase} "
                              f"checkpoint={out / 'state-latest.pt'}", flush=True))

    checkpoint, saved = out / "state-latest.pt", None
    if primary(checkpoint.exists):
        with meter.phase("checkpoint_load", gpu_count=world), meter.stage("read"):
            value = torch.load(checkpoint, weights_only=False, map_location="cpu")
            if any(value.get(k) != v for k, v in expected.items()):
                raise ValueError("checkpoint identity differs")
            saved = value["state"]

    if condition.kind == "features":
        if saved is None:
            with backend.operation("feature-comparison"), meter.phase("evaluation", 0, world):
                result = feature_comparison(backend, data, config)
            save({"result": result}, 0, "feature-comparison-complete")
        else:
            result = saved["result"]
    elif condition.kind == "cache":
        if saved is None:
            with backend.operation("initial-sr-cache"), meter.phase("cache_generation", 0, world):
                backend.evaluate(data["candidate_ids"], responses=8, seed=stream_seed(seed, 0, "sr-cache"))
                rewards = backend.evaluation_samples
            save({"cached_rewards": rewards}, 0, "cache-complete")
        else:
            rewards = saved["cached_rewards"]
        with meter.phase("cache_export", 0, world), meter.stage("write"):
            primary(lambda: atomic_json(folder / "sr-cache.json", {**expected, "cached_rewards": rewards}))
            cache_hash = primary(lambda: digest(folder / "sr-cache.json"))
        result = {"prompts": len(rewards), "responses_per_prompt": 8, "cache_policy": "initial-policy-t0",
                  "cache_sha256": cache_hash,
                  "reuse": "one shared cache per dataset/seed; charged once to each cold SR/Switch comparison"}
    else:
        if condition.needs_cache:
            with meter.phase("preparation", gpu_count=world), meter.stage("cache_read"):
                cache = json.loads((folder / "sr-cache.json").read_text())
                if any(cache.get(k) != manifest[k] for k in ("input_sha256", "implementation_sha256", "seed")):
                    raise ValueError("SR cache identity differs")
                data["cached_rewards"] = cache["cached_rewards"]
        else:
            data["cached_rewards"] = {i: [0] * 8 for i in data["candidate_ids"]}
        if condition.kind == "anchors":
            with meter.phase("preparation", gpu_count=world), meter.stage("selector_setup"):
                engine = NestedEngine(backend, data["candidate_ids"], data["ranking_validation_ids"],
                                      data["cached_rewards"], config=config, arm="on_policy")
            if saved:
                with meter.phase("checkpoint_load", gpu_count=world), meter.stage("restore"):
                    engine.load_state_dict(saved)
            while True:
                if engine.step in STAGES:
                    with meter.phase("checkpoint_save", engine.step, world), meter.stage("anchor_write"):
                        state = engine.state_dict()
                        primary(lambda state=state: publish_anchor(folder, manifest, state))
                if engine.step == condition.updates:
                    break
                with backend.operation(f"carrier-{engine.step}"):
                    engine.update()
                save(engine.state_dict(), engine.step, "carrier")
            result = {"stages": list(STAGES), "carrier_updates": engine.step}
        else:
            anchor = None
            if condition.kind == "diagnostic":
                with meter.phase("checkpoint_load", gpu_count=world), meter.stage("anchor_read"):
                    path = primary(lambda: str(verify_anchor(folder, manifest, condition.stage)))
                    anchor = torch.load(path, weights_only=False, map_location="cpu")
            with meter.phase("preparation", gpu_count=world), meter.stage("study_setup"):
                study = (Diagnostic(backend, data, config, condition, anchor) if anchor is not None
                         else Trajectory(backend, data, config, condition))
            if saved:
                with meter.phase("checkpoint_load", gpu_count=world), meter.stage("restore"):
                    study.load_state_dict(saved)
            while not study.done:
                phase = study.advance()
                if phase == "evaluation" and hasattr(study, "curve"):
                    study.curve[-1]["costs_to_checkpoint"] = primary(ledger.totals)
                with meter.phase("checkpoint_save", study.step, world), meter.stage("state_snapshot"):
                    state = study.state_dict()
                save(state, study.step, phase)
            result = study.result()
    costs = primary(ledger.totals)
    endpoint = {**expected, "status": "complete", "result": result, "cost_receipts": costs,
                "cost_measurement_complete": costs["complete"],
                "cost_role": "diagnostic" if condition.kind in {"diagnostic", "anchors", "features"} else "deployment",
                "initial_state": manifest["initial_state"]}
    primary(lambda: atomic_json(out / "endpoint.json", endpoint))
    primary(lambda: complete(folder, manifest, condition))
    return endpoint


def run(folder, condition):
    configure_cache()
    from srgc_rebuttal.existing_runtime import (
        load_model,
        runtime_packages,
        verifier_environment,
    )
    verifier_environment(os.environ)
    from scripts.srgc_child_tuning import count_progress
    from scripts.srgc_verifier_fallback import install
    install()
    count_progress()
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model

    from srgc_rebuttal.cost_ledger import PhaseLedger
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.runtime import atomic_json, lease
    from srgc_rebuttal.timing import invocation, torch_meter

    from .backend import ResearchBackend
    manifest = json.loads((folder / "manifest.json").read_text())
    verify_runtime(Path(manifest["runtime"]), manifest["implementation_sha256"])
    data = verify_inputs(folder, manifest)
    plan, seed = manifest["plan"], manifest["seed"]
    _, local = initialize(plan["world_size"])
    out = folder / condition.key
    meter = torch_meter(PhaseLedger(out / "cost-receipts").record)
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            def claim():
                stack.enter_context(lease(out / ".execution.lock"))
            primary(claim)
            stack.enter_context(invocation(PhaseLedger(out / "invocations"), meter, 4, started))
            if primary(lambda: complete(folder, manifest, condition)):
                return
            with meter.phase("startup", gpu_count=4):
                with meter.stage("model_and_adapter_load"):
                    torch.manual_seed(seed)
                    model, tokenizer = load_model(plan["model"], plan["model_revision"], torch.device("cuda", local),
                                                  attention=manifest["attention"])
                    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                        lora_dropout=0., bias="none", task_type="CAUSAL_LM"))
                module, function = plan["verifier"].split(":")
                backend = ResearchBackend(model, tokenizer, data["records"], getattr(importlib.import_module(module), function),
                    projection_dim=plan["projection_dim"], projection_seed=plan["projection_seed"],
                    max_new_tokens=plan["max_new_tokens"], logprob_micro_batch=plan.get("logprob_micro_batch", 2),
                    logit_chunk_tokens=plan.get("logit_chunk_tokens", 512), cost_meter=meter, rollout_root=out / "rollouts")
            primary(lambda: atomic_json(out / "packages.json", runtime_packages()))
            execute(folder, condition, manifest, data, backend)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--condition", required=True, type=Path)
    args = parser.parse_args()
    run(args.folder, Condition(**json.loads(args.condition.read_text())))


if __name__ == "__main__":
    main()
