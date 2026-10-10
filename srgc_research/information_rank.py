"""Four-GPU information collector, using the existing model and GRPO runtime."""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

from srgc_rebuttal.plan import load_plan, validate_inputs
from srgc_research.information_report import digest
from srgc_research.storage import verify_runtime


def run(output):
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model

    from scripts.srgc_verifier_fallback import install
    from srgc_rebuttal.cost_ledger import PhaseLedger
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.existing_runtime import (
        load_model,
        runtime_packages,
        verifier_environment,
    )
    from srgc_rebuttal.runtime import atomic_json, lease
    from srgc_rebuttal.srgc import Config
    from srgc_rebuttal.timing import invocation, torch_meter
    from srgc_research.backend import ResearchBackend
    from srgc_research.information import (
        InformationStudy,
        checkpoint_state,
        source_identity,
    )

    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    identity = manifest["identity"]
    verify_runtime(Path(manifest["runtime"]), identity["measurement_sha256"])
    for filename, expected in (("inputs.json", identity["input_sha256"]), ("plan.json", identity["plan_sha256"]),
                               ("source-checkpoint.pt", identity["source_checkpoint_sha256"])):
        if expected is not None and digest(output / filename) != expected:
            raise ValueError("frozen source identity differs")
    plan = load_plan(output / "plan.json")
    data = json.loads((output / "inputs.json").read_text())
    validate_inputs(data, recorded_rewards=True)
    config = Config(seed=identity["seed"], objective="grpo", responses=plan["responses"],
        training_prompts=plan["training_prompts"], scoring_prompts=plan["scoring_prompts_per_set"],
        projection_dim=plan["projection_dim"], selection_interval=plan["selection_interval"],
        check_interval=plan["check_interval"], first_check=plan["first_check"])
    saved = None
    backend_state = None
    if identity["source_checkpoint_sha256"] is not None:
        saved = torch.load(output / "source-checkpoint.pt", weights_only=False, map_location="cpu")
        source_identity(saved, input_hash=identity["input_sha256"], plan_hash=identity["plan_sha256"], seed=config.seed)
        backend_state = checkpoint_state(saved, data, config, identity["stage"])
    source_attention = (saved or {}).get("checkpoint_policy", {}).get("attention")
    attention = identity["requested_attention"] or source_attention or "eager"
    if source_attention and attention != source_attention:
        raise ValueError("measurement attention differs from source checkpoint")
    _, local = initialize(plan["world_size"])
    try:
        verifier_environment(os.environ)
        install()
        ledger = PhaseLedger(output / "cost-receipts")
        meter = torch_meter(ledger.record)
        started = time.perf_counter()
        from contextlib import ExitStack
        with ExitStack() as stack:
            primary(lambda: stack.enter_context(lease(output / ".execution.lock")) and None)
            stack.enter_context(invocation(PhaseLedger(output / "invocations"), meter, plan["world_size"], started))
            with meter.phase("startup", gpu_count=plan["world_size"]):
                torch.manual_seed(config.seed)
                model, tokenizer = load_model(plan["model"], plan["model_revision"], torch.device("cuda", local), attention=attention)
                model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                    lora_dropout=0., bias="none", task_type="CAUSAL_LM"))
                module, function = plan["verifier"].split(":")
                backend = ResearchBackend(model, tokenizer, data["records"], getattr(importlib.import_module(module), function),
                    projection_dim=plan["projection_dim"], projection_seed=plan["projection_seed"],
                    max_new_tokens=plan["max_new_tokens"], logprob_micro_batch=plan.get("logprob_micro_batch", 2),
                    logit_chunk_tokens=plan.get("logit_chunk_tokens", 512), cost_meter=meter,
                    rollout_root=output / "resume-rollouts")
                if backend_state is not None:
                    backend.load_state_dict(backend_state)
            primary(lambda: atomic_json(output / "packages.json", runtime_packages()))
            source_engine = (saved or {}).get("state", saved or {})
            source_anchor = source_engine.get("anchor") or source_engine
            primary(lambda: atomic_json(output / "measurement-settings.json", {"attention": attention,
                "source_attention": source_attention, "source_arm": source_engine.get("arm", "fresh-base"),
                "model_lineage_arm": source_anchor.get("arm", "fresh-base"),
                "source_implementation_sha256": (saved or {}).get("implementation_sha256"),
                "parameter_scope": "actual-trainable-LoRA", "seed": config.seed, "stage": identity["stage"]}))
            study = InformationStudy(backend, data, config, identity["stage"], output, identity,
                                     probe_prompts=identity["probe_prompts"])
            result = study.run(publish_endpoint=False)
        # The invocation receipt must be finished before a completion marker.
        costs = primary(ledger.totals)
        invocations = primary(lambda: PhaseLedger(output / "invocations").totals())
        result["cost_receipts"] = costs
        result["invocation_receipts"] = invocations
        result["cost_measurement_complete"] = costs["complete"] and invocations["complete"]
        primary(lambda: atomic_json(output / "costs.json", costs))
        primary(lambda: atomic_json(output / "endpoint.json", result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
