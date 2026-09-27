"""Run a shared prefix or one continuation on four GPUs; tasks may use different nodes."""

import argparse
from contextlib import ExitStack
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from .plan import DEFAULT_PLAN, digest, input_path, load_plan, validate_inputs
from .runtime import (arm_complete, atomic_json, finalize_seed, identity, lease, matches, prefix_ready, run_root)
from .srgc import Config, Engine, stream_seed
from .cost_ledger import PhaseLedger
from .timing import invocation, torch_meter
from .distributed import initialize, primary
from .progress import record as progress


def math_reward(record: dict, response: str) -> float:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    extraction = [LatexExtractionConfig(), ExprExtractionConfig()]
    gold = parse(str(record["answer"]), extraction_config=extraction)
    if not gold:
        raise ValueError("gold answer could not be parsed; fix input before training")
    return float(bool(verify(gold, parse(response, extraction_config=extraction))))


def main() -> None:
    invocation_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", choices=["all", "prefix", "random", "sr", "on_policy", "switch"], default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.seed not in plan["seeds"]:
        parser.error("seed is not in the frozen plan")
    folder = run_root(args.plan, plan) / f"seed-{args.seed}"
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .torch_backend import TorchBackend
    world = plan["world_size"]
    rank, local_rank = initialize(world)
    with ExitStack() as locks:
        # On a rank-local fault, exit promptly so torchrun terminates its peers.
        locks.push(lambda exc_type, *_: dist.destroy_process_group() if exc_type is None else None)
        def startup():
            data = json.loads(input_path(args.plan, plan, args.seed).read_text())
            validate_inputs(data)
            expected = identity(args.plan, plan, args.seed)
            ready = prefix_ready(folder, expected, plan["shared_prefix_updates"])
            complete = args.task == "prefix" and ready
            if args.task in plan["arms"] and arm_complete(folder, expected, args.task, plan["total_updates"]):
                finalize_seed(folder, expected, plan["arms"], plan["total_updates"])
                complete = True
            if not complete and args.task not in {"all", "prefix"} and not ready:
                raise ValueError("the verified shared prefix must finish before this arm can start")
            return data, expected, complete
        data, expected, complete = primary(startup)
        if complete:
            return
        result = [None]
        if rank == 0:
            try:
                tasks = ["prefix", *plan["arms"]] if args.task == "all" else [args.task]
                for task in sorted(tasks):
                    locks.enter_context(lease(folder / f".{task}.execution.lock"))
                with lease(folder / ".manifest.lock", wait=True):
                    marker = folder / "run.json"
                    if marker.exists():
                        if not args.resume or not matches(json.loads(marker.read_text()), expected):
                            raise ValueError("resume requires identical inputs, code and plan")
                    else:
                        atomic_json(marker, {**expected, "plan": plan, "status": "running",
                            "input_provenance": data["provenance"],
                            "ranking_validation_ids": data["ranking_validation_ids"],
                            "packages": {p: importlib.metadata.version(p) for p in
                                ("torch", "transformers", "peft", "math-verify")}})
            except Exception as exc:
                result[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(result, src=0)
        if result[0]:
            raise RuntimeError(result[0])
        scope = "shared-prefix" if args.task == "prefix" else args.task
        meter = torch_meter(lambda event: PhaseLedger(folder / "cost-receipts" / scope).record(event))
        # This inclusive receipt also captures imports, timer I/O and orchestration gaps.
        locks.enter_context(invocation(PhaseLedger(folder / "invocations" / args.task),
                                       meter, world, invocation_started))
        with meter.phase("startup", gpu_count=world):
            torch.manual_seed(args.seed)
            with meter.stage("tokenizer_load"):
                tokenizer = AutoTokenizer.from_pretrained(plan["model"], revision=plan["model_revision"])
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
            with meter.stage("model_load"):
                model = AutoModelForCausalLM.from_pretrained(plan["model"], revision=plan["model_revision"], torch_dtype=torch.bfloat16,
                                                            attn_implementation="eager")
                model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
                model.to(torch.device("cuda", local_rank))
            module, function = plan["verifier"].split(":", 1)
            backend = TorchBackend(model, tokenizer, data["records"], getattr(importlib.import_module(module), function),
                projection_dim=plan["projection_dim"], projection_seed=plan["projection_seed"],
                max_new_tokens=plan["max_new_tokens"], logprob_micro_batch=plan.get("logprob_micro_batch", 2),
                logit_chunk_tokens=plan.get("logit_chunk_tokens", 512), cost_meter=meter)
        progress("model_ready")
        config = Config(seed=args.seed, objective=plan["objective"],
                        selection_interval=plan["selection_interval"], check_interval=plan["check_interval"],
                        first_check=plan["first_check"], scoring_prompts=plan["scoring_prompts_per_set"],
                        training_prompts=plan["training_prompts"], responses=plan["responses"],
                        projection_dim=plan["projection_dim"])

        def engine(arm, ledger_name=None):
            with meter.phase("preparation", gpu_count=world), meter.stage("cached_sr_and_random_sets"):
                return Engine(backend, data["candidate_ids"], data["ranking_validation_ids"],
                              data["cached_rewards"], arm=arm, config=config)

        def save(path, current):
            with meter.phase("checkpoint_save", current.step, world):
                with meter.stage("state_snapshot"):
                    state = current.state_dict()
                def write():
                    with meter.stage("write"):
                        temporary = path.with_suffix(".tmp")
                        torch.save(state, temporary)
                        temporary.replace(path)
                primary(write)

        def load(path):
            with meter.phase("checkpoint_load", gpu_count=world), meter.stage("read"):
                return torch.load(path, map_location="cpu", weights_only=False)

        def restore(current, state, **kwargs):
            with meter.phase("checkpoint_load", state["step"], world), meter.stage("restore"):
                current.load_state_dict(state, **kwargs)

        prefix_path = folder / "prefix.pt"
        ready = primary(lambda: prefix_ready(folder, expected, plan["shared_prefix_updates"]))
        if args.task in {"all", "prefix"} and not ready:
            scope = "shared-prefix"
            prefix = engine("on_policy", "shared-prefix")
            partial = folder / "prefix-latest.pt"
            if primary(partial.exists):
                state = load(partial)
                if state["arm"] != "on_policy" or state["step"] > plan["shared_prefix_updates"]:
                    raise ValueError("invalid prefix resume checkpoint")
                restore(prefix, state)
            while prefix.step < plan["shared_prefix_updates"]:
                prefix.update()
                progress("update", arm="prefix", step=prefix.step)
                if prefix.step % 5 == 0:
                    save(partial, prefix)
            save(prefix_path, prefix)
            primary(lambda: atomic_json(folder / "prefix-ready.json", {**expected,
                    "completed_updates": prefix.step, "checkpoint_sha256": digest(prefix_path)}))
        if args.task == "prefix":
            return
        scope = args.task
        shared = load(prefix_path)
        if shared["arm"] != "on_policy" or shared["step"] != plan["shared_prefix_updates"]:
            raise ValueError("shared prefix has the wrong arm or update count")
        prefix_hash = json.loads((folder / "prefix-ready.json").read_text())["checkpoint_sha256"]
        arms = plan["arms"] if args.task == "all" else [args.task]
        for arm in arms:
            if primary(lambda: arm_complete(folder, expected, arm, plan["total_updates"])):
                continue
            scope = arm
            current = engine(arm)
            checkpoint_path = folder / f"{arm}-latest.pt"
            if primary(checkpoint_path.exists):
                state = load(checkpoint_path)
                if state["arm"] != arm or not plan["shared_prefix_updates"] <= state["step"] <= plan["total_updates"]:
                    raise ValueError("resume checkpoint has the wrong arm or update count")
                restore(current, state)
            else:
                restore(current, shared, fork_arm=arm)
            while current.step < plan["total_updates"]:
                current.update()
                progress("update", arm=arm, step=current.step)
                if current.step % 25 == 0 or current.step == plan["total_updates"]:
                    save(checkpoint_path, current)
                    primary(lambda: atomic_json(folder / f"{arm}-progress.json", {"seed": args.seed, "arm": arm,
                            "step": current.step, "switched_at": current.switched_at,
                            "costs": current.costs, "history": current.history}))
            with meter.phase("evaluation", current.step, world):
                per_question = backend.evaluate(data["evaluation_ids"],
                    seed=stream_seed(args.seed, current.step, "reporting-evaluation"), responses=8)
            def endpoint():
                ledger = PhaseLedger(folder / "cost-receipts" / arm).totals()
                measured_costs = {**current.costs, **ledger["known_gpu_seconds"]}
                atomic_json(folder / f"{arm}-endpoint.json", {**expected, "arm": arm,
                    "prefix_checkpoint_sha256": prefix_hash,
                    "total_updates": current.step, "shared_prefix_updates": plan["shared_prefix_updates"],
                    "switched_at": current.switched_at,
                    "reward": sum(per_question.values()) / len(per_question), "per_question_reward": per_question,
                    "costs": measured_costs, "cost_measurement_complete": ledger["complete"],
                    "cost_receipts": ledger,
                    "evaluation_gpu_seconds": ledger["known_gpu_seconds"]["evaluation_gpu_seconds"],
                    "selection_interval": plan["selection_interval"],
                    "selection_steps": [r["checkpoint"] for r in current.history if r["selection_refreshed"]],
                    "checks": [{"step": r["checkpoint"], "d": r["d"]} for r in current.history if r["d"] is not None]})
            primary(endpoint)
        primary(lambda: finalize_seed(folder, expected, plan["arms"], plan["total_updates"]))


if __name__ == "__main__":
    main()
