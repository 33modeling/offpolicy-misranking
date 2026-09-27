"""Generate the eight initial-policy cached rewards per candidate for a bundle (four GPUs).

The initial policy is the pretrained base model of the plan (no adapter), sampled
with the experiment's settings (temperature 1, top-p 1, no top-k, at most
``max_new_tokens`` new tokens). Each candidate's eight binary rewards are written
into the bundle's ``cached_rewards``; the raw responses are kept beside it for
provenance. Rerunning skips candidates already cached.

    torchrun --standalone --nproc_per_node=4 -m srgc_rebuttal.build_cache \
        --bundle srgc_rebuttal/inputs/mbpp-seed-5.json --verifier srgc_rebuttal.verifiers:code_reward
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

from .plan import DEFAULT_PLAN, load_plan, validate_inputs
from .runtime import atomic_json, lease
from .srgc import stream_seed
from .cost_ledger import PhaseLedger
from .timing import CostMeter, invocation, torch_meter
from .distributed import initialize, primary
from .progress import record as progress
from .existing_runtime import load_model

DEFAULT_MODEL = "allenai/Olmo-3-1025-7B"


def candidate_seed(cache_seed: int, candidate: str) -> int:
    """Independent of how many earlier candidates finished before a restart."""
    return stream_seed(cache_seed, 0, f"cache:{candidate}") & 0x7FFFFFFFFFFFFFFF


class CacheStore:
    """Immutable generation settings and atomic per-prompt response receipts."""

    def __init__(self, bundle_path: Path, bundle: dict, protocol: dict):
        validate_inputs(bundle, require_cache=False)
        self.root = bundle_path.with_suffix(".cache")
        self.candidates = tuple(bundle["candidate_ids"])
        self.protocol = {**protocol, "candidate_ids": list(self.candidates),
                         "records_sha256": hashlib.sha256(json.dumps(
                             {i: bundle["records"][i] for i in self.candidates},
                             sort_keys=True, ensure_ascii=False).encode()).hexdigest()}

    def bind(self):
        marker = self.root / "protocol.json"
        if marker.exists():
            if json.loads(marker.read_text()) != self.protocol:
                raise ValueError("cache generation settings or candidate data changed")
        else:
            atomic_json(marker, self.protocol)

    def path(self, candidate):
        if candidate not in self.candidates:
            raise ValueError("unknown cache candidate")
        return self.root / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json"

    def read(self, candidate):
        path = self.path(candidate)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if (value["id"] != candidate or len(value["rewards"]) != 8 or
                any(v not in (0, 1) for v in value["rewards"]) or
                len(value["responses"]) != 8 or
                value["sampling_seed"] != candidate_seed(self.protocol["cache_seed"], candidate)):
            raise ValueError("invalid cached response receipt")
        return value

    def write(self, candidate, rewards, texts, elapsed):
        if len(rewards) != 8 or len(texts) != 8 or any(r not in (0, 1) for r in rewards):
            raise ValueError("a cache receipt requires eight binary rewards and responses")
        atomic_json(self.path(candidate), {"id": candidate, "rewards": rewards, "responses": texts,
                    "sampling_seed": candidate_seed(self.protocol["cache_seed"], candidate),
                    "generation_and_verification_wall_seconds": elapsed})


class GenerationProgress:
    """Report completed decode work without stopping or changing sampling."""

    def __init__(self, prompt_tokens, candidate, *, clock=time.monotonic):
        self.prompt_tokens, self.candidate, self.clock = prompt_tokens, candidate, clock
        self.last_report = None

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids.shape[-1] - self.prompt_tokens
        now = self.clock()
        if generated > 0 and (self.last_report is None or now - self.last_report >= 30):
            progress("cache_generation", prompt=self.candidate, generated_steps=generated)
            print(f"[cache] rank={os.environ.get('RANK', '0')} prompt={self.candidate} "
                  f"generated_steps={generated}", flush=True)
            self.last_report = now
        return False


def generate_rewards(model, tokenizer, record: dict, verifier, *, responses: int, seed: int,
                     max_new_tokens: int, meter=None, candidate=None) -> tuple[list[float], list[str]]:
    import torch
    from transformers import StoppingCriteriaList
    meter = meter or CostMeter()
    meter.count("prompts")
    meter.count("responses", responses)
    with meter.stage("tokenization"):
        tokens = tokenizer(record["prompt"], return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(model.device) for k, v in tokens.items()}
    start = inputs["input_ids"].shape[1]
    torch.manual_seed(seed)
    if model.device.type == "cuda":
        torch.cuda.manual_seed(seed)
    with meter.stage("generation"), torch.no_grad():
        generated = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                   num_return_sequences=responses, max_new_tokens=max_new_tokens,
                                   use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id,
                                   stopping_criteria=StoppingCriteriaList([GenerationProgress(start, candidate)]))
    with meter.stage("decode"):
        texts = [tokenizer.decode(sequence[start:], skip_special_tokens=True) for sequence in generated]
    with meter.stage("reward_verification"):
        rewards = []
        for index, text in enumerate(texts):
            rewards.append(float(verifier(record, text)))
            progress("cache_verification", prompt=candidate, verified_responses=index + 1)
    if any(r not in (0.0, 1.0) for r in rewards):
        raise ValueError("verifier must return binary rewards")
    return rewards, texts


def main() -> None:
    invocation_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--verifier", help="module:function; defaults to the pinned plan")
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--responses", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--cache-seed", type=int, default=0, help="sampling seed base for the initial-policy cache")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    args.model = args.model or plan["model"]
    args.model_revision = args.model_revision or plan["model_revision"]
    args.verifier = args.verifier or plan["verifier"]
    if args.responses != 8:
        parser.error("the experiment requires eight cached responses per candidate")
    protocol = {k: getattr(args, k) for k in
                ("model", "model_revision", "verifier", "responses", "max_new_tokens", "cache_seed", "attention")}
    import torch
    import torch.distributed as dist
    world = plan["world_size"]
    rank, _ = initialize(world)
    with ExitStack() as locks:
        locks.push(lambda exc_type, *_: dist.destroy_process_group() if exc_type is None else None)
        def prepare():
            locks.enter_context(lease(args.bundle.with_suffix(".cache") / "execution.lock"))
            bundle = json.loads(args.bundle.read_text())
            validate_inputs(bundle, require_cache=False)
            complete = len(bundle.get("cached_rewards", {})) == len(bundle["candidate_ids"])
            if complete:
                validate_inputs(bundle)
                prior = bundle.get("provenance", {}).get("cache", {})
                if not isinstance(prior, dict) or any(prior.get(k) != v for k, v in protocol.items()):
                    raise ValueError("the complete cache has different or unverified generation settings")
                write_cost_summary(args.bundle, bundle)
            elif bundle.get("cached_rewards"):
                raise ValueError("partial rewards without complete raw receipts cannot be mixed with a new cache")
            return bundle, complete
        bundle, complete = primary(prepare)
        if complete:
            if rank == 0:
                print(f"PASS: {args.bundle} already contains a complete cache; no generation")
            return
        store = CacheStore(args.bundle, bundle, protocol)
        primary(store.bind)
        module, function = args.verifier.split(":", 1)
        verifier = getattr(importlib.import_module(module), function)
        todo = primary(lambda: [i for i in bundle["candidate_ids"] if store.read(i) is None])
        ledger = PhaseLedger(store.root / "cost-receipts")
        sessions = PhaseLedger(store.root / "invocations")
        meter = torch_meter(ledger.record)
        with invocation(sessions, meter, world, invocation_started):
            if todo:
                with meter.phase("startup", gpu_count=world):
                    with meter.stage("model_and_tokenizer_load"):
                        print(f"[cache] rank={rank} loading model attention={args.attention}", flush=True)
                        model, tokenizer = load_model(args.model, args.model_revision,
                            torch.device("cuda", torch.cuda.current_device()), attention=args.attention)
                progress("model_ready")
            elif rank == 0:
                print("[cache] all response receipts saved; exporting without model reload", flush=True)
            with meter.phase("cache_generation", gpu_count=world):
                rank_total = len(todo[rank::world])
                rank_done, rank_seconds = 0, 0.0
                print(f"[cache] rank={rank} ready; pending_prompts={rank_total}", flush=True)
                for index, candidate in enumerate(todo):
                    if index % world != rank:
                        continue
                    with lease(store.path(candidate).with_suffix(".lock")):
                        if store.read(candidate) is not None:
                            continue
                        seed = candidate_seed(args.cache_seed, candidate)
                        print(f"[cache] rank={rank} generating prompt {rank_done + 1}/{rank_total}", flush=True)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        rewards, responses = generate_rewards(model, tokenizer, bundle["records"][candidate], verifier,
                            responses=args.responses, seed=seed, max_new_tokens=args.max_new_tokens,
                            meter=meter, candidate=candidate)
                        meter.count(f"candidate:{candidate}")
                        torch.cuda.synchronize()
                        with meter.stage("receipt_write"):
                            store.write(candidate, rewards, responses, time.perf_counter() - started)
                        atomic_json(store.root / "live-costs" / f"{meter.event['id']}-rank-{rank}.json",
                                    meter.live_snapshot())
                        progress("cache_candidate", prompt=candidate)
                        elapsed = time.perf_counter() - started
                        rank_done += 1
                        rank_seconds += elapsed
                        eta = rank_seconds / rank_done * (rank_total - rank_done)
                        print(f"[cache] rank={rank} prompts={rank_done}/{rank_total} "
                              f"last={elapsed:.1f}s remaining_estimate={eta / 60:.1f}min", flush=True)
            with meter.phase("cache_export", gpu_count=world):
                primary(lambda: export_cache(args, bundle, store))
        primary(lambda: write_cost_summary(args.bundle, bundle))
        if rank == 0:
            print(f"PASS: cached {len(todo)} candidates into {args.bundle}; timings in {store.root}")


def write_cost_summary(bundle_path, bundle):
    root = bundle_path.with_suffix(".cache")
    report = PhaseLedger(root / "cost-receipts").totals()
    sessions = PhaseLedger(root / "invocations").totals()
    report["candidate_count"] = len(bundle["candidate_ids"])
    report["measured_prompts"] = report["counts"].get("cache_generation.prompts", 0)
    covered = {name.removeprefix("cache_generation.candidate:") for name in report["counts"]
               if name.startswith("cache_generation.candidate:")}
    report["complete"] = (report["complete"] and sessions["complete"] and bool(sessions["recorded_phases"]) and
                          covered == set(bundle["candidate_ids"]))
    report["invocations"] = sessions
    report["bundle_sha256"] = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if not report["complete"]:
        report["total_gpu_seconds"] = None
    atomic_json(root / "cost-summary.json", report)


def export_cache(args, bundle, store):
    receipts = {i: store.read(i) for i in bundle["candidate_ids"]}
    if any(r is None for r in receipts.values()):
        raise RuntimeError("incomplete cache; resume generation before training")
    bundle["cached_rewards"] = {i: r["rewards"] for i, r in receipts.items()}
    bundle["provenance"]["cache"] = {"model": args.model, "model_revision": args.model_revision,
                                     "responses": args.responses, "temperature": 1.0, "top_p": 1.0,
                                     "max_new_tokens": args.max_new_tokens, "cache_seed": args.cache_seed,
                                     "verifier": args.verifier}
    bundle["provenance"]["cache"]["attention"] = args.attention
    responses_path = args.bundle.with_suffix(".cache-responses.jsonl")
    temporary = responses_path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for record in receipts.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(responses_path)
    validate_inputs(bundle)
    atomic_json(args.bundle, bundle)


if __name__ == "__main__":
    main()
