"""Small, fixed candidate panel before any full Gemma cache or optimizer updates."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

from srgc_rebuttal.plan import digest, input_path
from srgc_rebuttal.runtime import atomic_json, lease

from .protocol import SUPPRESS_TOKENS, sampling

PANEL_PROMPTS = 8


def identity(plan_path, plan, bundle):
    candidates = bundle["candidate_ids"][:PANEL_PROMPTS]
    content = {candidate: bundle["records"][candidate] for candidate in candidates}
    return {"schema": "gemma4-text-preflight-v1", "plan_sha256": digest(plan_path),
            "seed": plan["seeds"][0], "candidate_ids": candidates,
            "records_sha256": hashlib.sha256(json.dumps(content, sort_keys=True,
                ensure_ascii=False).encode()).hexdigest(),
            "sampling": sampling(), "responses": plan["responses"],
            "max_new_tokens": plan["max_new_tokens"], "verifier": plan["verifier"]}


def folder(plan_path, plan):
    return Path(plan_path).parent.parent / "preflight" / plan["dataset"]


def read_receipt(path, protocol, candidate):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("protocol") != protocol or value.get("id") != candidate:
        raise ValueError("preflight receipt belongs to another model/input/protocol")
    count = protocol["responses"]
    rewards, texts, tokens = (value.get(key) for key in ("rewards", "responses", "token_ids"))
    if (not isinstance(rewards, list) or len(rewards) != count
            or any(type(v) not in (int, float) or v not in (0, 1) for v in rewards)
            or not isinstance(texts, list) or len(texts) != count
            or any(not isinstance(v, str) for v in texts)
            or not isinstance(tokens, list) or len(tokens) != count
            or any(not isinstance(row, list) or not row or len(row) > protocol["max_new_tokens"]
                   or any(type(v) is not int or v < 0 or v in SUPPRESS_TOKENS for v in row)
                   for row in tokens)):
        raise ValueError("invalid preflight raw responses/rewards/tokens")
    return value


def summarize(protocol, rows):
    if [row["id"] for row in rows] != protocol["candidate_ids"]:
        raise ValueError("preflight panel coverage is incomplete")
    rewards = [v for row in rows for v in row["rewards"]]
    texts = [v for row in rows for v in row["responses"]]
    correct, empty = sum(rewards), sum(not text.strip() for text in texts)
    return {"protocol": protocol, "prompts": len(rows), "responses": len(rewards),
            "correct": correct, "success_percent": 100 * correct / len(rewards),
            "empty_responses": empty, "passed": correct > 0,
            "scope": "fixed small candidate panel; not full initial-cache accuracy"}


def validate_gate(plan_path):
    from .adapter import validate_bundle_model, validate_extension

    plan = validate_extension(plan_path)
    bundle = json.loads(input_path(plan_path, plan, plan["seeds"][0]).read_text())
    validate_bundle_model(bundle, plan, plan["seeds"][0])
    protocol = identity(plan_path, plan, bundle)
    root = folder(plan_path, plan)
    rows = [read_receipt(root / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json",
                         protocol, candidate) for candidate in protocol["candidate_ids"]]
    if any(row is None for row in rows):
        raise ValueError("Gemma preflight is missing; run scripts/run_srgc_gemma4.sh all preflight first")
    expected = summarize(protocol, rows)
    saved = json.loads((root / "summary.json").read_text())
    if saved != expected or not expected["passed"]:
        raise ValueError("Gemma preflight has no verified correct response; full run is paused")
    return expected


def run(plan_path):
    import torch
    import torch.distributed as dist

    from srgc_rebuttal.build_cache import candidate_seed, generate_rewards
    from srgc_rebuttal.distributed import initialize, primary

    from .adapter import (
        MODEL,
        REVISION,
        load_model,
        validate_bundle_model,
        validate_extension,
    )

    plan_path = Path(plan_path)
    plan = validate_extension(plan_path)
    rank, local = initialize(4)
    try:
        bundle = primary(lambda: json.loads(input_path(plan_path, plan, plan["seeds"][0]).read_text()))
        validate_bundle_model(bundle, plan, plan["seeds"][0])
        protocol = identity(plan_path, plan, bundle)
        root = folder(plan_path, plan)
        # All nodes serialize the one shared panel. Each rank takes the same
        # branch after rank zero acquires the lease or reports a filesystem error.
        lock = None
        try:
            def acquire():
                nonlocal lock
                context = lease(root / ".lock", wait=True)
                context.__enter__()
                lock = context
            primary(acquire)

            def pending():
                return [candidate for candidate in protocol["candidate_ids"]
                        if read_receipt(root / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json",
                                        protocol, candidate) is None]

            todo = primary(pending)
            if todo:
                model, tokenizer = load_model(MODEL, REVISION, torch.device("cuda", local),
                                              attention=plan["attention"])
                module, function = plan["verifier"].split(":", 1)
                verifier = getattr(importlib.import_module(module), function)
                original = model.generate
                captured = []

                def capture(*args, **kwargs):
                    result = original(*args, **kwargs)
                    start = kwargs["input_ids"].shape[-1]
                    for sequence in result:
                        ids = sequence[start:].tolist()
                        if 1 in ids:
                            ids = ids[:ids.index(1) + 1]
                        captured.append(ids)
                    return result

                model.generate = capture
                for candidate in todo[rank::4]:
                    captured.clear()
                    rewards, texts = generate_rewards(model, tokenizer, bundle["records"][candidate],
                        verifier, responses=protocol["responses"],
                        seed=candidate_seed(protocol["seed"], candidate),
                        max_new_tokens=protocol["max_new_tokens"], candidate=candidate)
                    path = root / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json"
                    atomic_json(path, {"protocol": protocol, "id": candidate,
                                      "rewards": rewards, "responses": texts,
                                      "token_ids": list(captured)})
                    read_receipt(path, protocol, candidate)
                    print(f"GEMMA PREFLIGHT {candidate}: {sum(rewards):g}/{len(rewards)} correct", flush=True)
            dist.barrier()

            def finish():
                rows = [read_receipt(root / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json",
                                     protocol, candidate) for candidate in protocol["candidate_ids"]]
                report = summarize(protocol, rows)
                atomic_json(root / "summary.json", report)
                print(f"GEMMA PREFLIGHT {plan['dataset']}: {report['correct']:g}/{report['responses']} "
                      f"correct ({report['success_percent']:.2f}%), empty={report['empty_responses']}; "
                      f"{'PASS' if report['passed'] else 'PAUSED'}; report={root / 'summary.json'}", flush=True)
                print("This is a small panel, not full initial-cache accuracy.", flush=True)
                if not report["passed"]:
                    raise ValueError("zero correct responses in the small panel; no full cache/training launched")
                return report

            return primary(finish)
        finally:
            if rank == 0 and lock is not None:
                lock.__exit__(*sys.exc_info())
    finally:
        if sys.exc_info()[0] is None:
            dist.destroy_process_group()
