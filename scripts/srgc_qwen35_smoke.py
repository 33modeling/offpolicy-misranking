#!/usr/bin/env python3
"""Short Qwen readiness admission, separate from the frozen experiment workload."""

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
SMOKE_TOKENS = 32
SMOKE_RESPONSES = 8


def candidate_prompt(data, tokenizer):
    selected = None
    for candidate in data["candidate_ids"]:
        prompt = data["records"][candidate]["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
        if tokens and (selected is None or tokens < selected[2]):
            selected = candidate, prompt, tokens
    if selected is None:
        raise ValueError("Qwen admission needs a nonempty candidate prompt in the first seed bundle")
    return selected


def _stage(name, **details):
    print("[qwen-smoke] " + json.dumps({"stage": name, "rank": os.environ.get("RANK"), **details}), flush=True)


def lightweight_smoke(plan_path):
    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.plan import input_path
    from srgc_qwen35 import MODEL, REVISION, attach_adapter, load_model, validate_extension
    from srgc_qwen35_memory import QwenBackend

    plan_path = Path(plan_path)
    plan = validate_extension(plan_path)
    _stage("init", protocol="candidate-readiness-v1", response_tokens=SMOKE_TOKENS,
           responses=SMOKE_RESPONSES)
    rank, local = initialize(4)
    try:
        _stage("startup_collective")
        data = primary(lambda: json.loads(input_path(plan_path, plan, plan["seeds"][0]).read_text()))
        torch.manual_seed(104729)
        torch.cuda.reset_peak_memory_stats()
        _stage("model_load")
        model, tokenizer = load_model(MODEL, REVISION, torch.device("cuda", local))
        model = attach_adapter(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                               bias="none", task_type="CAUSAL_LM"), get_peft_model)
        candidate, prompt, tokens = candidate_prompt(data, tokenizer)
        records = {f"p{i}": {"prompt": prompt, "answer": "2"} for i in range(4)}
        backend = QwenBackend(model, tokenizer, records, lambda record, text: 0.0,
                              max_new_tokens=SMOKE_TOKENS, projection_dim=plan["projection_dim"],
                              projection_seed=plan["projection_seed"])
        _stage("rollout", candidate=candidate, prompt_tokens=tokens,
               response_tokens=SMOKE_TOKENS, responses=SMOKE_RESPONSES)
        generated, _, start = backend._rollout(f"p{rank}", SMOKE_RESPONSES, 17)
        prefix = generated[0][:start]
        suffixes = [tokenizer.encode(text, add_special_tokens=False) for text in (" 2", " 3")]
        forced = [(suffix * SMOKE_TOKENS)[:SMOKE_TOKENS] for suffix in suffixes]
        if not all(forced) or forced[0] == forced[1]:
            raise ValueError("Qwen admission needs two distinct nonempty synthetic suffixes")
        sequences = [torch.cat((prefix, torch.tensor(forced[i % 2], device=prefix.device,
                                                     dtype=torch.long))) for i in range(SMOKE_RESPONSES)]
        del generated
        backend._rollout = lambda *args: (sequences, np.array([0., 1.] * (SMOKE_RESPONSES // 2)), start)
        _stage("scoring")
        gradients = backend.score_gradients(list(records), responses=SMOKE_RESPONSES, group_size=4, seed=17)
        if not all(np.isfinite(g).all() for g in gradients.values()) or not any(np.any(g) for g in gradients.values()):
            raise ValueError("Qwen scoring backward produced zero/nonfinite synthetic gradients")
        before = [p.detach().cpu().clone() for _, p in backend.train_parameters]
        _stage("update")
        backend.train(list(records), responses=SMOKE_RESPONSES, objective="grpo", seed=17)
        if not all(torch.isfinite(p).all() for _, p in backend.train_parameters):
            raise ValueError("nonfinite Qwen adapter update")
        if not any(not torch.equal(a, p.detach().cpu()) for a, (_, p) in zip(before, backend.train_parameters)):
            raise ValueError("synthetic Qwen admission did not update adapters")
        _stage("final_barrier", peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved())
        dist.barrier()
        if rank == 0:
            print("PASS: Qwen four-rank generation/scoring/GRPO readiness; 32-token synthetic smoke", flush=True)
    except BaseException:
        # Let torchrun terminate peers promptly if a rank fails.
        raise
    else:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke",), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.parse_args()
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import srgc_qwen35
    from srgc_qwen35_rank import main as rank_main
    with patch.object(srgc_qwen35, "smoke", lightweight_smoke):
        return rank_main()


if __name__ == "__main__":
    main()
