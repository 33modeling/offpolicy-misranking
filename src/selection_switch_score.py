"""Paid current-policy fresh_r: eight candidates, two LOO groups, ranking validation only."""

from __future__ import annotations

import argparse
from pathlib import Path

import selection_gate as core
import selection_gate_gpu as base


def matched_score(groups, direction):
    from grads import cosine
    if groups.ndim != 2 or groups.shape[0] != 2:
        raise ValueError("fresh_r requires exactly two four-response gradient groups")
    return cosine(groups.mean(dim=0), direction)


def layout(config, prompts):
    if config["fresh_k"] != 32 or config["micro_group"] != 4 or config["behavior_k"] != 8:
        raise ValueError("unsupported primary fresh_r grouping")
    n = len(prompts["val"])
    if n < 8 or n % 4:
        raise ValueError("ranking validation requires the original R/A/B partition")
    return {"candidate_k": 8, "micro_group": 4, "ranking_validation_n": n//2,
            "val_k": config["val_k"], "score": "cosine(mean(two LOO4 gradients), mean(R validation gradients))"}


def worker(root, stage, shard):
    import torch
    import evidence_downstream as ed
    from experiment import read_rollouts
    from grads import ProjectionSpec, grad_params, loo_advantages, prompt_gradient
    from net_gate_memory_worker import checkpoint_decoder_layers
    from rollout import collect_rollouts, load_policy

    c = core.read(root / "scoring.json")
    cfg = c["config"]
    parent = Path(c["parent"])
    if base.digest(parent / "adapter_model.safetensors") != c["adapter_sha256"]:
        raise ValueError("scoring parent changed")
    prompts = core.read(Path(c["prompts"]))
    if base.digest(Path(c["prompts"])) != c["prompts_sha256"]:
        raise ValueError("scoring prompts changed")
    contract = layout(cfg, prompts)
    n = contract["ranking_validation_n"] if stage == "validation" else len(prompts["train"])
    indices = range(n*shard//4, n*(shard+1)//4)
    done = root / f"{stage}-{shard}.done.json"
    payload = root / f"{stage}-{shard}.json"
    binding = {"contract_sha256": base.digest(root / "scoring.json"), "stage": stage, "shard": shard}
    if done.exists():
        if core.read(done) != {**binding, "sha256": base.digest(payload)}:
            raise ValueError("scoring shard changed")
        return
    with base.lease(root / f".{stage}-{shard}.lock"):
        base.bind(root / f"{stage}-{shard}.contract.json", binding)
        model, tok = load_policy(cfg["model"], parent)
        rows_path = root / f"{stage}-{shard}.jsonl"
        k = contract["val_k"] if stage == "validation" else 8
        pool = prompts["val"] if stage == "validation" else prompts["train"]
        collect_rollouts(model, tok, pool[indices.start:indices.stop], k, cfg["max_new_tokens"],
                         cfg["temperature"], rows_path, idx_offset=indices.start,
                         sampling_seed_base=c["sampling_seed"]+(0 if stage == "candidate" else 90000001))
        ed.reward_rows(rows_path, indices, k)
        groups = read_rollouts(rows_path)
        params = grad_params(model, cfg["grad_layers"])
        checkpoint_decoder_layers(model)
        spec = ProjectionSpec(dim=cfg["proj_dim"])
        direction = torch.tensor(core.read(root / "direction.json")["direction"]) if stage == "candidate" else None
        output = {}
        for index in indices:
            saved = root / stage / f"prompt-{index}.json"
            binding_i = {**binding, "rollouts_sha256": base.digest(rows_path)}
            if saved.exists():
                row = core.read(saved)
                if row["binding"] != binding_i:
                    raise ValueError("partial gradient belongs to different inputs")
                output[str(index)] = row["value"]
                continue
            rows = sorted(groups[index], key=lambda r: r["rollout_idx"])
            size = k if stage == "validation" else 4
            vectors = []
            for start in range(0, k, size):
                chunk = rows[start:start+size]
                advantages = loo_advantages(torch.tensor([r["reward"] for r in chunk]))
                weights = [torch.full((r["input_ids"].numel()-r["resp_start"],), float(a))
                           for r, a in zip(chunk, advantages, strict=True)]
                vectors.append(prompt_gradient(model, params, chunk, weights, spec, micro_batch=1).cpu())
            value = vectors[0].tolist() if stage == "validation" else matched_score(torch.stack(vectors), direction)
            base.bind(saved, {"binding": binding_i, "value": value})
            output[str(index)] = value
            print(f"[fresh_r] {stage} shard={shard} prompt={index} ({len(output)}/{len(indices)})", flush=True)
        base.bind(payload, output)
        base.bind(done, {**binding, "sha256": base.digest(payload)})


def merge(root, stage):
    import numpy as np
    from select_rules import jittered_topk, topk_count
    c = core.read(root / "scoring.json")
    prompts = core.read(Path(c["prompts"]))
    contract = layout(c["config"], prompts)
    n = contract["ranking_validation_n"] if stage == "validation" else len(prompts["train"])
    rows = {}
    for shard in range(4):
        payload = root / f"{stage}-{shard}.json"
        if core.read(root / f"{stage}-{shard}.done.json") != {
                "contract_sha256": base.digest(root / "scoring.json"), "stage": stage,
                "shard": shard, "sha256": base.digest(payload)}:
            raise ValueError("missing or invalid score shard")
        part = core.read(payload)
        if set(part) != {str(i) for i in range(n*shard//4, n*(shard+1)//4)}:
            raise ValueError("shard prompt coverage differs")
        rows.update(part)
    values = np.array([rows[str(i)] for i in range(n)])
    if not np.isfinite(values).all():
        raise ValueError("non-finite gradient or score")
    if stage == "validation":
        if values.shape != (n, c["config"]["proj_dim"]):
            raise ValueError("validation projection shape differs")
        base.bind(root / "direction.json", {"direction": values.mean(axis=0).tolist(), "prompts": n})
    else:
        k = topk_count(n, c["config"]["topk_frac"])
        indices = sorted(jittered_topk({int(i): float(v) for i, v in rows.items()}, k, c["config"]["seed"]+1000))
        base.bind(root / "selected.json", {"indices": indices, "scores": rows})
        base.bind(root / "selected.sha256.json", {"sha256": base.digest(root / "selected.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("validation", "candidate"), required=True)
    p.add_argument("--shard", type=int, choices=range(4), required=True)
    a = p.parse_args()
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    worker(a.root, a.stage, a.shard)
