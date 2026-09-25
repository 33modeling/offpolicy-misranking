"""Instrument existing scoring workers without changing their gradients or draws."""
from __future__ import annotations

import argparse
import contextlib
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selection_gate as core
import selection_gate_gpu as base


class Timings:
    def __init__(self, synchronize, clock=time.perf_counter):
        self.synchronize, self.clock = synchronize, clock
        self.seconds, self.calls = {}, {}

    def wrap(self, name, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            self.synchronize()
            start = self.clock()
            try:
                return function(*args, **kwargs)
            finally:
                self.synchronize()
                self.seconds[name] = self.seconds.get(name, 0.) + self.clock() - start
                self.calls[name] = self.calls.get(name, 0) + 1
        return measured


def evaluate(root, shard):
    import evidence_downstream as ed
    from rollout import collect_rollouts, load_policy
    c = core.read(root / "evaluation.json")
    parent = Path(c["parent"])
    if base.digest(parent / "adapter_model.safetensors") != c["adapter_sha256"]:
        raise ValueError("evaluation adapter changed")
    n = len(c["questions"])
    indices = range(n * shard // 4, n * (shard + 1) // 4)
    model, tokenizer = load_policy(c["config"]["model"], parent)
    path = root / f"evaluation-{shard}.jsonl"
    collect_rollouts(model, tokenizer, c["questions"][indices.start:indices.stop], c["k"],
                     c["config"]["max_new_tokens"], c["config"]["temperature"], path,
                     idx_offset=indices.start, sampling_seed_base=c["sampling_seed"])
    ed.reward_rows(path, indices, c["k"])
    base.bind(root / f"evaluation-{shard}.done.json", {
        "evaluation_sha256": base.digest(root / "evaluation.json"), "shard": shard,
        "sha256": base.digest(path)})


def worker(root, kind, stage, shard):
    from unittest.mock import patch

    import torch

    import grads
    import rollout

    def synchronize():
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()

    timings = Timings(synchronize)
    with contextlib.ExitStack() as stack:
        for module, name, label in ((rollout, "load_policy", "model_setup"),
                                    (rollout, "collect_rollouts", "response_generation"),
                                    (grads, "prompt_gradient", "gradient_computation")):
            stack.enter_context(patch.object(module, name, timings.wrap(label, getattr(module, name))))
        if kind == "ranking":
            import selection_switch_score
            if stage not in ("validation", "candidate"):
                raise ValueError("invalid ranking stage")
            selection_switch_score.worker(root, stage, shard)
        elif kind == "check":
            import selector_pair_srgc_score
            if stage not in ("validation-a", "candidate-a"):
                raise ValueError("online checks use one reference only")
            selector_pair_srgc_score.worker(root, stage, shard)
        else:
            if stage != "evaluation":
                raise ValueError("invalid evaluation stage")
            evaluate(root, shard)
    base.bind(root / f"timing-{stage}-{shard}.json", {
        "schema": "selector-pair-cost/function-timing-v1", "kind": kind, "stage": stage,
        "shard": shard, "gpus": 1, "seconds": timings.seconds, "calls": timings.calls,
        "scope": "CUDA-synchronized elapsed function time on one GPU, including CPU work; "
                 "gradient_computation includes gradient projection, not isolated backward kernels. "
                 "Components are part of, not additional to, the outer allocation meter."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("ranking", "check", "evaluation"), required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    worker(args.root, args.kind, args.stage, args.shard)


if __name__ == "__main__":
    main()
