"""CPU smoke cohorts built on the code repository's own backends (no GPU).

toy   srgc_rebuttal.toy_backend.ToyBackend (categorical policy, AdamW, fresh sampling),
      full 400/40/4/8 shapes and the 25/275 schedule; virtual 4-GPU allocation so the
      cost-receipt path is exercised (measurement_status=simulated_cpu).
tiny  srgc_rebuttal.torch_backend.TorchBackend on a randomly initialized 4-layer OLMo-3
      with LoRA (real GRPO loss, autograd scoring gradients, AdamW state), reduced shapes
      and a 5-update schedule; requires torch/transformers/peft (CPU build).
Neither produces paper results.
"""

from __future__ import annotations

from . import coderepo


def toy_problem(seed=0, candidates=400, validation=100, evaluation=100):
    coderepo.activate()
    from srgc_rebuttal.toy_backend import make_problem
    features, answers, cand, val, ev, cache = make_problem(seed, candidates, validation, evaluation)
    return features, answers, {"candidate_ids": list(cand), "validation_pool_ids": list(val),
                               "ranking_validation_ids": list(val[:50]), "dev_evaluation_ids": list(val[50:]),
                               "evaluation_ids": list(ev), "cached_rewards": cache}


def toy_backend(features, answers, *, projection_dim=64, seed=0, learning_rate=0.05, virtual_gpus=4, record=None):
    coderepo.activate()
    import numpy as np
    from srgc_rebuttal.toy_backend import ToyBackend
    from srgc_rebuttal.timing import CostMeter
    from srgc_rebuttal.srgc import stream_seed

    class V8ToyBackend(ToyBackend):
        gpu_count = virtual_gpus
        world = 1

        def evaluate(self, ids, *, seed, responses=8):
            return {i: float(self._rollout(i, responses, stream_seed(seed, 0, "evaluate"))[0].mean()) for i in ids}

        def evaluate_counts(self, ids, *, seed, responses=8):
            return {i: int(self._rollout(i, responses, stream_seed(seed, 0, "evaluate"))[0].sum()) for i in ids}

    backend = V8ToyBackend(features, answers, projection_dim=projection_dim, seed=seed, learning_rate=learning_rate)
    backend.cost_meter = CostMeter(rank=0, local_gpu_count=virtual_gpus, record=record or (lambda e: None))
    _ = np
    return backend


def tiny_problem(n_candidates=24, n_val=8, n_eval=6):
    ids = [f"t{i:03d}" for i in range(n_candidates + n_val + n_eval)]
    records = {i: {"prompt": f"problem {i}", "answer": "3", "question": f"problem {i}"} for i in ids}
    cand, val, ev = ids[:n_candidates], ids[n_candidates:n_candidates + n_val], ids[n_candidates + n_val:]
    return records, {"candidate_ids": cand, "validation_pool_ids": val, "ranking_validation_ids": val[:4],
                     "dev_evaluation_ids": val[4:], "evaluation_ids": ev}


def tiny_backend(records, *, projection_dim=16, record=None, seed=7):
    coderepo.activate()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Olmo3Config, Olmo3ForCausalLM
    from srgc_rebuttal.torch_backend import TorchBackend
    from srgc_rebuttal.timing import CostMeter

    class TinyTokenizer:
        pad_token_id, eos_token_id = 0, 7

        def __call__(self, text, **kwargs):
            # distinct prompts map to distinct two-token inputs
            k = 1 + (sum(map(ord, text)) % 5)
            return {"input_ids": torch.tensor([[1, k]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}

        def decode(self, ids, **kwargs):
            return " ".join(map(str, ids.tolist()))

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Olmo3ForCausalLM(Olmo3Config(vocab_size=8, hidden_size=16, intermediate_size=24, num_hidden_layers=4,
                                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                                         pad_token_id=0, eos_token_id=7, layer_types=["full_attention"] * 4,
                                         attention_dropout=0.0))
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))
    meter = CostMeter(rank=0, local_gpu_count=0, record=record or (lambda e: None))
    backend = TorchBackend(model, TinyTokenizer(), records, lambda rec, text: float("3" in text.split()),
                           projection_dim=projection_dim, max_new_tokens=3, cost_meter=meter)
    for p in backend.optimizer.param_groups:
        p["lr"] = 5e-3  # visible movement in a few updates; the production LR is unchanged (1e-5)
    return backend


def initial_cache(backend, ids, *, responses, seed, srgc):
    from .cache import rewards_from_means
    means = backend.evaluate(ids, seed=srgc.stream_seed(seed, 0, "initial-cache"), responses=responses)
    return rewards_from_means(means, responses)
