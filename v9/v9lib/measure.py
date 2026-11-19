"""X5 measurement on one retained On-trajectory state (no training, no optimizer step).

For every candidate of the 400-prompt pool the existing scoring path
(backend.score_gradients, the call Engine._vectors makes for On selection) runs twice with
four responses each (two independent 4-response groups, group_size 4 = the On selector's
leave-one-out group). The same rollouts give

  (a) Alignment score reliability: cosine(g_A, v) vs cosine(g_B, v) across prompts (split-half rho_on),
      the full 8-response score cosine(g_A + g_B, v) (= the 8-response LOO gradient up to scale),
      sigma_on = its standard deviation, and ||v|| (validation gradient, 50 ranking-validation prompts,
      8 responses, group 8, as in Engine.update);
  (b) SR reliability: split-half of the success-rate estimate (p_A vs p_B, and -|p-1/2| scores),
      plus the cache's own 4/4 half split;
  (c) p_t (8 rollouts) against the cached p_beta;
  (d) KL(pi_t || beta) estimated on the same pi_t samples: log pi_t(y|x) - log beta(y|x) per response,
      summed over response tokens (sequence KL) and averaged per token.

beta = the policy that produced the SR cache: the pretrained base model without the LoRA adapter
(srgc_rebuttal/build_cache.py: "The initial policy is the pretrained base model of the plan (no
adapter)"; seeds 5-9 OLMo MATH bundles import the selector-pair "initial-policy behavior rollouts",
srgc_rebuttal/import_pair_inputs.py). log beta is computed by the same forward with the adapter
disabled (peft disable_adapter). The bundle's cache provenance is copied into the summary.

Resumable: candidates are processed in fixed chunks of 40; a finished chunk is appended to
rows.jsonl and skipped on resume. Rollout seeds depend only on (seed, update, half, chunk index).
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import time
from pathlib import Path

import numpy as np

CHUNK = 40
HALF = 4
MEASURE_SPEC = "x5-splithalf-4x2-kl-v1"


# ------------------------------------------------------------------ rollout probes
class TorchProbe:
    """TorchBackend / LlamaBackend: capture (sequences, rewards, start) of each scoring rollout."""

    def __init__(self, backend):
        self.backend = backend

    @contextmanager
    def capture(self):
        records = {}
        original = self.backend._rollout

        def wrapped(prompt_id, responses, seed):
            out = original(prompt_id, responses, seed)
            sequences, rewards, start = out
            records[prompt_id] = {"sequences": [s.detach() for s in sequences],
                                  "rewards": np.asarray(rewards, dtype=float), "start": int(start)}
            return out
        self.backend._rollout = wrapped
        try:
            yield records
        finally:
            del self.backend._rollout  # back to the class (or protected subclass) method

    def log_ratios(self, rec):
        """Per response: (sum over tokens of log pi_t - log beta, token count)."""
        import torch
        b = self.backend
        model = b.model
        if not hasattr(model, "disable_adapter"):
            raise RuntimeError("model has no adapter to disable; beta (base model) log-probs are unavailable")
        seqs, start = rec["sequences"], rec["start"]
        step = max(1, int(getattr(b, "logprob_micro_batch", 1)))
        cur, ref = [], []
        with torch.no_grad():
            for i in range(0, len(seqs), step):
                cur += [lp.double().cpu() for lp in b._logps_batch(seqs[i:i + step], start)]
            with model.disable_adapter():
                for i in range(0, len(seqs), step):
                    ref += [lp.double().cpu() for lp in b._logps_batch(seqs[i:i + step], start)]
        out = []
        for a, r in zip(cur, ref):
            if a.shape != r.shape:
                raise RuntimeError("token count differs between pi_t and beta log-prob passes")
            out.append((float((a - r).sum()), int(a.numel())))
        return out


class ToyProbe:
    """ToyBackend (CPU software check): one-token responses, beta = the cache policy's weights."""

    def __init__(self, backend, beta_weights):
        self.backend, self.beta = backend, np.asarray(beta_weights, dtype=float)

    @contextmanager
    def capture(self):
        records = {}
        original = self.backend._rollout

        def wrapped(prompt, responses, seed):
            rewards, gradients = original(prompt, responses, seed)
            records[prompt] = {"rewards": np.asarray(rewards, dtype=float), "responses": responses, "seed": seed}
            return rewards, gradients
        self.backend._rollout = wrapped
        try:
            yield records
        finally:
            del self.backend._rollout

    def log_ratios(self, rec):
        from srgc_rebuttal.srgc import stream_seed
        b = self.backend
        prompt = rec["prompt"]
        probs = b._probabilities(prompt)
        rng = np.random.default_rng(stream_seed(rec["seed"], 0, prompt))
        actions = rng.choice(2, size=rec["responses"], p=probs)  # the draw _rollout made
        logits = b.features[prompt] @ self.beta
        ref = np.exp(logits - logits.max())
        ref = ref / ref.sum()
        return [(float(math.log(probs[a]) - math.log(ref[a])), 1) for a in actions]


def make_probe(backend, beta=None):
    if hasattr(backend, "model"):
        return TorchProbe(backend)
    if beta is None:
        raise ValueError("toy probe needs the cache policy's weights")
    return ToyProbe(backend, beta)


# ------------------------------------------------------------------ statistics
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    return pearson(rankdata(x[ok]), rankdata(y[ok]))


def spearman_brown(r):
    return None if r is None or r <= -1 else 2 * r / (1 + r)


def cosine(g, v):
    d = float(np.linalg.norm(g) * np.linalg.norm(v))
    return float(g @ v / d) if d > 0 else 0.0


def summarize(rows, *, v_norm, meta):
    sa = [r["score_a"] for r in rows]
    sb = [r["score_b"] for r in rows]
    full = np.asarray([r["score_full"] for r in rows], float)
    pa = [r["p_a"] for r in rows]
    pb = [r["p_b"] for r in rows]
    sra = [-abs(p - 0.5) for p in pa]
    srb = [-abs(p - 0.5) for p in pb]
    ca = [r["p_beta_a"] for r in rows]
    cb = [r["p_beta_b"] for r in rows]
    dp = np.asarray([abs(r["p_t"] - r["p_beta"]) for r in rows], float)
    kl_seq = np.asarray([r["kl_seq_mean"] for r in rows], float)
    tok_sum = sum(r["log_ratio_token_sum"] for r in rows)
    tok_n = sum(r["tokens"] for r in rows)
    pinsker = [abs(r["p_t"] - r["p_beta"]) <= math.sqrt(max(r["kl_seq_mean"], 0.0) / 2) for r in rows]
    rho_on = pearson(sa, sb)
    return {
        **meta, "n_prompts": len(rows),
        "rho_on_split_half_pearson": rho_on, "rho_on_split_half_spearman": spearman(sa, sb),
        "rho_on_full_spearman_brown": spearman_brown(rho_on),
        "sigma_on": float(full.std(ddof=1)) if len(full) > 1 else None,
        "mean_score_on": float(full.mean()) if len(full) else None,
        "validation_gradient_norm": v_norm,
        "rho_sr_split_half_pearson": pearson(sra, srb), "rho_p_split_half_pearson": pearson(pa, pb),
        "rho_sr_cache_split_half_pearson": pearson([-abs(p - 0.5) for p in ca], [-abs(p - 0.5) for p in cb]),
        "rho_p_cache_split_half_pearson": pearson(ca, cb),
        "abs_p_shift_mean": float(dp.mean()), "abs_p_shift_median": float(np.median(dp)), "abs_p_shift_max": float(dp.max()),
        "kl_seq_mean": float(kl_seq.mean()), "kl_seq_median": float(np.median(kl_seq)), "kl_seq_max": float(kl_seq.max()),
        "kl_token_mean": tok_sum / tok_n if tok_n else None, "tokens_total": tok_n,
        "fraction_within_pinsker_bound": float(np.mean(pinsker)),
        "mean_p_t": float(np.mean([r["p_t"] for r in rows])), "mean_p_beta": float(np.mean([r["p_beta"] for r in rows])),
    }


# ------------------------------------------------------------------ the task
def execute_measure(task, ctx, *, beta=None):
    """Run inside the cluster child (every rank) or in-process (toy/tiny CPU checks)."""
    from switching import snapshots
    from switching.common import JsonlLog, atomic_write_json, read_json, read_jsonl, utc_now, write_once_json
    from switching.ledger import Ledger
    srgc, backend = ctx.srgc, ctx.backend
    out = Path(ctx.out_root) / task["out_dir"]
    summary_path = out / "summary.json"
    if ctx.primary(lambda: summary_path.exists()):
        return {"status": "skipped_complete", "path": str(summary_path)}
    state = snapshots.load_engine_state(task["snapshot_path"], task["snapshot_sha256"])
    if state["step"] != task["completed_update"]:
        raise ValueError(f"snapshot holds update {state['step']}, task expects {task['completed_update']}")
    backend.load_state_dict(state["backend"])
    seed, u = task["training_seed"], task["completed_update"]
    data = ctx.data
    candidates = list(data["candidate_ids"])
    validation = list(data["ranking_validation_ids"])
    cache = data["cached_rewards"]
    probe = make_probe(backend, beta)
    gather = getattr(backend, "_gather", lambda x: x)
    resp = task.get("responses_half", HALF)
    started = time.time()
    ledger = None
    if ctx.is_primary:
        out.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(out / "events.jsonl", run_id=task["task_id"], attempt_id=f"a{int(started)}",
                        gpu_type=ctx.gpu_type, measurement_status=ctx.measurement_status)
    ctx.event_sink = ledger.on_meter_event if ledger else None
    gpu_count = getattr(backend, "gpu_count", 0)

    def phase():
        meter = getattr(backend, "cost_meter", None)
        return meter.phase("evaluation", u, gpu_count) if meter is not None else _null()
    try:
        vpath = out / "validation_vector.json"
        v = ctx.primary(lambda: read_json(vpath)["v"] if vpath.exists() else None)
        if v is None:
            with phase():
                val = backend.score_gradients(validation, responses=task.get("validation_responses", 8),
                                              group_size=task.get("validation_responses", 8),
                                              seed=srgc.stream_seed(seed, u, "x5-validation"))
            v = np.stack([np.asarray(val[i], float) for i in validation]).mean(axis=0).tolist()
            ctx.primary(lambda: write_once_json(vpath, {"v": v, "n_validation": len(validation)}))
        v = np.asarray(v, float)
        rows_path = out / "rows.jsonl"
        done_chunks = ctx.primary(lambda: sorted({r["chunk"] for r in read_jsonl(rows_path)}))
        for k in range(0, (len(candidates) + CHUNK - 1) // CHUNK):
            if k in done_chunks:
                continue
            chunk = candidates[k * CHUNK:(k + 1) * CHUNK]
            local = {}
            with phase():
                halves = {}
                for name in ("a", "b"):
                    with probe.capture() as recs:
                        grads = backend.score_gradients(chunk, responses=resp, group_size=resp,
                                                        seed=srgc.stream_seed(seed, u, f"x5-half-{name}-chunk{k}"))
                    halves[name] = (grads, dict(recs))
                for pid in halves["a"][1]:
                    ratios = []
                    for name in ("a", "b"):
                        rec = dict(halves[name][1][pid], prompt=pid)
                        ratios += probe.log_ratios(rec)
                    local[pid] = {
                        "rewards_a": halves["a"][1][pid]["rewards"].tolist(),
                        "rewards_b": halves["b"][1][pid]["rewards"].tolist(),
                        "seq": [r[0] for r in ratios], "tok": [r[1] for r in ratios]}
                merged = gather(local)
            ga, gb = halves["a"][0], halves["b"][0]
            if set(merged) != set(chunk) or set(ga) != set(chunk) or set(gb) != set(chunk):
                raise ValueError(f"chunk {k}: scoring or rollout capture returned a different prompt set")
            rows = []
            for pid in chunk:
                a, b = np.asarray(ga[pid], float), np.asarray(gb[pid], float)
                ra, rb = merged[pid]["rewards_a"], merged[pid]["rewards_b"]
                c = [float(x) for x in cache[pid]]
                seq, tok = merged[pid]["seq"], merged[pid]["tok"]
                rows.append({"chunk": k, "prompt_id": pid, "score_a": cosine(a, v), "score_b": cosine(b, v),
                             "score_full": cosine(a + b, v), "dot_a": float(a @ v), "dot_b": float(b @ v),
                             "norm_a": float(np.linalg.norm(a)), "norm_b": float(np.linalg.norm(b)),
                             "p_a": float(np.mean(ra)), "p_b": float(np.mean(rb)), "p_t": float(np.mean(ra + rb)),
                             "p_beta": float(np.mean(c)), "p_beta_a": float(np.mean(c[:len(c) // 2])),
                             "p_beta_b": float(np.mean(c[len(c) // 2:])),
                             "kl_seq_mean": float(np.mean(seq)), "kl_seq_per_response": seq,
                             "log_ratio_token_sum": float(np.sum(seq)), "tokens": int(np.sum(tok))})

            def append(rows=rows):
                log = JsonlLog(rows_path, ("prompt_id",))
                for r in rows:
                    log.append(r)
            ctx.primary(append)
        def finish():
            rows = read_jsonl(rows_path)
            if {r["prompt_id"] for r in rows} != set(candidates):
                raise ValueError("measurement rows do not cover the candidate pool")
            meta = {"task_id": task["task_id"], "cohort": task["cohort_id"], "seed": seed, "completed_update": u,
                    "source": task["source"], "snapshot_sha256": task["snapshot_sha256"], "spec": MEASURE_SPEC,
                    "responses_per_half": resp, "beta": task["beta"],
                    "cache_provenance": (data.get("provenance") or {}).get("cache"),
                    "measured_at": utc_now(), "measurement_status": ctx.measurement_status,
                    "wall_seconds_this_attempt": time.time() - started}
            s = summarize(rows, v_norm=float(np.linalg.norm(v)), meta=meta)
            atomic_write_json(summary_path, s)
            return s
        s = ctx.primary(finish)
    finally:
        ctx.event_sink = None
        if ledger is not None:
            ledger.close_interrupted()
    return {"status": "measured", "path": str(summary_path), "rho_on": s["rho_on_split_half_pearson"]}


@contextmanager
def _null():
    yield
