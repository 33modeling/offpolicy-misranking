#!/usr/bin/env python3
"""Synthetic RLVR pool study (registered extension E4, 2026-09-07).

Every candidate prompt is a ``horizon``-token verifiable environment whose
token probabilities are logistic functions of a *shared* parameter vector, so
that one policy update moves every prompt at once, as an RLVR update does. At
step ``t`` the token ``A_t`` is Bernoulli with success probability

    p_{i,t}(A_{t-1}) = sigmoid(psi_{i,t,A_{t-1}} . theta),      A_0 = 0,

and the terminal verifier reward is ``R = A_T``. The current policy has
parameters ``theta_pi``; the behavior policy is ``theta_beta = theta_pi + delta
* eta`` with ``eta ~ N(0, I)``, so ``delta`` is the standard deviation of the
per-logit change. The score function of the current policy at token ``t`` is
``(A_t - p_{i,t}) psi_{i,t,A_{t-1}}``.

Prompts share structure through ``n_skills`` latent skills: every prompt's
feature vectors are a ``skill_weight`` mixture of its skill's shared features
and prompt-specific noise. Candidate prompts draw their skill uniformly;
validation prompts draw it with weights proportional to
``validation_mix_decay ** skill``, so the validation objective favours some
skills and prompt selection has a real target: candidates of the favoured
skills carry gradients aligned with the validation direction.

The measurement mirrors the pipeline (EXPERIMENT_PLAN Section 6):

* behavior pool: ``K_b`` rollouts per candidate sampled under ``theta_beta``,
  leave-one-out advantages, token importance weights for ``g00/g10/g01/g11``
  (token ratio; prefix product; suffix product; full product), each clipped
  to ``[1/clip_cap, clip_cap]``;
* current pool: 32 rollouts per candidate under ``theta_pi`` in eight
  four-rollout micro-groups: R = first two groups, A = groups five and six,
  B = groups seven and eight;
* validation: 100 prompts with eight rollouts each, split R/A/B as 50/25/25,
  giving the three directions ``v_R, v_A, v_B``;
* scores are cosines with ``v_R``; the held-out utility of a prompt is the mean
  of its cosines with ``v_A`` and ``v_B``; retention, precision, split-half
  reliability, measurability, and reversal rate follow the plan.

Only numpy is required. The default grid runs in about a minute on one CPU.

    PYTHONPATH=src python3 src/synthetic_pool_study.py --output-dir results/synthetic
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ESTIMATORS = ("g00", "g10", "g01", "g11")
SELECTORS = ESTIMATORS + ("fresh_r",)


@dataclass(frozen=True)
class StudyConfig:
    n_candidates: int = 400
    n_validation: int = 100
    dim: int = 64
    horizon: int = 8
    logit_scale: float = 1.0
    n_skills: int = 4
    skill_weight: float = 0.7
    validation_mix_decay: float = 0.3
    fresh_k: int = 32
    val_k: int = 8
    micro_group: int = 4
    topk_frac: float = 0.10
    clip_cap: float = 10.0
    tie_pairs: int = 20


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _topk_count(n: int, frac: float) -> int:
    return min(n, max(1, int(n * frac)))


def _topk_set(scores: np.ndarray, k: int, rng: np.random.Generator) -> set[int]:
    """Top-k indices with a random tie stream, as ``select_rules.jittered_topk``."""
    jitter = rng.random(scores.shape[0])
    order = np.lexsort((jitter, -scores))
    return set(int(i) for i in order[:k])


def _cosine_rows(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(direction)
    out = matrix @ direction
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(norms > 0, out / np.where(norms > 0, norms, 1.0), 0.0)
    return out


def loo_advantages(rewards: np.ndarray) -> np.ndarray:
    """Leave-one-out advantages along the last axis; zero when K < 2."""
    k = rewards.shape[-1]
    if k < 2:
        return np.zeros_like(rewards, dtype=float)
    total = rewards.sum(axis=-1, keepdims=True)
    return rewards - (total - rewards) / (k - 1)


def token_log_weights(log_r: np.ndarray, estimator: str) -> np.ndarray:
    """``grads.log_weights`` for a trajectory: ``log_r`` has shape (..., T)."""
    if estimator == "g00":
        return log_r
    if estimator == "g10":
        return np.cumsum(log_r, axis=-1)
    if estimator == "g01":
        total = log_r.sum(axis=-1, keepdims=True)
        return total - np.cumsum(log_r, axis=-1) + log_r
    if estimator == "g11":
        return np.broadcast_to(log_r.sum(axis=-1, keepdims=True), log_r.shape)
    raise ValueError(f"unknown estimator {estimator}")


class Environment:
    """One replicate: features, current parameters, and the drifted behavior."""

    def __init__(self, config: StudyConfig, delta: float, rng: np.random.Generator):
        self.config = config
        d, T = config.dim, config.horizon
        n_total = config.n_candidates + config.n_validation
        scale = 1.0 / math.sqrt(d)
        candidate_skills = rng.integers(0, config.n_skills, size=config.n_candidates)
        mix = np.array([config.validation_mix_decay ** s for s in range(config.n_skills)])
        validation_skills = rng.choice(config.n_skills, size=config.n_validation, p=mix / mix.sum())
        skills = np.concatenate([candidate_skills, validation_skills])
        skill_psi = rng.normal(0.0, scale, size=(config.n_skills, T, 2, d))
        w = config.skill_weight
        noise = math.sqrt(max(0.0, 1.0 - w * w))
        self.skills = skills
        self.psi = w * skill_psi[skills] + noise * rng.normal(0.0, scale, size=(n_total, T, 2, d))
        self.theta_pi = rng.normal(0.0, config.logit_scale, size=d)
        self.theta_beta = self.theta_pi + delta * rng.normal(0.0, 1.0, size=d)
        self.candidates = np.arange(config.n_candidates)
        self.validation = np.arange(config.n_candidates, n_total)

    def logits(self, theta: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Token success logits for both previous states: shape (n, T, 2)."""
        return self.psi[ids] @ theta

    def sample(self, theta: np.ndarray, ids: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """Trajectories under ``theta``: actions of shape (n, k, T)."""
        probs = _sigmoid(self.logits(theta, ids))  # (n, T, 2)
        n, T = ids.shape[0], self.config.horizon
        actions = np.zeros((n, k, T), dtype=np.int64)
        previous = np.zeros((n, k), dtype=np.int64)
        rows = np.arange(n)[:, None]
        for t in range(T):
            p = probs[rows, t, previous]  # (n, k)
            actions[:, :, t] = (rng.random((n, k)) < p).astype(np.int64)
            previous = actions[:, :, t]
        return actions

    def _previous(self, actions: np.ndarray) -> np.ndarray:
        previous = np.zeros_like(actions)
        previous[:, :, 1:] = actions[:, :, :-1]
        return previous

    def score_function(self, ids: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Per-token score functions of the current policy, shape (n, k, T, dim)."""
        probs = _sigmoid(self.logits(self.theta_pi, ids))  # (n, T, 2)
        previous = self._previous(actions)
        n, k, T = actions.shape
        rows = np.arange(n)[:, None, None]
        steps = np.arange(T)[None, None, :]
        p_taken = probs[rows, steps, previous]  # (n, k, T)
        psi_taken = self.psi[ids][rows, steps, previous]  # (n, k, T, dim)
        return (actions - p_taken)[..., None] * psi_taken

    def log_ratios(self, ids: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Per-token log pi/beta for behavior rollouts, shape (n, k, T)."""
        previous = self._previous(actions)
        n, k, T = actions.shape
        rows = np.arange(n)[:, None, None]
        steps = np.arange(T)[None, None, :]

        def token_logp(theta: np.ndarray) -> np.ndarray:
            p = _sigmoid(self.logits(theta, ids))[rows, steps, previous]
            return np.where(actions == 1, np.log(p), np.log1p(-p))

        return token_logp(self.theta_pi) - token_logp(self.theta_beta)


def stale_gradients(env: Environment, ids: np.ndarray, k_b: int, rng: np.random.Generator):
    actions = env.sample(env.theta_beta, ids, k_b, rng)
    rewards = actions[:, :, -1].astype(float)
    advantages = loo_advantages(rewards)  # (n, k)
    z = env.score_function(ids, actions)  # (n, k, T, dim)
    log_r = env.log_ratios(ids, actions)  # (n, k, T)
    cap = math.log(env.config.clip_cap)
    out = {}
    clip_fraction = {}
    for estimator in ESTIMATORS:
        log_w = token_log_weights(log_r, estimator)
        clip_fraction[estimator] = float(np.mean(np.abs(log_w) > cap))
        weights = np.exp(np.clip(log_w, -cap, cap)) * advantages[:, :, None]
        out[estimator] = (weights[..., None] * z).sum(axis=2).mean(axis=1)
    return out, rewards.mean(axis=1), clip_fraction


def current_group_gradients(env: Environment, ids: np.ndarray, k: int, micro: int, rng):
    """Per-prompt micro-group gradients, shape (n, k // micro, dim)."""
    actions = env.sample(env.theta_pi, ids, k, rng)
    z = env.score_function(ids, actions).sum(axis=2)  # (n, k, dim)
    rewards = actions[:, :, -1].astype(float)
    groups = k // micro
    out = np.zeros((ids.shape[0], groups, env.config.dim))
    for g in range(groups):
        sl = slice(g * micro, (g + 1) * micro)
        adv = loo_advantages(rewards[:, sl])
        out[:, g] = (adv[:, :, None] * z[:, sl]).mean(axis=1)
    return out


def exact_gradients(env: Environment, ids: np.ndarray) -> np.ndarray:
    """Population policy gradient of E[R] under theta_pi by forward dynamic programming."""
    probs = _sigmoid(env.logits(env.theta_pi, ids))  # (n, T, 2)
    n, T = probs.shape[0], env.config.horizon
    psi = env.psi[ids]  # (n, T, 2, dim)
    # state distribution over previous action and accumulated score expectation
    state = np.zeros((n, 2))
    state[:, 0] = 1.0
    score = np.zeros((n, 2, env.config.dim))  # E[sum_{u<=t} z_u | A_t = a] * P(A_t = a)
    for t in range(T):
        next_state = np.zeros_like(state)
        next_score = np.zeros_like(score)
        for prev in (0, 1):
            p = probs[:, t, prev][:, None]  # P(A_t = 1 | prev)
            for action, p_action in ((1, p), (0, 1 - p)):
                z = (action - p) * psi[:, t, prev]  # (n, dim)
                weight = state[:, prev][:, None] * p_action
                next_state[:, action] += weight[:, 0]
                next_score[:, action] += p_action * score[:, prev] + weight * z
        state, score = next_state, next_score
    return score[:, 1]  # reward is A_T = 1


def run_replicate(config: StudyConfig, delta: float, k_b: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    env = Environment(config, delta, rng)
    cand, val = env.candidates, env.validation
    n = cand.shape[0]
    k = _topk_count(n, config.topk_frac)

    val_groups = current_group_gradients(env, val, config.val_k, config.val_k, rng)[:, 0]
    half = val.shape[0] // 2
    quarter = val.shape[0] // 4
    v_r = val_groups[:half].mean(axis=0)
    v_a = val_groups[half : half + quarter].mean(axis=0)
    v_b = val_groups[half + quarter :].mean(axis=0)

    groups = current_group_gradients(env, cand, config.fresh_k, config.micro_group, rng)
    fresh = _cosine_rows(groups[:, :2].mean(axis=1), v_r)
    score_a = _cosine_rows(groups[:, 4:6].mean(axis=1), v_a)
    score_b = _cosine_rows(groups[:, 6:8].mean(axis=1), v_b)
    truth = (score_a + score_b) / 2.0

    stale, behavior_rate, clip_fraction = stale_gradients(env, cand, k_b, rng)
    scores = {est: _cosine_rows(stale[est], v_r) for est in ESTIMATORS}
    scores["fresh_r"] = fresh
    exact_scores = _cosine_rows(exact_gradients(env, cand), v_r)

    tie_rng = np.random.default_rng(seed + 1_000)
    fresh_top = [_topk_set(fresh, k, tie_rng) for _ in range(config.tie_pairs)]
    a_top = [_topk_set(score_a, k, tie_rng) for _ in range(config.tie_pairs)]
    b_top = [_topk_set(score_b, k, tie_rng) for _ in range(config.tie_pairs)]
    floor = float(np.mean([len(sa & sb) / k for sa, sb in zip(a_top, b_top)]))
    chance = k / n
    random_utility = float(truth.mean())
    fresh_utility = float(np.mean([truth[list(s)].mean() for s in fresh_top]))
    fresh_gain = fresh_utility - random_utility
    measurable = floor >= 2.0 * chance and fresh_gain > 0
    mixed = float(np.mean((behavior_rate > 0.0) & (behavior_rate < 1.0)))

    rows: dict[str, dict[str, float]] = {}
    for name in SELECTORS:
        sc = scores[name]
        tops = [_topk_set(sc, k, tie_rng) for _ in range(config.tie_pairs)]
        precision = float(np.mean([len(t & f) / k for t, f in zip(tops, fresh_top)]))
        utility = float(np.mean([truth[list(t)].mean() for t in tops]))
        gain = utility - random_utility
        nonzero = (sc != 0) & (fresh != 0)
        reversal = float(np.mean(sc[nonzero] * fresh[nonzero] < 0)) if nonzero.any() else 0.0
        exact_nonzero = (sc != 0) & (exact_scores != 0)
        exact_reversal = (
            float(np.mean(sc[exact_nonzero] * exact_scores[exact_nonzero] < 0)) if exact_nonzero.any() else 0.0
        )
        rows[name] = {
            "precision": precision,
            "utility_gain": gain,
            "fresh_gain": fresh_gain,
            "retention": gain / fresh_gain if measurable else float("nan"),
            "reversal_vs_fresh": reversal,
            "reversal_vs_exact": exact_reversal,
            "clip_fraction": clip_fraction.get(name, 0.0),
            "floor": floor,
            "chance": chance,
            "measurable": float(measurable),
            "mixed_reward_fraction": mixed,
            "k": float(k),
        }
    return rows


AGGREGATE_KEYS = ("precision", "utility_gain", "fresh_gain", "retention", "reversal_vs_fresh",
                  "reversal_vs_exact", "clip_fraction", "floor", "measurable", "mixed_reward_fraction")


def _aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for rec in records:
        grouped.setdefault((rec["delta"], rec["k_b"], rec["selector"]), []).append(rec)
    out = []
    for (delta, k_b, selector), items in sorted(grouped.items()):
        row = {"delta": delta, "k_b": k_b, "selector": selector, "replicates": len(items)}
        for key in AGGREGATE_KEYS:
            values = np.array([it[key] for it in items], dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{key}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{key}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            row[f"{key}_n"] = int(finite.size)
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_dat(path: Path, rows: list[dict], x: str, series: list[str], y: str) -> None:
    """pgfplots table: one x column and mean/sd columns per series."""
    xs = sorted({row[x] for row in rows})
    lines = [x + " " + " ".join(f"{s}_mean {s}_sd" for s in series)]
    for value in xs:
        cells = [f"{value:g}"]
        for s in series:
            match = [row for row in rows if row[x] == value and row["selector"] == s]
            if match:
                cells.append(f"{match[0][y + '_mean']:.5f} {match[0][y + '_sd']:.5f}")
            else:
                cells.append("nan nan")
        lines.append(" ".join(cells))
    path.write_text("\n".join(lines) + "\n")


def run_study(config: StudyConfig, deltas: list[float], behavior_ks: list[int], replicates: int,
              seed: int, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for delta in deltas:
        for k_b in behavior_ks:
            for rep in range(replicates):
                rep_seed = seed + 100_003 * rep + int(round(delta * 1e4)) * 7 + k_b * 31
                for selector, metrics in run_replicate(config, delta, k_b, rep_seed).items():
                    records.append({"delta": delta, "k_b": k_b, "replicate": rep, "seed": rep_seed,
                                    "selector": selector, **metrics})
    summary = _aggregate(records)
    _write_csv(output_dir / "synthetic_pool_records.csv", records)
    _write_csv(output_dir / "synthetic_pool_summary.csv", summary)
    (output_dir / "synthetic_pool_config.json").write_text(json.dumps({
        "config": asdict(config), "deltas": deltas, "behavior_ks": behavior_ks,
        "replicates": replicates, "seed": seed}, indent=1))
    reference_k = 8 if 8 in behavior_ks else behavior_ks[0]
    at_reference_k = [row for row in summary if row["k_b"] == reference_k]
    for metric in ("retention", "precision", "reversal_vs_fresh", "clip_fraction", "utility_gain"):
        _write_dat(output_dir / f"{metric}_vs_delta.dat", at_reference_k, "delta", list(ESTIMATORS), metric)
    at_zero = [row for row in summary if row["delta"] == deltas[0]]
    _write_dat(output_dir / "retention_vs_kb.dat", at_zero, "k_b", list(ESTIMATORS), "retention")
    _write_dat(output_dir / "floor_vs_kb.dat", at_zero, "k_b", ["fresh_r"], "floor")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--behavior-ks", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_907)
    parser.add_argument("--n-candidates", type=int, default=400)
    parser.add_argument("--n-validation", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--clip-cap", type=float, default=10.0)
    parser.add_argument("--n-skills", type=int, default=4)
    parser.add_argument("--skill-weight", type=float, default=0.7)
    parser.add_argument("--validation-mix-decay", type=float, default=0.3)
    args = parser.parse_args(argv)
    if args.n_validation < 8 or args.n_validation % 4:
        parser.error("--n-validation must be a multiple of four and at least eight")
    if any(k < 2 for k in args.behavior_ks):
        parser.error("behavior budgets need at least two rollouts for leave-one-out advantages")
    if args.horizon < 1 or args.clip_cap < 1.0:
        parser.error("--horizon must be positive and --clip-cap at least 1")
    if not 0.0 <= args.skill_weight <= 1.0 or args.n_skills < 1:
        parser.error("--skill-weight must lie in [0, 1] and --n-skills must be positive")
    if not 0.0 < args.validation_mix_decay <= 1.0:
        parser.error("--validation-mix-decay must lie in (0, 1]")
    config = StudyConfig(n_candidates=args.n_candidates, n_validation=args.n_validation, dim=args.dim,
                         horizon=args.horizon, clip_cap=args.clip_cap, n_skills=args.n_skills,
                         skill_weight=args.skill_weight, validation_mix_decay=args.validation_mix_decay)
    summary = run_study(config, args.deltas, args.behavior_ks, args.replicates, args.seed, args.output_dir)
    print(f"{'delta':>6} {'K_b':>4} {'selector':>8} {'retention':>10} {'(n)':>4} {'precision':>10} "
          f"{'reversal':>9} {'clip':>6} {'floor':>6} {'measurable':>10}")
    for row in summary:
        print(f"{row['delta']:>6g} {row['k_b']:>4d} {row['selector']:>8} "
              f"{row['retention_mean']:>10.3f} {row['retention_n']:>4d} {row['precision_mean']:>10.3f} "
              f"{row['reversal_vs_fresh_mean']:>9.3f} {row['clip_fraction_mean']:>6.3f} "
              f"{row['floor_mean']:>6.3f} {row['measurable_mean']:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
