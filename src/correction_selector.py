"""Experimental cross-fitted correction selection (CFCS), NumPy only.

The score is a directional derivative, not cosine or observed training gain.
Weights are calibrated from between-prompt signal and within-prompt noise.
This is an empirical shrinkage rule, not a top-k certificate or a DR estimator.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from select_rules import jittered_topk

ENDPOINTS = ("g00", "g10", "g01", "g11")
SCHEMA = "crossfit-correction-selector/v1"


def _finite(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def directional_difference(plus, minus, epsilon):
    """Central difference of token log probabilities along one fixed direction.

    The caller evaluates the SAME stored tokens under theta +/- epsilon*v,
    with dropout disabled and the original parameters restored afterwards.
    An exact center evaluation is still required for importance ratios.
    """
    plus, minus = _finite(plus, "plus"), _finite(minus, "minus")
    if plus.shape != minus.shape or not plus.size:
        raise ValueError("plus and minus must have matching nonempty shapes")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    return (plus - minus) / (2 * epsilon)


def endpoint_groups(log_ratios, directional_tokens, rewards, *, group_size=2, cap=10.0):
    """Reduce shared stale trajectories to (prompt, independent group, endpoint).

    Inputs are (N,K,T), (N,K,T), (N,K). Padding must have zero log ratio and
    zero directional term. LOO advantages are local to disjoint micro-groups.
    All four endpoints share the same samples, preserving their covariance.
    """
    ratios = _finite(log_ratios, "log_ratios")
    direction = _finite(directional_tokens, "directional_tokens")
    reward = _finite(rewards, "rewards")
    if ratios.ndim != 3 or min(ratios.shape) < 1 or direction.shape != ratios.shape:
        raise ValueError("token inputs must have matching nonempty (N,K,T) shapes")
    n, k, _ = ratios.shape
    if reward.shape != (n, k):
        raise ValueError("rewards must have shape (N,K)")
    if group_size < 2 or k % group_size:
        raise ValueError("K must be divisible by group_size >= 2")
    if not np.isfinite(cap) or cap < 1:
        raise ValueError("cap must be finite and >= 1")
    groups = k // group_size
    r = reward.reshape(n, groups, group_size)
    advantages = r - (r.sum(axis=-1, keepdims=True) - r) / (group_size - 1)
    advantages = advantages.reshape(n, k)
    prefix = np.cumsum(ratios, axis=-1)
    total = ratios.sum(axis=-1, keepdims=True)
    log_weights = (
        ratios,
        prefix,
        total - prefix + ratios,
        np.broadcast_to(total, ratios.shape),
    )
    outputs = []
    for log_w in log_weights:
        weights = np.exp(np.clip(log_w, -np.log(cap), np.log(cap)))
        sample_scores = advantages * (weights * direction).sum(axis=-1)
        outputs.append(sample_scores.reshape(n, groups, group_size).mean(axis=-1))
    return np.stack(outputs, axis=-1)


def correction_components(groups):
    """g00, prefix increment, suffix increment, remaining interaction."""
    g = _finite(groups, "groups")
    if g.ndim != 3 or g.shape[-1] != 4 or g.shape[0] < 2:
        raise ValueError("groups must have shape (N>=2,G,4)")
    b, p, s, f = np.moveaxis(g, -1, 0)
    return np.stack((b, p - b, s - b, f - p - s + b), axis=-1)


def box_quadratic(matrix, target):
    """Solve min x'Hx - 2b'x over [0,1]^3 by enumerating active faces."""
    h, b = _finite(matrix, "matrix"), _finite(target, "target")
    if h.shape != (3, 3) or b.shape != (3,):
        raise ValueError("quadratic must have dimension three")
    scale = max(float(np.abs(h).max()), float(np.abs(b).max()))
    if scale == 0:
        return np.zeros(3)
    h, b = (h + h.T) / (2 * scale), b / scale
    if np.linalg.eigvalsh(h).min() < -1e-9:
        raise ValueError("quadratic must be positive semidefinite")
    best, best_loss = np.zeros(3), 0.0
    for face in itertools.product((0, 1, 2), repeat=3):
        x = np.array([1.0 if v == 1 else 0.0 for v in face])
        free = np.flatnonzero(np.array(face) == 2)
        fixed = np.flatnonzero(np.array(face) != 2)
        if free.size:
            rhs = b[free] - h[np.ix_(free, fixed)] @ x[fixed]
            x[free] = np.linalg.lstsq(h[np.ix_(free, free)], rhs, rcond=None)[0]
            if np.any(x[free] < -1e-8) or np.any(x[free] > 1 + 1e-8):
                continue
            x = np.clip(x, 0, 1)
        loss = float(x @ h @ x - 2 * b @ x)
        if loss < best_loss - 1e-12:
            best, best_loss = x, loss
    return best


def fit_corrections(calibration_groups, evaluation_group_count):
    """Estimate a shared correction vector using calibration samples only.

    For correction means c_i, M estimates their BETWEEN-prompt covariance.
    Noise covariance comes from independent groups, accounting for covariances
    with the base score. The plug-in risk target is the clipped full endpoint.
    It cannot detect clipping bias, missing support, or a bad validation target.
    """
    z = correction_components(calibration_groups)
    n, m, _ = z.shape
    if m < 2 or evaluation_group_count < 1:
        raise ValueError("need >=2 calibration groups and >=1 evaluation group")
    means = z.mean(axis=1)
    centered = z - means[:, None, :]
    single_cov = np.einsum("ngi,ngj->ij", centered, centered) / (n * (m - 1))
    corrections = means[:, 1:]
    between = np.cov(corrections, rowvar=False, ddof=1)
    signal = between - single_cov[1:, 1:] / m
    eigenvalues, eigenvectors = np.linalg.eigh((signal + signal.T) / 2)
    signal = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    noise = single_cov / evaluation_group_count
    h = signal + noise[1:, 1:]
    target = signal @ np.ones(3) - noise[1:, 0]
    coefficients = box_quadratic(h, target)
    return coefficients, {
        "signal_trace": float(np.trace(signal)),
        "noise_trace": float(np.trace(noise[1:, 1:])),
        "calibration_groups": m,
        "evaluation_groups": evaluation_group_count,
    }


def fit_anchor_corrections(
    calibration_groups, anchor_ids, anchor_scores, *, discrete=False
):
    """Fit the same bounded correction rule to a small independent fresh pilot.

    Fresh labels must use the SAME frozen direction as the stale scores. They
    are calibration data, never evaluation evidence. Centering absorbs a common
    offset; each coefficient stays in [0,1]. This can discover clipping/common
    bias absent from stale-only covariance estimation, but has no universal guarantee.
    """
    z = correction_components(calibration_groups).mean(axis=1)
    ids = np.asarray(anchor_ids)
    y = _finite(anchor_scores, "anchor_scores")
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 8:
        raise ValueError("anchor_ids must contain at least eight integer IDs")
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(z)):
        raise ValueError("anchor_ids must be unique valid row indices")
    if y.shape != ids.shape:
        raise ValueError("anchor_scores must match anchor_ids")
    x = z[ids, 1:]
    residual = y - z[ids, 0]
    x = x - x.mean(axis=0)
    residual = residual - residual.mean()
    h = x.T @ x / (len(ids) - 1)
    rhs = x.T @ residual / (len(ids) - 1)
    if discrete:
        choices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=float
        )
        losses = np.einsum("ni,ij,nj->n", choices, h, choices) - 2 * choices @ rhs
        coefficients = choices[np.argmin(losses)]
    else:
        coefficients = box_quadratic(h, rhs)
    return coefficients, {
        "calibration_source": "independent fresh anchor responses",
        "anchor_count": len(ids),
        "centered_fit_mse": float(np.mean((x @ coefficients - residual) ** 2)),
    }


def crossfit_scores(
    groups, *, seed=0, anchor_ids=None, anchor_scores=None, anchor_discrete=False
):
    g = _finite(groups, "groups")
    z = correction_components(g)
    m = z.shape[1]
    if m < 4:
        raise ValueError("CFCS requires at least four independent groups")
    order = np.random.default_rng(seed).permutation(m)
    halves = (order[: m // 2], order[m // 2 :])
    scores = np.zeros(g.shape[0])
    folds = []
    if (anchor_ids is None) != (anchor_scores is None):
        raise ValueError("anchor_ids and anchor_scores must be supplied together")
    for fit_ids, eval_ids in (halves, halves[::-1]):
        if anchor_ids is None:
            coefficients, diagnostics = fit_corrections(g[:, fit_ids], len(eval_ids))
        else:
            coefficients, diagnostics = fit_anchor_corrections(
                g[:, fit_ids], anchor_ids, anchor_scores, discrete=anchor_discrete
            )
        evaluated = z[:, eval_ids].mean(axis=1)
        scores += (
            len(eval_ids) / m * (evaluated[:, 0] + evaluated[:, 1:] @ coefficients)
        )
        folds.append(
            {
                **diagnostics,
                "coefficients": coefficients.tolist(),
                "fit_groups": fit_ids.tolist(),
                "score_groups": eval_ids.tolist(),
            }
        )
    return scores, folds


def select(
    groups, k, *, seed=0, exploration_fraction=0.0, anchor_ids=None, anchor_scores=None
):
    scores, folds = crossfit_scores(
        groups, seed=seed, anchor_ids=anchor_ids, anchor_scores=anchor_scores
    )
    n = len(scores)
    if not 1 <= k <= n:
        raise ValueError("k must be in [1,N]")
    if not np.isfinite(exploration_fraction) or not 0 <= exploration_fraction <= 1:
        raise ValueError("exploration_fraction must be in [0,1]")
    n_random = min(k, int(np.ceil(k * exploration_fraction)))
    n_scored = k - n_random
    chosen = (
        jittered_topk(dict(enumerate(scores)), n_scored, seed) if n_scored else set()
    )
    if n_random:
        remaining = sorted(set(range(n)) - chosen)
        chosen.update(
            np.random.default_rng(seed + 104729)
            .choice(remaining, n_random, replace=False)
            .tolist()
        )
    return {
        "schema": SCHEMA,
        "method": "anchor_cfcs" if anchor_ids is not None else "cfcs",
        "selected_indices": sorted(chosen),
        "scores": scores.tolist(),
        "folds": folds,
        "seed": seed,
        "exploration_count": n_random,
        "score_target": "fresh directional pilot"
        if anchor_ids is not None
        else "clipped-full directional shrinkage target",
        "guarantee": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="NPZ: endpoint_groups (N,G,4), optional prompt_ids"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--anchors", type=Path, help="NPZ: anchor_ids (row indices), anchor_scores"
    )
    parser.add_argument("--exploration-fraction", type=float, default=0.0)
    args = parser.parse_args()
    anchors = {}
    if args.anchors is not None:
        with np.load(args.anchors, allow_pickle=False) as data:
            anchors = {
                "anchor_ids": data["anchor_ids"],
                "anchor_scores": data["anchor_scores"],
            }
    with np.load(args.input, allow_pickle=False) as data:
        result = select(
            data["endpoint_groups"],
            args.k,
            seed=args.seed,
            exploration_fraction=args.exploration_fraction,
            **anchors,
        )
        ids = (
            data["prompt_ids"].tolist()
            if "prompt_ids" in data
            else list(range(len(result["scores"])))
        )
    if len(ids) != len(result["scores"]) or len(set(ids)) != len(ids):
        raise ValueError("prompt_ids must be unique and cover every score")
    result["selected_prompt_ids"] = [ids[i] for i in result["selected_indices"]]
    result["input"] = str(args.input.resolve())
    if args.anchors is not None:
        result["anchors_input"] = str(args.anchors.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            {"selected": len(result["selected_indices"]), "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
