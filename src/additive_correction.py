"""Experimental scoring weights; neither estimator is a new GRPO optimizer.

Unclipped additive weights omit only prefix/suffix cross interactions.
The terminal-reward TayPO-2 comparator differentiates a degree-two trajectory
ratio polynomial, not the full discounted/critic-based published algorithm.
"""

from __future__ import annotations

import math

import torch

from grads import log_weights, token_weights

METHODS = ("gadd", "tay2_terminal")


def validate_logps(pi: torch.Tensor, beta: torch.Tensor) -> None:
    if pi.ndim != 1 or pi.shape != beta.shape or not pi.numel():
        raise ValueError("log probabilities must be nonempty equal-length vectors")
    if not torch.isfinite(pi).all() or not torch.isfinite(beta).all():
        raise ValueError("log probabilities must be finite")


def correction_weights(pi, beta, advantage=1.0, *, method="gadd", cap=10.0):
    """Detached, possibly signed weights. cap=None is the raw analytic variant.

gadd: C(rP) + C(rS) - C(r), with C(w)=clip(w, 1/cap, cap).
tay2_terminal: C(r_t) * (1 + sum_{u!=t}(C(r_u)-1)).
Clipping precedes composition in both cases. It introduces separate bias;
neither clipped formula inherits the raw remainder identity or its bounds.
"""
    validate_logps(pi, beta)
    if method not in METHODS:
        raise ValueError(f"unknown correction method: {method}")
    if not math.isfinite(float(advantage)):
        raise ValueError("advantage must be finite")
    if cap is not None and (not math.isfinite(cap) or cap < 1):
        raise ValueError("cap must be finite and >=1, or None")
    if cap is None:
        pi, beta = pi.detach().double(), beta.detach().double()
        ratio = torch.exp(pi - beta)
        if method == "gadd":
            weights = (log_weights(pi, beta, "g10").exp()
                       + log_weights(pi, beta, "g01").exp() - ratio)
        else:
            deviation = ratio - 1
            weights = ratio * (1 + deviation.sum() - deviation)
    else:
        ratio = token_weights(pi, beta, 1., "g00", cap)
        if method == "gadd":
            weights = (token_weights(pi, beta, 1., "g10", cap)
                       + token_weights(pi, beta, 1., "g01", cap) - ratio)
        else:
            deviation = ratio - 1
            weights = ratio * (1 + deviation.sum() - deviation)
    weights = weights * float(advantage)
    if not torch.isfinite(weights).all():
        raise ValueError("nonfinite composite weights; raw product overflow is not silently clipped")
    return weights.detach()


def algebra_audit():
    """Deterministic identities and limitations, not a real-model experiment."""
    maximum_error = 0.
    for ratios in ([.7], [.6, 1.4], [.8, 1.1, 1.3], [1.3, .8, 1.2, .6]):
        r = torch.tensor(ratios, dtype=torch.float64)
        lp, lb = r.log(), torch.zeros_like(r)
        p = torch.exp(torch.cumsum(lp, 0) - lp)
        s = torch.exp(lp.sum() - torch.cumsum(lp, 0))
        full = r.prod().expand_as(r)
        add = correction_weights(lp, lb, cap=None)
        error = float((full - add - r * (p - 1) * (s - 1)).abs().max())
        maximum_error = max(maximum_error, error)
        if len(ratios) <= 2:
            assert torch.allclose(full, add, atol=1e-12, rtol=0)
            assert torch.allclose(add, correction_weights(lp, lb, method="tay2_terminal", cap=None))
    # Counterexample to "the additive construction is just TayPO-2".
    r = torch.tensor([2., 3., 4.], dtype=torch.float64)
    zero = torch.zeros_like(r)
    add = correction_weights(r.log(), zero, cap=None)
    tay = correction_weights(r.log(), zero, method="tay2_terminal", cap=None)
    assert torch.allclose(add, torch.tensor([24., 15., 24.], dtype=torch.float64))
    assert torch.allclose(tay, torch.tensor([12., 15., 16.], dtype=torch.float64))
    scaling = []
    for delta in (.1, .05, .025):
        p = .5 + delta
        full = p**3 * (1-p)
        partial = p**2 * (1-p) / 2
        additive = p * (1-p) * (p-.25)
        assert math.isclose(full-additive, p*(1-p)*delta**2, abs_tol=1e-12)
        scaling.append({"delta": delta, "partial_bias": full-partial,
                        "additive_bias": full-additive})
    assert maximum_error < 1e-12
    return {"schema": "offpolicy-additive-algebra/v1", "identity_max_error": maximum_error,
            "three_token_weights": {"ratios": r.tolist(), "gadd": add.tolist(), "tay2_terminal": tay.tolist()},
            "exact_three_token_bernoulli_example": scaling,
            "scope": "Exact unclipped examples only. No variance, cosine, downstream, or novelty guarantee."}
