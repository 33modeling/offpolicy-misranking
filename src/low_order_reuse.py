"""Finite-G GRPO reuse moments. No rollout generation or policy optimization.

The target is the initial unclipped, length-normalized GRPO loss gradient.
Full response IS is retained; no clipping or self-normalization is hidden here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

METHODS = ("low_order", "pair_u2")


@dataclass(frozen=True)
class Approximation:
    group_size: int
    epsilon: float
    a: float
    b: float
    max_absolute_error: float
    min_coefficient: float

    def record(self):
        return {**asdict(self), "relative_coefficient_error_bound":
                self.max_absolute_error / self.min_coefficient,
                "scope": "Population normalization coefficient only; not finite-sample or benchmark error."}


def normalization(p, *, group_size=8, epsilon=1e-4):
    if type(group_size) is not int or group_size < 2:
        raise ValueError("group size must be an integer >=2")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("success probability must be in [0,1]")
    n = group_size - 2
    return sum((group_size - 1) / group_size * math.comb(n, l)
               * p**l * (1-p)**(n-l)
               / (math.sqrt((l+1)/group_size * (1-(l+1)/group_size)) + epsilon)
               for l in range(n+1))


@lru_cache(maxsize=32)
def approximation(*, group_size=8, epsilon=1e-4):
    """Analytic secant-error calculation; constants depend only on G and epsilon."""
    if group_size != 8:
        raise ValueError("the reviewed low-order approximation supports G=8 only")
    normalization(0., group_size=group_size, epsilon=epsilon)
    bernstein = [(group_size-1)/group_size /
                 (math.sqrt((l+1)/group_size*(1-(l+1)/group_size)) + epsilon)
                 for l in range(7)]
    power = np.zeros(7, dtype=np.float64)
    for l, value in enumerate(bernstein):
        for j in range(7-l):
            power[l+j] += value * math.comb(6, l) * math.comb(6-l, j) * (-1)**j
    c0, c1, c2, c3 = power[0], power[1], power[2]+power[1], -power[6]
    # f(p) is symmetric: f = c0+c1*t+c2*t^2+c3*t^3, t=p(1-p).
    f = lambda t: c0+c1*t+c2*t*t+c3*t*t*t
    if min(2*c2, 2*c2+1.5*c3) <= 0 or max(c1, c1+.5*c2+.1875*c3) >= 0:
        raise ValueError("normalization does not satisfy the reviewed convex/decreasing contract")
    slope = (f(.25)-f(0.))/.25
    roots = np.roots([3*c3, 2*c2, c1-slope])
    candidates = [float(r.real) for r in roots if abs(r.imag) < 1e-10 and 0 < r.real < .25]
    if len(candidates) != 1:
        raise ValueError("cannot certify the normalization secant error")
    t = candidates[0]
    error = (c0+slope*t-f(t))/2
    # A conservative allowance for coefficient arithmetic, not a sampling bound.
    return Approximation(8, epsilon, float(c0-error), float(-slope),
                         float(error+1e-10), float(f(.25)-1e-10))


def reuse_score(rewards, directional, log_ratios, *, group_size=8, epsilon=1e-4):
    """O(K^2) two/four-response moments; inputs must refer to the same responses."""
    r, z, lr = (np.asarray(value, dtype=np.float64) for value in
                (rewards, directional, log_ratios))
    if r.ndim != 1 or r.size < 4 or z.shape != r.shape or lr.shape != r.shape:
        raise ValueError("need >=4 matched one-dimensional response arrays")
    if not all(np.isfinite(value).all() for value in (r, z, lr)):
        raise ValueError("response arrays must be finite")
    if not np.isin(r, [0., 1.]).all():
        raise ValueError("binary verifier rewards required")
    spec = approximation(group_size=group_size, epsilon=epsilon)
    k = len(r)
    j, l = np.triu_indices(k, 1)
    contrast = r[j]-r[l]
    mixed = contrast != 0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            # Excluding a dominant weight by subtraction loses small contrasts.
            # Binary rewards allow positive prefix/suffix sums in log space instead.
            excluded = []
            for reward in (0., 1.):
                logs = np.where(r == reward, lr, -np.inf)
                left = np.r_[-np.inf, np.logaddexp.accumulate(logs)[:-1]]
                right = np.r_[np.logaddexp.accumulate(logs[::-1])[::-1][1:], -np.inf]
                excluded.append(np.logaddexp(left, right))
            jj, ll = j[mixed], l[mixed]
            pos = np.where(r[jj] == 1, jj, ll)
            neg = np.where(r[jj] == 0, jj, ll)
            pair_logs = lr[jj]+lr[ll]
            difference = contrast[mixed]*(z[jj]-z[ll])
            h = .5*np.exp(pair_logs)*difference
            terms4 = .25*np.exp(pair_logs+excluded[1][pos]+excluded[0][neg])*difference
            u2 = float(h.sum()/math.comb(k, 2))
            u4 = float(terms4.sum()/(math.comb(k, 2)*math.comb(k-2, 2)))
            score = spec.a*u2-spec.b*u4
    except FloatingPointError as exc:
        raise ValueError("importance moments overflowed; weights were not clipped") from exc
    if not all(math.isfinite(x) for x in (u2, u4, score)):
        raise ValueError("nonfinite importance moments; weights were not clipped")
    scaled = np.exp(lr-lr.max())
    ess = float(scaled.sum()**2 / np.dot(scaled, scaled))
    return {"methods": {"low_order": score, "pair_u2": u2}, "u2": u2, "u4": u4,
            "responses": k, "successes": int(r.sum()), "pairs": len(j),
            "mixed_pairs": int(np.count_nonzero(contrast)), "ess": ess,
            "normalized_ess": ess/k, "max_log_ratio": float(lr.max()),
            "min_log_ratio": float(lr.min()), "weights_clipped": False,
            "normalization": spec.record()}


def algebra_audit():
    """Enumerate all binary caches: a mathematical check, not learning evidence."""
    spec = approximation()
    rows = []
    for p, beta in ((.1, .1), (.3, .3), (.5, .5), (.8, .8), (.6, .3), (.1, .8)):
        means = np.zeros(2)
        for mask in range(256):
            r = np.array([(mask >> j) & 1 for j in range(8)])
            probability = beta**int(r.sum())*(1-beta)**(8-int(r.sum()))
            z = (r-p)/np.where(r == 1, 1., 3.)
            lr = np.where(r == 1, math.log(p/beta), math.log((1-p)/(1-beta)))
            value = reuse_score(r, z, lr)
            means += probability*np.array([value["u2"], value["u4"]])
        t = p*(1-p)
        covariance = t*((1-p)+p/3)
        error = float(np.max(np.abs(means-[covariance, t*covariance])))
        if error > 1e-10:
            raise ValueError("moment expectation identity failed")
        rows.append({"pi_success": p, "beta_success": beta, "identity_max_error": error})
    return {"normalization": spec.record(), "enumeration": rows,
            "scope": "CPU algebra only; no LLM performance or cost claim."}
