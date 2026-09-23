"""CPU-only uncertainty diagnostic for saved per-prompt SR-GC projections."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import selector_pair_srgc_score as score


def _jackknife_variance(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("jackknife stratum requires at least two prompts")
    return float((n - 1) / n * np.square(values - values.mean()).sum())


def estimate(directory, sets, check_index, *, family_alpha=0.05):
    """Approximate one-sided bound; shared candidate IDs stay paired across A/B."""
    if type(check_index) is not int or check_index < 1:
        raise ValueError("check index must start at one")
    if not math.isfinite(family_alpha) or not 0 < family_alpha < 1:
        raise ValueError("family alpha must be between zero and one")
    on_ids, sr_ids = set(sets["on_policy"]), set(sets["cached"])
    candidate_ids = sorted(on_ids | sr_ids)
    if len(on_ids) < 2 or len(sr_ids) < 2:
        raise ValueError("both candidate subsets require at least two prompts")
    candidates = [score.projections(directory, "candidate-" + half) for half in ("a", "b")]
    validations = [score.projections(directory, "validation-" + half) for half in ("a", "b")]
    if any(set(rows) != set(candidate_ids) for rows in candidates):
        raise ValueError("candidate projection IDs differ from the frozen sets")
    c = [np.stack([rows[i] for i in candidate_ids]) for rows in candidates]
    v = [np.stack(list(rows.values())) for rows in validations]
    if any(len(rows) < 2 for rows in v):
        raise ValueError("each validation reference requires at least two prompts")
    on = np.asarray([i in on_ids for i in candidate_ids])
    sr = np.asarray([i in sr_ids for i in candidate_ids])
    means_v = [rows.mean(axis=0) for rows in v]
    deltas = [rows[on].mean(axis=0) - rows[sr].mean(axis=0) for rows in c]
    d = float(sum(delta @ direction for delta, direction in zip(deltas, means_v)) / 2)

    candidate_replicates = []
    for i in range(len(candidate_ids)):
        contrasts = []
        for rows, direction in zip(c, means_v):
            on_sum, sr_sum = rows[on].sum(axis=0), rows[sr].sum(axis=0)
            on_mean = (on_sum - rows[i] * on[i]) / (on.sum() - int(on[i]))
            sr_mean = (sr_sum - rows[i] * sr[i]) / (sr.sum() - int(sr[i]))
            contrasts.append(float((on_mean - sr_mean) @ direction))
        candidate_replicates.append(sum(contrasts) / 2)
    variance = _jackknife_variance(candidate_replicates)
    for half in range(2):
        other = float(deltas[1 - half] @ means_v[1 - half])
        total = v[half].sum(axis=0)
        replicates = [float((deltas[half] @ ((total - row) / (len(v[half]) - 1)) + other) / 2)
                      for row in v[half]]
        variance += _jackknife_variance(replicates)
    se = math.sqrt(max(0., variance))
    alpha_k = family_alpha / (check_index * (check_index + 1))
    upper = d + NormalDist().inv_cdf(1 - alpha_k) * se
    if not all(math.isfinite(value) for value in (d, se, upper)):
        raise ValueError("non-finite SR-GC uncertainty estimate")
    return {"d": d, "standard_error": se, "upper": upper,
            "alpha_at_check": alpha_k, "family_alpha": family_alpha,
            "check_index": check_index, "candidate_prompts": len(candidate_ids),
            "validation_prompts": [len(rows) for rows in v],
            "method": "stratified leave-one-prompt-out jackknife; one-sided normal approximation",
            "confirmed_sr": upper < 0}
