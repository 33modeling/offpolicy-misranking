"""Gaussian top-k measurement-ceiling lookup for the registered designs.

The lookup curves are Monte Carlo estimates of ``Ovl_{n,k}(rho)`` with 20,000
replicates per correlation.  Runtime analysis only interpolates these frozen,
monotone curves, so every run uses the same mapping and seed contract.
"""

from __future__ import annotations

import math

SCHEMA = "gaussian-topk-ceiling/v1"
LOOKUP_SEED = 20_260_820
LOOKUP_REPLICATES = 20_000
CORRELATIONS = (
    0.00,
    0.02,
    0.05,
    0.10,
    0.12,
    0.198029508595,
    0.20,
    0.308606699924,
    0.40,
    0.426401432711,
    0.462910049886,
    0.577350269190,
    0.60,
    0.755928946018,
    0.80,
    0.866025403784,
    0.90,
    0.942809041582,
    0.95,
    0.973328526785,
    0.987096233586,
    0.99,
    0.997484272744,
    1.00,
)

# The rho=0 endpoints are set to the exact chance rate; the other entries are
# frozen Monte Carlo means.  Seeds are LOOKUP_SEED + 1_000_003*j + n for the
# j-th correlation, which makes every entry independently reproducible.
REGISTERED_CURVES = {
    (400, 40): (
        0.10000000,
        0.10633125,
        0.11581625,
        0.13276250,
        0.14080750,
        0.17144125,
        0.17177625,
        0.21895750,
        0.26518250,
        0.28005625,
        0.29955875,
        0.37306500,
        0.38747500,
        0.51594750,
        0.56051125,
        0.63755000,
        0.68666125,
        0.76125125,
        0.77552250,
        0.83545125,
        0.88422750,
        0.89765875,
        0.94652625,
        1.00000000,
    ),
    (512, 51): (
        51 / 512,
        0.10587549,
        0.11511667,
        0.13275196,
        0.14006471,
        0.17058922,
        0.17110686,
        0.21956863,
        0.26541667,
        0.28007647,
        0.30129902,
        0.37231176,
        0.38838627,
        0.51600392,
        0.56027157,
        0.63795980,
        0.68677941,
        0.76123431,
        0.77600098,
        0.83543922,
        0.88427059,
        0.89831176,
        0.94734118,
        1.00000000,
    ),
}


def _interpolate(value: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for index in range(1, len(xs)):
        if value <= xs[index]:
            width = xs[index] - xs[index - 1]
            fraction = (value - xs[index - 1]) / width
            return ys[index - 1] + fraction * (ys[index] - ys[index - 1])
    raise AssertionError("interpolation bracket not found")


def gaussian_ceiling_from_reliability(
    reliability: float, n: int, k: int
) -> dict[str, float | int | str | bool] | None:
    """Map split-half top-k reliability to the Gaussian oracle ceiling.

    Unsupported pool designs return ``None`` rather than silently substituting
    a large-sample approximation.  Reliability below chance is clamped to the
    nonnegative-correlation boundary and marked in the result.
    """
    curve = REGISTERED_CURVES.get((n, k))
    if curve is None:
        return None
    if not math.isfinite(reliability):
        raise ValueError("reliability must be finite")
    chance = k / n
    bounded = min(1.0, max(chance, reliability))
    rho_half = _interpolate(bounded, curve, CORRELATIONS)
    rho_full = 2.0 * rho_half / (1.0 + rho_half)
    ceiling_correlation = math.sqrt(rho_full)
    ceiling = _interpolate(ceiling_correlation, CORRELATIONS, curve)
    return {
        "schema": SCHEMA,
        "n": n,
        "k": k,
        "lookup_seed": LOOKUP_SEED,
        "lookup_replicates": LOOKUP_REPLICATES,
        "reliability": reliability,
        "reliability_clamped": reliability != bounded,
        "rho_half": rho_half,
        "rho_full": rho_full,
        "ceiling_correlation": ceiling_correlation,
        "ceiling": ceiling,
    }


def gaussian_ceiling_interval(
    lower_reliability: float,
    point_reliability: float,
    upper_reliability: float,
    n: int,
    k: int,
) -> dict[str, float | int | str | bool] | None:
    """Propagate a reliability interval through the monotone Gaussian mapping."""
    if lower_reliability > upper_reliability:
        raise ValueError("reliability interval endpoints are reversed")
    lower = gaussian_ceiling_from_reliability(lower_reliability, n, k)
    point = gaussian_ceiling_from_reliability(point_reliability, n, k)
    upper = gaussian_ceiling_from_reliability(upper_reliability, n, k)
    if point is None:
        return None
    assert lower is not None and upper is not None
    return {
        **point,
        "reliability_lower_two_sided_95": lower_reliability,
        "reliability_upper_two_sided_95": upper_reliability,
        "ceiling_lower_two_sided_95": lower["ceiling"],
        "ceiling_upper_two_sided_95": upper["ceiling"],
    }


def simulate_registered_curve(
    n: int,
    k: int,
    *,
    replicates: int = LOOKUP_REPLICATES,
    chunk_size: int = 1_000,
) -> tuple[float, ...]:
    """Reproduce a registered lookup curve from its frozen RNG contract."""
    if (n, k) not in REGISTERED_CURVES:
        raise ValueError(f"unregistered Gaussian ceiling design: n={n}, k={k}")
    if replicates < 1 or chunk_size < 1:
        raise ValueError("replicates and chunk_size must be positive")
    import numpy as np

    overlaps = []
    for index, correlation in enumerate(CORRELATIONS):
        if correlation == 0.0:
            overlaps.append(k / n)
            continue
        if correlation == 1.0:
            overlaps.append(1.0)
            continue
        rng = np.random.default_rng(LOOKUP_SEED + 1_000_003 * index + n)
        total = 0
        completed = 0
        while completed < replicates:
            draws = min(chunk_size, replicates - completed)
            latent = rng.standard_normal((draws, n))
            noisy = correlation * latent + math.sqrt(1.0 - correlation**2) * (
                rng.standard_normal((draws, n))
            )
            latent_top = np.argpartition(latent, -k, axis=1)[:, -k:]
            noisy_top = np.argpartition(noisy, -k, axis=1)[:, -k:]
            latent_mask = np.zeros((draws, n), dtype=bool)
            np.put_along_axis(latent_mask, latent_top, True, axis=1)
            total += int(np.take_along_axis(latent_mask, noisy_top, axis=1).sum())
            completed += draws
        overlaps.append(total / (replicates * k))
    return tuple(overlaps)
