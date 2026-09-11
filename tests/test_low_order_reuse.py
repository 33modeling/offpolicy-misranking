import itertools
import math

import numpy as np
import pytest

from low_order_reuse import algebra_audit, approximation, normalization, reuse_score


def brute(r, z, logw):
    pairs = list(itertools.combinations(range(len(r)), 2))
    h = lambda j, l: .5*math.exp(logw[j]+logw[l])*(r[j]-r[l])*(z[j]-z[l])
    v = lambda j, l: .5*math.exp(logw[j]+logw[l])*(r[j]-r[l])**2
    u2 = np.mean([h(*p) for p in pairs])
    u4 = np.mean([h(*p)*v(*q) for p in pairs for q in pairs if not set(p)&set(q)])
    return u2, u4


@pytest.mark.parametrize("k", [4, 5, 8, 13])
def test_disjoint_pair_reduction_matches_enumeration_and_permutation(k):
    rng = np.random.default_rng(809+k)
    r, z, logw = rng.integers(2, size=k), rng.normal(size=k), rng.normal(size=k)
    value = reuse_score(r, z, logw)
    assert [value["u2"], value["u4"]] == pytest.approx(brute(r, z, logw), rel=1e-11, abs=1e-12)
    order = rng.permutation(k)
    permuted = reuse_score(r[order], z[order], logw[order])
    assert permuted["methods"] == pytest.approx(value["methods"], abs=1e-12)
    assert value["weights_clipped"] is False


def test_population_coefficient_approximation_is_uniformly_bounded():
    spec = approximation()
    p = np.linspace(0, 1, 10001)
    error = np.abs(normalization(p)-(spec.a-spec.b*p*(1-p)))
    assert error.max() <= spec.max_absolute_error
    assert spec.max_absolute_error < .018162
    assert spec.record()["relative_coefficient_error_bound"] < .00979
    assert spec.a == pytest.approx(2.626790565102)
    assert spec.b == pytest.approx(3.158769570035)


@pytest.mark.parametrize("p", [.02, .1, .5, .91])
def test_finite_group_identity_keeps_response_length_weighting(p):
    expected = 0.
    for rewards in itertools.product([0., 1.], repeat=8):
        r = np.array(rewards)
        q = r.mean()
        z = (r-p)/np.where(r == 1, 1., 3.)
        group = np.mean((r-q)/(math.sqrt(q*(1-q))+1e-4)*z)
        expected += p**int(r.sum())*(1-p)**(8-int(r.sum()))*group
    covariance = p*(1-p)*((1-p)+p/3)
    assert expected == pytest.approx(float(normalization(p))*covariance, abs=1e-12)


def test_exact_enumeration_checks_offpolicy_moments():
    result = algebra_audit()
    assert len(result["enumeration"]) == 6
    assert max(r["identity_max_error"] for r in result["enumeration"]) < 1e-10


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), .5, -1, 2])
def test_invalid_rewards_rejected(bad):
    with pytest.raises(ValueError):
        reuse_score([0, 1, 0, bad], [1, 2, 3, 4], [0]*4)


@pytest.mark.parametrize("args", [([0, 1], [1, 2], [0, 0]),
                                  ([0]*4, [1]*3, [0]*4),
                                  ([0]*4, [float("nan")]*4, [0]*4)])
def test_invalid_response_arrays(args):
    with pytest.raises(ValueError):
        reuse_score(*args)


def test_extreme_weights_are_not_silently_clipped():
    with pytest.raises(ValueError, match="overflow"):
        reuse_score([0, 1, 0, 1], [1, 3, -1, 5], [1000]*4)


def test_dominant_weight_does_not_erase_disjoint_pairs():
    r = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    z = np.array([.2, -.7, .3, .2, -.1, .8, .7, -.2])
    lw = np.array([40., -40., 0., 1., -1., 2., -2., 3.])
    value = reuse_score(r, z, lw)
    assert [value["u2"], value["u4"]] == pytest.approx(brute(r, z, lw), rel=1e-12)


@pytest.mark.parametrize("reward", [0, 1])
def test_homogeneous_cache_has_zero_contrast(reward):
    row = reuse_score([reward]*8, list(range(8)), [0]*8)
    assert row["methods"] == {"low_order": 0., "pair_u2": 0.}
    assert row["mixed_pairs"] == 0
    assert row["ess"] == 8


def test_only_reviewed_group_size_and_positive_epsilon():
    with pytest.raises(ValueError, match="G=8"):
        approximation(group_size=4)
    for eps in (0, -1, float("inf")):
        with pytest.raises(ValueError):
            approximation(epsilon=eps)
