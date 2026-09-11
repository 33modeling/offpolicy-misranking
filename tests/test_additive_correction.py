import itertools
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from additive_correction import METHODS, algebra_audit, correction_weights
from grads import token_weights


def test_exact_audit_and_quadratic_remainder():
    result = algebra_audit()
    assert result["identity_max_error"] < 1e-12


@pytest.mark.parametrize("length", [1, 2, 3, 16])
def test_raw_identity_and_two_token_full_correction(length):
    generator = torch.Generator().manual_seed(length)
    lp = torch.randn(length, generator=generator, dtype=torch.float64) * .3
    lb = torch.zeros_like(lp)
    p = (lp.cumsum(0)-lp).exp()
    s = (lp.sum()-lp.cumsum(0)).exp()
    add = correction_weights(lp, lb, cap=None)
    full = lp.sum().exp().expand_as(lp)
    torch.testing.assert_close(full-add, lp.exp()*(p-1)*(s-1))
    if length <= 2:
        torch.testing.assert_close(add, full)


def test_tay2_is_the_derivative_of_degree_two_terminal_ratio_polynomial():
    logits = torch.tensor([-.4, .3, .7, -.2], dtype=torch.float64, requires_grad=True)
    lp, lb = logits.sigmoid().log(), torch.full((4,), -.6931471805599453, dtype=torch.float64)
    delta = (lp-lb).exp()-1
    polynomial = 1 + delta.sum() + sum(delta[i]*delta[j] for i, j in itertools.combinations(range(4), 2))
    expected, = torch.autograd.grad(polynomial, logits, retain_graph=True)
    weights = correction_weights(lp, lb, method="tay2_terminal", cap=None)
    actual, = torch.autograd.grad((weights*lp).sum(), logits)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cap", [1., 3., 10.])
def test_component_clipping_combination_exactly_matches_existing_estimators(cap):
    lp = torch.tensor([-20., .6, 12., -4.])
    lb = torch.zeros_like(lp)
    for advantage in (-.8, 0., 1.):
        actual = correction_weights(lp, lb, advantage, cap=cap)
        expected = (token_weights(lp, lb, advantage, "g10", cap)
                    + token_weights(lp, lb, advantage, "g01", cap)
                    - token_weights(lp, lb, advantage, "g00", cap))
        torch.testing.assert_close(actual, expected)


def test_one_combined_backward_equals_three_component_backwards():
    lp = torch.tensor([-.4, .2, .3, -.7], requires_grad=True)
    lb = torch.zeros_like(lp)
    combination = correction_weights(lp, lb)
    g1, = torch.autograd.grad((combination * lp).sum(), lp, retain_graph=True)
    separate = []
    for method in ("g10", "g01", "g00"):
        g, = torch.autograd.grad((token_weights(lp, lb, 1., method) * lp).sum(), lp, retain_graph=True)
        separate.append(g)
    torch.testing.assert_close(g1, separate[0]+separate[1]-separate[2])
    assert not combination.requires_grad


def test_negative_weights_are_not_silently_removed():
    lp = torch.tensor([.1, 2., .1], dtype=torch.float64).log()
    lb = torch.zeros_like(lp)
    assert correction_weights(lp, lb)[1] < 0
    assert correction_weights(lp, lb, method="tay2_terminal")[1] < 0


def test_clipped_identity_can_fail_and_is_not_claimed():
    lp = torch.tensor([5., -5., 5.], dtype=torch.float64)
    lb = torch.zeros_like(lp)
    p = (lp.cumsum(0)-lp).exp()
    s = (lp.sum()-lp.cumsum(0)).exp()
    clipped_error = token_weights(lp, lb, 1., "g11") - correction_weights(lp, lb)
    assert not torch.allclose(clipped_error, lp.exp()*(p-1)*(s-1))


@pytest.mark.parametrize("method", METHODS)
def test_equal_policies_produce_unit_weights(method):
    lp = torch.tensor([-4., -3., -2., -1.])
    torch.testing.assert_close(correction_weights(lp, lp, method=method), torch.ones_like(lp))


@pytest.mark.parametrize("cap", [.9, float("nan"), float("inf")])
def test_invalid_caps_rejected(cap):
    with pytest.raises(ValueError, match="cap"):
        correction_weights(torch.zeros(2), torch.zeros(2), cap=cap)


@pytest.mark.parametrize("bad", [torch.tensor([]), torch.zeros(2, 2), torch.tensor([float("nan")])])
def test_invalid_log_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        correction_weights(bad, bad)


def test_unclipped_overflow_is_an_error_not_a_plausible_score():
    with pytest.raises(ValueError, match="nonfinite"):
        correction_weights(torch.full((8,), 100.), torch.zeros(8), cap=None)


def test_additive_is_not_generally_a_scalar_objective_gradient():
    r = torch.tensor([1.2, 1.4, 1.6], dtype=torch.float64, requires_grad=True)
    def field(x):
        return torch.stack((x[1]*x[2], x[0]+x[2]-1, x[0]*x[1]))
    jacobian = torch.autograd.functional.jacobian(field, r)
    assert jacobian[0, 1] == pytest.approx(1.6)
    assert jacobian[1, 0] == pytest.approx(1.)


def test_raw_additive_and_tay2_share_first_order_terms_but_not_all_orders():
    errors = []
    for epsilon in (.01, .005):
        r = 1 + epsilon * torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
        zero = torch.zeros_like(r)
        difference = correction_weights(r.log(), zero, cap=None) - correction_weights(r.log(), zero, cap=None, method="tay2_terminal")
        errors.append(float(difference.abs().max()))
    assert 3.8 < errors[0]/errors[1] < 4.2


def test_actual_small_olmo_projection_matches_three_component_backwards():
    from test_logit_chunking import _tiny_olmo3

    from grads import (
        ProjectionSpec,
        grad_params,
        prompt_gradient,
        sequence_logprobs_batch,
    )

    model = _tiny_olmo3(seed=17)
    params = grad_params(model, 2)
    rows = [{"input_ids": torch.tensor([2, 3, 4, 5, 6, 7, 8]), "resp_start": 2},
            {"input_ids": torch.tensor([2, 3, 4, 5, 6, 7, 8, 9]), "resp_start": 2}]
    lp = sequence_logprobs_batch(model, rows, micro_batch=1)
    lb = [x - torch.linspace(-1.7, .5, x.numel()) for x in lp]
    advantages = [1., -1.]
    spec = ProjectionSpec(dim=64)
    composed = [correction_weights(x, y, a) for x, y, a in zip(lp, lb, advantages, strict=True)]
    combined = prompt_gradient(model, params, rows, composed, spec, micro_batch=1)
    separate = []
    for method in ("g10", "g01", "g00"):
        weights = [token_weights(x, y, a, method) for x, y, a in zip(lp, lb, advantages, strict=True)]
        separate.append(prompt_gradient(model, params, rows, weights, spec, micro_batch=1))
    torch.testing.assert_close(combined, separate[0]+separate[1]-separate[2], atol=1e-5, rtol=1e-4)
