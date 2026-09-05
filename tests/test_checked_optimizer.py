import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_policy_grpo import checked_optimizer_step


@pytest.mark.parametrize("gradient,loss,active", [
    (float("inf"), 1., True), (float("nan"), 1., True),
    (1., float("nan"), True), (0., 1., True),
    (float("inf"), 0., False),
])
def test_invalid_update_preserves_parameters_and_optimizer(gradient, loss, active):
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([gradient])
    with pytest.raises(RuntimeError, match="optimizer not applied"):
        checked_optimizer_step(optimizer, [parameter], 1., loss, active)
    assert parameter.item() == 1.
    assert not optimizer.state
    assert parameter.grad is None


def test_valid_update_is_applied():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.ones(1)
    checked_optimizer_step(optimizer, [parameter], 1., 1., True)
    assert parameter.item() < 1.


def test_peer_rank_failure_vetoes_local_finite_update(monkeypatch):
    from train_policy_grpo import dist
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda flag, op: flag.fill_(1))
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.ones(1)
    with pytest.raises(RuntimeError, match="optimizer not applied"):
        checked_optimizer_step(optimizer, [parameter], 1., 1., True)
    assert parameter.item() == 1.
    assert not optimizer.state
