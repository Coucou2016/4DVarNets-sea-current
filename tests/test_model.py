"""Model architecture and gradient flow tests."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fourdvarnet.model import FourDVarNetUV
from fourdvarnet.observation import ObservationOperator
from fourdvarnet.solver import VariationalCost


def test_observation_synergy_shape():
    obs = ObservationOperator(n_state=3, dT_sst=5, feat_dim=8)
    B, H, W = 2, 16, 16
    x = torch.randn(B, 3, H, W, requires_grad=True)
    y_ssh = torch.randn(B, 1, H, W)
    z = torch.randn(B, 5, H, W)
    m = torch.ones(B, 1, H, W)
    ms = torch.ones(B, 5, H, W)
    dy = obs(x, y_ssh, z, m, ms, use_sst=True)
    assert len(dy) == 2
    assert dy[1].shape == (B, 8, H, W)


def test_variational_cost_backward():
    from fourdvarnet.prior import PhiPrior

    phi = PhiPrior(3, hidden=8)
    obs = ObservationOperator(dT_sst=5, feat_dim=8)
    cost = VariationalCost(phi, obs, use_sst=True)
    x = torch.randn(1, 3, 12, 12, requires_grad=True)
    loss = cost(x, torch.randn(1, 1, 12, 12), torch.randn(1, 5, 12, 12), torch.ones(1, 1, 12, 12), torch.ones(1, 5, 12, 12))
    loss.backward()
    assert x.grad is not None


def test_4dvarnet_end_to_end_grad():
    m = FourDVarNetUV(dT_sst=5, n_iter=3, hidden_lstm=24, feat_dim=8)
    B, H, W = 1, 12, 12
    pred = m(
        torch.randn(B, 1, H, W),
        torch.randn(B, 5, H, W),
        torch.ones(B, 1, H, W),
        torch.ones(B, 5, H, W),
        torch.randn(B, 1, H, W),
        torch.randn(B, 1, H, W),
    )
    assert pred.shape == (B, 3, H, W)
    pred.sum().backward()
    assert any(p.grad is not None for p in m.parameters())


def test_variational_cost_sqg_adv_backward():
    from fourdvarnet.prior import PhiPrior

    phi = PhiPrior(3, hidden=8)
    obs = ObservationOperator(dT_sst=5, feat_dim=8)
    dx = 5e3
    cost = VariationalCost(
        phi, obs, use_sst=True, use_sqg=True, use_adv=True, dx=dx, dy=dx, Ld=30e3, f0=7e-5
    )
    x = torch.randn(1, 3, 12, 12, requires_grad=True)
    loss = cost(
        x,
        torch.randn(1, 1, 12, 12),
        torch.randn(1, 5, 12, 12),
        torch.ones(1, 1, 12, 12),
        torch.ones(1, 5, 12, 12),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None


def test_physics_flags_do_not_change_state_dict_keys():
    a = FourDVarNetUV(dT_sst=5, n_iter=2, hidden_lstm=16, feat_dim=8, use_sst=False)
    b = FourDVarNetUV(
        dT_sst=5, n_iter=2, hidden_lstm=16, feat_dim=8, use_sst=True, use_sqg=True, use_adv=True
    )
    assert set(a.state_dict()) == set(b.state_dict())
