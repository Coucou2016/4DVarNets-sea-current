import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fourdvarnet.losses import TrainingLoss
from fourdvarnet.prior import PhiPrior


def test_training_loss_uncert_logs_nll():
    pred = torch.randn(2, 3, 8, 8, requires_grad=True)
    truth = torch.randn(2, 3, 8, 8)
    phi = PhiPrior(3, hidden=8)
    crit = TrainingLoss(use_uncert=True, sigma0=0.05, alpha=1e4, dx=5e3, dy=5e3)
    loss, logs = crit(pred, truth, phi)
    assert torch.isfinite(loss)
    assert "l_uv_nll" in logs
    assert "l_uv" in logs
    loss.backward()
    assert pred.grad is not None


def test_uncert_nll_not_reduced_by_inflating_pred_strain():
    """Regression for M4 collapse: σ must not reward wild predicted strain."""
    torch.manual_seed(0)
    truth = torch.randn(2, 3, 16, 16)
    calm = truth + 0.05 * torch.randn_like(truth)
    wild = calm.clone()
    wild[:, 1:] = wild[:, 1:] + 3.0 * torch.randn_like(wild[:, 1:])

    phi = PhiPrior(3, hidden=8)
    crit = TrainingLoss(
        use_uncert=True,
        sigma0=0.05,
        alpha=1e4,
        dx=5e3,
        dy=5e3,
        uncert_from_truth=True,
        uncert_mse_mix=0.0,
    )
    _, logs_calm = crit(calm, truth, phi)
    _, logs_wild = crit(wild, truth, phi)
    # With σ from truth, wild UV must not look better than calm under NLL alone.
    assert logs_wild["l_uv_nll"] > logs_calm["l_uv_nll"]


def test_uncert_mse_mix_keeps_mse_in_total_grad():
    pred = torch.zeros(1, 3, 8, 8, requires_grad=True)
    truth = torch.ones(1, 3, 8, 8)
    phi = PhiPrior(3, hidden=8)
    crit = TrainingLoss(
        use_uncert=True,
        sigma0=0.05,
        alpha=1e4,
        dx=5e3,
        dy=5e3,
        uncert_mse_mix=0.5,
    )
    loss, _ = crit(pred, truth, phi)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_uncert_nll_scaled_near_mse_when_sigma_near_sigma0():
    """σ-normalized NLL stays MSE-comparable when σ≈σ0 (M4 SSH-swamp fix)."""
    torch.manual_seed(1)
    # Uniform UV field → strain≈0 → σ≈σ0 after uncertainty map.
    truth = torch.zeros(2, 3, 16, 16)
    truth[:, 1:] = 0.2
    pred = truth + 0.05 * torch.randn_like(truth)
    phi = PhiPrior(3, hidden=8)
    sigma0 = 0.05
    crit_mse = TrainingLoss(use_uncert=False, sigma0=sigma0, dx=5e3, dy=5e3)
    crit_nll = TrainingLoss(
        use_uncert=True,
        sigma0=sigma0,
        alpha=1e4,
        dx=5e3,
        dy=5e3,
        uncert_mse_mix=0.0,
    )
    _, logs_mse = crit_mse(pred, truth, phi)
    _, logs_nll = crit_nll(pred, truth, phi)
    # Pure-NLL UV contribution should match MSE closely when σ≈σ0 (log term ~0).
    assert abs(logs_nll["l_uv_nll"] - logs_mse["l_uv"]) < 0.25 * logs_mse["l_uv"] + 1e-4
    assert logs_nll["l_uv_nll"] > 0.0
