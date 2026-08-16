import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fourdvarnet.metrics import (
    batch_metrics,
    isotropic_psd_2d,
    lagrangian_separation,
    resolved_scale_km,
    resolved_timescale,
    rmse,
)


def test_rmse_zero():
    x = torch.ones(2, 3, 4, 4)
    assert rmse(x, x) < 1e-6


def test_batch_metrics_keys():
    pred = torch.randn(2, 3, 8, 8)
    truth = torch.randn(2, 3, 8, 8)
    m = batch_metrics(pred, truth, dx=5e3, dy=5e3, dx_km=5.0)
    for key in ("tau_uv", "tau_div", "tau_vort", "tau_strain", "rmse_u", "rmse_v", "rmse_ssh"):
        assert key in m
    assert "lambda_x_ssh_km" in m
    assert "lambda_x_ssh_min_ratio" in m
    assert "lambda_x_ssh_resolved" in m


def test_resolved_scale_unresolved_exposes_min_ratio():
    """When λ_x is NaN, min_ratio must still be a finite diagnostic (≥ threshold)."""
    from fourdvarnet.metrics import resolved_scale_report

    rng = np.random.default_rng(1)
    truth = rng.normal(size=(16, 16))
    # Huge error ⇒ err/signal never below 0.5
    pred = truth + 50.0 * rng.normal(size=(16, 16))
    rep = resolved_scale_report(pred, truth, dx_km=5.0)
    assert rep["lambda_km"] != rep["lambda_km"]  # NaN
    assert rep["resolved"] == 0.0
    assert np.isfinite(rep["min_ratio"])
    assert rep["min_ratio"] >= 0.5


def test_resolved_scale_perfect_match_is_small_or_nan():
    rng = np.random.default_rng(0)
    y = np.linspace(0, 2 * np.pi, 32)
    x = np.linspace(0, 2 * np.pi, 32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    field = np.sin(2 * xx) * np.cos(3 * yy)
    lam = resolved_scale_km(field, field, dx_km=5.0)
    assert lam != lam or lam >= 0.0


def test_isotropic_psd_shape():
    field = torch.randn(3, 24, 24)
    psd, k = isotropic_psd_2d(field, dx=5.0)
    assert psd.shape == k.shape
    assert np.all(np.isfinite(psd))
    assert np.all(k >= 0)


def test_resolved_timescale_identical_series():
    t = np.sin(np.linspace(0, 8 * np.pi, 64))
    lam = resolved_timescale(t, t, dt=3600.0)
    assert lam != lam or lam >= 0.0


def test_lagrangian_separation_truth_is_zero():
    ny, nx = 16, 16
    u = np.ones((ny, nx)) * 0.1
    v = np.zeros((ny, nx))
    x0 = np.array([4.0, 8.0, 12.0])
    y0 = np.array([4.0, 8.0, 10.0])
    out = lagrangian_separation(u, v, x0, y0, n_steps=5, dt=60.0, dx=1000.0, dy=1000.0, u_truth=u, v_truth=v)
    assert out["sep_mean"][-1] < 1e-6
