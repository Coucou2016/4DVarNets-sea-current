"""Physics consistency tests (QG relation, SQG, advection, divergence)."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic import SyntheticOSSEConfig, generate_synthetic_osse
from fourdvarnet.physics import (
    explained_variance,
    geostrophic_velocity,
    sqg_velocity,
    sst_advection_residual,
    strain_uncertainty,
)


def test_geostrophic_recovers_ssh_gradient():
    data = generate_synthetic_osse(SyntheticOSSEConfig(n_time=3, height=32, width=32, seed=0))
    ssh = torch.from_numpy(data["ssh"][0:1])
    f = float(data["f"][0])
    dx = float(data["dx"][0]) * 111e3
    u_g, v_g = geostrophic_velocity(ssh, f, dx=dx, dy=dx)
    # Non-periodic reference (matches geometry.grad_*)
    s = torch.from_numpy(data["ssh"][0:1])
    ddy = torch.empty_like(s)
    ddx = torch.empty_like(s)
    ddy[..., 1:-1, :] = (s[..., 2:, :] - s[..., :-2, :]) / (2 * dx)
    ddy[..., 0, :] = (s[..., 1, :] - s[..., 0, :]) / dx
    ddy[..., -1, :] = (s[..., -1, :] - s[..., -2, :]) / dx
    ddx[..., :, 1:-1] = (s[..., :, 2:] - s[..., :, :-2]) / (2 * dx)
    ddx[..., :, 0] = (s[..., :, 1] - s[..., :, 0]) / dx
    ddx[..., :, -1] = (s[..., :, -1] - s[..., :, -2]) / dx
    u_ref = -(9.81 / f) * ddy
    v_ref = (9.81 / f) * ddx
    assert torch.allclose(u_g, u_ref, atol=1e-4)
    assert torch.allclose(v_g, v_ref, atol=1e-4)


def test_total_currents_differ_from_geostrophic():
    data = generate_synthetic_osse(SyntheticOSSEConfig(n_time=5, height=24, width=24, seed=1))
    ssh = torch.from_numpy(data["ssh"][2:3])
    u, v = data["u"][2], data["v"][2]
    f, dx = float(data["f"][0]), float(data["dx"][0]) * 111e3
    u_g, v_g = geostrophic_velocity(ssh, f, dx=dx, dy=dx)
    diff = np.mean(np.abs(u - u_g.numpy()) + np.abs(v - v_g.numpy()))
    assert diff > 0.01, "ageostrophic component should be non-negligible"


def test_explained_variance_perfect():
    x = torch.randn(2, 3, 8, 8)
    tau = explained_variance(x, x)
    assert abs(tau - 1.0) < 1e-5


def test_sqg_uniform_sst_near_zero_currents():
    sst = torch.ones(1, 1, 32, 32)
    ssh = torch.ones(1, 1, 32, 32) * 0.1
    u, v = sqg_velocity(sst, ssh, Ld=30e3, f=7e-5, dx=5e3, dy=5e3)
    assert u.shape == sst.shape and v.shape == sst.shape
    assert torch.isfinite(u).all() and torch.isfinite(v).all()
    assert u.abs().max().item() < 1e-4
    assert v.abs().max().item() < 1e-4


def test_sqg_magnitude_comparable_to_geostrophy():
    data = generate_synthetic_osse(SyntheticOSSEConfig(n_time=3, height=32, width=32, seed=0))
    ssh = torch.from_numpy(data["ssh"][0:1])
    sst = torch.from_numpy(data["sst"][0:1])
    f = float(data["f"][0])
    dx = float(data["dx"][0]) * 111e3
    u_g, _ = geostrophic_velocity(ssh, f, dx=dx, dy=dx)
    u_s, v_s = sqg_velocity(sst, ssh, Ld=30e3, f=f, dx=dx, dy=dx)
    assert torch.isfinite(u_s).all()
    assert u_s.abs().mean().item() < 20.0 * max(u_g.abs().mean().item(), 1e-3)
    assert u_s.abs().mean().item() > 0.0


def test_sqg_even_odd_shapes_finite():
    for h, w in ((32, 32), (31, 33), (16, 17)):
        sst = torch.randn(2, 1, h, w)
        ssh = torch.randn(2, 1, h, w)
        u, v = sqg_velocity(sst, ssh, Ld=30e3, f=7e-5, dx=5e3, dy=5e3)
        assert u.shape == sst.shape
        assert torch.isfinite(u).all() and torch.isfinite(v).all()


def test_advection_uniform_t_uniform_flow_near_zero():
    t = torch.ones(2, 7, 16, 16) * 12.0
    u = torch.ones(2, 1, 16, 16) * 0.4
    v = torch.ones(2, 1, 16, 16) * -0.2
    r = sst_advection_residual(t, u, v, kappa=50.0, dt=86400.0, dx=5e3, dy=5e3)
    assert r.shape == (2, 16, 16)
    assert r.abs().max().item() < 1e-6


def test_advection_accepts_unbatched_sequence():
    t = torch.randn(5, 12, 12)
    u = torch.zeros(12, 12)
    v = torch.zeros(12, 12)
    r = sst_advection_residual(t, u, v, kappa=0.0, dt=1.0, dx=1.0, dy=1.0)
    assert r.shape == (12, 12)
    assert torch.isfinite(r).all()


def test_strain_uncertainty_at_least_sigma0():
    u = torch.randn(1, 1, 16, 16)
    v = torch.randn(1, 1, 16, 16)
    sig = strain_uncertainty(u, v, sigma0=0.05, alpha=1e4, dx=5e3, dy=5e3)
    assert torch.isfinite(sig).all()
    assert (sig >= 0.05 - 1e-8).all()
