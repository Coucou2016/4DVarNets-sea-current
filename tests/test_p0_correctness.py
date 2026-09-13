"""Phase-1 P0 correctness tests (peer-review blockers)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.natl60 import check_natl60_paths, load_natl60
from fourdvarnet.geometry import grad_x
from fourdvarnet.physics import sst_advection_residual
from fourdvarnet.solver import Solver4DVarNet, Solver4DVarNetTruncated, _masked_mse
from fourdvarnet.model import FourDVarNetUV


def test_paper_mode_requires_obs_and_oi():
    paths = {
        "ssh_ref": str(ROOT / "missing_ssh.nc"),
        "sst_ref": str(ROOT / "missing_sst.nc"),
        "u_ref": str(ROOT / "missing_u.nc"),
        "v_ref": str(ROOT / "missing_v.nc"),
    }
    missing = check_natl60_paths(paths, allow_truth_background=False)
    assert any("obs" in m for m in missing)
    assert any("oi" in m for m in missing)
    # Debug mode drops obs/oi requirement
    missing_dbg = check_natl60_paths(paths, allow_truth_background=True)
    assert not any(m.startswith("obs:") for m in missing_dbg)
    assert not any(m.startswith("oi:") for m in missing_dbg)


def test_no_truth_background_fallback_raises(tmp_path):
    """Mock loader path: refs exist, oi/obs absent → RuntimeError, never y=truth."""
    # Create dummy files so path check for refs passes when allow_truth_background=True
    # but paper mode still requires obs/oi at check_natl60_paths.
    refs = {}
    for k in ("ssh_ref", "sst_ref", "u_ref", "v_ref"):
        p = tmp_path / f"{k}.nc"
        p.write_text("dummy")
        refs[k] = str(p)
    with pytest.raises(FileNotFoundError, match="obs|oi|paper mode|required"):
        load_natl60(refs, allow_truth_background=False)


def test_truth_background_only_with_explicit_flag():
    """allow_truth_background=False is the default on check_natl60_paths."""
    missing = check_natl60_paths({})
    assert any("obs" in m for m in missing)


def test_masked_mse_denominator_valid_only():
    # residual already zeroed by mask; zeros must not dilute the mean
    residual = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    # mean of 1² over 1 valid cell = 1.0 (not 0.25)
    assert abs(_masked_mse(residual, mask).item() - 1.0) < 1e-6
    # unmasked residual + mask form: sum(r²*m)/sum(m)
    err = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    m = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    expected = (4.0 + 9.0) / 2.0
    assert abs(_masked_mse(err, m).item() - expected) < 1e-6
    # empty mask → 0
    assert _masked_mse(err, torch.zeros_like(m)).item() == 0.0


def test_nonperiodic_grad_boundary_no_wrap():
    # Field increases only in x; periodic roll would wrap and pollute boundaries.
    field = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5).expand(1, 1, 3, 5).clone()
    field = field + torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1)
    gx = grad_x(field, dx=1.0)
    # Interior central ≈ 1
    assert torch.allclose(gx[..., 1:-1], torch.ones_like(gx[..., 1:-1]), atol=1e-5)
    # Boundaries one-sided ≈ 1, and must NOT equal wrap-around value
    assert torch.allclose(gx[..., :, 0], torch.ones_like(gx[..., :, 0]), atol=1e-5)
    assert torch.allclose(gx[..., :, -1], torch.ones_like(gx[..., :, -1]), atol=1e-5)
    # Contrast: torch.roll periodic would give (f0 - f_{-1})/2 = (0 - 4)/2 = -2 at west
    periodic_west = (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1)) / 2.0
    assert not torch.allclose(gx[..., :, 0], periodic_west[..., :, 0])


def test_advection_final_time_backward_scheme():
    # Construct T so that (T_t - T_{t-1})/dt = 1, u=v=0, kappa=0 → residual = 1
    b, dT, h, w = 1, 4, 8, 8
    sst = torch.zeros(b, dT, h, w)
    sst[:, -1] = 2.0
    sst[:, -2] = 1.0
    u = torch.zeros(b, 1, h, w)
    v = torch.zeros(b, 1, h, w)
    r = sst_advection_residual(sst, u, v, kappa=0.0, dt=1.0, dx=1.0, dy=1.0)
    assert torch.allclose(r, torch.ones_like(r), atol=1e-5)

    # Mask: invalidate t-1 → residual must be zeroed
    mask = torch.ones(b, dT, h, w)
    mask[:, -2] = 0.0
    r2 = sst_advection_residual(sst, u, v, kappa=0.0, dt=1.0, dx=1.0, dy=1.0, mask_sst=mask)
    assert r2.abs().max().item() < 1e-8


def test_solver_unrolled_create_graph_no_detach_smoke():
    """Faithful solver: create_graph path runs and grads flow past iter 0."""
    torch.manual_seed(0)
    model = FourDVarNetUV(dT_sst=3, n_iter=2, hidden_lstm=8, feat_dim=4, use_sst=True)
    y = torch.randn(1, 1, 8, 8, requires_grad=True)
    z = torch.randn(1, 3, 8, 8)
    m_ssh = torch.ones(1, 1, 8, 8)
    m_sst = torch.ones(1, 3, 8, 8)
    u_g = torch.zeros(1, 1, 8, 8)
    v_g = torch.zeros(1, 1, 8, 8)
    pred = model(y, z, m_ssh, m_sst, u_g, v_g)
    loss = (pred**2).mean()
    loss.backward()
    # Gradients reached model parameters (unrolled graph wired)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_truncated_solver_exists_as_ablation():
    assert issubclass(Solver4DVarNetTruncated, Solver4DVarNet)
    s = Solver4DVarNetTruncated(n_channels=3, n_iter=2, hidden_lstm=8, feat_dim=4, dT_sst=3)
    x0 = torch.randn(1, 3, 8, 8)
    y = torch.randn(1, 1, 8, 8)
    z = torch.randn(1, 3, 8, 8)
    out = s(x0, y, z, torch.ones(1, 1, 8, 8), torch.ones(1, 3, 8, 8))
    assert out.shape == x0.shape


def test_dataset_uses_full_sst_mask_window():
    from data.dataset import SSTSSHCurrentDataset

    dT = 4
    n, h, w = 10, 6, 6
    mask_sst = np.zeros((n, h, w), dtype=np.float32)
    # Distinct pattern per time so broadcast-from-last would fail equality check
    for t in range(n):
        mask_sst[t] = float(t + 1)
    data = {
        "ssh": np.zeros((n, h, w), dtype=np.float32),
        "u": np.zeros((n, h, w), dtype=np.float32),
        "v": np.zeros((n, h, w), dtype=np.float32),
        "sst": np.zeros((n, h, w), dtype=np.float32),
        "y_ssh": np.zeros((n, h, w), dtype=np.float32),
        "mask_ssh": np.ones((n, h, w), dtype=np.float32),
        "mask_sst": mask_sst,
        "f": np.array([7e-5], dtype=np.float32),
        "dx": np.array([0.05], dtype=np.float32),
    }
    ds = SSTSSHCurrentDataset(data, dT=dT, time_indices=[dT - 1])
    batch = ds[0]
    assert batch["mask_sst"].shape[0] == dT
    # Last window times are 0..dT-1 → mask values 1..dT
    assert torch.allclose(
        batch["mask_sst"][:, 0, 0],
        torch.arange(1, dT + 1, dtype=torch.float32),
    )
