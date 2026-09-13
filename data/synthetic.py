"""
Synthetic OSSE with QG-like SSH, geostrophic + ageostrophic currents, SST synergy.

Designed for unit tests and smoke training without NATL60 downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SyntheticOSSEConfig:
    n_time: int = 32
    height: int = 48
    width: int = 48
    dT: int = 7
    f0: float = 7.0e-5  # Coriolis ~40°N
    g: float = 9.81
    dx_deg: float = 0.05
    seed: int = 42
    altimetry_gap_fraction: float = 0.85
    sst_noise_std: float = 0.05


def _streamfunction_ssh(ny: int, nx: int, t: int, rng: np.random.Generator) -> np.ndarray:
    y = np.linspace(0, 2 * np.pi, ny)
    x = np.linspace(0, 2 * np.pi, nx)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    psi = (
        np.sin(xx + 0.3 * t)
        * np.cos(yy + 0.2 * t)
        + 0.4 * np.sin(2 * xx - 0.1 * t) * np.cos(3 * yy)
    )
    ssh = 0.1 * psi + 0.02 * rng.standard_normal((ny, nx))
    return ssh.astype(np.float32)


def _geostrophic_uv(ssh: np.ndarray, f: float, g: float, dy: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    dssh_dy = (np.roll(ssh, -1, axis=0) - np.roll(ssh, 1, axis=0)) / (2 * dy)
    dssh_dx = (np.roll(ssh, -1, axis=1) - np.roll(ssh, 1, axis=1)) / (2 * dx)
    u_g = -(g / f) * dssh_dy
    v_g = (g / f) * dssh_dx
    return u_g.astype(np.float32), v_g.astype(np.float32)


def _ageostrophic_uv(ny: int, nx: int, t: int, rng: np.random.Generator, scale: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Divergent ageostrophic component (paper target ~47% divergence recovery)."""
    ua = scale * np.sin(3 * np.linspace(0, 2 * np.pi, nx) + 0.5 * t)
    va = scale * np.cos(3 * np.linspace(0, 2 * np.pi, ny)[:, None] - 0.3 * t)
    ua = ua * np.ones((ny, nx)) + 0.01 * rng.standard_normal((ny, nx))
    va = va + 0.01 * rng.standard_normal((ny, nx))
    return ua.astype(np.float32), va.astype(np.float32)


def _sst_from_ssh(ssh: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SST ~ Laplacian(SSH) + noise (SQG-inspired synergy)."""
    lap = (
        np.roll(ssh, 1, 0)
        + np.roll(ssh, -1, 0)
        + np.roll(ssh, 1, 1)
        + np.roll(ssh, -1, 1)
        - 4 * ssh
    )
    return (0.5 * lap + 0.02 * rng.standard_normal(ssh.shape)).astype(np.float32)


def generate_synthetic_osse(cfg: SyntheticOSSEConfig, out_path: str | Path | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    ny, nx = cfg.height, cfg.width
    dy = dx = cfg.dx_deg * 111e3  # approx meters per grid cell

    ssh = np.zeros((cfg.n_time, ny, nx), dtype=np.float32)
    u = np.zeros_like(ssh)
    v = np.zeros_like(ssh)
    sst = np.zeros_like(ssh)

    for t in range(cfg.n_time):
        s = _streamfunction_ssh(ny, nx, t, rng)
        ug, vg = _geostrophic_uv(s, cfg.f0, cfg.g, dy, dx)
        ua, va = _ageostrophic_uv(ny, nx, t, rng)
        ssh[t] = s
        u[t] = ug + ua
        v[t] = vg + va
        sst[t] = _sst_from_ssh(s, rng)

    # Gappy altimetry mask (along-track-like stripes)
    mask_ssh = np.ones_like(ssh)
    for t in range(cfg.n_time):
        stripes = rng.random(nx) < cfg.altimetry_gap_fraction
        mask_ssh[t, :, stripes] = 0.0
        mask_ssh[t, rng.random((ny, nx)) < 0.3] = 0.0

    y_ssh = ssh * mask_ssh
    # DUACS-like OI: light smoothing of gappy field
    from scipy.ndimage import gaussian_filter

    y_oi = np.zeros_like(ssh)
    for t in range(cfg.n_time):
        filled = np.where(mask_ssh[t] > 0, ssh[t], np.nanmean(ssh[t]))
        y_oi[t] = gaussian_filter(filled, sigma=1.0)

    mask_sst = np.ones_like(sst)
    sst_obs = sst + cfg.sst_noise_std * rng.standard_normal(sst.shape).astype(np.float32)

    data = {
        "ssh": ssh,
        "u": u,
        "v": v,
        "sst": sst_obs,
        "sst_truth": sst,
        "y_ssh": y_oi,
        "y_oi": y_oi.copy(),  # synthetic path: OI-only == model SSH input
        "mask_ssh": mask_ssh.astype(np.float32),
        "mask_sst": mask_sst.astype(np.float32),
        "f": np.array([cfg.f0], dtype=np.float32),
        "dx": np.array([cfg.dx_deg], dtype=np.float32),
        "dx_m": np.array([dx], dtype=np.float32),
        "dy_m": np.array([dy], dtype=np.float32),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **data)
    return data


def load_synthetic_npz(path: str | Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))
