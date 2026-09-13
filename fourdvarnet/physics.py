"""Geostrophic, eSQG-style, advection, and finite-difference operators.

Geostrophy follows Fablet et al. JAMES 2024 Eq. 5. The SQG helper is an
*effective* eSQG-style mix of SSH geostrophy and a spectral SST streamfunction
ψ̂(k) ∝ θ̂(k)/(k + 1/L_d). It is not a full 3D SQG inversion.

Spatial derivatives use **non-periodic** operators from ``fourdvarnet.geometry``
(shared by train and eval). SST advection uses a **final-time backward** scheme
for single-time velocity state: (T_t − T_{t−1})/Δt + u_t·∇T_t − κ∇²T_t.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fourdvarnet.geometry import grad_x, grad_y, laplacian_nonperiodic, stencil_valid_mask


def geostrophic_velocity(
    ssh: torch.Tensor,
    f: float | torch.Tensor,
    g: float = 9.81,
    dx: float | torch.Tensor = 1.0,
    dy: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SSH-derived geostrophic currents (Eq. 5): u_g = -(g/f)*d_y SSH, v_g = (g/f)*d_x SSH.

    ``f``, ``dx``, ``dy`` may be scalars or tensors broadcastable to ``ssh``.
    """
    dssh_dy = central_diff_y(ssh, dy)
    dssh_dx = central_diff_x(ssh, dx)
    u_g = -(g / f) * dssh_dy
    v_g = (g / f) * dssh_dx
    return u_g, v_g


def central_diff_x(field: torch.Tensor, dx: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Non-periodic ∂/∂x (alias of ``geometry.grad_x``)."""
    return grad_x(field, dx)


def central_diff_y(field: torch.Tensor, dy: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Non-periodic ∂/∂y (alias of ``geometry.grad_y``)."""
    return grad_y(field, dy)


def laplacian(field: torch.Tensor, dx: float | torch.Tensor = 1.0, dy: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Non-periodic five-point Laplacian (alias of ``geometry.laplacian_nonperiodic``)."""
    return laplacian_nonperiodic(field, dx, dy)


def _rfft2_wavenumbers(
    height: int,
    width: int,
    dx: float,
    dy: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2π-wavenumbers for rfft2. Works for even and odd H, W."""
    ky = (2.0 * math.pi * torch.fft.fftfreq(height, d=dy)).to(device=device, dtype=dtype)
    kx = (2.0 * math.pi * torch.fft.rfftfreq(width, d=dx)).to(device=device, dtype=dtype)
    ky = ky.view(height, 1)
    kx = kx.view(1, width // 2 + 1)
    k = torch.sqrt(kx**2 + ky**2)
    return kx, ky, k


def sqg_velocity(
    sst: torch.Tensor,
    ssh: torch.Tensor,
    Ld: float,
    f: float,
    g: float = 9.81,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Effective eSQG-style surface current from SST + SSH.

    Mixes SSH geostrophy with an SST-derived streamfunction whose spectrum is
        ψ̂(k) ∝ θ̂(k) / (k + 1/L_d)
    SST is first mapped to an effective height (RMS-matched to SSH, floor 0.05 m)
    so the filter ``1/(k L_d + 1)`` stays dimensionless. Implemented with
    ``rfft2`` / ``irfft2`` on the last two axes. Leading batch and channel
    dimensions are preserved. Uniform SST contributes ~zero SQG current
    (and uniform SSH ⇒ near-zero total). This is **not** a full 3D SQG inversion.

    Parameters
    ----------
    sst, ssh:
        Same spatial size; any leading dims, e.g. (H, W), (B, H, W), (B, 1, H, W).
    Ld:
        Deformation radius in the same length unit as ``dx`` / ``dy`` (meters).
    f, g:
        Coriolis and gravity; geostrophic branch uses the same constants.
    """
    sst, ssh = torch.broadcast_tensors(sst, ssh)
    orig_shape = sst.shape
    height, width = int(sst.shape[-2]), int(sst.shape[-1])
    sst_f = sst.reshape(-1, height, width)
    ssh_f = ssh.reshape(-1, height, width)

    u_g, v_g = geostrophic_velocity(ssh_f, f=f, g=g, dx=dx, dy=dy)

    ld = max(float(Ld), 1e-6)
    sst_a = sst_f - sst_f.mean(dim=(-2, -1), keepdim=True)
    ssh_a = ssh_f - ssh_f.mean(dim=(-2, -1), keepdim=True)
    sst_std = torch.sqrt(torch.mean(sst_a**2, dim=(-2, -1), keepdim=True) + 1e-16)
    ssh_std = torch.sqrt(torch.mean(ssh_a**2, dim=(-2, -1), keepdim=True) + 1e-16)
    # Map SST anomalies to an effective height (meters) so the SST branch is
    # in the same units as SSH geostrophy. Uniform SST => h_sst = 0.
    ref_h = torch.clamp(ssh_std.detach(), min=0.05)
    h_sst = torch.where(sst_std > 1e-8, sst_a * (ref_h / sst_std), torch.zeros_like(sst_a))

    h_hat = torch.fft.rfft2(h_sst)
    kx, ky, k = _rfft2_wavenumbers(
        height, width, dx, dy, device=sst.device, dtype=sst_f.dtype
    )
    # ψ̂(k) ∝ θ̂(k)/(k + 1/L_d) with θ the effective height; (g/f)/(k L_d + 1)
    # is the dimensionless eSQG filter times the geostrophic streamfunction scale.
    psi_hat = (g / f) * h_hat / (k * ld + 1.0)
    psi_hat = psi_hat.clone()
    psi_hat[:, 0, 0] = 0.0

    u_sst = torch.fft.irfft2((-1j * ky) * psi_hat, s=(height, width)).to(dtype=sst_f.dtype)
    v_sst = torch.fft.irfft2((1j * kx) * psi_hat, s=(height, width)).to(dtype=sst_f.dtype)

    u = (u_g + u_sst).reshape(orig_shape)
    v = (v_g + v_sst).reshape(orig_shape)
    return u, v


def _to_bhw(field: torch.Tensor, batch: int, height: int, width: int, time_index: int | None) -> torch.Tensor:
    """Coerce u/v to (B, H, W). 4-D tensors with a time axis use ``time_index``."""
    t = field
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim == 4:
        # (B, C_or_T, H, W)
        if t.shape[1] == 1:
            t = t[:, 0]
        elif time_index is not None:
            idx = time_index if time_index >= 0 else t.shape[1] + time_index
            idx = min(max(idx, 0), t.shape[1] - 1)
            t = t[:, idx]
        else:
            t = t[:, -1]
    if t.ndim != 3:
        raise ValueError(f"expected u/v to reduce to (B,H,W), got {tuple(field.shape)}")
    if t.shape[-2:] != (height, width):
        raise ValueError(f"u/v spatial shape {tuple(t.shape[-2:])} != {(height, width)}")
    if t.shape[0] == 1 and batch > 1:
        t = t.expand(batch, -1, -1)
    return t


def sst_advection_residual(
    sst_seq: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    kappa: float,
    dt: float,
    dx: float | torch.Tensor,
    dy: float | torch.Tensor,
    mask_sst: torch.Tensor | None = None,
) -> torch.Tensor:
    """Heat-budget residual at final analysis time (single-time velocity state).

    Scheme (backward in time, collocated with ``u_t, v_t``)::

        (T_t − T_{t−1}) / Δt + u_t · ∇T_t − κ ∇²T_t

    ``sst_seq`` is (B, dT, H, W) or (dT, H, W). When ``mask_sst`` is provided
    with the same time axis, the residual is zeroed wherever the time stencil
    (t and t−1) or the spatial FD stencil is invalid.
    """
    squeeze_batch = sst_seq.ndim == 3
    if squeeze_batch:
        sst_seq = sst_seq.unsqueeze(0)
    if sst_seq.ndim != 4:
        raise ValueError(f"sst_seq must be (B,dT,H,W) or (dT,H,W), got {tuple(sst_seq.shape)}")

    batch, d_t, height, width = sst_seq.shape
    if d_t < 2:
        z = sst_seq.new_zeros(batch, height, width)
        return z[0] if squeeze_batch else z

    # Final-time backward: align ∂t and advection with analysis velocity (u_t, v_t).
    t_field = sst_seq[:, -1]
    dt_t = (sst_seq[:, -1] - sst_seq[:, -2]) / dt

    u_b = _to_bhw(u, batch, height, width, time_index=-1)
    v_b = _to_bhw(v, batch, height, width, time_index=-1)
    tx = central_diff_x(t_field, dx)
    ty = central_diff_y(t_field, dy)
    residual = dt_t + u_b * tx + v_b * ty - float(kappa) * laplacian(t_field, dx, dy)

    if mask_sst is not None:
        m = mask_sst
        if m.ndim == 3:
            m = m.unsqueeze(0)
        if m.ndim != 4:
            raise ValueError(f"mask_sst must be (B,dT,H,W) or (dT,H,W), got {tuple(mask_sst.shape)}")
        if m.shape[0] == 1 and batch > 1:
            m = m.expand(batch, -1, -1, -1)
        m_t = m[:, -1] > 0.5
        m_tm1 = m[:, -2] > 0.5
        spatial_ok = stencil_valid_mask(m_t)
        valid = m_t & m_tm1 & spatial_ok
        residual = torch.where(valid, residual, residual.new_zeros(()))

    return residual[0] if squeeze_batch else residual


def strain_uncertainty(
    u: torch.Tensor,
    v: torch.Tensor,
    sigma0: float,
    alpha: float,
    dx: float | torch.Tensor = 1.0,
    dy: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Strain-aware spatial reweighting scale σ = σ0 (1 + α strain).

    This is **not** a learned uncertainty head (no σ network). M4 uses it only
    to down-weight high-strain cells in the supervised UV term. Always ≥ σ0
    for α ≥ 0. Callers should pass truth velocities or detached predictions so
    the optimizer cannot inflate strain to shrink the NLL (M4 collapse mode).
    """
    return float(sigma0) * (1.0 + float(alpha) * strain(u, v, dx, dy))


class FixedGradient2d(nn.Module):
    """Non-trainable Sobel-like gradients for SSH (loss L_grad_SSH)."""

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        ky = kx.t()
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = x.shape[1]
        # Match input device/dtype (TrainingLoss buffers must follow .to(device)).
        kx = self.kx.to(device=x.device, dtype=x.dtype).expand(c, 1, 3, 3)
        ky = self.ky.to(device=x.device, dtype=x.dtype).expand(c, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1, groups=c)
        gy = F.conv2d(x, ky, padding=1, groups=c)
        return gx, gy


class FixedDivergence2d(nn.Module):
    """Non-trainable divergence for loss L_div (Eq. 13). Uses shared non-periodic grads."""

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        dx: float | torch.Tensor = 1.0,
        dy: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        du_dx = central_diff_x(u, dx)
        dv_dy = central_diff_y(v, dy)
        return du_dx + dv_dy


def vorticity(
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float | torch.Tensor = 1.0,
    dy: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    return central_diff_x(v, dx) - central_diff_y(u, dy)


def strain(
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float | torch.Tensor = 1.0,
    dy: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    du_dx = central_diff_x(u, dx)
    dv_dy = central_diff_y(v, dy)
    du_dy = central_diff_y(u, dy)
    dv_dx = central_diff_x(v, dx)
    return torch.sqrt((du_dx - dv_dy) ** 2 + (du_dy + dv_dx) ** 2 + 1e-12)


def explained_variance(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """τ metric: 1 - MSE/Var(truth)."""
    err = pred - truth
    mse = torch.mean(err**2).item()
    var = torch.var(truth).item()
    if var < 1e-12:
        return float("nan")
    return 1.0 - mse / var
