"""Lat-dependent Coriolis / metric spacing and non-periodic finite differences.

Train and eval share these operators (imported by ``physics`` and the dataset).
Gulf Stream / NATL60 boxes are **not** periodic; ``torch.roll`` is not used here.
"""

from __future__ import annotations

import math

import torch

# SI constants
OMEGA_EARTH = 7.292115e-5  # rad/s
R_EARTH = 6_371_000.0  # m
DEG_TO_RAD = math.pi / 180.0


def coriolis(lat_deg: torch.Tensor | float) -> torch.Tensor | float:
    """f = 2 Ω sin(φ). ``lat_deg`` in degrees; returns same container type."""
    if isinstance(lat_deg, torch.Tensor):
        return (2.0 * OMEGA_EARTH) * torch.sin(lat_deg * DEG_TO_RAD)
    return (2.0 * OMEGA_EARTH) * math.sin(float(lat_deg) * DEG_TO_RAD)


def metric_dx(dlon_deg: float | torch.Tensor, lat_deg: torch.Tensor | float) -> torch.Tensor | float:
    """Eastward grid spacing (m) for longitude step ``dlon_deg`` at latitude ``lat_deg``."""
    if isinstance(lat_deg, torch.Tensor) or isinstance(dlon_deg, torch.Tensor):
        lat_t = lat_deg if isinstance(lat_deg, torch.Tensor) else torch.as_tensor(lat_deg)
        dlon_t = dlon_deg if isinstance(dlon_deg, torch.Tensor) else torch.as_tensor(dlon_deg, dtype=lat_t.dtype)
        return dlon_t * R_EARTH * torch.cos(lat_t * DEG_TO_RAD) * DEG_TO_RAD
    return float(dlon_deg) * R_EARTH * math.cos(float(lat_deg) * DEG_TO_RAD) * DEG_TO_RAD


def metric_dy(dlat_deg: float | torch.Tensor) -> torch.Tensor | float:
    """Northward grid spacing (m) for latitude step ``dlat_deg``."""
    if isinstance(dlat_deg, torch.Tensor):
        return dlat_deg * R_EARTH * DEG_TO_RAD
    return float(dlat_deg) * R_EARTH * DEG_TO_RAD


def _as_spacing(spacing: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor | float:
    if isinstance(spacing, torch.Tensor):
        return spacing.to(device=ref.device, dtype=ref.dtype)
    return spacing


def _broadcast_metric(spacing: float | torch.Tensor, field: torch.Tensor) -> torch.Tensor | float:
    """Broadcast scalar/map spacing to ``field.shape`` (supports lat-row or HxW maps).

    Accepts scalars, ``(..., H, 1)``, ``(..., 1, W)``, or full ``(..., H, W)`` tensors.
    Leading dims are padded/truncated to match ``field.ndim`` before broadcast.
    """
    if not isinstance(spacing, torch.Tensor):
        return spacing
    s = spacing.to(device=field.device, dtype=field.dtype)
    while s.ndim < field.ndim:
        s = s.unsqueeze(0)
    if s.ndim > field.ndim:
        s = s.reshape(*s.shape[-field.ndim :])
    try:
        return torch.broadcast_to(s, field.shape)
    except RuntimeError as exc:
        raise RuntimeError(
            f"cannot broadcast spacing shape {tuple(spacing.shape)} to field {tuple(field.shape)}"
        ) from exc


def grad_x(field: torch.Tensor, dx: float | torch.Tensor = 1.0) -> torch.Tensor:
    """∂/∂x with interior central differences and one-sided boundaries (no wrap).

    ``dx`` may be a scalar or a metric map broadcastable to ``field`` (e.g. lat-row
    ``(..., H, 1)`` or full ``(..., H, W)``). Denominators are **sliced** to the
    same spatial cells as the numerator (avoids HxW broadcast bugs).
    """
    dx_b = _broadcast_metric(dx, field)
    out = torch.empty_like(field)
    if isinstance(dx_b, torch.Tensor):
        # Interior: (f[..., i+1] - f[..., i-1]) / (2 dx_i)
        out[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) / (2.0 * dx_b[..., :, 1:-1])
        out[..., :, 0] = (field[..., :, 1] - field[..., :, 0]) / dx_b[..., :, 0]
        out[..., :, -1] = (field[..., :, -1] - field[..., :, -2]) / dx_b[..., :, -1]
    else:
        out[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) / (2.0 * dx_b)
        out[..., :, 0] = (field[..., :, 1] - field[..., :, 0]) / dx_b
        out[..., :, -1] = (field[..., :, -1] - field[..., :, -2]) / dx_b
    return out


def grad_y(field: torch.Tensor, dy: float | torch.Tensor = 1.0) -> torch.Tensor:
    """∂/∂y with interior central differences and one-sided boundaries (no wrap).

    ``dy`` may be a scalar or a metric map; denominators are sliced like ``grad_x``.
    """
    dy_b = _broadcast_metric(dy, field)
    out = torch.empty_like(field)
    if isinstance(dy_b, torch.Tensor):
        out[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dy_b[..., 1:-1, :])
        out[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / dy_b[..., 0, :]
        out[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / dy_b[..., -1, :]
    else:
        out[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dy_b)
        out[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / dy_b
        out[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / dy_b
    return out


def laplacian_nonperiodic(field: torch.Tensor, dx: float | torch.Tensor = 1.0, dy: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Five-point Laplacian with one-sided second differences at boundaries."""
    dx_b = _broadcast_metric(dx, field)
    dy_b = _broadcast_metric(dy, field)
    dxx = torch.empty_like(field)
    dyy = torch.empty_like(field)
    if isinstance(dx_b, torch.Tensor):
        dx_i = dx_b[..., :, 1:-1]
        dxx[..., :, 1:-1] = (field[..., :, 2:] - 2.0 * field[..., :, 1:-1] + field[..., :, :-2]) / (dx_i * dx_i)
        dx0 = dx_b[..., :, 0]
        dxn = dx_b[..., :, -1]
        dxx[..., :, 0] = (field[..., :, 0] - 2.0 * field[..., :, 1] + field[..., :, 2]) / (dx0 * dx0)
        dxx[..., :, -1] = (field[..., :, -1] - 2.0 * field[..., :, -2] + field[..., :, -3]) / (dxn * dxn)
    else:
        dxx[..., :, 1:-1] = (field[..., :, 2:] - 2.0 * field[..., :, 1:-1] + field[..., :, :-2]) / (dx_b * dx_b)
        dxx[..., :, 0] = (field[..., :, 0] - 2.0 * field[..., :, 1] + field[..., :, 2]) / (dx_b * dx_b)
        dxx[..., :, -1] = (field[..., :, -1] - 2.0 * field[..., :, -2] + field[..., :, -3]) / (dx_b * dx_b)
    if isinstance(dy_b, torch.Tensor):
        dy_i = dy_b[..., 1:-1, :]
        dyy[..., 1:-1, :] = (field[..., 2:, :] - 2.0 * field[..., 1:-1, :] + field[..., :-2, :]) / (dy_i * dy_i)
        dy0 = dy_b[..., 0, :]
        dyn = dy_b[..., -1, :]
        dyy[..., 0, :] = (field[..., 0, :] - 2.0 * field[..., 1, :] + field[..., 2, :]) / (dy0 * dy0)
        dyy[..., -1, :] = (field[..., -1, :] - 2.0 * field[..., -2, :] + field[..., -3, :]) / (dyn * dyn)
    else:
        dyy[..., 1:-1, :] = (field[..., 2:, :] - 2.0 * field[..., 1:-1, :] + field[..., :-2, :]) / (dy_b * dy_b)
        dyy[..., 0, :] = (field[..., 0, :] - 2.0 * field[..., 1, :] + field[..., 2, :]) / (dy_b * dy_b)
        dyy[..., -1, :] = (field[..., -1, :] - 2.0 * field[..., -2, :] + field[..., -3, :]) / (dy_b * dy_b)
    return dxx + dyy


def stencil_valid_mask(mask: torch.Tensor) -> torch.Tensor:
    """True where a 5-point spatial stencil is fully valid (and mask itself).

    For interior points requires N/S/E/W neighbors; boundaries require the
    one-sided neighbour used by ``grad_x`` / ``grad_y`` / Laplacian.
    """
    m = mask.to(dtype=torch.bool)
    # Neighbour presence (False outside domain)
    zeros = torch.zeros_like(m)
    m_e = torch.cat([m[..., :, 1:], zeros[..., :, :1]], dim=-1)
    m_w = torch.cat([zeros[..., :, :1], m[..., :, :-1]], dim=-1)
    m_n = torch.cat([m[..., 1:, :], zeros[..., :1, :]], dim=-2)
    m_s = torch.cat([zeros[..., :1, :], m[..., :-1, :]], dim=-2)

    out = m.clone()
    # Interior: need all four neighbours
    out[..., 1:-1, 1:-1] = (
        m[..., 1:-1, 1:-1]
        & m_e[..., 1:-1, 1:-1]
        & m_w[..., 1:-1, 1:-1]
        & m_n[..., 1:-1, 1:-1]
        & m_s[..., 1:-1, 1:-1]
    )
    # Edges: require the one-sided neighbour(s) used by FD
    out[..., 1:-1, 0] = m[..., 1:-1, 0] & m_e[..., 1:-1, 0] & m_n[..., 1:-1, 0] & m_s[..., 1:-1, 0]
    out[..., 1:-1, -1] = m[..., 1:-1, -1] & m_w[..., 1:-1, -1] & m_n[..., 1:-1, -1] & m_s[..., 1:-1, -1]
    out[..., 0, 1:-1] = m[..., 0, 1:-1] & m_n[..., 0, 1:-1] & m_e[..., 0, 1:-1] & m_w[..., 0, 1:-1]
    out[..., -1, 1:-1] = m[..., -1, 1:-1] & m_s[..., -1, 1:-1] & m_e[..., -1, 1:-1] & m_w[..., -1, 1:-1]
    # Corners
    out[..., 0, 0] = m[..., 0, 0] & m_e[..., 0, 0] & m_n[..., 0, 0]
    out[..., 0, -1] = m[..., 0, -1] & m_w[..., 0, -1] & m_n[..., 0, -1]
    out[..., -1, 0] = m[..., -1, 0] & m_e[..., -1, 0] & m_s[..., -1, 0]
    out[..., -1, -1] = m[..., -1, -1] & m_w[..., -1, -1] & m_s[..., -1, -1]
    return out


def grid_metrics_from_latlon(
    lat: torch.Tensor | "np.ndarray",
    lon: torch.Tensor | "np.ndarray",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(f, dx_m, dy_m)`` broadcastable to (H, W) from 1-D or 2-D lat/lon."""
    import numpy as np

    lat_t = torch.as_tensor(np.asarray(lat), dtype=torch.float32)
    lon_t = torch.as_tensor(np.asarray(lon), dtype=torch.float32)
    if lat_t.ndim == 1 and lon_t.ndim == 1:
        dlat = float((lat_t[1] - lat_t[0]).abs().item()) if lat_t.numel() > 1 else 0.05
        dlon = float((lon_t[1] - lon_t[0]).abs().item()) if lon_t.numel() > 1 else 0.05
        f = coriolis(lat_t).view(-1, 1).expand(lat_t.numel(), lon_t.numel())
        dx = metric_dx(dlon, lat_t).view(-1, 1).expand_as(f)
        dy = torch.full_like(f, float(metric_dy(dlat)))
        return f, dx, dy
    if lat_t.ndim == 2 and lon_t.ndim == 2:
        dlat = (lat_t[1:, :] - lat_t[:-1, :]).abs().mean().clamp_min(1e-6)
        dlon = (lon_t[:, 1:] - lon_t[:, :-1]).abs().mean().clamp_min(1e-6)
        f = coriolis(lat_t)
        dx = metric_dx(dlon, lat_t)
        dy = torch.full_like(f, float(metric_dy(dlat)))
        return f, dx, dy
    raise ValueError(f"expected 1-D or 2-D lat/lon, got {tuple(lat_t.shape)} / {tuple(lon_t.shape)}")
