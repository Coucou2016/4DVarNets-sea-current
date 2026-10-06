"""Evaluation metrics aligned with Le Guillou et al. (2020) / JAMES paper §4.3."""

from __future__ import annotations

import numpy as np
import torch

from fourdvarnet.physics import (
    central_diff_x,
    central_diff_y,
    explained_variance,
    strain,
    vorticity,
)


def rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - truth) ** 2)).item()


def _as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def isotropic_psd_2d(field: torch.Tensor | np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged 2D power spectrum (Hann window, rfft2).

    ``field`` is (..., H, W). Leading axes are averaged (Welch-like segments).
    Returns ``(psd, k)`` with ``k`` in rad / (same unit as ``dx``).
    """
    arr = np.asarray(_as_numpy(field), dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError("isotropic_psd_2d expects at least a 2D field")
    height, width = int(arr.shape[-2]), int(arr.shape[-1])
    win = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    stacked = arr.reshape(-1, height, width)
    acc = np.zeros((height, width // 2 + 1), dtype=np.float64)
    for slab in stacked:
        spec = np.fft.rfft2(slab * win)
        acc += np.abs(spec) ** 2
    acc /= max(len(stacked), 1)

    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dx)
    kx = 2.0 * np.pi * np.fft.rfftfreq(width, d=dx)
    k_map = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    k_max = float(min(np.max(np.abs(kx)), np.max(np.abs(ky))))
    n_bins = max(min(height, width) // 2, 8)
    edges = np.linspace(0.0, k_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    psd = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        mask = (k_map >= edges[i]) & (k_map < edges[i + 1])
        if mask.any():
            psd[i] = acc[mask].mean()
    return psd, centers


def error_signal_min_ratio(
    err_psd: np.ndarray,
    signal_psd: np.ndarray,
    k: np.ndarray,
) -> float:
    """Minimum finite err/signal PSD ratio over positive wavenumbers.

    Useful when ``resolved_scale_from_psd`` returns NaN (no bin below threshold):
    ``min_ratio`` still shows how far the spectrum is from the 0.5 cutoff.
    """
    err_psd = np.asarray(err_psd, dtype=np.float64)
    signal_psd = np.asarray(signal_psd, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    ratio = err_psd / (signal_psd + 1e-12)
    valid = np.isfinite(ratio) & (k > 0) & (signal_psd > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.min(ratio[valid]))


def resolved_scale_from_psd(
    err_psd: np.ndarray,
    signal_psd: np.ndarray,
    k: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Wavelength (2π/k, same length unit as 1/k) of the finest still-resolved scale.

    Le Guillou style: among wavenumbers with err/signal < threshold, take the
    largest k (smallest spatial scale). Returns NaN if **no** bin meets the
    threshold (field unresolved at all analysed scales) — pair with
    ``error_signal_min_ratio`` for a finite diagnostic.
    """
    err_psd = np.asarray(err_psd, dtype=np.float64)
    signal_psd = np.asarray(signal_psd, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    ratio = err_psd / (signal_psd + 1e-12)
    valid = np.isfinite(ratio) & (k > 0) & (signal_psd > 0)
    below = valid & (ratio < threshold)
    if not np.any(below):
        return float("nan")
    k_star = float(np.max(k[below]))
    if k_star <= 0:
        return float("nan")
    return float(2.0 * np.pi / k_star)


def resolved_scale_km(
    pred: torch.Tensor | np.ndarray,
    truth: torch.Tensor | np.ndarray | None = None,
    dx_km: float | None = None,
    threshold: float = 0.5,
    k: np.ndarray | None = None,
) -> float:
    """Minimum spatial scale (km) where error PSD / signal PSD < ``threshold``.

    Preferred call: ``resolved_scale_km(pred, truth, dx_km)``.
    Also accepts precomputed ``(err_psd, k)`` as the first two positional args
    when ``truth`` is a wavenumber array (legacy helper).
    """
    if truth is not None and dx_km is None and k is None and np.ndim(truth) == 1 and np.ndim(pred) == 1:
        # Legacy: resolved_scale_km(err_psd, k, dx_km=..., threshold=...)
        raise TypeError("pass dx_km as a keyword when using precomputed PSDs")

    if k is not None and truth is not None and np.ndim(pred) == 1 and np.ndim(truth) == 1:
        lam = resolved_scale_from_psd(np.asarray(pred), np.asarray(truth), np.asarray(k), threshold)
        return float(lam * dx_km) if dx_km is not None else lam

    if truth is None or dx_km is None:
        raise TypeError("resolved_scale_km(pred, truth, dx_km) requires fields and grid scale")

    err = _as_numpy(pred) - _as_numpy(truth)
    err_psd, k_rad = isotropic_psd_2d(err, dx_km)
    sig_psd, _ = isotropic_psd_2d(truth, dx_km)
    return resolved_scale_from_psd(err_psd, sig_psd, k_rad, threshold)


def resolved_scale_report(
    pred: torch.Tensor | np.ndarray,
    truth: torch.Tensor | np.ndarray,
    dx_km: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """λ_x plus ``min_ratio`` so unresolved (NaN λ_x) cases stay interpretable."""
    err = _as_numpy(pred) - _as_numpy(truth)
    err_psd, k_rad = isotropic_psd_2d(err, dx_km)
    sig_psd, _ = isotropic_psd_2d(truth, dx_km)
    lam = resolved_scale_from_psd(err_psd, sig_psd, k_rad, threshold)
    return {
        "lambda_km": float(lam),
        "min_ratio": error_signal_min_ratio(err_psd, sig_psd, k_rad),
        "resolved": 1.0 if lam == lam else 0.0,  # NaN ⇒ unresolved
    }


def resolved_timescale(
    pred_ts: torch.Tensor | np.ndarray,
    truth_ts: torch.Tensor | np.ndarray,
    dt: float,
    threshold: float = 0.5,
) -> float:
    """1D analogue of λ_x: period where error/signal PSD ratio stays below threshold."""
    pred_a = np.asarray(_as_numpy(pred_ts), dtype=np.float64).reshape(-1)
    truth_a = np.asarray(_as_numpy(truth_ts), dtype=np.float64).reshape(-1)
    n = min(len(pred_a), len(truth_a))
    if n < 8:
        return float("nan")
    pred_a, truth_a = pred_a[:n], truth_a[:n]
    win = np.hanning(n)
    freq = np.fft.rfftfreq(n, d=dt)
    omega = 2.0 * np.pi * freq
    err_psd = np.abs(np.fft.rfft((pred_a - truth_a) * win)) ** 2
    sig_psd = np.abs(np.fft.rfft(truth_a * win)) ** 2
    return resolved_scale_from_psd(err_psd, sig_psd, omega, threshold)


def _bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample (H, W) at fractional indices (x along W, y along H), periodic wrap."""
    height, width = field.shape[-2], field.shape[-1]
    xw = np.mod(x, width)
    yw = np.mod(y, height)
    x0 = np.floor(xw).astype(np.int64)
    y0 = np.floor(yw).astype(np.int64)
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    sx = xw - x0
    sy = yw - y0
    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]
    return (1 - sy) * ((1 - sx) * f00 + sx * f10) + sy * ((1 - sx) * f01 + sx * f11)


def _rk2_step(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    u1 = _bilinear_sample(u, x, y)
    v1 = _bilinear_sample(v, x, y)
    x_m = x + 0.5 * dt * u1 / dx
    y_m = y + 0.5 * dt * v1 / dy
    u2 = _bilinear_sample(u, x_m, y_m)
    v2 = _bilinear_sample(v, x_m, y_m)
    x_n = x + dt * u2 / dx
    y_n = y + dt * v2 / dy
    return x_n, y_n


def _uv_frame(uv: np.ndarray, t: int) -> np.ndarray:
    if uv.ndim == 2:
        return uv
    return uv[min(t, uv.shape[0] - 1)]


def lagrangian_separation(
    u: torch.Tensor | np.ndarray,
    v: torch.Tensor | np.ndarray,
    x0: torch.Tensor | np.ndarray,
    y0: torch.Tensor | np.ndarray,
    n_steps: int,
    dt: float,
    dx: float,
    dy: float,
    u_truth: torch.Tensor | np.ndarray | None = None,
    v_truth: torch.Tensor | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """RK2 particle advection on a regular grid.

    Positions ``x0, y0`` are fractional grid indices (x along width).
    ``u, v`` are m/s with shape (H, W) or (T, H, W). If truth currents are
    given, also advect a twin set and return mean separation (meters) vs time.
    """
    u_a = np.asarray(_as_numpy(u), dtype=np.float64)
    v_a = np.asarray(_as_numpy(v), dtype=np.float64)
    x = np.asarray(_as_numpy(x0), dtype=np.float64).reshape(-1).copy()
    y = np.asarray(_as_numpy(y0), dtype=np.float64).reshape(-1).copy()
    x_t = x.copy()
    y_t = y.copy()
    have_truth = u_truth is not None and v_truth is not None
    if have_truth:
        u_gt = np.asarray(_as_numpy(u_truth), dtype=np.float64)
        v_gt = np.asarray(_as_numpy(v_truth), dtype=np.float64)
    else:
        u_gt = v_gt = None

    sep = np.zeros(n_steps + 1, dtype=np.float64)
    height, width = u_a.shape[-2], u_a.shape[-1]
    for t in range(n_steps):
        x, y = _rk2_step(_uv_frame(u_a, t), _uv_frame(v_a, t), x, y, dt, dx, dy)
        if have_truth:
            x_t, y_t = _rk2_step(_uv_frame(u_gt, t), _uv_frame(v_gt, t), x_t, y_t, dt, dx, dy)
            dxi = np.minimum(np.abs(x - x_t), width - np.abs(x - x_t))
            dyi = np.minimum(np.abs(y - y_t), height - np.abs(y - y_t))
            sep[t + 1] = float(np.mean(np.hypot(dxi * dx, dyi * dy)))
    out: dict[str, np.ndarray] = {"x": x, "y": y, "sep_mean": sep}
    if have_truth:
        out["x_truth"] = x_t
        out["y_truth"] = y_t
    return out


def batch_metrics(
    pred: torch.Tensor,
    truth: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    dx_km: float | None = None,
) -> dict[str, float]:
    """pred/truth: (B, 3, H, W) — SSH, u, v."""
    u_p, v_p = pred[:, 1], pred[:, 2]
    u_t, v_t = truth[:, 1], truth[:, 2]
    curl_p = vorticity(u_p, v_p, dx, dy)
    curl_t = vorticity(u_t, v_t, dx, dy)
    div_p = central_diff_x(u_p, dx) + central_diff_y(v_p, dy)
    div_t = central_diff_x(u_t, dx) + central_diff_y(v_t, dy)
    tau_vort = explained_variance(curl_p, curl_t)
    out: dict[str, float] = {
        "rmse_ssh": rmse(pred[:, 0:1], truth[:, 0:1]),
        "rmse_u": rmse(u_p, u_t),
        "rmse_v": rmse(v_p, v_t),
        "rmse_uv": rmse(torch.stack([u_p, v_p], dim=1), torch.stack([u_t, v_t], dim=1)),
        "tau_uv": explained_variance(
            torch.stack([u_p, v_p], dim=1), torch.stack([u_t, v_t], dim=1)
        ),
        "tau_div": explained_variance(div_p, div_t),
        "tau_vort": tau_vort,
        "tau_curl": tau_vort,
        "tau_strain": explained_variance(strain(u_p, v_p, dx, dy), strain(u_t, v_t, dx, dy)),
    }
    if dx_km is not None:
        ssh_rep = resolved_scale_report(pred[:, 0], truth[:, 0], dx_km)
        out["lambda_x_ssh_km"] = ssh_rep["lambda_km"]
        out["lambda_x_ssh_min_ratio"] = ssh_rep["min_ratio"]
        out["lambda_x_ssh_resolved"] = ssh_rep["resolved"]
        speed_p = torch.sqrt(u_p**2 + v_p**2)
        speed_t = torch.sqrt(u_t**2 + v_t**2)
        uv_rep = resolved_scale_report(speed_p, speed_t, dx_km)
        out["lambda_x_uv_km"] = uv_rep["lambda_km"]
        out["lambda_x_uv_min_ratio"] = uv_rep["min_ratio"]
        out["lambda_x_uv_resolved"] = uv_rep["resolved"]
    return out
