"""PyTorch dataset for synthetic or NATL60-style NetCDF stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.synthetic import SyntheticOSSEConfig, generate_synthetic_osse, load_synthetic_npz
from fourdvarnet.geometry import coriolis, metric_dx, metric_dy
from fourdvarnet.physics import geostrophic_velocity


class SSTSSHCurrentDataset(Dataset):
    def __init__(
        self,
        data: dict[str, np.ndarray],
        dT: int = 7,
        time_indices: list[int] | None = None,
        f: float | None = None,
    ) -> None:
        self.dT = dT
        self.ssh = data["ssh"]
        self.u = data["u"]
        self.v = data["v"]
        self.sst = data["sst"]
        self.y_ssh = data["y_ssh"]
        # Pure OI/DUACS SSH for fair geostrophic baseline (falls back to y_ssh).
        self.y_oi = data["y_oi"] if "y_oi" in data else data["y_ssh"]
        self.mask_ssh = data["mask_ssh"]
        self.mask_sst = data["mask_sst"]
        self.lat = data.get("lat")
        self.lon = data.get("lon")
        n = self.ssh.shape[0]
        self.indices = time_indices if time_indices is not None else list(range(dT - 1, n))
        self.f = float(data["f"][0]) if f is None and "f" in data else (f or 7e-5)
        self.dx_deg = float(data["dx"][0]) if "dx" in data else 0.05
        # Prefer meter spacings from loader when present (lat-aware).
        if "dx_m" in data and "dy_m" in data:
            self.dx_m = float(data["dx_m"][0])
            self.dy_m = float(data["dy_m"][0])
        else:
            self.dx_m = self.dx_deg * 111e3
            self.dy_m = self.dx_deg * 111e3
        # Legacy attribute used by older call sites
        self.dx = self.dx_deg

    def __len__(self) -> int:
        return len(self.indices)

    def _geo_scales(self) -> tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor]:
        """Coriolis and metric spacing shared with train/eval physics."""
        lat = self.lat
        if lat is None:
            return self.f, self.dx_m, self.dy_m
        lat_arr = np.asarray(lat, dtype=np.float32)
        if lat_arr.ndim == 1 and lat_arr.size == self.ssh.shape[1]:
            f_map = torch.tensor([float(coriolis(float(y))) for y in lat_arr], dtype=torch.float32)
            f_map = f_map.view(1, 1, -1, 1)
            lon = np.asarray(self.lon, dtype=np.float32) if self.lon is not None else None
            dlon = self.dx_deg
            if lon is not None and lon.ndim == 1 and lon.size > 1:
                dlon = float(np.abs(np.diff(lon)).mean())
            dlat = float(np.abs(np.diff(lat_arr)).mean()) if lat_arr.size > 1 else self.dx_deg
            dx_map = torch.tensor(
                [float(metric_dx(dlon, float(y))) for y in lat_arr], dtype=torch.float32
            ).view(1, 1, -1, 1)
            dy_v = float(metric_dy(dlat))
            return f_map, dx_map, dy_v
        return self.f, self.dx_m, self.dy_m

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t = self.indices[idx]
        sst_win = self.sst[t - self.dT + 1 : t + 1]
        ssh_t = self.ssh[t : t + 1]
        u_t = self.u[t : t + 1]
        v_t = self.v[t : t + 1]
        y = self.y_ssh[t : t + 1]
        y_oi = self.y_oi[t : t + 1]
        m_ssh = self.mask_ssh[t : t + 1]
        # Full dT SST mask window (do not broadcast last-frame mask only).
        m_sst = self.mask_sst[t - self.dT + 1 : t + 1]
        if m_sst.shape[0] != self.dT:
            raise RuntimeError(
                f"mask_sst window length {m_sst.shape[0]} != dT={self.dT} at index {t}"
            )

        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
        y_oi_t = torch.from_numpy(np.asarray(y_oi, dtype=np.float32))
        f_s, dx_s, dy_s = self._geo_scales()
        # Model init: geostrophy from hybrid (or OI) observation background.
        u_g, v_g = geostrophic_velocity(y_t, f_s, dx=dx_s, dy=dy_s)
        # Fair baseline: geostrophy from pure OI/DUACS SSH only.
        u_g_oi, v_g_oi = geostrophic_velocity(y_oi_t, f_s, dx=dx_s, dy=dy_s)

        # Scalar mean spacing for TrainingLoss / VariationalCost / metrics.
        if isinstance(dx_s, torch.Tensor):
            dx_mean = float(dx_s.mean().item())
        else:
            dx_mean = float(dx_s)
        dy_mean = float(dy_s.mean().item()) if isinstance(dy_s, torch.Tensor) else float(dy_s)

        return {
            "truth": torch.from_numpy(np.concatenate([ssh_t, u_t, v_t], axis=0).astype(np.float32)),
            "y_ssh": y_t,
            "y_oi": y_oi_t,
            "z_sst": torch.from_numpy(sst_win.astype(np.float32)),
            "mask_ssh": torch.from_numpy(m_ssh.astype(np.float32)),
            "mask_sst": torch.from_numpy(np.asarray(m_sst, dtype=np.float32)),
            "u_geo": u_g,
            "v_geo": v_g,
            "u_geo_oi": u_g_oi,
            "v_geo_oi": v_g_oi,
            "dx": torch.tensor(dx_mean, dtype=torch.float32),
            "dy": torch.tensor(dy_mean, dtype=torch.float32),
        }


def make_synthetic_datasets(
    cfg: SyntheticOSSEConfig | None = None,
    cache_path: str | Path | None = None,
) -> tuple[SSTSSHCurrentDataset, SSTSSHCurrentDataset, SSTSSHCurrentDataset]:
    cfg = cfg or SyntheticOSSEConfig()
    if cache_path and Path(cache_path).exists():
        data = load_synthetic_npz(cache_path)
    else:
        data = generate_synthetic_osse(cfg, cache_path)
    n = data["ssh"].shape[0]
    train_idx = list(range(cfg.dT - 1, int(0.7 * n)))
    val_idx = list(range(int(0.7 * n), int(0.85 * n)))
    test_idx = list(range(int(0.85 * n), n))
    return (
        SSTSSHCurrentDataset(data, cfg.dT, train_idx),
        SSTSSHCurrentDataset(data, cfg.dT, val_idx),
        SSTSSHCurrentDataset(data, cfg.dT, test_idx),
    )


def _center_crop_spatial(data: dict[str, np.ndarray], crop_size: int) -> dict[str, np.ndarray]:
    """Center-crop 3-D (T,H,W) fields for CPU/NATL60 smoke. Leaves 1-D coords alone."""
    out = dict(data)
    for key in ("ssh", "u", "v", "sst", "y_ssh", "y_oi", "mask_ssh", "mask_sst"):
        arr = out.get(key)
        if arr is None or not hasattr(arr, "ndim") or arr.ndim != 3:
            continue
        _, height, width = arr.shape
        if height < crop_size or width < crop_size:
            raise ValueError(f"{key} spatial {height}x{width} smaller than crop_size={crop_size}")
        y0 = (height - crop_size) // 2
        x0 = (width - crop_size) // 2
        out[key] = arr[:, y0 : y0 + crop_size, x0 : x0 + crop_size]
    # Crop 1-D lat/lon to match spatial crop when present
    if "lat" in out and np.asarray(out["lat"]).ndim == 1 and out["ssh"].ndim == 3:
        height = out["ssh"].shape[1]
        lat = np.asarray(out["lat"])
        if lat.size >= height + (np.asarray(data["ssh"]).shape[1] - height):
            y0 = (np.asarray(data["ssh"]).shape[1] - crop_size) // 2
            out["lat"] = lat[y0 : y0 + crop_size]
    if "lon" in out and np.asarray(out["lon"]).ndim == 1 and out["ssh"].ndim == 3:
        width = out["ssh"].shape[2]
        lon = np.asarray(out["lon"])
        if lon.size >= width:
            x0 = (np.asarray(data["ssh"]).shape[2] - crop_size) // 2
            out["lon"] = lon[x0 : x0 + crop_size]
    return out


def _cap_indices(indices: list[int], max_samples: int | None) -> list[int]:
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return indices
    # Even stride so smoke still covers the split calendar range.
    step = max(len(indices) // max_samples, 1)
    capped = indices[::step][:max_samples]
    return capped


def _purge_train_vs_val(
    train_idx: list[int],
    val_idx: list[int],
    dT: int,
    purge_days: int,
) -> list[int]:
    """Drop train windows whose SST history overlaps any val analysis day.

    With ``purge_days = dT - 1``, require ``t > max(val_idx) + purge_days`` so the
    dT-day SST window does not share calendar days with val targets.
    """
    if purge_days <= 0 or not val_idx:
        return train_idx
    val_set = set(val_idx)
    last_val = max(val_idx)
    kept: list[int] = []
    for t in train_idx:
        if any(i in val_set for i in range(t - dT + 1, t + 1)):
            continue
        if t - purge_days <= last_val:
            continue
        kept.append(t)
    return kept


def make_natl60_datasets(
    paths: dict[str, str],
    dT: int = 7,
    root: str | Path | None = None,
    f0: float = 7.0e-5,
    dx_deg: float = 0.05,
    crop_size: int | None = None,
    max_samples: int | None = None,
    allow_truth_background: bool = False,
    purge_days: int | None = None,
) -> tuple[SSTSSHCurrentDataset, SSTSSHCurrentDataset, SSTSSHCurrentDataset]:
    """Sliding dT windows on the Gulf Stream NATL60 OSSE with paper date splits.

    ``crop_size`` / ``max_samples`` are CPU smoke helpers only — not for paper tables.
    ``allow_truth_background`` is debug-only (default False; never for paper claims).
    ``purge_days`` (default ``dT-1`` in paper mode) drops train windows whose SST
    history overlaps val analysis times.
    """
    from data.natl60 import indices_for_split, load_natl60

    data = load_natl60(
        paths,
        root=root,
        f0=f0,
        dx_deg=dx_deg,
        allow_truth_background=allow_truth_background,
    )
    if crop_size is not None:
        data = _center_crop_spatial(data, int(crop_size))
    times = data["time"]
    train_idx = indices_for_split(times, "train", dT)
    val_idx = indices_for_split(times, "val", dT)
    test_idx = indices_for_split(times, "test", dT)

    if purge_days is None and not allow_truth_background:
        purge_days = max(int(dT) - 1, 0)
    if purge_days is not None and int(purge_days) > 0:
        train_idx = _purge_train_vs_val(train_idx, val_idx, dT, int(purge_days))

    train_idx = _cap_indices(train_idx, max_samples)
    val_idx = _cap_indices(val_idx, max_samples)
    test_idx = _cap_indices(test_idx, max_samples)
    empty = [name for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)) if not idx]
    if empty:
        raise RuntimeError(
            "NATL60 split(s) "
            + ", ".join(empty)
            + " have no windows. Check that the NetCDF time axis covers "
            "2012-10-20 to 2013-09-30 (paper §3.2)."
        )
    return (
        SSTSSHCurrentDataset(data, dT, train_idx, f=f0),
        SSTSSHCurrentDataset(data, dT, val_idx, f=f0),
        SSTSSHCurrentDataset(data, dT, test_idx, f=f0),
    )


def make_datasets(
    cfg: dict,
    root: str | Path,
    paths: dict[str, str] | None = None,
) -> tuple[SSTSSHCurrentDataset, SSTSSHCurrentDataset, SSTSSHCurrentDataset]:
    """Dispatch on ``cfg['data']['source']`` (default: synthetic)."""
    source = str((cfg.get("data") or {}).get("source", "synthetic")).lower()
    dcfg = cfg.get("data") or {}
    phys = cfg.get("physics") or {}
    root_p = Path(root)
    if source == "natl60":
        if not paths:
            raise FileNotFoundError("NATL60 source requested but paths.yaml mapping is empty")
        crop = dcfg.get("crop_size")
        max_s = dcfg.get("max_samples")
        purge = dcfg.get("purge_days", None)
        return make_natl60_datasets(
            paths,
            dT=int(dcfg.get("dT", 7)),
            root=root_p,
            f0=float(phys.get("f0", 7.0e-5)),
            dx_deg=float(phys.get("dx_deg", 0.05)),
            crop_size=int(crop) if crop is not None else None,
            max_samples=int(max_s) if max_s is not None else None,
            allow_truth_background=bool(dcfg.get("allow_truth_background", False)),
            purge_days=int(purge) if purge is not None else None,
        )
    scfg = SyntheticOSSEConfig(
        n_time=int(dcfg.get("n_time", 40)),
        height=int(dcfg.get("height", 48)),
        width=int(dcfg.get("width", 48)),
        dT=int(dcfg.get("dT", 7)),
        f0=float(phys.get("f0", 7.0e-5)),
        dx_deg=float(phys.get("dx_deg", 0.05)),
    )
    cache = dcfg.get("cache")
    cache_path = root_p / cache if cache else None
    return make_synthetic_datasets(scfg, cache_path)


def scales_from_dataset(
    ds: SSTSSHCurrentDataset,
    cfg: dict | None = None,
) -> dict[str, float]:
    """Prefer dataset meter spacings over isotropic ``dx_deg * 111e3`` config fallback."""
    from fourdvarnet.model import physics_scales_from_config

    base = physics_scales_from_config(cfg or {})
    base["dx"] = float(ds.dx_m)
    base["dy"] = float(ds.dy_m)
    if getattr(ds, "f", None) is not None:
        base["f0"] = float(ds.f)
    return base
