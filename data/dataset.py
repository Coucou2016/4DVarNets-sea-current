"""PyTorch dataset for synthetic or NATL60-style NetCDF stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.synthetic import SyntheticOSSEConfig, generate_synthetic_osse, load_synthetic_npz
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
        self.mask_ssh = data["mask_ssh"]
        self.mask_sst = data["mask_sst"]
        n = self.ssh.shape[0]
        self.indices = time_indices if time_indices is not None else list(range(dT - 1, n))
        self.f = float(data["f"][0]) if f is None and "f" in data else (f or 7e-5)
        self.dx = float(data["dx"][0]) if "dx" in data else 0.05

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t = self.indices[idx]
        sst_win = self.sst[t - self.dT + 1 : t + 1]
        ssh_t = self.ssh[t : t + 1]
        u_t = self.u[t : t + 1]
        v_t = self.v[t : t + 1]
        y = self.y_ssh[t : t + 1]
        m_ssh = self.mask_ssh[t : t + 1]
        m_sst = self.mask_sst[t : t + 1]
        m_sst = np.broadcast_to(m_sst, (self.dT, *m_sst.shape[1:]))

        y_t = torch.from_numpy(y)
        u_g, v_g = geostrophic_velocity(y_t, self.f, dx=self.dx * 111e3, dy=self.dx * 111e3)

        return {
            "truth": torch.from_numpy(np.concatenate([ssh_t, u_t, v_t], axis=0)),
            "y_ssh": y_t,
            "z_sst": torch.from_numpy(sst_win.astype(np.float32)),
            "mask_ssh": torch.from_numpy(m_ssh.astype(np.float32)),
            "mask_sst": torch.from_numpy(m_sst.astype(np.float32)),
            "u_geo": u_g,
            "v_geo": v_g,
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
    for key in ("ssh", "u", "v", "sst", "y_ssh", "mask_ssh", "mask_sst"):
        arr = out[key]
        if arr.ndim != 3:
            continue
        _, height, width = arr.shape
        if height < crop_size or width < crop_size:
            raise ValueError(f"{key} spatial {height}x{width} smaller than crop_size={crop_size}")
        y0 = (height - crop_size) // 2
        x0 = (width - crop_size) // 2
        out[key] = arr[:, y0 : y0 + crop_size, x0 : x0 + crop_size]
    return out


def _cap_indices(indices: list[int], max_samples: int | None) -> list[int]:
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return indices
    # Even stride so smoke still covers the split calendar range.
    step = max(len(indices) // max_samples, 1)
    capped = indices[::step][:max_samples]
    return capped


def make_natl60_datasets(
    paths: dict[str, str],
    dT: int = 7,
    root: str | Path | None = None,
    f0: float = 7.0e-5,
    dx_deg: float = 0.05,
    crop_size: int | None = None,
    max_samples: int | None = None,
) -> tuple[SSTSSHCurrentDataset, SSTSSHCurrentDataset, SSTSSHCurrentDataset]:
    """Sliding dT windows on the Gulf Stream NATL60 OSSE with paper date splits.

    ``crop_size`` / ``max_samples`` are CPU smoke helpers only — not for paper tables.
    """
    from data.natl60 import indices_for_split, load_natl60

    data = load_natl60(paths, root=root, f0=f0, dx_deg=dx_deg)
    if crop_size is not None:
        data = _center_crop_spatial(data, int(crop_size))
    times = data["time"]
    train_idx = _cap_indices(indices_for_split(times, "train", dT), max_samples)
    val_idx = _cap_indices(indices_for_split(times, "val", dT), max_samples)
    test_idx = _cap_indices(indices_for_split(times, "test", dT), max_samples)
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
        return make_natl60_datasets(
            paths,
            dT=int(dcfg.get("dT", 7)),
            root=root_p,
            f0=float(phys.get("f0", 7.0e-5)),
            dx_deg=float(phys.get("dx_deg", 0.05)),
            crop_size=int(crop) if crop is not None else None,
            max_samples=int(max_s) if max_s is not None else None,
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
