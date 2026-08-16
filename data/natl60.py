"""
NATL60 / OSSE data loader (Fablet et al. JAMES 2024 §3).

Download URLs (from CIA-Oceanix/4dvarnet-james-uv-ssc README):
  obs: https://s3.eu-central-1.wasabisys.com/melody/NATL/data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc
  oi:  https://s3.eu-central-1.wasabisys.com/melody/NATL/oi/ssh_NATL60_swot_4nadir.nc
  ref SSH/SST/u/v: https://s3.eu-central-1.wasabisys.com/melody/NATL/ref/...

Set paths in config/paths.yaml after download. Importing this module does not
touch the filesystem or require the NetCDF files to exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

NATL60_URLS = {
    "obs": "https://s3.eu-central-1.wasabisys.com/melody/NATL/data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc",
    "oi": "https://s3.eu-central-1.wasabisys.com/melody/NATL/oi/ssh_NATL60_swot_4nadir.nc",
    "ssh_ref": "https://s3.eu-central-1.wasabisys.com/melody/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc",
    "sst_ref": "https://s3.eu-central-1.wasabisys.com/melody/NATL/ref/NATL60-CJM165_NATL_sst_y2013.1y.nc",
    "u_ref": "https://s3.eu-central-1.wasabisys.com/melody/NATL/ref/NATL60-CJM165_NATL_u_y2013.1y.nc",
    "v_ref": "https://s3.eu-central-1.wasabisys.com/melody/NATL/ref/NATL60-CJM165_NATL_v_y2013.1y.nc",
}

# Paper §3.2 Gulf Stream OSSE box
GULF_STREAM_BOX = {"lat_min": 33.0, "lat_max": 43.0, "lon_min": -65.0, "lon_max": -55.0}

# Paper §3.2 date splits (analysis time of each window)
NATL60_SPLITS = {
    "train": ("2013-02-04", "2013-09-30"),
    "val": ("2013-01-01", "2013-02-04"),
    "test": ("2012-10-20", "2012-12-04"),
}

_SSH_NAMES = ("ssh", "sossheig", "sla", "adt", "ssh_obs", "ssh_mod")
_SST_NAMES = ("sst", "sosstsst", "thetao", "analysed_sst", "temperature")
_U_NAMES = ("u", "vozocrtx", "uo", "ssu", "u_obs")
_V_NAMES = ("v", "vomecrty", "vo", "ssv", "v_obs")
_LAT_NAMES = ("lat", "latitude", "nav_lat", "y")
_LON_NAMES = ("lon", "longitude", "nav_lon", "x")
_TIME_NAMES = ("time", "time_counter", "date")


def check_natl60_paths(paths: dict[str, str], required: tuple[str, ...] | None = None) -> list[str]:
    """Return missing path messages. Default required = SSH/SST/u/v refs (obs/oi optional)."""
    required = required if required is not None else ("ssh_ref", "sst_ref", "u_ref", "v_ref")
    missing = []
    for key in required:
        p = paths.get(key)
        if not p or not Path(p).exists():
            missing.append(f"{key}: {p or '(unset)'}")
    return missing


def missing_files_message(paths: dict[str, str], missing: list[str] | None = None) -> str:
    missing = missing if missing is not None else check_natl60_paths(paths)
    lines = [
        "NATL60 NetCDF files are missing (required: ssh_ref, sst_ref, u_ref, v_ref).",
        "obs / oi are optional (OI background + along-track mask).",
        "Download with: python scripts/download_natl60.py",
        "Resume is supported (re-run the same command; *.part files continue).",
        "Then set paths in config/paths.yaml.",
        "Missing:",
    ]
    for m in missing:
        lines.append(f"  {m}")
        try:
            _key, _, path_s = m.partition(": ")
            part = Path(path_s.strip()).with_suffix(Path(path_s.strip()).suffix + ".part")
            if part.exists():
                lines.append(f"    (partial download: {part} {part.stat().st_size} bytes)")
        except OSError:
            pass
    lines.append("URLs:")
    lines.extend(f"  {k}: {url}" for k, url in NATL60_URLS.items())
    return "\n".join(lines)


def _resolve_paths(paths: dict[str, str], root: str | Path | None = None) -> dict[str, str]:
    root_p = Path(root) if root is not None else None
    out = {}
    for k, p in paths.items():
        path = Path(p)
        if not path.is_absolute() and root_p is not None:
            path = root_p / path
        out[k] = str(path)
    return out


def _coord_name(ds: Any, names: tuple[str, ...]) -> str | None:
    coords: set[str] = set()
    for attr in ("coords", "dims", "data_vars", "indexes"):
        if hasattr(ds, attr):
            try:
                coords.update(map(str, list(getattr(ds, attr))))
            except (TypeError, ValueError):
                pass
    if hasattr(ds, "name") and ds.name is not None:
        coords.add(str(ds.name))
    for n in names:
        if n in coords:
            return n
    return None


def _first_var(ds: Any, names: tuple[str, ...], kind: str):
    for n in names:
        if n in ds.data_vars or n in ds.variables:
            return ds[n]
    data_vars = [v for v in ds.data_vars if not str(v).startswith("crs")]
    if len(data_vars) == 1:
        return ds[data_vars[0]]
    raise KeyError(f"could not find {kind} in {list(ds.data_vars)}; tried {names}")


def _to_lon180(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def _subset_box(da: Any, box: dict[str, float]) -> Any:
    lat_name = _coord_name(da, _LAT_NAMES)
    lon_name = _coord_name(da, _LON_NAMES)
    if lat_name is None or lon_name is None:
        return da
    lat = da[lat_name]
    lon = da[lon_name]
    lon180 = _to_lon180(np.asarray(lon.values))
    # 1-D lat/lon
    if lat.ndim == 1 and lon.ndim == 1:
        lat_vals = np.asarray(lat.values)
        lat_mask = (lat_vals >= box["lat_min"]) & (lat_vals <= box["lat_max"])
        lon_mask = (lon180 >= box["lon_min"]) & (lon180 <= box["lon_max"])
        if not lat_mask.any() or not lon_mask.any():
            return da
        return da.isel({lat_name: lat_mask, lon_name: lon_mask})
    # 2-D nav grid: mask then crop bounding indices
    lat2 = np.asarray(lat.values)
    lon2 = _to_lon180(np.asarray(lon.values))
    inside = (
        (lat2 >= box["lat_min"])
        & (lat2 <= box["lat_max"])
        & (lon2 >= box["lon_min"])
        & (lon2 <= box["lon_max"])
    )
    if not inside.any():
        return da
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    y_dim, x_dim = lat.dims
    return da.isel({y_dim: slice(rows.min(), rows.max() + 1), x_dim: slice(cols.min(), cols.max() + 1)})


def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 3:
        a = np.take(a, 0, axis=0)
    if a.ndim == 2:
        a = a[None, ...]
    return np.asarray(a, dtype=np.float32)


# NATL60 OSSE cubes store time as seconds since this epoch (attrs often empty).
NATL60_TIME_EPOCH = np.datetime64("2012-10-01T00:00:00")


def _time_values(da: Any) -> np.ndarray:
    tname = _coord_name(da, _TIME_NAMES)
    if tname is None:
        n = int(da.shape[0])
        return (NATL60_TIME_EPOCH + np.arange(n).astype("timedelta64[D]")).astype("datetime64[ns]")
    t = np.asarray(da[tname].values)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]")
    # Numeric: prefer CF units if present, else seconds since 2012-10-01
    units = ""
    try:
        units = str(da[tname].attrs.get("units", "") or "")
    except Exception:
        units = ""
    if units.startswith("seconds since") or np.issubdtype(t.dtype, np.floating) or np.issubdtype(t.dtype, np.integer):
        # Treat as seconds from NATL60_TIME_EPOCH when units missing/unparsed
        if "since" in units:
            try:
                import cftime  # noqa: F401
                import xarray as xr

                decoded = xr.decode_cf(xr.Dataset({tname: da[tname]}))[tname].values
                return np.asarray(decoded).astype("datetime64[ns]")
            except Exception:
                pass
        secs = t.astype(np.float64)
        return (NATL60_TIME_EPOCH + (secs * 1e9).astype("timedelta64[ns]")).astype("datetime64[ns]")
    try:
        return t.astype("datetime64[ns]")
    except (TypeError, ValueError):
        return t


def _interp_like(da: Any, ref: Any) -> Any:
    kwargs = {}
    for names in (_TIME_NAMES, _LAT_NAMES, _LON_NAMES):
        src = _coord_name(da, names)
        dst = _coord_name(ref, names)
        if src and dst and src in da.coords and dst in ref.coords:
            kwargs[src] = ref[dst]
            if src != dst:
                da = da.rename({src: dst})
                kwargs = {dst if k == src else k: v for k, v in kwargs.items()}
    if not kwargs:
        return da
    try:
        return da.interp(**kwargs)
    except (ValueError, TypeError):
        return da


def load_natl60(
    paths: dict[str, str],
    box: dict[str, float] | None = None,
    root: str | Path | None = None,
    f0: float = 7.0e-5,
    dx_deg: float = 0.05,
) -> dict[str, np.ndarray]:
    """Load NATL60 OSSE stacks. Raises FileNotFoundError with download URLs if missing."""
    paths = _resolve_paths(paths, root)
    missing = check_natl60_paths(paths)
    if missing:
        raise FileNotFoundError(missing_files_message(paths, missing))

    import xarray as xr

    box = box or GULF_STREAM_BOX
    required = ("ssh_ref", "sst_ref", "u_ref", "v_ref")
    for key in required:
        if key not in paths:
            raise KeyError(f"paths must include {required}, missing {key}")

    ssh_ds = xr.open_dataset(paths["ssh_ref"])
    sst_ds = xr.open_dataset(paths["sst_ref"])
    u_ds = xr.open_dataset(paths["u_ref"])
    v_ds = xr.open_dataset(paths["v_ref"])
    ssh = _subset_box(_first_var(ssh_ds, _SSH_NAMES, "SSH"), box)
    sst = _interp_like(_subset_box(_first_var(sst_ds, _SST_NAMES, "SST"), box), ssh)
    u = _interp_like(_subset_box(_first_var(u_ds, _U_NAMES, "u"), box), ssh)
    v = _interp_like(_subset_box(_first_var(v_ds, _V_NAMES, "v"), box), ssh)

    ssh_np = _squeeze_hw(ssh.values)
    sst_np = _squeeze_hw(sst.values)
    u_np = _squeeze_hw(u.values)
    v_np = _squeeze_hw(v.values)
    n = min(ssh_np.shape[0], sst_np.shape[0], u_np.shape[0], v_np.shape[0])
    ssh_np, sst_np, u_np, v_np = ssh_np[:n], sst_np[:n], u_np[:n], v_np[:n]

    y_ssh = np.zeros_like(ssh_np)
    mask_ssh = np.ones_like(ssh_np, dtype=np.float32)
    if paths.get("oi") and Path(paths["oi"]).exists():
        oi_ds = xr.open_dataset(paths["oi"])
        oi = _interp_like(_subset_box(_first_var(oi_ds, _SSH_NAMES + ("ssh_oi", "oi"), "OI SSH"), box), ssh)
        y_ssh = _squeeze_hw(oi.values)[:n]
        oi_ds.close()
    else:
        y_ssh = ssh_np.copy()

    if paths.get("obs") and Path(paths["obs"]).exists():
        obs_ds = xr.open_dataset(paths["obs"])
        try:
            obs = _interp_like(_subset_box(_first_var(obs_ds, _SSH_NAMES, "obs SSH"), box), ssh)
            obs_np = _squeeze_hw(obs.values)[:n]
            mask_ssh = np.isfinite(obs_np).astype(np.float32)
            mask_ssh[np.abs(obs_np) < 1e-12] = 0.0
            # Prefer along-track values on the OI background
            y_ssh = np.where(mask_ssh > 0, obs_np, y_ssh)
        except KeyError:
            pass
        obs_ds.close()

    mask_sst = np.isfinite(sst_np).astype(np.float32)
    sst_np = np.where(np.isfinite(sst_np), sst_np, 0.0).astype(np.float32)
    times = _time_values(ssh)[:n]

    lat_name = _coord_name(ssh, _LAT_NAMES)
    lon_name = _coord_name(ssh, _LON_NAMES)
    lat = np.asarray(ssh[lat_name].values) if lat_name else np.array([38.0], dtype=np.float32)
    lon = np.asarray(ssh[lon_name].values) if lon_name else np.array([-60.0], dtype=np.float32)

    ssh_ds.close()
    sst_ds.close()
    u_ds.close()
    v_ds.close()

    return {
        "ssh": ssh_np,
        "u": u_np,
        "v": v_np,
        "sst": sst_np,
        "y_ssh": y_ssh.astype(np.float32),
        "mask_ssh": mask_ssh.astype(np.float32),
        "mask_sst": mask_sst.astype(np.float32),
        "time": times,
        "lat": np.asarray(lat, dtype=np.float32),
        "lon": np.asarray(lon, dtype=np.float32),
        "f": np.array([f0], dtype=np.float32),
        "dx": np.array([dx_deg], dtype=np.float32),
    }


def indices_for_split(
    times: np.ndarray,
    split: str,
    dT: int,
    splits: dict[str, tuple[str, str]] | None = None,
) -> list[int]:
    """Window end indices whose analysis time falls in ``split`` (paper §3.2).

    Val is [2013-01-01, 2013-02-04); train is [2013-02-04, 2013-09-30] so the
    shared calendar day is assigned to train only.
    """
    splits = splits or NATL60_SPLITS
    if split not in splits:
        raise KeyError(f"unknown split {split}; expected {list(splits)}")
    start_s, end_s = splits[split]
    start = np.datetime64(start_s)
    end = np.datetime64(end_s)
    t = np.asarray(times)
    try:
        t = t.astype("datetime64[D]")
    except (TypeError, ValueError) as exc:
        raise ValueError("NATL60 time coordinate is not datetime-like; cannot apply paper splits") from exc
    idx: list[int] = []
    for i in range(dT - 1, len(t)):
        ti = t[i]
        if split == "val":
            if start <= ti < end:
                idx.append(i)
        else:
            if start <= ti <= end:
                idx.append(i)
    return idx
