"""Real OSE product stubs (SWOT L3, OSTIA, GDP). No live downloads on import.

These identifiers are for the follow-on observing-system experiment, not the
NATL60 OSSE used in Fablet et al. JAMES 2024. Functions never hit the network;
callers pass a local file or get a documented placeholder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# --- Product catalogue (IDs / DOIs; verify against the latest catalogue) ---
PRODUCTS: dict[str, dict[str, str]] = {
    "swot_l3": {
        "name": "SWOT Level-3 LR sea surface height (AVISO/DUACS)",
        "doi": "10.24400/527896/A01-2023.017",
        "url": "https://www.aviso.altimetry.fr/en/data/products/sea-surface-height-products/global/swot-l3-ocean-products.html",
        "note": "Use L3 unsmoothed or 2 km product for Gulf Stream OSE collocation.",
    },
    "ostia": {
        "name": "OSTIA global SST (UK Met Office / CMEMS L4)",
        "product_id": "SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",
        "doi": "10.48670/moi-00165",
        "url": "https://data.marine.copernicus.eu/",
        "note": "Daily L4 foundation SST; collocate with SWOT/SSH on the model grid.",
    },
    "gdp": {
        "name": "NOAA AOML Global Drifter Program (6-hourly QC)",
        "doi": "10.25921/x46c-3620",
        "erddap": "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc",
        "erddap_html": "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc.html",
        "note": "Subset lon/lat/time for the Gulf Stream box; do not download the global archive blindly.",
    },
}


def load_gdp_erddap_placeholder(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load a local GDP CSV/NetCDF if present. Does not query ERDDAP.

    Expected columns if CSV: time, latitude, longitude, ve, vn, id
    Returns None when ``path`` is missing so callers can fall back to a
    synthetic-drifter smoke test.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() in {".nc", ".nc4"}:
        import xarray as xr

        ds = xr.open_dataset(p)
        out = {k: np.asarray(ds[k].values) for k in ds.data_vars}
        ds.close()
        return out
    # CSV / whitespace table
    try:
        data = np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")
    except OSError:
        return None
    if data.dtype.names is None:
        return {"table": np.asarray(data)}
    return {name: np.asarray(data[name]) for name in data.dtype.names}


def collocate_uv_at_points(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    lon_p: np.ndarray,
    lat_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample of gridded (u, v) at drifter locations.

    ``lon, lat`` are 1-D axes (or 2-D nav coords reduced along the other axis).
    Out-of-domain points return NaN.
    """
    from scipy.interpolate import RegularGridInterpolator

    u_grid = np.asarray(u_grid, dtype=np.float64)
    v_grid = np.asarray(v_grid, dtype=np.float64)
    if u_grid.ndim != 2 or v_grid.ndim != 2:
        raise ValueError("u_grid and v_grid must be 2-D (H, W)")
    lat_a = np.asarray(lat, dtype=np.float64)
    lon_a = np.asarray(lon, dtype=np.float64)
    if lat_a.ndim == 2:
        lat_a = lat_a[:, 0] if lat_a.shape[0] == u_grid.shape[0] else lat_a[0, :]
    if lon_a.ndim == 2:
        lon_a = lon_a[0, :] if lon_a.shape[1] == u_grid.shape[1] else lon_a[:, 0]
    if lat_a.shape[0] != u_grid.shape[0] or lon_a.shape[0] != u_grid.shape[1]:
        # try swapped
        if lat_a.shape[0] == u_grid.shape[1] and lon_a.shape[0] == u_grid.shape[0]:
            u_grid, v_grid = u_grid.T, v_grid.T
        else:
            raise ValueError(
                f"grid {u_grid.shape} incompatible with lat {lat_a.shape} lon {lon_a.shape}"
            )

    if lat_a[0] > lat_a[-1]:
        lat_a = lat_a[::-1].copy()
        u_grid = u_grid[::-1]
        v_grid = v_grid[::-1]
    if lon_a[0] > lon_a[-1]:
        lon_a = lon_a[::-1].copy()
        u_grid = u_grid[:, ::-1]
        v_grid = v_grid[:, ::-1]

    fu = RegularGridInterpolator((lat_a, lon_a), u_grid, bounds_error=False, fill_value=np.nan)
    fv = RegularGridInterpolator((lat_a, lon_a), v_grid, bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([np.asarray(lat_p, dtype=np.float64).reshape(-1), np.asarray(lon_p, dtype=np.float64).reshape(-1)])
    return fu(pts), fv(pts)
