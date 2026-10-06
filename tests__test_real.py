import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.real import PRODUCTS, collocate_uv_at_points, load_gdp_erddap_placeholder


def test_product_catalogue_has_dois():
    assert "doi" in PRODUCTS["swot_l3"]
    assert "erddap" in PRODUCTS["gdp"]
    assert "product_id" in PRODUCTS["ostia"]


def test_gdp_placeholder_missing_returns_none():
    assert load_gdp_erddap_placeholder(ROOT / "no_such_gdp.csv") is None
    assert load_gdp_erddap_placeholder(None) is None


def test_collocate_uv_bilinear_center():
    lon = np.linspace(-65.0, -55.0, 11)
    lat = np.linspace(33.0, 43.0, 11)
    u = np.broadcast_to(lon[None, :], (11, 11)).copy()
    v = np.broadcast_to(lat[:, None], (11, 11)).copy()
    lon_p = np.array([-60.0])
    lat_p = np.array([38.0])
    u_s, v_s = collocate_uv_at_points(u, v, lon, lat, lon_p, lat_p)
    assert abs(u_s[0] - (-60.0)) < 1e-6
    assert abs(v_s[0] - 38.0) < 1e-6


def test_collocate_out_of_domain_is_nan():
    lon = np.linspace(-65.0, -55.0, 5)
    lat = np.linspace(33.0, 43.0, 5)
    u = np.zeros((5, 5))
    v = np.zeros((5, 5))
    u_s, v_s = collocate_uv_at_points(u, v, lon, lat, np.array([0.0]), np.array([0.0]))
    assert np.isnan(u_s[0]) and np.isnan(v_s[0])
