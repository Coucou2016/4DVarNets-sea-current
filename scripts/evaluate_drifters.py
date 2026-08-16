#!/usr/bin/env python3
"""Lagrangian evaluation vs GDP drifters, with a synthetic-drifter fallback.

If no local GDP file exists (config/paths.yaml ose.gdp), seed particles on the
synthetic OSSE, advect with truth vs prediction (or geostrophy), and print
mean separation. This path is testable without multi-GB downloads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import make_synthetic_datasets
from data.real import collocate_uv_at_points, load_gdp_erddap_placeholder
from data.synthetic import SyntheticOSSEConfig
from fourdvarnet.metrics import lagrangian_separation
from fourdvarnet.model import build_fourdvarnet, physics_scales_from_config


def _seed_particles(height: int, width: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    margin = max(min(height, width) // 8, 2)
    x = rng.uniform(margin, width - 1 - margin, size=n)
    y = rng.uniform(margin, height - 1 - margin, size=n)
    return x, y


def _synthetic_smoke(ckpt: Path | None, cfg: dict, n_particles: int, n_steps: int, dt: float) -> dict:
    scfg = SyntheticOSSEConfig(
        n_time=cfg["data"]["n_time"],
        height=cfg["data"]["height"],
        width=cfg["data"]["width"],
        dT=cfg["data"]["dT"],
    )
    _, _, test_ds = make_synthetic_datasets(scfg, ROOT / cfg["data"]["cache"] if (ROOT / cfg["data"]["cache"]).exists() else None)
    batch = next(iter(DataLoader(test_ds, batch_size=1)))
    truth = batch["truth"][0]
    u_t = truth[1].numpy()
    v_t = truth[2].numpy()
    u_geo = batch["u_geo"][0, 0].numpy()
    v_geo = batch["v_geo"][0, 0].numpy()

    pred_uv = None
    if ckpt is not None and ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu")
        model = build_fourdvarnet(
            blob.get("config") or cfg,
            use_sst=blob.get("use_sst", True),
            use_sqg=blob.get("use_sqg", False),
            use_adv=blob.get("use_adv", False),
            use_uncert=blob.get("use_uncert", False),
        )
        model.load_state_dict(blob["model"])
        model.eval()
        pred = model(
            batch["y_ssh"],
            batch["z_sst"],
            batch["mask_ssh"],
            batch["mask_sst"],
            batch["u_geo"],
            batch["v_geo"],
        )
        pred_uv = (pred[0, 1].detach().numpy(), pred[0, 2].detach().numpy())

    scales = physics_scales_from_config(cfg)
    rng = np.random.default_rng(0)
    x0, y0 = _seed_particles(u_t.shape[0], u_t.shape[1], n_particles, rng)
    u_p, v_p = pred_uv if pred_uv is not None else (u_geo, v_geo)
    label = "4DVarNet" if pred_uv is not None else "geostrophy"
    sep = lagrangian_separation(
        u_p, v_p, x0, y0, n_steps=n_steps, dt=dt, dx=scales["dx"], dy=scales["dy"], u_truth=u_t, v_truth=v_t
    )
    mean_end = float(sep["sep_mean"][-1])
    print(f"synthetic drifters: {n_particles} particles, {n_steps} steps, dt={dt:.0f}s")
    print(f"  mean separation vs truth ({label}): {mean_end:.1f} m")
    print(f"  sep curve (m): {np.round(sep['sep_mean'], 1)}")
    return {"mean_sep_m": mean_end, "source": "synthetic", "compared": label}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--ckpt", default="checkpoints/4dvarnet-ssh-sst-sqg-adv-uncert-best.pt")
    p.add_argument("--gdp", default=None, help="override path to local GDP table")
    p.add_argument("--n-particles", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--dt", type=float, default=3600.0)
    args = p.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        paths = yaml.safe_load(f) or {}
    gdp_path = args.gdp or (paths.get("ose") or {}).get("gdp")
    gdp = load_gdp_erddap_placeholder(ROOT / gdp_path if gdp_path else None)

    if gdp is None:
        print("No local GDP file - running synthetic-drifter smoke test.")
        print("Place a GDP extract at config/paths.yaml ose.gdp to enable OSE collocation.")
        ckpt = ROOT / args.ckpt
        _synthetic_smoke(ckpt if ckpt.exists() else None, cfg, args.n_particles, args.n_steps, args.dt)
        return

    lat = np.asarray(gdp.get("latitude", gdp.get("lat")))
    lon = np.asarray(gdp.get("longitude", gdp.get("lon")))
    print(f"Loaded GDP placeholder with {lat.size} points from {gdp_path}")
    # Without a full SSH/SST analysis cube, collocation is demonstrated on synthetic grid.
    scfg = SyntheticOSSEConfig(n_time=8, height=24, width=24, dT=5)
    from data.synthetic import generate_synthetic_osse

    data = generate_synthetic_osse(scfg)
    lon_g = np.linspace(-65.0, -55.0, data["u"].shape[-1])
    lat_g = np.linspace(33.0, 43.0, data["u"].shape[-2])
    u_s, v_s = collocate_uv_at_points(data["u"][0], data["v"][0], lon_g, lat_g, lon[: min(16, lon.size)], lat[: min(16, lat.size)])
    print(f"collocated {u_s.size} samples; finite={np.isfinite(u_s).sum()}")


if __name__ == "__main__":
    main()
