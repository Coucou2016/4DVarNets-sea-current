#!/usr/bin/env python3
"""Stage H: SST coarsening + altimetry sparsity sensitivity (B2 vs M3).

Applies transforms at eval time on frozen post_p0 checkpoints (no retrain).
Saves under results/post_p0/sensitivity/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import make_datasets, scales_from_dataset  # noqa: E402
from fourdvarnet.metrics import batch_metrics  # noqa: E402
from fourdvarnet.model import build_fourdvarnet  # noqa: E402
from fourdvarnet.physics import strain  # noqa: E402
from fourdvarnet.repro import git_commit  # noqa: E402


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    return obj


def _batch_scales(batch, fallback):
    dx = batch.get("dx")
    dy = batch.get("dy")
    if dx is None or dy is None:
        return float(fallback["dx"]), float(fallback["dy"])
    if isinstance(dx, torch.Tensor):
        dx = float(dx.float().mean().item())
    if isinstance(dy, torch.Tensor):
        dy = float(dy.float().mean().item())
    return float(dx), float(dy)


def coarsen_sst(z_sst: torch.Tensor, factor: int) -> torch.Tensor:
    """Average-pool by ``factor`` then bilinear upsample back to native grid."""
    if factor <= 1:
        return z_sst
    b, t, h, w = z_sst.shape
    x = z_sst.reshape(b * t, 1, h, w)
    # Pad so H,W divisible by factor
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    pooled = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    up = F.interpolate(pooled, size=(h + pad_h, w + pad_w), mode="bilinear", align_corners=False)
    up = up[..., :h, :w]
    return up.reshape(b, t, h, w)


def thin_mask(mask_ssh: torch.Tensor, keep_every: int) -> torch.Tensor:
    """Keep every N-th valid obs pixel in raster order (deterministic sparsity)."""
    if keep_every <= 1:
        return mask_ssh
    out = torch.zeros_like(mask_ssh)
    for bi in range(mask_ssh.shape[0]):
        for ci in range(mask_ssh.shape[1]):
            m = mask_ssh[bi, ci]
            idx = (m > 0.5).nonzero(as_tuple=False)
            if idx.numel() == 0:
                continue
            keep = idx[::keep_every]
            out[bi, ci, keep[:, 0], keep[:, 1]] = 1.0
    return out


def load_model(ckpt_path: Path, cfg_fallback: dict, scales: dict, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config") or cfg_fallback
    mcfg = ckpt_cfg.get("model") or cfg_fallback["model"]
    flags = {
        "use_sst": ckpt.get("use_sst", mcfg.get("use_sst", True)),
        "use_sqg": ckpt.get("use_sqg", mcfg.get("use_sqg", False)),
        "use_adv": ckpt.get("use_adv", mcfg.get("use_adv", False)),
        "use_uncert": ckpt.get("use_uncert", mcfg.get("use_uncert", False)),
    }
    model = build_fourdvarnet(
        ckpt_cfg,
        **flags,
        dx=scales["dx"],
        dy=scales["dy"],
        f0=scales["f0"],
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), flags, ckpt


def eval_with_transform(
    model,
    loader,
    scales,
    device,
    *,
    sst_factor: int = 1,
    mask_keep_every: int = 1,
    strain_bins: bool = False,
):
    """Evaluate with optional SST coarsening / mask thinning.

    Inner 4DVar still needs autograd — do not wrap in no_grad / inference_mode.
    """
    pred_chunks, truth_chunks = [], []
    dx_km = float(scales["dx"]) / 1000.0
    dx_acc = dy_acc = n = 0.0
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        dx_b, dy_b = _batch_scales(batch, scales)
        dx_acc += dx_b
        dy_acc += dy_b
        n += 1
        z = coarsen_sst(batch["z_sst"], sst_factor)
        m = thin_mask(batch["mask_ssh"], mask_keep_every)
        pred = model(
            batch["y_ssh"],
            z,
            m,
            batch["mask_sst"],
            batch["u_geo"],
            batch["v_geo"],
            dx=dx_b,
            dy=dy_b,
        )
        truth = batch["truth"]
        pred_chunks.append(pred.detach().cpu())
        truth_chunks.append(truth.detach().cpu())

    dx_pool = dx_acc / max(n, 1)
    dy_pool = dy_acc / max(n, 1)
    pred_all = torch.cat(pred_chunks, dim=0)
    truth_all = torch.cat(truth_chunks, dim=0)
    metrics = batch_metrics(pred_all, truth_all, dx=dx_pool, dy=dy_pool, dx_km=dx_km)

    out = {"model": metrics, "sst_factor": sst_factor, "mask_keep_every": mask_keep_every}
    if strain_bins:
        s_all = strain(truth_all[:, 1:2], truth_all[:, 2:3], dx=dx_pool, dy=dy_pool)
        med = float(s_all.median().item())
        hi_mask = (s_all >= med).squeeze(1)  # (N,H,W)
        err_u = (pred_all[:, 1] - truth_all[:, 1]) ** 2
        err_v = (pred_all[:, 2] - truth_all[:, 2]) ** 2
        hi = hi_mask
        lo = ~hi_mask
        out["strain_binning"] = {
            "median_strain": med,
            "rmse_uv_high_strain": float(torch.sqrt(0.5 * (err_u[hi].mean() + err_v[hi].mean())).item())
            if hi.any()
            else None,
            "rmse_uv_low_strain": float(torch.sqrt(0.5 * (err_u[lo].mean() + err_v[lo].mean())).item())
            if lo.any()
            else None,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--device", default="cuda")
    p.add_argument("--b2-ckpt", default="checkpoints/4dvarnet-B2-s0-best.pt")
    p.add_argument("--m3-ckpt", default="checkpoints/4dvarnet-M3-s0-best.pt")
    p.add_argument("--out-dir", default="results/post_p0/sensitivity")
    p.add_argument("--sst-factors", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--mask-keep-every", type=int, default=3)
    args = p.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data", {})["source"] = "natl60"
    cfg["data"]["crop_size"] = args.crop_size
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        paths_all = yaml.safe_load(f) or {}
    natl_paths = paths_all.get("natl60") if isinstance(paths_all.get("natl60"), dict) else paths_all

    _, _, test_ds = make_datasets(cfg, ROOT, paths=natl_paths)
    loader = DataLoader(test_ds, batch_size=1)
    scales = scales_from_dataset(test_ds, cfg)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device
    )

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stage": "H",
        "git_commit": git_commit(ROOT),
        "crop_size": args.crop_size,
        "sst_coarsening": [],
        "altimetry_sparsity": [],
        "strain_binning": [],
        "notes": {
            "sst_factor": (
                "factor=1 native (~0.05°); factor=4 ≈ 0.2°; factor=8 ≈ 0.4° "
                "(average-pool then upsample; approximates 1/4° and ~1/2° bands on this grid)."
            ),
            "mask_keep_every": "keep every N-th valid altimetry pixel (deterministic thin mask).",
        },
    }

    for label, ckpt_rel in (("B2", args.b2_ckpt), ("M3", args.m3_ckpt)):
        ckpt_path = ROOT / ckpt_rel
        if not ckpt_path.is_file():
            print(f"SKIP {label}: missing {ckpt_path}")
            continue
        model, flags, ckpt = load_model(ckpt_path, cfg, scales, device)
        for fac in args.sst_factors:
            print(f"SST coarsen factor={fac} model={label}", flush=True)
            met = eval_with_transform(
                model, loader, scales, device, sst_factor=fac, mask_keep_every=1
            )
            results["sst_coarsening"].append(
                {"model": label, "ckpt": str(ckpt_rel), "flags": flags, **met}
            )
        print(f"Altimetry thin keep_every={args.mask_keep_every} model={label}", flush=True)
        met = eval_with_transform(
            model,
            loader,
            scales,
            device,
            sst_factor=1,
            mask_keep_every=args.mask_keep_every,
        )
        results["altimetry_sparsity"].append(
            {"model": label, "ckpt": str(ckpt_rel), "flags": flags, **met}
        )
        print(f"Strain binning model={label}", flush=True)
        met = eval_with_transform(
            model,
            loader,
            scales,
            device,
            sst_factor=1,
            mask_keep_every=1,
            strain_bins=True,
        )
        results["strain_binning"].append(
            {
                "model": label,
                "ckpt": str(ckpt_rel),
                "flags": flags,
                "tau_uv": (met.get("model") or {}).get("tau_uv"),
                **(met.get("strain_binning") or {}),
            }
        )

    out = out_dir / "sensitivity_summary.json"
    out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
