#!/usr/bin/env python3
"""Evaluate trained model vs geostrophic baseline on test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import make_datasets, scales_from_dataset
from fourdvarnet.metrics import batch_metrics, resolved_timescale
from fourdvarnet.model import build_fourdvarnet


def _fmt(v: float) -> str:
    if v != v:
        return "nan"
    return f"{v:.4f}"


def _mean_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {k: sum(d[k] for d in items) / len(items) for k in keys}


def _jsonable(obj):
    """Convert floats for strict JSON (NaN/Inf → null)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    return obj


def _batch_scales(batch: dict, fallback: dict[str, float]) -> tuple[float, float]:
    dx = batch.get("dx")
    dy = batch.get("dy")
    if dx is None or dy is None:
        return float(fallback["dx"]), float(fallback["dy"])
    if isinstance(dx, torch.Tensor):
        dx = float(dx.float().mean().item())
    if isinstance(dy, torch.Tensor):
        dy = float(dy.float().mean().item())
    return float(dx), float(dy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--ckpt", default="checkpoints/4dvarnet-ssh-sst-sqg-adv-uncert-best.pt")
    p.add_argument("--source", default=None, help="synthetic | natl60")
    p.add_argument("--crop-size", type=int, default=None, help="NATL60 center crop (CPU smoke)")
    p.add_argument("--max-samples", type=int, default=None, help="cap test windows (CPU smoke)")
    p.add_argument("--out", default=None, help="optional JSON path for metrics")
    args = p.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.source:
        cfg.setdefault("data", {})["source"] = args.source
    if args.crop_size is not None:
        cfg.setdefault("data", {})["crop_size"] = args.crop_size
    if args.max_samples is not None:
        cfg.setdefault("data", {})["max_samples"] = args.max_samples
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        paths_all = yaml.safe_load(f) or {}
    natl_paths = paths_all.get("natl60") if isinstance(paths_all.get("natl60"), dict) else paths_all

    try:
        _, _, test_ds = make_datasets(cfg, ROOT, paths=natl_paths)
    except FileNotFoundError as exc:
        raise SystemExit(f"Data load failed:\n{exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(f"Dataset split error:\n{exc}") from exc
    loader = DataLoader(test_ds, batch_size=min(4, max(len(test_ds), 1)))

    # torch>=2 supports weights_only=; faceswap env is 1.12.1 — omit for compat
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu")
    ckpt_cfg = ckpt.get("config") or cfg
    mcfg = ckpt_cfg.get("model") or cfg["model"]
    use_sst = ckpt.get("use_sst", mcfg.get("use_sst", True))
    use_sqg = ckpt.get("use_sqg", mcfg.get("use_sqg", False))
    use_adv = ckpt.get("use_adv", mcfg.get("use_adv", False))
    use_uncert = ckpt.get("use_uncert", mcfg.get("use_uncert", False))

    scales = scales_from_dataset(test_ds, ckpt_cfg)
    model = build_fourdvarnet(
        ckpt_cfg,
        use_sst=use_sst,
        use_sqg=use_sqg,
        use_adv=use_adv,
        use_uncert=use_uncert,
        dx=scales["dx"],
        dy=scales["dy"],
        f0=scales["f0"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    dx_km = float(scales["dx"]) / 1000.0
    dt_s = float(scales.get("dt", (ckpt_cfg.get("model") or {}).get("dt_seconds", 86400.0)))

    model_ms, geo_ms = [], []
    # Domain-mean UV time series for temporal λ (need ≥8 samples).
    pred_uv_ts: list[float] = []
    truth_uv_ts: list[float] = []
    # NOTE: metrics below are **batch-mean of per-batch batch_metrics** (fine for
    # smoke / short runs). Full-test pooled statistics (concat all windows then
    # score once) are a Stage-B refinement — not claimed as paper Table numbers.
    # Inner 4DVar loop needs autograd even at inference - do not wrap in no_grad.
    for batch in loader:
        dx_b, dy_b = _batch_scales(batch, scales)
        pred = model(
            batch["y_ssh"],
            batch["z_sst"],
            batch["mask_ssh"],
            batch["mask_sst"],
            batch["u_geo"],
            batch["v_geo"],
            dx=dx_b,
            dy=dy_b,
        )
        truth = batch["truth"]
        pred_d = pred.detach()
        # Fair geostrophic baseline: pure OI/DUACS SSH + OI-only geostrophy.
        y_oi = batch.get("y_oi", batch["y_ssh"])
        u_g = batch.get("u_geo_oi", batch["u_geo"])
        v_g = batch.get("v_geo_oi", batch["v_geo"])
        geo_state = torch.cat([y_oi, u_g, v_g], dim=1)
        kw = dict(dx=dx_b, dy=dy_b, dx_km=dx_km)
        model_ms.append(batch_metrics(pred_d, truth, **kw))
        geo_ms.append(batch_metrics(geo_state, truth, **kw))

        # Per-sample domain-mean |UV| proxies for temporal resolved scale.
        for i in range(pred_d.shape[0]):
            pu = pred_d[i, 1].mean().item()
            pv = pred_d[i, 2].mean().item()
            tu = truth[i, 1].mean().item()
            tv = truth[i, 2].mean().item()
            pred_uv_ts.append(0.5 * (pu + pv))
            truth_uv_ts.append(0.5 * (tu + tv))

    mean_m = _mean_dicts(model_ms)
    mean_geo = _mean_dicts(geo_ms)

    # Temporal λ: only meaningful with a long enough ordered series.
    lam_t_note = "待补充"
    if len(pred_uv_ts) >= 8:
        lam_t_model = float(resolved_timescale(pred_uv_ts, truth_uv_ts, dt=dt_s))
        mean_m["lambda_t_uv_s"] = lam_t_model
        mean_m["lambda_t_uv_days"] = lam_t_model / 86400.0 if lam_t_model == lam_t_model else float("nan")
        lam_t_note = "resolved_timescale on domain-mean UV series"
    else:
        mean_m["lambda_t_uv_s"] = float("nan")
        mean_m["lambda_t_uv_days"] = float("nan")
        mean_m["lambda_t_status"] = "待补充"
        print(
            f"NOTE: resolved_timescale 待补充 "
            f"(need ≥8 test windows, got {len(pred_uv_ts)}; short smoke / crop runs skip temporal λ)"
        )

    print("=== 4DVarNet test metrics ===")
    print(f"  physics scales: dx={scales['dx']:.1f} m  dy={scales['dy']:.1f} m")
    print("  geostrophic baseline: OI/DUACS-only (not hybrid OI+sparse)")
    for k, v in mean_m.items():
        g = mean_geo.get(k, float("nan"))
        print(f"  {k}: {_fmt(v)}   (geostrophic {_fmt(g)})")
    print(f"  tau_uv (geostrophic baseline): {_fmt(mean_geo.get('tau_uv', float('nan')))}")
    print(f"  tau_uv (4DVarNet): {_fmt(mean_m.get('tau_uv', float('nan')))}")
    if len(pred_uv_ts) >= 8:
        print(f"  lambda_t_uv_days (model): {_fmt(mean_m.get('lambda_t_uv_days', float('nan')))}  [{lam_t_note}]")
    if mean_m.get("tau_uv", float("nan")) > mean_geo.get("tau_uv", float("-inf")):
        print("OK: model beats geostrophic baseline on this test split")
    else:
        print("WARN: model did not beat geostrophic baseline (needs more training or NATL60)")

    if args.out:
        payload = {
            "ckpt": str(Path(args.ckpt)),
            "source": str((cfg.get("data") or {}).get("source", "synthetic")),
            "flags": {
                "use_sst": bool(use_sst),
                "use_sqg": bool(use_sqg),
                "use_adv": bool(use_adv),
                "use_uncert": bool(use_uncert),
            },
            "physics_scales": {"dx": scales["dx"], "dy": scales["dy"], "f0": scales["f0"]},
            "geostrophic_baseline": "oi_only",
            "model": mean_m,
            "geostrophic": mean_geo,
            "lambda_t_note": lam_t_note if len(pred_uv_ts) >= 8 else "待补充 (need ≥8 windows)",
        }
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
