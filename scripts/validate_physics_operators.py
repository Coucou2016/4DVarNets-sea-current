#!/usr/bin/env python3
"""Stage E: validate SQG / advection operators on NATL60 truth (no network training).

Uses paper-mode loader (obs+oi required). Writes JSON + SciencePlots figures under
``results/physics_ops/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.natl60 import load_natl60  # noqa: E402
from fourdvarnet.metrics import isotropic_psd_2d, rmse  # noqa: E402
from fourdvarnet.physics import (  # noqa: E402
    explained_variance,
    sqg_velocity,
    sst_advection_residual,
)
from fourdvarnet.plotting import apply_science_style, save_figure  # noqa: E402
from fourdvarnet.repro import git_commit  # noqa: E402


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1).float()
    bb = b.reshape(-1).float()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = torch.sqrt((aa**2).sum() * (bb**2).sum()) + 1e-12
    return float((aa * bb).sum() / denom)


def _spectral_coherence(u_pred: np.ndarray, u_truth: np.ndarray, dx: float) -> dict:
    """Crude magnitude-squared coherence proxy via PSD cross-ratio."""
    p_pred, k = isotropic_psd_2d(u_pred, dx)
    p_truth, _ = isotropic_psd_2d(u_truth, dx)
    # Cross-spectrum proxy: |FFT(pred) conj FFT(truth)| averaged radially is expensive;
    # report band-mean PSD ratio as a coarse spectral agreement diagnostic.
    valid = (k > 0) & np.isfinite(p_pred) & np.isfinite(p_truth) & (p_truth > 0)
    if not np.any(valid):
        return {"mean_psd_ratio": float("nan"), "n_bins": 0}
    ratio = p_pred[valid] / (p_truth[valid] + 1e-12)
    return {
        "mean_psd_ratio": float(np.mean(ratio)),
        "median_psd_ratio": float(np.median(ratio)),
        "n_bins": int(np.sum(valid)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Stage E physics operator validation")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--crop-size", type=int, default=96, help="center crop (VRAM/IO); null via 0=full")
    p.add_argument("--n-samples", type=int, default=24, help="number of analysis days to score")
    p.add_argument("--stride", type=int, default=3, help="sample every N days in test window")
    p.add_argument("--out-dir", default="results/physics_ops")
    p.add_argument("--device", default="cpu", help="cpu recommended for Stage E")
    args = p.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        paths_all = yaml.safe_load(f) or {}
    natl_paths = paths_all.get("natl60") if isinstance(paths_all.get("natl60"), dict) else paths_all
    mcfg = cfg.get("model") or {}
    phys = cfg.get("physics") or {}

    print("Loading NATL60 paper-mode (obs+oi required)...")
    data = load_natl60(natl_paths, root=ROOT, allow_truth_background=False)
    crop = None if int(args.crop_size) <= 0 else int(args.crop_size)
    if crop is not None:
        from data.dataset import _center_crop_spatial

        data = _center_crop_spatial(data, crop)

    ssh = data["ssh"]
    sst = data["sst"]
    u = data["u"]
    v = data["v"]
    mask_sst = data["mask_sst"]
    times = data["time"]
    dx = float(np.asarray(data["dx_m"]).reshape(-1)[0])
    dy = float(np.asarray(data["dy_m"]).reshape(-1)[0])
    f0 = float(np.asarray(data["f"]).reshape(-1)[0]) if "f" in data else float(phys.get("f0", 7e-5))
    Ld = float(mcfg.get("Ld_km", 30.0)) * 1e3
    kappa = float(mcfg.get("kappa", 50.0))
    dt = float(mcfg.get("dt_seconds", 86400.0))
    dT = int((cfg.get("data") or {}).get("dT", 7))

    # Prefer test-split days when available
    from data.natl60 import indices_for_split

    try:
        test_idx = indices_for_split(times, "test", dT)
    except Exception:
        test_idx = list(range(dT - 1, len(times)))
    if not test_idx:
        test_idx = list(range(dT - 1, len(times)))
    sampled = test_idx[:: max(int(args.stride), 1)][: int(args.n_samples)]
    if len(sampled) < 3:
        sampled = test_idx[: max(3, min(len(test_idx), int(args.n_samples)))]

    device = torch.device(args.device)
    sqg_rows = []
    adv_rows = []

    for t in sampled:
        ssh_t = torch.from_numpy(ssh[t : t + 1].astype(np.float32)).to(device)
        sst_t = torch.from_numpy(sst[t : t + 1].astype(np.float32)).to(device)
        u_t = torch.from_numpy(u[t : t + 1].astype(np.float32)).to(device)
        v_t = torch.from_numpy(v[t : t + 1].astype(np.float32)).to(device)
        sst_seq = torch.from_numpy(sst[t - dT + 1 : t + 1].astype(np.float32)).to(device)
        m_sst = torch.from_numpy(mask_sst[t - dT + 1 : t + 1].astype(np.float32)).to(device)

        u_sqg, v_sqg = sqg_velocity(sst_t, ssh_t, Ld=Ld, f=f0, dx=dx, dy=dy)
        row = {
            "t_index": int(t),
            "rmse_u": rmse(u_sqg, u_t),
            "rmse_v": rmse(v_sqg, v_t),
            "rmse_uv": float(
                torch.sqrt(0.5 * ((u_sqg - u_t) ** 2 + (v_sqg - v_t) ** 2).mean()).item()
            ),
            "corr_u": _corr(u_sqg, u_t),
            "corr_v": _corr(v_sqg, v_t),
            "tau_u": explained_variance(u_sqg, u_t),
            "tau_v": explained_variance(v_sqg, v_t),
            "tau_uv": explained_variance(
                torch.stack([u_sqg, v_sqg], dim=0),
                torch.stack([u_t, v_t], dim=0),
            ),
        }
        coh = _spectral_coherence(
            u_sqg.detach().cpu().numpy(),
            u_t.detach().cpu().numpy(),
            dx=dx,
        )
        row["spectral_coherence_u"] = coh
        sqg_rows.append(row)

        # Advection residual: truth flow vs scrambled vs zero
        res_truth = sst_advection_residual(
            sst_seq, u_t, v_t, kappa=kappa, dt=dt, dx=dx, dy=dy, mask_sst=m_sst
        )
        # Scramble: spatially shuffle u,v (same marginals, destroy alignment)
        flat_u = u_t.reshape(-1)
        flat_v = v_t.reshape(-1)
        perm = torch.randperm(flat_u.numel(), device=device)
        u_scr = flat_u[perm].reshape_as(u_t)
        v_scr = flat_v[perm].reshape_as(v_t)
        res_scr = sst_advection_residual(
            sst_seq, u_scr, v_scr, kappa=kappa, dt=dt, dx=dx, dy=dy, mask_sst=m_sst
        )
        res_zero = sst_advection_residual(
            sst_seq,
            torch.zeros_like(u_t),
            torch.zeros_like(v_t),
            kappa=kappa,
            dt=dt,
            dx=dx,
            dy=dy,
            mask_sst=m_sst,
        )
        adv_rows.append(
            {
                "t_index": int(t),
                "rms_truth": float(torch.sqrt((res_truth**2).mean()).item()),
                "rms_scrambled": float(torch.sqrt((res_scr**2).mean()).item()),
                "rms_zero": float(torch.sqrt((res_zero**2).mean()).item()),
            }
        )

    def _mean_key(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    sqg_summary = {
        "n_samples": len(sqg_rows),
        "rmse_uv_mean": _mean_key(sqg_rows, "rmse_uv"),
        "corr_u_mean": _mean_key(sqg_rows, "corr_u"),
        "corr_v_mean": _mean_key(sqg_rows, "corr_v"),
        "tau_uv_mean": _mean_key(sqg_rows, "tau_uv"),
        "spectral_psd_ratio_mean": float(
            np.mean(
                [
                    r["spectral_coherence_u"]["mean_psd_ratio"]
                    for r in sqg_rows
                    if r["spectral_coherence_u"].get("mean_psd_ratio")
                    == r["spectral_coherence_u"].get("mean_psd_ratio")
                ]
            )
        )
        if sqg_rows
        else float("nan"),
    }
    adv_summary = {
        "n_samples": len(adv_rows),
        "rms_truth_mean": _mean_key(adv_rows, "rms_truth"),
        "rms_scrambled_mean": _mean_key(adv_rows, "rms_scrambled"),
        "rms_zero_mean": _mean_key(adv_rows, "rms_zero"),
        "truth_beats_scrambled": _mean_key(adv_rows, "rms_truth")
        < _mean_key(adv_rows, "rms_scrambled"),
        "truth_beats_zero": _mean_key(adv_rows, "rms_truth") < _mean_key(adv_rows, "rms_zero"),
    }

    tau = sqg_summary["tau_uv_mean"]
    near_zero_skill = (tau != tau) or (abs(tau) < 0.05)
    caution = None
    if near_zero_skill:
        caution = (
            "SQG operator has near-zero / negative skill vs truth UV on this sample. "
            "Treat lam_sqg as a soft prior with caution; do not claim SQG recovers currents alone."
        )

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "E",
        "git_commit": git_commit(ROOT),
        "crop_size": crop,
        "scales": {"dx": dx, "dy": dy, "f0": f0, "Ld": Ld, "kappa": kappa, "dt": dt},
        "sampled_t_indices": [int(t) for t in sampled],
        "sqg": {"summary": sqg_summary, "per_sample": sqg_rows},
        "advection": {"summary": adv_summary, "per_sample": adv_rows},
        "lam_sqg_caution": caution,
        "near_zero_sqg_skill": near_zero_skill,
    }
    out_json = out_dir / "physics_ops_validation.json"
    out_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"SQG tau_uv_mean={sqg_summary['tau_uv_mean']:.4f}  rmse_uv={sqg_summary['rmse_uv_mean']:.4f}")
    print(
        f"Adv RMS truth={adv_summary['rms_truth_mean']:.4e}  "
        f"scrambled={adv_summary['rms_scrambled_mean']:.4e}  "
        f"zero={adv_summary['rms_zero_mean']:.4e}"
    )
    if caution:
        print("CAUTION:", caution)

    # Figures
    try:
        import matplotlib.pyplot as plt

        apply_science_style()
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        labels = ["truth u,v", "scrambled", "zero flow"]
        vals = [
            adv_summary["rms_truth_mean"],
            adv_summary["rms_scrambled_mean"],
            adv_summary["rms_zero_mean"],
        ]
        ax.bar(labels, vals, color=["#2c7bb6", "#fdae61", "#d7191c"])
        ax.set_ylabel("SST advection residual RMS")
        ax.set_title("Stage E: advection residual sanity")
        save_figure(fig, out_dir / "fig_adv_residual_sanity")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        keys = ["rmse_uv", "corr_u", "corr_v", "tau_uv"]
        means = [sqg_summary["rmse_uv_mean"], sqg_summary["corr_u_mean"], sqg_summary["corr_v_mean"], sqg_summary["tau_uv_mean"]]
        ax.bar(keys, means, color="#4daf4a")
        ax.set_title("Stage E: SQG operator vs truth UV")
        ax.axhline(0.0, color="k", lw=0.6)
        save_figure(fig, out_dir / "fig_sqg_skill")
        plt.close(fig)
        print(f"Figures under {out_dir}")
    except Exception as exc:  # pragma: no cover
        print(f"WARN: figure generation failed: {exc}")


if __name__ == "__main__":
    main()
