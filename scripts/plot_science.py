#!/usr/bin/env python3
"""Regenerate SciencePlots figures from metrics JSON / training history."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fourdvarnet.plotting import (  # noqa: E402
    apply_science_style,
    bar_compare,
    loss_curves,
    mirror_to,
    save_figure,
)


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metric(blob: dict, key: str):
    m = blob.get("model") or {}
    v = m.get(key)
    return None if v is None else float(v)


def _plot_gpu96_loss(exp: str, fig_dir: Path, written: list) -> None:
    """exp like B2-GPU96; history file history-B2-GPU96.json; fig fig_B2_GPU96_loss."""
    hist_path = ROOT / "checkpoints" / f"history-{exp}.json"
    if not hist_path.is_file():
        return
    hist = _load(hist_path)
    epochs = [int(h["epoch"]) for h in hist]
    train = [float(h["train"]) for h in hist]
    val = [float(h["val"]) for h in hist]
    fig = loss_curves(
        epochs,
        train,
        val,
        title=f"{exp} NATL60 crop96 (20 ep) - not paper Table",
    )
    fig_stem = exp.replace("-", "_")
    written += save_figure(fig, fig_dir / f"fig_{fig_stem}_loss")
    fig.clf()


def main() -> None:
    apply_science_style()
    results = ROOT / "results"
    fig_dir = results / "figures"
    paper_fig = ROOT / "docs" / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paper_fig.mkdir(parents=True, exist_ok=True)

    written = []

    # --- Synthetic ablation bars (directional only) ---
    ids = ["B2", "M3", "M4"]
    tau, rmse, geo_tau = [], [], []
    for name in ids:
        blob = _load(results / f"metrics_{name}.json")
        tau.append(_metric(blob, "tau_uv"))
        rmse.append(_metric(blob, "rmse_uv"))
        geo = blob.get("geostrophic") or {}
        geo_tau.append(float(geo.get("tau_uv", float("nan"))))

    fig = bar_compare(
        ids,
        {"Model tau_uv": tau, "Geostrophic tau_uv": geo_tau},
        ylabel=r"$\tau_{uv}$",
        title="Synthetic OSSE (8 ep) - directional only",
    )
    written += save_figure(fig, fig_dir / "fig_synth_ablation_tau_uv")
    fig.clf()

    fig = bar_compare(
        ids,
        {"Model RMSE_uv": rmse},
        ylabel=r"RMSE$_{uv}$",
        title="Synthetic OSSE (8 ep) - directional only",
    )
    written += save_figure(fig, fig_dir / "fig_synth_ablation_rmse_uv")
    fig.clf()

    # --- GPU96 loss curves (history uses hyphenated exp names) ---
    for exp in ("B2-GPU96", "M3-GPU96", "M4-GPU96"):
        _plot_gpu96_loss(exp, fig_dir, written)

    # --- GPU96 metrics bars; JSON uses underscores: metrics_B2_GPU96.json ---
    gpu_ids = []
    gpu_tau = []
    gpu_geo = []
    gpu_rmse = []
    for name in ("B2_GPU96", "M3_GPU96", "M4_GPU96"):
        path = results / f"metrics_{name}.json"
        if not path.is_file():
            continue
        blob = _load(path)
        label = name.replace("_", "-")  # B2-GPU96 for axis labels
        gpu_ids.append(label)
        gpu_tau.append(_metric(blob, "tau_uv"))
        gpu_rmse.append(_metric(blob, "rmse_uv"))
        gpu_geo.append(float((blob.get("geostrophic") or {}).get("tau_uv", float("nan"))))

    if gpu_ids:
        fig = bar_compare(
            gpu_ids,
            {"Model tau_uv": gpu_tau, "Geostrophic tau_uv": gpu_geo},
            ylabel=r"$\tau_{uv}$",
            title="NATL60 crop96/20ep - NOT paper Table row",
        )
        written += save_figure(fig, fig_dir / "fig_GPU96_tau_uv")
        fig.clf()

        fig = bar_compare(
            gpu_ids,
            {"Model RMSE_uv": gpu_rmse},
            ylabel=r"RMSE$_{uv}$",
            title="NATL60 crop96/20ep - NOT paper Table row",
        )
        written += save_figure(fig, fig_dir / "fig_GPU96_rmse_uv")
        fig.clf()

        if "B2-GPU96" in gpu_ids:
            i = gpu_ids.index("B2-GPU96")
            fig = bar_compare(
                ["B2-GPU96"],
                {
                    "Model tau_uv": [gpu_tau[i]],
                    "Geostrophic tau_uv": [gpu_geo[i]],
                },
                ylabel=r"$\tau_{uv}$",
                title="NATL60 crop96/20ep - NOT paper Table row",
            )
            written += save_figure(fig, fig_dir / "fig_B2_GPU96_tau_uv")
            fig.clf()

    notes = results / "figures" / "FIGURE_NOTES.md"
    notes.write_text(
        "\n".join(
            [
                "# Figure notes",
                "",
                "- Style: SciencePlots (`science`+`ieee`) + Times New Roman; CJK fallback SimSun/Microsoft YaHei (`fourdvarnet/plotting.py`). DPI 300.",
                "- Synthetic ablation figures are **directional** only (8-epoch synthetic OSSE).",
                "- GPU96 figures are **cropped / short-epoch** NATL60 — **not** JAMES Table rows.",
                "- Do not invent or paste JAMES paper table numbers into these plots.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mirrored = mirror_to(written, paper_fig)
    print("Wrote:")
    for p in written + mirrored:
        print(f"  {p}")


if __name__ == "__main__":
    main()