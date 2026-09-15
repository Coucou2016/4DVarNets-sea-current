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

    # --- GPU96 metrics bars; JSON under results/legacy_pre_review2/ ---
    gpu_ids = []
    gpu_tau = []
    gpu_geo = []
    gpu_rmse = []
    gpu_tau_div = []
    gpu_lam_uv = []
    legacy = results / "legacy_pre_review2"
    for name in ("B2_GPU96", "M3_GPU96", "M4_GPU96"):
        path = legacy / f"metrics_{name}.json"
        if not path.is_file():
            path = results / f"metrics_{name}.json"
        if not path.is_file():
            continue
        blob = _load(path)
        label = name.replace("_", "-")  # B2-GPU96 for axis labels
        gpu_ids.append(label)
        gpu_tau.append(_metric(blob, "tau_uv"))
        gpu_rmse.append(_metric(blob, "rmse_uv"))
        gpu_tau_div.append(_metric(blob, "tau_div"))
        gpu_lam_uv.append(_metric(blob, "lambda_x_uv_km"))
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

        fig = bar_compare(
            gpu_ids,
            {r"Model $\tau_{div}$": gpu_tau_div},
            ylabel=r"$\tau_{div}$",
            title="NATL60 crop96/20ep - NOT paper Table (diagnostics)",
        )
        written += save_figure(fig, fig_dir / "fig_GPU96_tau_div")
        fig.clf()

        fig = bar_compare(
            gpu_ids,
            {r"$\lambda_{x,uv}$ (km)": gpu_lam_uv},
            ylabel=r"$\lambda_{x,uv}$ (km)",
            title="NATL60 crop96/20ep - NOT paper Table (diagnostics)",
        )
        written += save_figure(fig, fig_dir / "fig_GPU96_lambda_x_uv")
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
                "- GPU96 / legacy figures under `results/legacy_pre_review2/` are **pre_p0_fix** - do not cite as Table rows.",
                "- **post_p0** figures (`fig_post_p0_*`) are crop96 / 15ep / multi-seed directional evidence after P0 fixes - still **not** full-grid JAMES Table rows.",
                "- Stage E physics-operator figures live under `results/physics_ops/` and are mirrored here.",
                "- Do not invent or paste JAMES paper table numbers into these plots.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # --- Mirror Stage E physics figures into results/figures ---
    phys_dir = results / "physics_ops"
    for stem in ("fig_sqg_skill", "fig_adv_residual_sanity"):
        for ext in (".png", ".pdf"):
            src = phys_dir / f"{stem}{ext}"
            if src.is_file():
                dst = fig_dir / f"{stem}{ext}"
                dst.write_bytes(src.read_bytes())
                written.append(dst)

    # --- post_p0 ablation summary (mean±std) ---
    post = results / "post_p0" / "ablation_summary.json"
    if post.is_file():
        ledger = _load(post)
        summary = ledger.get("summary") or {}
        # OI-only geo from any seed-0 metrics file (same protocol).
        geo_ref = float("nan")
        for seed_name in ("B2-s0", "B1-s0", "M3-s0"):
            mp = results / "post_p0" / f"metrics_{seed_name}.json"
            if mp.is_file():
                geo_ref = float((_load(mp).get("geostrophic") or {}).get("tau_uv", float("nan")))
                break

        ids_p: list[str] = []
        tau_m: list[float] = []
        tau_s: list[float] = []
        rmse_m: list[float] = []
        rmse_s: list[float] = []
        ssh_m: list[float] = []
        ssh_s: list[float] = []
        geo_m: list[float] = []
        for name in ("B1", "B2", "M1", "M2", "M3", "M4", "R0"):
            block = summary.get(name)
            if not block:
                continue
            mets = block.get("metrics") or {}
            m_tau = mets.get("tau_uv") or {}
            if m_tau.get("mean") is None:
                continue
            ids_p.append(name)
            tau_m.append(float(m_tau["mean"]))
            tau_s.append(float(m_tau.get("std") or 0.0))
            m_rmse = mets.get("rmse_uv") or {}
            rmse_m.append(float(m_rmse.get("mean") or float("nan")))
            rmse_s.append(float(m_rmse.get("std") or 0.0))
            m_ssh = mets.get("rmse_ssh") or {}
            ssh_m.append(float(m_ssh.get("mean") or float("nan")))
            ssh_s.append(float(m_ssh.get("std") or 0.0))
            g = (block.get("geostrophic_tau_uv") or {}).get("mean")
            geo_m.append(float(g) if g is not None else geo_ref)

        if ids_p:
            fig = bar_compare(
                ids_p,
                {"Model $\\tau_{uv}$ (mean)": tau_m, "Geostrophic $\\tau_{uv}$": geo_m},
                ylabel=r"$\tau_{uv}$",
                title="post_p0 NATL60 crop96/15ep (3 seeds) — not JAMES Table",
                yerr={"Model $\\tau_{uv}$ (mean)": tau_s},
            )
            written += save_figure(fig, fig_dir / "fig_post_p0_tau_uv")
            fig.clf()

            fig = bar_compare(
                ids_p,
                {"Model RMSE$_{uv}$ (mean)": rmse_m},
                ylabel=r"RMSE$_{uv}$",
                title="post_p0 NATL60 crop96/15ep (3 seeds) — not JAMES Table",
                yerr={"Model RMSE$_{uv}$ (mean)": rmse_s},
            )
            written += save_figure(fig, fig_dir / "fig_post_p0_rmse_uv")
            fig.clf()

            fig = bar_compare(
                ids_p,
                {"Model RMSE$_{ssh}$ (mean)": ssh_m},
                ylabel=r"RMSE$_{ssh}$",
                title="post_p0 NATL60 crop96/15ep (3 seeds) — not JAMES Table",
                yerr={"Model RMSE$_{ssh}$ (mean)": ssh_s},
            )
            written += save_figure(fig, fig_dir / "fig_post_p0_rmse_ssh")
            fig.clf()

            (results / "post_p0" / "figure_tau_uv_means.json").write_text(
                json.dumps(
                    {
                        "ids": ids_p,
                        "tau_uv_mean": tau_m,
                        "tau_uv_std": tau_s,
                        "rmse_uv_mean": rmse_m,
                        "rmse_uv_std": rmse_s,
                        "rmse_ssh_mean": ssh_m,
                        "rmse_ssh_std": ssh_s,
                        "geo_tau_uv": geo_m,
                        "protocol": ledger.get("protocol"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    report_fig = ROOT / "docs" / "report" / "figures"
    mirrored = mirror_to(written, paper_fig)
    mirrored += mirror_to(written, report_fig)
    print("Wrote:")
    for p in written + mirrored:
        print(f"  {p}")


if __name__ == "__main__":
    main()