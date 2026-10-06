# Figure notes

- Style: SciencePlots (`science`+`ieee`) + Times New Roman for Latin; CJK fallback SimSun / Microsoft YaHei (`fourdvarnet/plotting.py`). DPI 300.
- Synthetic ablation figures are **directional** only (8-epoch synthetic OSSE).
- GPU96 figures are **cropped / short-epoch** NATL60 — **not** JAMES Table rows.
- Do not invent or paste JAMES paper table numbers into these plots.

## Paths

| Figure | `results/figures/` | `docs/paper/figures/` |
|--------|--------------------|------------------------|
| Synthetic τ_uv | `fig_synth_ablation_tau_uv.{png,pdf}` | same basename |
| Synthetic rmse_uv | `fig_synth_ablation_rmse_uv.{png,pdf}` | same |
| GPU96 τ_uv bars | `fig_GPU96_tau_uv.{png,pdf}` | same |
| GPU96 rmse_uv bars | `fig_GPU96_rmse_uv.{png,pdf}` | same |
| B2/M3/M4 loss | `fig_{B2,M3,M4}_GPU96_loss.{png,pdf}` | same |

## Caption drafts (for manuscript)

1. **Synthetic ablation.** Explained variance / RMSE of surface currents for B2, M3, M4 on synthetic OSSE (8 epochs). Directional wiring check only.
2. **GPU96 ablation.** Same metrics on NATL60 OSSE with spatial crop 96 and 20 training epochs. Preliminary; not comparable to full-domain JAMES table protocol. B2 leads; M3/M4 remain above geostrophy; M4 post-fix SSH ~0.063.
3. **Loss curves.** Validation loss vs epoch for B2/M3/M4-GPU96. M4 curve is **post-NLL-fix** retrain (best val ~4.83); pre-fix (~682) archived in `metrics_M4_GPU96_pre_nllfix.json`.
