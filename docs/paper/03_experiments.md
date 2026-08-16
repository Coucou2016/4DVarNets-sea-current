# 3. Experiments (setup + preliminary results)

**Status:** working draft. All numeric values below are **local measured** results.  
**Hard caveat:** GPU96 = NATL60 **crop_size=96**, **20 epochs** — **not** JAMES Table / full-domain 200-ep protocol.

Figures: SciencePlots + Times New Roman via `scripts/plot_science.py` → `results/figures/` (mirrored in `docs/paper/figures/`).

---

## 3.1 OSSE protocols

### Synthetic (directional)

- Source: `synthetic` cache (`data/synthetic_osse.npz`).
- Short training (8 epochs in directional ablation) for wiring / ranking checks only.
- See `results/SYNTHETIC_ABLATION_NOTES.md` and `results/metrics_{B2,M3,M4}.json`.

### NATL60 Gulf Stream OSSE (paper protocol vs this draft)

Paper-ready rows require (`docs/PAPER_EXPERIMENTS.md`):

- Full Gulf Stream box **33–43°N, 65–55°W**
- Splits: train 2013-02-04→2013-09-30; val 2013-01-01→2013-02-04; test 2012-10-20→2012-12-04
- Finished ckpt + `results/metrics_<ID>.json` on **test**
- **Uncropped**, long epoch schedule (target ~200) for Table claims

**This draft reports only GPU96 crop96/20ep** for B2/M3/M4 — preliminary ranking evidence.

Hardware note: GTX 950M 4GB, `faceswap` env (torch 1.12.1+cu113).

---

## 3.2 Ablation matrix

| ID | SST | SQG | Adv | Uncert | Role |
|----|-----|-----|-----|--------|------|
| B2 | ✓ | | | | Multimodal synergy baseline (Fablet-like) |
| M3 | ✓ | ✓ | ✓ | | Physics residuals in cost |
| M4 | ✓ | ✓ | ✓ | ✓ | + strain UV uncertainty in supervised loss |
| geo | — | — | — | — | Geostrophic baseline (always co-reported) |

---

## 3.3 Preliminary results — GPU96 crop96 / 20ep

Source JSON: `results/metrics_{B2,M3,M4}_GPU96.json`.

| ID | τ_uv ↑ | rmse_uv ↓ | rmse_ssh ↓ | Notes |
|----|--------|-----------|------------|-------|
| **B2** | **0.848** | **0.184** | **0.059** | Best UV & SSH among learned models |
| M3 | 0.811 | 0.206 | 0.064 | SQG+adv slightly below B2 |
| M4 | 0.801 | 0.211 | 0.063 | **Post-NLL-fix retrain**; SSH recovered (~B2/M3); UV ≈ M3 |
| M4 (pre-fix) | 0.806 | 0.209 | **0.122** | Archived `metrics_M4_GPU96_pre_nllfix.json` — SSH starved by raw NLL |
| geo | −3.74 | 1.029 | 0.059 | All learned models beat geo on τ_uv / rmse_uv |

Ranking (this protocol, **post-fix**): **B2 > M3 ≳ M4** on currents; M4 SSH no longer degraded.

Training diagnostics (val loss at best ckpt): B2 ≈ 4.53; M3 ≈ 5.19; M4 **post-fix** ≈ **4.83** @ep15 (was ≈682 pre-fix). Pre-fix backup: `results/metrics_M4_GPU96_pre_nllfix.json`.

### Figures (captions draft)

**Figure 2 (synthetic).** `fig_synth_ablation_tau_uv`, `fig_synth_ablation_rmse_uv`  
*Caption:* Synthetic OSSE ablation (8 epochs). Directional only; not NATL60 Table metrics. On this short synthetic run, M4 ranked best among B2/M3/M4 after the truth-σ uncertainty fix.

**Figure 3 (GPU96).** `fig_GPU96_tau_uv`, `fig_GPU96_rmse_uv`, `fig_{B2,M3,M4}_GPU96_loss`  
*Caption:* NATL60 OSSE, **cropped 96×96, 20 epochs**. Explained variance and RMSE of surface currents for B2/M3/M4 vs geostrophy. **Preliminary / not paper Table.** B2 leads; M3 and M4 remain far above geostrophy but do not improve on B2. After σ-normalized NLL retrain, M4 SSH RMSE recovers (~0.063); UV still trails B2 (see `metrics_M4_GPU96.json`).

---

## 3.4 Contrast with synthetic 8-ep ranking

| Setting | Ranking (τ_uv) |
|---------|----------------|
| Synthetic 8ep (directional) | M4 best among B2/M3/M4 |
| NATL60 GPU96 crop96/20ep | B2 > M3 ≳ M4 |

This discrepancy is treated as a **scientific result**, not a bug to hide: physics extras can help under some regimes/scales/training lengths and hurt or be neutral under others (see Discussion).

---

## 3.5 What is *not* claimed yet

- No full-grid NATL60 200-ep Table.
- No OSE / drifter scores in this draft.
- No copying of Fablet 2024 JAMES table numbers as ours.
- GPU96 metrics must not be promoted to “demonstrates improvement of SQG+adv over SST synergy.”
