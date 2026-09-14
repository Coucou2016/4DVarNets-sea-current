# 3. Experiments (setup + preliminary results)

**Status:** matured draft. All numeric values below are **local measured** results.  
**Hard caveat:** GPU96 = NATL60 **crop_size=96**, **20 epochs** — **not** JAMES Table / full-domain 200-ep protocol.

> **P0 (2026-09):** GPU96 / synthetic metrics JSON are tagged `pre_p0_fix` and are
> **obsolete for formal claims** after Phase-1 correctness fixes. See
> `docs/REVIEW_RESPONSE_P0.md`. Numbers below are **historical pre-fix** context only.

Figures: SciencePlots + Times New Roman via `scripts/plot_science.py` → `results/figures/` (mirrored in `docs/paper/figures/`). Expanded summary JSON: `results/legacy_pre_review2/metrics_GPU96_expanded_summary.json` (`legacy_pre_review2` / `pre_p0_fix`).

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

**Hardware honesty.** GTX 950M 4GB (`faceswap` env, torch 1.12.1+cu113) cannot host uncropped ~200×200 / ~200-ep Table training in this environment. We therefore (i) label crop96/20ep as preliminary, (ii) expand measured diagnostics (τ_div, λ_x, loss curves), and (iii) do **not** invent JAMES Table scores. B1 SSH-only crop96/20ep remains a desirable completeness run when GPU memory is free.

---

## 3.2 Ablation matrix

| ID | SST | SQG | Adv | Uncert | Role | Status (this draft) |
|----|-----|-----|-----|--------|------|---------------------|
| B1 | | | | | SSH-only baseline | **Not measured** this session |
| B2 | ✓ | | | | Multimodal synergy (Fablet-like) | Measured GPU96 |
| M3 | ✓ | ✓ | ✓ | | Physics residuals in cost | Measured GPU96 |
| M4 | ✓ | ✓ | ✓ | ✓ | + strain-aware spatial reweighting | Measured GPU96 (post-NLL-fix; **pre_p0_fix**) |
| geo | — | — | — | — | Geostrophic baseline | Always co-reported |

---

## 3.3 Preliminary results — GPU96 crop96 / 20ep

Source JSON: `results/legacy_pre_review2/metrics_{B2,M3,M4}_GPU96.json` (+ pre-fix archive).

### Table A — primary UV / SSH skill

| ID | τ_uv ↑ | rmse_uv ↓ | rmse_ssh ↓ | Notes |
|----|--------|-----------|------------|-------|
| **B2** | **0.848** | **0.184** | **0.059** | Best UV & SSH among learned models |
| M3 | 0.811 | 0.206 | 0.064 | SQG+adv slightly below B2 |
| M4 | 0.801 | 0.211 | 0.063 | **Post-NLL-fix retrain**; SSH recovered; UV ≈ M3 |
| M4 (pre-fix) | 0.806 | 0.209 | **0.122** | Archived — SSH starved by raw NLL |
| geo | −3.74 | 1.029 | 0.059 | All learned models beat geo on τ_uv / rmse_uv |

Ranking (this protocol, **post-fix**): **B2 > M3 ≳ M4** on currents; M4 SSH no longer degraded.

Training diagnostics (val loss at best ckpt): B2 ≈ 4.53; M3 ≈ 5.19; M4 **post-fix** ≈ **4.83** @ep15 (was ≈682 pre-fix).

### Table B — expanded diagnostics (same JSON; still crop96/20ep)

| ID | τ_div | τ_vort | τ_strain | λ_x,ssh (km) | λ_x,uv (km) |
|----|-------|--------|----------|--------------|-------------|
| B2 | **0.354** | **0.535** | **0.571** | 102.4 | 90.3 |
| M3 | −1.317 | −0.635 | −0.908 | 97.6 | 79.6 |
| M4 | −1.480 | −0.858 | −1.316 | 83.3 | 86.1 |
| geo | ≈0 | −56.1 | −80.5 | 122.2 | 101.6 |

**Reading Table B.** On this preliminary protocol, B2 alone shows positive explained variance for divergence/vorticity/strain; M3/M4 are negative on these diagnostics despite competitive τ_uv. This strengthens the partial-result narrative: physics residuals do not automatically improve dynamical structure scores under crop96/20ep. λ_x values are reported for completeness; they are **not** promoted to paper-Table resolved-scale claims.

### Figures (captions)

**Figure 2 (synthetic).** `fig_synth_ablation_tau_uv`, `fig_synth_ablation_rmse_uv`  
*Caption:* Synthetic OSSE ablation (8 epochs). Directional only; not NATL60 Table metrics. On this short synthetic run, M4 ranked best among B2/M3/M4 after the truth-σ reweight fix.

**Figure 3 (GPU96 primary).** `fig_GPU96_tau_uv`, `fig_GPU96_rmse_uv`, `fig_{B2,M3,M4}_GPU96_loss`  
*Caption:* NATL60 OSSE, **cropped 96×96, 20 epochs**. Explained variance and RMSE of surface currents for B2/M3/M4 vs geostrophy; loss curves for each run. **Preliminary / not paper Table.** B2 leads; M3 and M4 remain far above geostrophy but do not improve on B2.

**Figure 3b (GPU96 diagnostics).** `fig_GPU96_tau_div`, `fig_GPU96_lambda_x_uv`  
*Caption:* Same protocol. Divergence explained variance and λ_x,uv from measured JSON. Diagnostic only; still ≠ JAMES Table.

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
- No B1 GPU96 row in this draft.
- No OSE / drifter scores in this draft.
- No copying of Fablet 2024 JAMES table numbers as ours.
- GPU96 metrics must not be promoted to “demonstrates improvement of SQG+adv over SST synergy.”
- No fabricated multi-seed confidence intervals.
