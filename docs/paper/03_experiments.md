# 3. Experiments and results

**Protocol caveat.** Primary NATL60 numbers are **crop_size=96**, **15 epochs**, **seeds {0,1,2}** (tag `post_p0`). They are directional evidence on limited VRAM and are **not** JAMES full-domain ~200-epoch Table rows. Full-grid Table claims: **待补充**.

Canonical paths: `results/physics_ops/` (Stage E), `results/post_p0/` (Stages F–G). Legacy crop96/20ep JSON under `results/legacy_pre_review2/` is quarantined (`pre_p0_fix`) and is not used for scientific ranking claims.

## 3.1 Stage E — physics operators on NATL60 truth

Operators are evaluated against truth SST/SSH/UV on paper-mode NATL60 (obs+oi), crop96, 23 test days (`scripts/validate_physics_operators.py` → `results/physics_ops/physics_ops_validation.json`).

| Quantity | Measured |
|----------|---------:|
| SQG τ_uv mean | −0.011 |
| SQG corr_u / corr_v | 0.876 / 0.915 |
| SQG rmse_uv | 0.478 |
| Adv RMS truth / scrambled / zero | 6.06×10⁻⁶ / 2.22×10⁻⁵ / 4.97×10⁻⁶ |

Standalone SQG recovers correlated structure but near-zero explained variance of UV on this crop (`lam_sqg_caution`). Truth advection residuals beat scrambled fields; zero-flow residuals can be ≤ truth when \(\partial_t - \kappa\nabla^2\) dominates. Soft λ_sqg inside a learned solver must therefore not be equated with hard SQG current recovery.

## 3.2 Stages F–G — learned ablations (post_p0)

- Source: NATL60 paper mode; Gulf Stream box; paper date splits; OI-only geostrophic baseline.
- Train: crop96, batch size 1, 15 epochs, seeds {0,1,2}, CUDA (GTX 950M 4GB).
- Summary: `results/post_p0/ablation_summary.json` (mean ± std over seeds).

### Table 1. Measured post_p0 scores (mean of three seeds)

| ID | τ_uv ↑ | rmse_uv ↓ | rmse_ssh ↓ |
|----|-------:|----------:|-----------:|
| B1 | 0.861 ± 0.004 | 0.178 ± 0.003 | 0.059 ± 0.001 |
| B2 | 0.878 ± 0.027 | 0.166 ± 0.019 | 0.059 ± 0.006 |
| M1 | 0.916 ± 0.005 | 0.139 ± 0.004 | 0.049 ± 0.002 |
| M2 | 0.880 ± 0.025 | 0.165 ± 0.018 | 0.056 ± 0.003 |
| M3 | 0.914 ± 0.005 | 0.140 ± 0.004 | 0.050 ± 0.001 |
| M4 | 0.917 ± 0.003 | 0.137 ± 0.003 | 0.051 ± 0.002 |
| R0 | 0.850 ± 0.013 | 0.185 ± 0.008 | 0.059 ± 0.000 |
| geo (OI-only, B2-s0) | 0.846 | 0.188 | — |

Acceptance checks in the same JSON: geostrophic τ_uv is sane (~0.85; the pre-P0 pathological geo τ_uv ≈ −3.7 is fixed); B2 mean τ_uv exceeds B1.

**Interpretation (bounded).** Under this post_p0 protocol, soft SQG-containing configurations (M1, M3, M4) outperform multimodal B2 on mean τ_uv, while B2 still beats B1 and geostrophy. Advection alone (M2) is close to B2. Compact larger-capacity R0 does not beat B2 here and is not a byte-faithful official R0. These ranks reverse the quarantined pre-P0 crop96/20ep historical order (B2 > M3 ≳ M4), underscoring protocol dependence and the need for full-grid long-train confirmation (**待补充**).

Figures: `fig_post_p0_tau_uv`, `fig_post_p0_rmse_uv`, `fig_post_p0_rmse_ssh` (SciencePlots, 300 dpi); Stage E `fig_sqg_skill`, `fig_adv_residual_sanity`.

## 3.3 Quarantined historical context (not for Table claims)

Legacy NATL60 crop96/20ep (`results/legacy_pre_review2/`): B2 τ_uv 0.848, M3 0.811, M4 0.801 (post-NLL-fix), all above a then-broken geo diagnostic. Those files remain for engineering history only.

Synthetic 8-epoch OSSE metrics (`results/metrics_B2.json`, `M3`, `M4`) are directional code-path checks only.

## 3.4 Stage H — sensitivity (eval-time, frozen ckpts)

Source: `results/post_p0/sensitivity/sensitivity_summary.json` (B2-s0 vs M3-s0; crop96).

SST coarsening (average-pool then upsample) and altimetry thinning change scores only mildly on this crop; high-strain bins show larger UV RMSE than low-strain bins for both models. These are diagnostic, not full-grid Table rows.

## 3.5 What is not claimed

- No full-grid NATL60 ~200-epoch JAMES Table from this workstation.
- No copying of Fablet 2024 JAMES table numbers as ours.
- No claim that SQG alone recovers currents (Stage E τ_uv ≈ 0).
- No claim that M4 estimates uncertainty (strain reweighting only).
- No byte-faithful Fablet R0 comparison.
