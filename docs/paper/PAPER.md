# Soft differentiable physics in a compact 4DVarNet for SST–SSH sea-surface current inversion

**Manuscript draft (methods / JAMES–GMD style).** Primary scores: post_p0 NATL60 crop96 / 15 epochs / 3 seeds — **not** full-grid JAMES Table rows.  
**Code:** https://github.com/Coucou2016/4DVarNets-sea-current  
**Assembled:** 2026-09-15

---

## Key Points

- Soft eSQG-style SQG and SST-advection residuals (plus optional strain-aware UV reweighting) can be embedded in a compact 4DVarNet-inspired unrolled cost without claiming a byte-faithful Fablet solver.
- On measured post_p0 crop96/15ep multi-seed NATL60 OSSEs, soft SQG-containing configurations (M1/M3/M4) improve mean τ_uv relative to SSH+SST (B2), while a standalone SQG current map has near-zero τ_uv skill—soft residuals ≠ hard SQG inversion.
- Full-domain ~200-epoch JAMES Table scores remain **待补充**; legacy pre-P0 crop96/20ep ranks are quarantined and are not used as formal claims.

## Plain Language Summary

Satellites observe sea level and sea-surface temperature more readily than ocean currents. Neural variational solvers that combine those observations can estimate currents better than the classical geostrophic approximation. We test whether adding soft physics checks—surface quasi-geostrophy and temperature advection—inside such a solver helps further. In our cropped NATL60 simulation experiments, those soft checks can improve current skill relative to using sea level and temperature alone, even though a stand-alone physics map of currents performs poorly. We treat this as protocol-dependent evidence and do not claim full-domain paper-table scores until longer, uncropped training is available.

## Abstract

Estimating sea-surface currents (SSC) from satellite sea-surface height (SSH) remains limited by altimeter resolution and by the geostrophic approximation. Multimodal 4DVarNet solvers that synergize SSH with sea-surface temperature (SST) improve SSC relative to geostrophy in NATL60 observing-system simulation experiments (OSSEs). Here we use a compact 4DVarNet-inspired ConvLSTM unrolled solver and extend the variational cost with soft, differentiable residuals: an effective eSQG-style SQG term, an SST advection term, and optional strain-aware spatial UV reweighting. In a controlled ablation under a post-P0 protocol (NATL60 crop_size=96, 15 epochs, three seeds), soft SQG-containing models achieve higher mean explained variance τ_uv than SSH+SST alone, while a standalone SQG operator exhibits near-zero τ_uv against truth currents on the same crop. We therefore frame physics residuals as useful soft constraints inside a learned solver, not as hard current maps, and contrast this setting with VarDyn-style dynamical joint SSH–SST mapping. Cropped short-epoch scores must not be read as JAMES Table rows; full-grid long-epoch claims remain gated (**待补充**). All reported numbers are computed in this repository; we do not reproduce external paper tables as our results.

---

# 1. Introduction

Sea-surface current (SSC) estimation from satellite altimetry remains limited by the effective resolution of sea-surface height (SSH) and by the geostrophic approximation, which under-represents ageostrophic and small-scale contributions in energetic western-boundary regimes. A long-standing strategy is to combine sea-surface temperature (SST) with SSH—through surface quasi-geostrophy (SQG / eSQG), heat-budget or advection inversions, and, more recently, multimodal deep learning (Lapeyre & Klein, 2006; Rio et al., 2016; Martin et al., 2023; Fablet et al., 2024).

Fablet et al. (2024, JAMES) established a multimodal 4DVarNet solver for SST–SSH → SSC in NATL60 Gulf Stream observing-system simulation experiments (OSSEs), with trainable observation and prior operators inside an unrolled variational loop. That baseline already demonstrates strong SST–SSH synergy relative to geostrophy. What it does not make explicit in the variational cost are soft physics residuals motivated by eSQG mixing and SST advection, nor a controlled ablation of when those terms help or hurt. A complementary line—VarDyn, a dynamical joint SSH–SST mapping method (Le Guillou, Chapron & Rio, 2025, JAMES)—constrains SSH and SST with reduced dynamical models in a variational scheme; we cite it as a dynamical counterpart that jointly maps tracers, not as a competing SSC table under our protocol.

**Gap.** Without explicit, ablatable physics residuals, it is difficult to attribute gains to dynamical constraints versus learned multimodal synergy alone, and harder to diagnose when SQG or advection priors are misspecified. Conversely, adding physics terms can degrade skill if operators are poorly scaled or regime-dependent—for example when SST is a weak proxy for surface density, mixed-layer motions dominate, or interior potential vorticity contributes strongly to surface velocity (transfer-function and SQG-regime literature; Miracca-Lage et al., 2022; Yassin & Griffies, 2023).

**This work.** We use a compact 4DVarNet-inspired ConvLSTM unrolled solver (not a byte-faithful Fablet reproduction) and extend the variational cost and supervised loss with (i) an effective eSQG-style SQG residual, (ii) an SST advection residual, and (iii) optional strain-aware spatial UV reweighting (configuration M4—reweighting, not a learned uncertainty head). We evaluate a controlled ablation matrix (B1, B2, M1–M4, R0) under an identical post-P0 training protocol on cropped NATL60 OSSEs, and we separately validate the physics operators against NATL60 truth (Stage E).

**Contributions**

1. Soft differentiable SQG and advection residuals inside an unrolled 4DVarNet-inspired cost, with code-mapped operators and geostrophy co-reported.
2. A fair ablation isolating SSH-only (B1), SST synergy (B2), SQG (M1), advection (M2), both (M3), and strain reweighting (M4).
3. Measured post-P0 evidence (crop96 / 15 epochs / three seeds) showing that, under this protocol, soft physics configurations (notably M1/M3/M4) can improve τ_uv relative to B2, while a standalone SQG map of currents has near-zero τ_uv skill—illustrating that soft residuals inside a learned solver are not equivalent to hard SQG inversion.
4. An explicit contrast with VarDyn: dynamical tracer mapping versus learned SSC with optional soft physics.

**Boundary.** All primary numerical claims below use metrics JSON produced in this repository. Cropped short-epoch scores are **directional** and are **not** JAMES full-domain Table rows. Full-grid ~200-epoch NATL60 Table scores remain **待补充**. We never report Fablet et al. (2024) table entries as our results. Historical pre-P0 crop96/20ep ranks (B2 > M3 ≳ M4) are quarantined under `legacy_pre_review2` / `pre_p0_fix`.

### Key citations (DOI verified)

- Fablet et al. 2024 JAMES — https://doi.org/10.1029/2023MS003609  
- Beauchamp et al. 2023 GMD — https://doi.org/10.5194/gmd-16-2119-2023  
- Lapeyre & Klein 2006 JPO — https://doi.org/10.1175/JPO2840.1  
- Rio et al. 2016 JTECH — https://doi.org/10.1175/JTECH-D-16-0017.1  
- Martin et al. 2023 JAMES — https://doi.org/10.1029/2022MS003589  
- Le Guillou, Chapron & Rio 2025 JAMES (VarDyn) — https://doi.org/10.1029/2024MS004689  
- González-Haro / Isern-Fontanet related — https://doi.org/10.1029/2019JC015958  
- Miracca-Lage et al. 2022 JGR Oceans — https://doi.org/10.1029/2021JC018001  
- Yassin & Griffies 2023 JPO — https://doi.org/10.1175/JPO-D-22-0040.1

# 2. Methods

We describe a compact 4DVarNet-inspired ConvLSTM unrolled solver (not a byte-faithful Fablet/IMT reproduction). Soft physics residuals enter the variational cost and the supervised loss. After Phase-1 correctness fixes, the solver uses full unrolled automatic differentiation without per-iteration `detach` (see `Solver4DVarNet`; a truncated variant remains available for ablation).

## 2.1 State and unrolled variational solver

State at each analysis time is \(x = (\eta, u, v)\) (SSH and surface currents). Observations provide gappy SSH \(y\) and an SST window \(z\) of length \(dT\) (default 7 days). The solver performs \(K\) unrolled gradient steps on a variational cost \(U(x)\), with LSTM-parameterized updates (ConvLSTM), in the spirit of 4DVarNet (Beauchamp et al., 2023; Fablet et al., 2024).

| Piece | Module |
|-------|--------|
| Model wrapper | `fourdvarnet/model.py` → `FourDVarNetUV` |
| Unrolled solver | `fourdvarnet/solver.py` → `Solver4DVarNet` |
| Prior \(\Phi\) | `fourdvarnet/prior.py` → `PhiPrior` |
| Obs operators \(G,H\) | `fourdvarnet/observation.py` → `ObservationOperator` |

The initial state uses observed or optimally interpolated (OI) SSH plus geostrophic \((u_g, v_g)\). The **geostrophic baseline** for metrics uses pure OI/DUACS SSH (not the hybrid OI+sparse model input). Grid spacings \(dx, dy\) for SQG, advection, and divergence come from dataset meter metrics when NATL60 provides latitude-aware scales; synthetic data fall back to an isotropic configuration scale.

## 2.2 Variational cost with soft physics residuals

Following the multimodal 4DVarNet cost and extending it with soft residuals,

\[
\begin{aligned}
U(x)
&= \lambda_{\mathrm{obs}}\,\| \eta - y \|^2_{\Omega}
+ \lambda_{\mathrm{sst}}\,\| G(\eta) - H(z) \|^2 \\
&+ \lambda_{\Phi}\,\| x - \Phi(x) \|^2 \\
&+ \mathbf{1}_{\mathrm{SQG}}\,\lambda_{\mathrm{sqg}}\,\| (u,v) - A_{\mathrm{SQG}}(z,\eta) \|^2 \\
&+ \mathbf{1}_{\mathrm{adv}}\,\lambda_{\mathrm{adv}}\,\| \partial_t T + \mathbf{u}\cdot\nabla T - \kappa\nabla^2 T \|^2 .
\end{aligned}
\]

Implementation: `VariationalCost` in `fourdvarnet/solver.py`. Ablation flags (`use_sst`, `use_sqg`, `use_adv`) and default λ weights live in `config/default.yaml`.

**Contrast to VarDyn.** VarDyn (Le Guillou et al., 2025) jointly maps SSH and SST with quasi-geostrophic and advection–diffusion constraints inside a classical variational scheme. Our formulation targets SSC \((u,v)\) through a learned unrolled solver and treats SQG/advection as *soft* residuals that can be switched off, rather than replacing the neural propagator with a reduced dynamical model.

## 2.3 Effective eSQG-style operator \(A_{\mathrm{SQG}}\)

We use an effective surface-QG-style mixing of SST and SSH anomalies to predict a velocity field (not a full three-dimensional SQG inversion). Code: `physics.sqg_velocity` in `fourdvarnet/physics.py`, with deformation radius \(L_d\), Coriolis \(f_0\), gravity \(g\), and grid spacings from the dataset.

**Assumptions and limits.** SST is treated as a usable surface-density proxy. Phase and amplitude relations may fail when mixed-layer or unbalanced motions dominate, or when interior potential vorticity contributes strongly to surface velocity. Stage E (Section 3) therefore reports standalone SQG skill against NATL60 truth before interpreting \(\lambda_{\mathrm{sqg}}\) as a hard constraint.

## 2.4 SST advection residual

Heat-budget style residual (final-time backward difference for a single-time state),

\[
r_{\mathrm{adv}} = \partial_t T + u\,\partial_x T + v\,\partial_y T - \kappa \nabla^2 T,
\]

with \(\partial_t T = (T_t - T_{t-1})/\Delta t\) from consecutive SST frames in the \(dT\) window (`physics.sst_advection_residual`). This requires \(dT \ge 2\).

## 2.5 Supervised training loss and strain reweighting (M4)

Outside the inner cost, training minimizes a weighted supervised loss on SSH, \(\nabla\)SSH, UV, divergence, and prior consistency (`fourdvarnet/losses.py` → `TrainingLoss`).

**M4 — strain-aware spatial UV reweighting (not uncertainty estimation).** Let \(\sigma = \sigma_0 (1 + \alpha\,\mathrm{strain}(u,v))\), clamped to \(\sigma \le \sigma_0 \cdot m_{\max}\). By default σ is computed from truth UV (`uncert_from_truth`) to avoid a collapse mode in which predicted strain is inflated to shrink a heteroscedastic term. There is no learned σ head.

Normalized UV reweight term (MSE units when \(\sigma \approx \sigma_0\)):

\[
L_{\mathrm{uv}}^{\sigma}
= \mathbb{E}\Big[ \|e_{uv}\|^2 \big(\sigma_0/\sigma\big)^2 \Big]
+ \sigma_0^2\,\mathbb{E}\big[\log(\sigma/\sigma_0)^2\big],
\]

mixed with MSE: \(L_{uv} = m\,L_{\mathrm{MSE}} + (1-m)\,L_{\mathrm{uv}}^{\sigma}\). Raw heteroscedastic NLL with \(\sigma_0 \ll 1\) can dominate SSH terms; the normalized form keeps UV and SSH on a comparable scale (documented historically on pre-P0 M4 runs).

## 2.6 Ablation identifiers

| ID | Meaning | Flags |
|----|---------|-------|
| B1 | SSH-only | sst=0, sqg=0, adv=0, uncert=0 |
| B2 | SSH+SST synergy | sst=1, sqg=0, adv=0, uncert=0 |
| M1 | + SQG residual | sst=1, sqg=1, adv=0, uncert=0 |
| M2 | + advection residual | sst=1, sqg=0, adv=1, uncert=0 |
| M3 | + SQG + advection | sst=1, sqg=1, adv=1, uncert=0 |
| M4 | + strain UV reweighting | sst=1, sqg=1, adv=1, uncert=1 |
| R0 | Larger compact SSH+SST capacity | not byte-faithful Fablet R0 |
| geo | OI-only geostrophy | evaluation baseline |

## 2.7 Metrics

Primary scores: \(\tau_{uv}\) (explained variance of surface currents), `rmse_uv`, `rmse_ssh`. Diagnostics from the same evaluation JSON include \(\tau_{\mathrm{div}}\), \(\tau_{\mathrm{vort}}\), \(\tau_{\mathrm{strain}}\), and resolved scales \(\lambda_x\) (SSH and UV). Every model score is co-reported with the OI-only geostrophic baseline on the same split (`fourdvarnet/metrics.py`, `scripts/evaluate.py`).

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

# 4. Discussion

## 4.1 Soft physics inside a learned solver

A central empirical point is the gap between Stage E and Stages F–G. The standalone eSQG-style map has near-zero τ_uv against truth UV on the same crop, yet enabling a soft SQG residual inside the unrolled cost (M1/M3/M4) improves mean post_p0 τ_uv relative to B2. In other words, the residual acts as a regularizer coupled to multimodal observations and the learned prior, not as a hard current estimate. This distinction matters for interpretation: “physics helps” here means soft constraints in a neural variational loop, not that SQG inversion alone solves SSC.

Advection alone (M2) yields scores close to B2, consistent with Stage E evidence that the heat-budget residual can be dominated by \(\partial_t - \kappa\nabla^2\) on daily frames. Combining SQG and advection (M3) behaves similarly to SQG alone (M1) under this protocol; strain reweighting (M4) yields a further small mean τ_uv gain with reduced seed scatter.

## 4.2 When physics helps or fails

**Helps (this protocol).** Soft SQG-containing losses improve UV skill and SSH RMSE relative to B2 on post_p0 crop96/15ep multi-seed runs, while all learned models beat OI-only geostrophy.

**Fails or is fragile.** (i) Hard SQG as a current map fails on τ_uv (Stage E). (ii) Pre-P0 historical crop96/20ep runs ranked B2 above M3/M4, showing that short protocols and correctness bugs can reverse rankings. (iii) Raw heteroscedastic UV weighting previously starved SSH; scale normalization was required before M4 was interpretable. (iv) Larger compact R0 capacity did not automatically beat B2.

Taken together, physics residuals are implementable and ablatable, but **regime- and hyperparameter-dependent**. Formal generalization to full-domain long training remains **待补充**.

## 4.3 Relation to VarDyn and Fablet et al.

Fablet et al. (2024) demonstrate learned SST–SSH synergy for SSC with 4DVarNets; we do not re-quote their Table scores. Our contribution is the soft residual ablation and operator diagnostics under a transparent protocol. VarDyn (Le Guillou et al., 2025) jointly reconstructs SSH and SST with dynamical constraints; it is a natural dynamical counterpart for tracer mapping, whereas our solver targets SSC with optional soft SQG/advection residuals. Cross-code benchmarking against VarDyn is out of scope here.

## 4.4 Limitations

Hardware (4GB GPU) precludes uncropped ~200×200 × ~200-epoch multi-seed tables. R0 is not byte-faithful to CIA-Oceanix. OSE/drifter evaluation hooks exist but formal scores are **待补充**. Stage H sensitivity (SST coarsening / sparsity) is reported when `results/post_p0/sensitivity/` is populated; otherwise marked incomplete in the audit package.

# 5. Conclusions

1. Soft eSQG-style SQG and SST-advection residuals, with optional strain-aware UV reweighting, can be embedded in a compact 4DVarNet-inspired unrolled cost.
2. Measured post_p0 crop96/15ep multi-seed NATL60 scores show soft SQG-containing configurations improving mean τ_uv relative to B2, while Stage E shows standalone SQG τ_uv ≈ 0—soft physics ≠ hard SQG inversion.
3. Rankings are protocol-dependent; legacy pre-P0 crop96/20ep results are quarantined. Full-grid ~200-epoch JAMES Table scores remain **待补充**.
4. Relative to VarDyn, our contribution is soft SSC-facing residuals inside a learned unrolled solver, not dynamical joint tracer mapping.

# Open Research

Code, metrics JSON, and figures: https://github.com/Coucou2016/4DVarNets-sea-current (excludes large `*.nc` / `*.pt` weights). NATL60 source data follow the OSSE distribution cited in Fablet et al. (2024). Evidence audit: `docs/AUDIT_EVIDENCE.md`.

# References (key DOIs verified)

- Fablet, R., Chapron, B., Le Sommer, J., & Sévellec, F. (2024). Inversion of sea surface currents from satellite-derived SST-SSH synergies with 4DVarNets. *Journal of Advances in Modeling Earth Systems*, 16, e2023MS003609. https://doi.org/10.1029/2023MS003609
- Beauchamp, M., Febvre, Q., Georgenthum, H., & Fablet, R. (2023). 4DVarNet-SSH: end-to-end learning of variational interpolation schemes for nadir and wide-swath satellite altimetry. *Geoscientific Model Development*, 16, 2119–2147. https://doi.org/10.5194/gmd-16-2119-2023
- Le Guillou, F., Chapron, B., & Rio, M.-H. (2025). VarDyn: Dynamical joint-reconstructions of sea surface height and temperature from multi-sensor satellite observations. *Journal of Advances in Modeling Earth Systems*, 17, e2024MS004689. https://doi.org/10.1029/2024MS004689
- Lapeyre, G., & Klein, P. (2006). Dynamics of the upper oceanic layers in terms of surface quasigeostrophy theory. *Journal of Physical Oceanography*, 36, 165–176. https://doi.org/10.1175/JPO2840.1
- Rio, M.-H., Santoleri, R., Bourdalle-Badie, R., et al. (2016). Improving the altimeter-derived surface currents using high-resolution sea surface temperature data: A feasibility study based on model outputs. *Journal of Atmospheric and Oceanic Technology*. https://doi.org/10.1175/JTECH-D-16-0017.1
- Martin, M. J., et al. (2023). Synergistic use of SSH and SST. *Journal of Advances in Modeling Earth Systems*. https://doi.org/10.1029/2022MS003589
- González-Haro, C., & Isern-Fontanet, J. (related). https://doi.org/10.1029/2019JC015958
- Miracca-Lage, M., et al. (2022). *Journal of Geophysical Research: Oceans*. https://doi.org/10.1029/2021JC018001
- Yassin, H., & Griffies, S. M. (2023). *Journal of Physical Oceanography*. https://doi.org/10.1175/JPO-D-22-0040.1
