# Physics-constrained 4DVarNet for SST–SSH sea-surface current inversion

**Assembled manuscript (working).** Crop96/20ep ≠ JAMES Table.  
**Public code:** https://github.com/Coucou2016/4DVarNets-sea-current  
**Assembled:** 2026-09-14

**Axes (nature-writing):** `task=manuscript`, `paper_type=methods`, `journal=generic` (JAMES / GMD-style methods paper; Nature-family *clarity*, not flagship Nature format), `language=en`.

---

## Key Points

- Explicit eSQG-style SQG, SST-advection, and optional strain-aware UV **reweighting** can be embedded in a compact 4DVarNet-**inspired** cost (solver graph was updated for full unrolled autodiff; not a byte-faithful Fablet clone).
- Historical NATL60 **crop96 / 20-epoch** OSSE scores (**`pre_p0_fix` / obsolete for claims**) ranked **B2 > M3 ≳ M4** on τ_uv; all beat geostrophy. Physics extras are not universally additive.
- A raw heteroscedastic NLL scale bug starved M4 SSH; σ-normalized UV reweight + retrain recovered SSH while UV still trailed B2. Cropped short runs are **not** paper Table rows. Post-P0 retrain required before any formal score.

## Plain Language Summary

Satellites measure sea level and sea-surface temperature more easily than they measure ocean currents directly. Learning models that combine those measurements can estimate currents better than the classical geostrophic approximation. We test whether adding explicit physics “checks” (surface quasi-geostrophy and temperature advection) inside a variational neural solver helps further. In our short, cropped simulation tests, the simpler sea-level + temperature model still wins; the physics checks remain useful to implement and diagnose, but they do not automatically improve scores. We also show how a poorly scaled strain-reweighting loss can quietly damage sea-level skill, and how to fix that scaling.

## Abstract

Estimating sea-surface currents (SSC) from satellite sea-surface height (SSH) remains limited by altimeter resolution and by the geostrophic approximation. Multimodal 4DVarNet solvers that synergize SSH with sea-surface temperature (SST) already improve SSC relative to geostrophy in NATL60 observing-system simulation experiments (OSSEs). Here we use a compact 4DVarNet-**inspired** ConvLSTM unrolled solver (not a byte-faithful Fablet reproduction; the solver autodiff graph was corrected) and extend the variational cost / supervised loss with (i) an effective eSQG-style SQG residual, (ii) an SST advection residual, and (iii) optional strain-aware spatial UV **reweighting** (M4 — not uncertainty estimation). Under a controlled ablation (B2: SSH+SST; M3: +SQG+advection; M4: +reweighting), synthetic short runs can favor physics extras, but historical NATL60 **crop_size=96, 20-epoch** scores (**`pre_p0_fix`**) ranked **B2 > M3 ≳ M4** on explained variance τ_uv, with all configurations beating geostrophy. An M4 SSH degradation is traced to raw NLL scale mismatch and repaired by a σ-normalized UV term. We therefore frame physics residuals as implementable, ablatable, and **regime-/hyperparameter-dependent**, not universally additive. Cropped short-epoch scores must not be read as JAMES Table rows; post-P0 retrain and uncropped long-epoch NATL60 claims remain gated.

---

# 1. Introduction

**Status:** matured draft (evidence-calibrated; crop96/20ep ≠ paper Table).  
**Nature-skills axes:** `task=manuscript`, `paper_type=methods`, `journal=generic` (JAMES/GMD-style), `language=en`.

---

Sea-surface current (SSC) estimation from satellite altimetry remains limited by the effective resolution of sea-surface height (SSH) and by the geostrophic approximation, which under-represents ageostrophic and small-scale contributions in energetic western-boundary regimes. Synergistic use of sea-surface temperature (SST) with SSH is a long-standing route to finer-scale surface velocities, via surface quasi-geostrophy (SQG / eSQG), heat-budget / advection inversions, and, more recently, multimodal deep learning (e.g. Lapeyre & Klein, 2006; Rio et al., 2016; Martin et al., 2023; Fablet et al., 2024).

Fablet et al. (2024, JAMES) established a multimodal 4DVarNet solver for SST–SSH → SSC in NATL60 Gulf Stream observing-system simulation experiments (OSSEs), with trainable observation/prior operators inside an unrolled variational loop. That baseline already demonstrates strong SST–SSH synergy relative to geostrophy. What it does **not** make explicit in the variational cost are physics residuals motivated by eSQG mixing and SST advection, nor a strain-aware spatial treatment of UV errors. A complementary line—VarDyn / dynamical joint SSH–SST mapping (Le Guillou, Chapron & Rio, 2025, JAMES)—constrains SSH and SST with reduced dynamical models rather than learning SSC through an unrolled neural solver; we cite it as a dynamical counterpart, not a competing SSC Table.

**Gap.** Without explicit physics residuals, it is hard to attribute gains to dynamical constraints versus learned multimodal synergy alone, and harder to diagnose when SQG/advection priors help or hurt. Conversely, naively adding physics terms can degrade skill if operators are misspecified, poorly scaled, or regime-dependent—e.g. when SST is a weak proxy for surface density, mixed-layer motions dominate, or interior potential vorticity contributes strongly to surface velocity (Isern-Fontanet / González-Haro transfer-function literature; Miracca-Lage et al., 2022; Yassin & Griffies, 2023, on variable-stratification SQG regimes).

**This work.** We use a compact 4DVarNet-**inspired** ConvLSTM unrolled solver (not a byte-faithful Fablet/IMT reproduction) and extend the variational cost / supervised loss with (i) an *effective* eSQG-style SQG residual, (ii) an SST advection residual (final-time backward for single-time state), and (iii) optional strain-aware spatial UV **reweighting** (M4 — not a learned uncertainty head). We evaluate a controlled ablation matrix (B2 / M3 / M4) under identical training protocols on synthetic and cropped NATL60 OSSEs.

**Contributions (honest).**

1. **Physics residuals inside the unrolled cost** — code-mapped operators for SQG, advection, and strain-aware UV reweighting.
2. **Ablation evidence under a fair protocol** — isolating SST synergy (B2) vs SQG+adv (M3) vs +reweighting (M4), with geostrophy co-reported.
3. **Partial / negative result on short cropped NATL60 (pre-P0 historical)** — on GPU96 crop96/20ep, **B2 outperformed M3 ≳ M4** on τ_uv; all still beat geostrophy. Those scores are **`pre_p0_fix` / obsolete for claims** after Phase-1 correctness fixes (`docs/REVIEW_RESPONSE_P0.md`). Physics extras are **regime- and hyperparameter-dependent**, not universally additive.
4. **M4 loss-scale diagnosis** — raw heteroscedastic NLL with σ₀≪1 can swamp SSH terms; we document and mitigate the scale mismatch (σ-normalized UV reweight + retrain).

**Boundary.** Cropped / short-epoch GPU runs are **preliminary** and are **not** JAMES Table rows. Full-domain NATL60 long training and OSE/drifter evaluation remain future evidence gates. Hardware (GTX 950M 4GB) currently precludes uncropped 200×200 / ~200-ep Table runs; we therefore strengthen what *is* measured (expanded diagnostics τ_div, λ_x; honest ranking) rather than invent Table scores.

### Key citations (verified DOI)

- Fablet et al. 2024 JAMES — https://doi.org/10.1029/2023MS003609  
- Beauchamp et al. 2023 GMD (4DVarNet-SSH) — https://doi.org/10.5194/gmd-16-2119-2023  
- Lapeyre & Klein 2006 JPO — https://doi.org/10.1175/JPO2840.1  
- Rio et al. 2016 JTECH — https://doi.org/10.1175/JTECH-D-16-0017.1  
- González-Haro & Isern-Fontanet / related — https://doi.org/10.1029/2019JC015958  
- Martin et al. 2023 JAMES — https://doi.org/10.1029/2022MS003589  
- Le Guillou, Chapron & Rio 2025 JAMES (VarDyn) — https://doi.org/10.1029/2024MS004689  
- Miracca-Lage et al. 2022 JGR Oceans (SQG regimes) — https://doi.org/10.1029/2021JC018001  
- Yassin & Griffies 2023 JPO (variable-stratification SQG) — https://doi.org/10.1175/JPO-D-22-0040.1

# 2. Methods

**Status:** matured draft; equations mapped to repository modules.  
Compact 4DVarNet-**inspired** ConvLSTM unrolled solver (not a byte-faithful Fablet/IMT
reproduction). Physics residuals live in the variational cost / supervised loss; the
solver graph **was changed** for Phase-1 P0 (full unrolled autodiff, no per-iter
`detach`; see `Solver4DVarNet` vs ablation `Solver4DVarNetTruncated`).

---

## 2.1 State and unrolled variational solver

State at each analysis time is \(x = (\eta, u, v)\) (SSH and surface currents). Observations provide gappy SSH \(y\) and an SST window \(z\) of length \(dT\) (default 7 days). The solver performs \(K\) unrolled gradient steps on a variational cost \(U(x)\), with LSTM-parameterized updates (ConvLSTM), in the spirit of 4DVarNet.

| Piece | Code |
|-------|------|
| Model wrapper | `fourdvarnet/model.py` → `FourDVarNetUV` |
| Unrolled solver | `fourdvarnet/solver.py` → `Solver4DVarNet` |
| Prior \(\Phi\) | `fourdvarnet/prior.py` → `PhiPrior` |
| Obs operators \(G,H\) | `fourdvarnet/observation.py` → `ObservationOperator` |

Initial state uses observed/OI SSH plus geostrophic \((u_g, v_g)\). The **geostrophic baseline** for metrics uses **pure OI/DUACS SSH** (not the hybrid OI+sparse model input).

Grid spacings \(dx, dy\) for SQG / advection / divergence come from dataset meter metrics when NATL60 provides lat-aware scales; synthetic falls back to config isotropic \(dx_{\deg}\times 111\,\mathrm{km}\).

---

## 2.2 Variational cost with physics residuals

Following Fablet’s multimodal cost and extending it:

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

Implementation: `VariationalCost` in `fourdvarnet/solver.py`.

**Flags (ablation):** `use_sst`, `use_sqg`, `use_adv` (config / CLI). Default λ weights in `config/default.yaml` (`lam_obs`, `lam_sst`, `lam_prior`, `lam_sqg`, `lam_adv`).

**Contrast to VarDyn.** VarDyn (Le Guillou et al., 2025) jointly maps SSH and SST with QG / advection–diffusion dynamical constraints inside a classical 4DVar-style scheme. Our cost adds *SSC-facing* eSQG-style and SST-advection residuals inside a **learned unrolled** 4DVarNet-inspired solver aimed at \((u,v)\), rather than replacing the neural solver with a reduced dynamical propagator.

---

## 2.3 Effective eSQG-style operator \(A_{\mathrm{SQG}}\)

We use an *effective* surface QG–style mixing of SST and SSH anomalies to predict a velocity field (not a full 3D SQG inversion). Code: `physics.sqg_velocity` (`fourdvarnet/physics.py`), with deformation radius `Ld` (`Ld_km` in config), Coriolis `f0`, gravity `g`, and grid spacings `dx, dy` from the dataset (or `physics.dx_deg` fallback).

**Assumptions / limits.** SST is treated as a usable surface-density proxy; phase/amplitude relations may fail in mixed-layer–dominated or strongly unbalanced regimes, or when interior PV contributes to surface velocity (Miracca-Lage et al., 2022; Yassin & Griffies, 2023). This motivates reporting cases where SQG residuals do **not** improve over pure SST synergy.

---

## 2.4 SST advection residual

Heat-budget style residual (final-time backward for single-time state):

\[
r_{\mathrm{adv}} = \partial_t T + u\,\partial_x T + v\,\partial_y T - \kappa \nabla^2 T,
\]

with \(\partial_t T = (T_t - T_{t-1})/\Delta t\) from consecutive SST frames in the \(dT\) window. Code: `physics.sst_advection_residual`. Requires \(dT \ge 2\) SST frames.

---

## 2.5 Supervised training loss and strain reweighting (M4)

Outside the inner cost, training minimizes a weighted supervised loss on SSH, \(\nabla\)SSH, UV, divergence, and prior consistency (`fourdvarnet/losses.py` → `TrainingLoss`; weights under `loss:` in config).

**M4 — strain-aware spatial UV reweighting (not uncertainty estimation).** Let \(\sigma = \sigma_0 (1 + \alpha\,\mathrm{strain}(u,v))\), clamped to \(\sigma \le \sigma_0 \cdot m_{\max}\). By default σ is computed from **truth** UV (`uncert_from_truth`) to block the collapse mode where predicted strain is inflated to shrink a heteroscedastic term. There is **no** learned σ head.

Normalized UV reweight term (MSE units when \(\sigma \approx \sigma_0\)):

\[
L_{\mathrm{uv}}^{\sigma}
= \mathbb{E}\Big[ \|e_{uv}\|^2 \big(\sigma_0/\sigma\big)^2 \Big]
+ \sigma_0^2\,\mathbb{E}\big[\log(\sigma/\sigma_0)^2\big],
\]

mixed with MSE: \(L_{uv} = m\,L_{\mathrm{MSE}} + (1-m)\,L_{\mathrm{uv}}^{\sigma}\) (`uncert_mse_mix`).

**Why normalize.** Raw \(\mathbb{E}[\|e\|^2/\sigma^2 + \log\sigma^2]\) with \(\sigma_0 \approx 0.05\) is \(\mathcal{O}(1/\sigma_0^2)\) larger than MSE and can dominate SSH terms (observed on GPU96 M4: best val \(\sim 682\), `rmse_ssh` roughly \(2\times\) B2). The normalized form keeps UV and SSH on a comparable scale.

Code map: `physics.strain_uncertainty`, `TrainingLoss` (`use_uncert`).

---

## 2.6 Ablation IDs

| ID | Meaning | Flags |
|----|---------|-------|
| B1 | SSH-only (no SST) | sst=0, sqg=0, adv=0, uncert=0 |
| B2 | SSH+SST (Fablet-like synergy) | sst=1, sqg=0, adv=0, uncert=0 |
| M3 | + SQG + advection | sst=1, sqg=1, adv=1, uncert=0 |
| M4 | + strain UV spatial reweighting | sst=1, sqg=1, adv=1, uncert=1 |

Full matrix B1/M1/M2 documented in `docs/PAPER_EXPERIMENTS.md`. **B1 crop96/20ep** is desirable for ablation completeness but was **not run in this session** (GPU occupied / 4GB memory budget shared with other jobs); we report measured B2/M3/M4 only. GPU96 scores are **`pre_p0_fix` / obsolete for formal claims**.

---

## 2.7 Metrics

Primary: \(\tau_{uv}\) (explained variance), `rmse_uv`, `rmse_ssh`. Diagnostics from the same JSON: \(\tau_{\mathrm{div}}\), \(\tau_{\mathrm{vort}}\), \(\tau_{\mathrm{strain}}\), resolved scales \(\lambda_x\) (SSH and UV). Temporal \(\lambda_t\) (`resolved_timescale`) is reported when ≥8 test windows are available; otherwise evaluate output marks it **待补充**. Always co-report **OI-only geostrophic** baseline on the same split (`fourdvarnet/metrics.py`, `scripts/evaluate.py`).

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

# 4. Discussion

**Status:** matured draft. Interprets measured crop96/20ep and synthetic evidence only.

---

## 4.1 Main interpretation

On the **short cropped NATL60 OSSE** (historical **`pre_p0_fix`** scores), explicit SQG + advection residuals (M3) and strain **reweighting** (M4) **do not improve** surface-current skill over the pure SST–SSH synergy baseline (B2). All three learned configurations still **strongly outperform geostrophy** on τ_uv and rmse_uv. Expanded diagnostics reinforce the message: only B2 shows positive τ_div / τ_vort / τ_strain on this protocol. The honest claim is therefore:

> Physics-constrained cost terms are implementable inside a 4DVarNet-**inspired** solver and remain competitive with geostrophy, but **SST synergy alone can dominate** under this cropped/short-epoch protocol; SQG/advection/reweighting are **not universally additive**.

This is a useful **partial / negative result** for methods papers: it bounds when “more physics in the cost” helps, and it documents a concrete loss-scale failure mode (raw NLL) that can masquerade as a dynamical failure.

---

## 4.2 Why B2 > M3 is plausible

Rival explanations (not mutually exclusive):

1. **Operator misspecification.** Our \(A_{\mathrm{SQG}}\) is eSQG-*style*, not full interior+surface QG. When SST poorly tracks surface density (mixed layer, unbalanced motions), or when interior PV contributes to surface velocity, the SQG residual may pull velocities toward a biased attractor (Miracca-Lage et al., 2022; Yassin & Griffies, 2023; isQG motivations in Wang et al., 2013).
2. **Redundant information.** Multimodal \(G/H\) SST–SSH terms (B2) may already capture much of the transferable SST structure; adding λ_sqg / λ_adv with fixed weights can over-constrain the unrolled trajectory on short training.
3. **Scale / crop artefacts.** Crop96 windows truncate mesoscale context that SQG/advection residuals assume; short 20-ep schedules may favor the simpler B2 loss landscape.
4. **Hyperparameter regime.** Default `lam_sqg=lam_adv=0.1` was not re-tuned on NATL60 GPU96; under- or over-weighted physics can erase gains.

Synthetic 8-ep ranking (M4 best) vs GPU96 ranking (B2 best) supports **regime dependence**: directional wiring success ≠ NATL60 cropped transfer.

---

## 4.3 M4 SSH degradation (diagnosed) and post-fix retrain

**Pre-fix GPU96** matched M3 on tau_uv but roughly **doubled** rmse_ssh (0.122 vs ~0.06). Best-val ~682 vs ~5 for B2/M3 was a **loss-scale bug**, not a deep dynamical failure:

- Raw heteroscedastic NLL `||e||^2/sigma^2` with `sigma_0≈0.05` is `~1/sigma_0^2` larger than MSE.
- UV gradients then dominate; SSH terms become relatively weak — SSH fit suffers while UV remains OK via the strain reweighting.

**Mitigation + retrain (2026-08-16):** sigma-normalized UV term + `uncert_mse_mix: 0.5`; full M4-GPU96 **20ep retrain** on faceswap CUDA. Post-fix: best val **4.83**, rmse_ssh **0.063**, tau_uv **0.801**, rmse_uv **0.211**. SSH recovered near B2/M3; currents still trail B2 (ranking unchanged: B2 > M3 ≳ M4). Pre-fix metrics archived as `results/legacy_pre_review2/metrics_M4_GPU96_pre_nllfix.json`.

---

## 4.4 Framing innovation without overclaim

Defensible:

- Explicit, ablatable physics residuals inside a Fablet-like 4DVarNet cost.
- Engineering diagnosis of reweighting / NLL scale mismatch.
- Evidence that multimodal SST synergy can outperform added SQG/adv on a constrained OSSE — a caution for “physics always helps” narratives.
- Transparent contrast to dynamical joint SSH–SST mapping (VarDyn) without claiming SSC Table superiority.

Not claimed:

- Universal superiority of M3/M4 over B2.
- Full 3D SQG inversion.
- Paper-table NATL60 scores from crop96/20ep.
- That B1 SSH-only was measured here (it was not).

---

## 4.5 Next evidence gates

1. **Full-grid / longer NATL60** (uncropped, ≥ paper epoch budget) for B2/M3/M4 with fixed seeds — **blocked on 4GB GPU** until hardware upgrades or off-machine compute.
2. **B1 crop96/20ep** when GPU is free, for ablation completeness (SSH-only vs SST synergy).
3. **λ_sqg / λ_adv / reweighting (M4) sweep** on a small grid before locking Table configs.
4. ~~Re-train M4 with normalized NLL~~ **Done** (SSH 0.063); next: multi-seed short runs if memory allows.
5. **OSE / drifters** (`scripts/evaluate_drifters.py`) once OSSE Table rows exist.
6. Optional: M1/M2 single-term ablations to separate SQG vs advection on full protocol.

---

## 4.6 Limitations checklist

- Single GPU class (4GB) forced crop96; may interact with physics residuals; uncropped 200-ep Table runs not feasible here.
- No formal significance / multi-seed intervals yet.
- Strain α / dx calibration still under review for physical units.
- B1 GPU96 missing from the measured matrix.
- ChatGPT browser automation and Codex advisory quota were blocked this session (see `docs/chatgpt_collaboration/SESSION_5ROUNDS.md`); literature DOIs verified via Cursor WebSearch; pastes left for human ChatGPT re-review.
- Public code+docs: https://github.com/Coucou2016/4DVarNets-sea-current

# 5. Conclusions

1. Explicit eSQG-style SQG, SST-advection, and optional strain-aware UV **reweighting** can be embedded in a compact 4DVarNet-**inspired** cost (solver autodiff graph corrected; not an unchanged / byte-faithful Fablet clone).
2. Historical NATL60 **crop96/20ep** scores (**`pre_p0_fix`**) ranked **B2 > M3 ≳ M4** on τ_uv; all beat geostrophy. Physics extras are **not universally additive** under that protocol.
3. M4 SSH degradation was an NLL scale bug; σ-normalized reweight + retrain recovered SSH while UV still trailed B2.
4. Paper-table claims require post-P0 retrain on uncropped, long-epoch NATL60 (+ optional OSE/drifters). On current 4GB hardware we strengthen measured crop96 diagnostics rather than invent Table scores.

# Open Research

Code, metrics JSON, and figures: https://github.com/Coucou2016/4DVarNets-sea-current (excludes large `*.nc` / `*.pt` weights). NATL60 source data follow the Fablet/Oceanix OSSE distribution cited in Fablet et al. (2024).

# References (key DOIs verified)

- Fablet et al. 2024 JAMES — https://doi.org/10.1029/2023MS003609
- Beauchamp et al. 2023 GMD — https://doi.org/10.5194/gmd-16-2119-2023
- Lapeyre & Klein 2006 JPO — https://doi.org/10.1175/JPO2840.1
- Rio et al. 2016 JTECH — https://doi.org/10.1175/JTECH-D-16-0017.1
- Martin et al. 2023 JAMES — https://doi.org/10.1029/2022MS003589
- González-Haro & Isern-Fontanet related — https://doi.org/10.1029/2019JC015958
- Le Guillou, Chapron & Rio 2025 JAMES (VarDyn) — https://doi.org/10.1029/2024MS004689
- Miracca-Lage et al. 2022 JGR Oceans — https://doi.org/10.1029/2021JC018001
- Yassin & Griffies 2023 JPO — https://doi.org/10.1175/JPO-D-22-0040.1
