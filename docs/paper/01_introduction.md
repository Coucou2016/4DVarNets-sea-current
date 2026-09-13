# 1. Introduction

**Status:** matured draft (evidence-calibrated; crop96/20ep ≠ paper Table).  
**Nature-skills axes:** `task=manuscript`, `paper_type=methods`, `journal=generic` (JAMES/GMD-style), `language=en`.

---

Sea-surface current (SSC) estimation from satellite altimetry remains limited by the effective resolution of sea-surface height (SSH) and by the geostrophic approximation, which under-represents ageostrophic and small-scale contributions in energetic western-boundary regimes. Synergistic use of sea-surface temperature (SST) with SSH is a long-standing route to finer-scale surface velocities, via surface quasi-geostrophy (SQG / eSQG), heat-budget / advection inversions, and, more recently, multimodal deep learning (e.g. Lapeyre & Klein, 2006; Rio et al., 2016; Martin et al., 2023; Fablet et al., 2024).

Fablet et al. (2024, JAMES) established a multimodal 4DVarNet solver for SST–SSH → SSC in NATL60 Gulf Stream observing-system simulation experiments (OSSEs), with trainable observation/prior operators inside an unrolled variational loop. That baseline already demonstrates strong SST–SSH synergy relative to geostrophy. What it does **not** make explicit in the variational cost are physics residuals motivated by eSQG mixing and SST advection, nor a strain-aware treatment of UV uncertainty. A complementary line—VarDyn / dynamical joint SSH–SST mapping (Le Guillou, Chapron & Rio, 2025, JAMES)—constrains SSH and SST with reduced dynamical models rather than learning SSC through an unrolled neural solver; we cite it as a dynamical counterpart, not a competing SSC Table.

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
