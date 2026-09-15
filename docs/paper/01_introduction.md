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
