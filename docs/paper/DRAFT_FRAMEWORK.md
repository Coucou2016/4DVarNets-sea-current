# Draft framework — Physics-constrained multimodal 4DVarNet

This document is the working manuscript framework after local literature verification and nature-skills (`methods` paper type) guidance. It is **not** a submission draft; no invented metrics.

## Nature-skills routing used

- Skill root: `C:\Users\Administrator\.cursor\skills\nature-skills\`
- Followed: `nature-writing` (methods playbook + workflow one-sentence argument + terminology ledger) and `nature-figure` (Python backend; user override: SciencePlots + Times New Roman instead of default Arial)
- Stance: no fabricated novelty, calibrate verbs to evidence, surface missing inputs
- Section drafts: `01_introduction.md`, `02_methods.md`, `03_experiments.md`, `04_discussion.md`

## Core claim (bounded, updated)

We extend the Fablet et al. (2024) multimodal 4DVarNet cost with **explicit physics residuals** (eSQG-style SQG, SST advection, optional strain-aware UV uncertainty). On **GPU96 crop96/20ep**, B2 (SST synergy) **outperforms** M3 ≳ M4 on τ_uv. M4 SSH degradation from raw NLL scale mismatch is **fixed** (post-fix rmse_ssh ≈ 0.063); UV ranking unchanged. Innovation includes the ablatable physics cost **and** an honest partial/negative result on when extras help.

## What Fablet 2024 already did (do not rehash as ours)

Trainable observation/prior operators inside 4DVarNet for SST–SSH → SSC; NATL60 Gulf Stream OSSE; SWOT/nadir configs; ageostrophic recovery claims. Cite: https://doi.org/10.1029/2023MS003609

## Our technical modules (map to code)

| Claim element | Module |
|---------------|--------|
| Unrolled solver (unchanged) | `fourdvarnet/model.py`, `convlstm.py`, `solver.py` |
| eSQG-style residual | `physics.sqg_velocity` |
| Advection residual | `physics.sst_advection_residual` |
| Strain σ / UV term | `physics.strain_uncertainty`, `losses.TrainingLoss` |
| Metrics | `metrics.py` (τ, RMSE, λ_x, geostrophic) |

## Results narrative (evidence-calibrated)

**Show (local measured):**
- Synthetic 8-ep: M4 best among B2/M3/M4 (directional)
- GPU96 crop96/20ep (post-NLL-fix M4): B2 τ_uv 0.848 > M3 0.811 ≳ M4 0.801; geo −3.74; M4 rmse_ssh 0.063 (pre-fix archive 0.122)
- M4 σ-normalized UV term + mse_mix shipped; **20ep retrain done** (best val ≈ 4.83)

**Do not claim:**
- JAMES Table scores from Fablet paper as our numbers
- Crop96/20ep as paper-ready NATL60 rows
- That SQG+adv improve over B2 on NATL60 (prelim evidence: no)
- Full 3D SQG inversion

## Writing order (methods paper)

1. Methods equations + algorithm box — **drafted**
2. Experiment protocol + measured GPU96 table — **drafted**
3. Introduction/Discussion with honest framing — **drafted**
4. Abstract / Key Points — last, after full-grid evidence

## Figure/table hygiene

- All quantitative figures via `scripts/plot_science.py` (SciencePlots, Times New Roman, `text.usetex=False`)
- Caption every crop/short run as non-Table
- Geostrophic baseline always co-reported

## Next evidence gates

1. Retrain M4 short (optional smoke) with σ-normalized loss — confirm SSH recovers
2. Uncropped long NATL60 for Table rows
3. λ_sqg/λ_adv sweep; OSE/drifters
