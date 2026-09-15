# Paper outline — Physics-constrained 4DVarNet for SST–SSH current inversion

**Axes (nature-writing):** `task=manuscript`, `paper_type=methods`, `journal=generic` (target: **JAMES / GMD-style** methods paper), `language=en`.

**One-sentence argument:**  
We embed soft, differentiable eSQG-style SQG and SST-advection residuals (plus optional strain-aware UV reweighting) inside a compact 4DVarNet-inspired unrolled cost, and show—with **our measured** post-P0 crop96 multi-seed NATL60 scores and Stage E operator diagnostics—when those soft physics terms help relative to SSH+SST synergy alone, when a standalone SQG map fails, and how this differs from VarDyn-style dynamical joint SSH–SST mapping.

**Evidence strength (this package):**
| Protocol | Artefacts | Role in claims |
|----------|-----------|----------------|
| **post_p0** NATL60 crop96 / 15 ep / seeds {0,1,2} | `results/post_p0/ablation_summary.json`, `metrics_*-s*.json` | Primary directional ablation (B1–M4, R0) |
| **Stage E** physics ops on NATL60 truth | `results/physics_ops/physics_ops_validation.json` | Operator skill / λ_sqg caution |
| **legacy_pre_review2** crop96/20ep | `results/legacy_pre_review2/metrics_*_GPU96.json` | Quarantined pre-P0 historical context only |
| Full-grid ~200 ep JAMES Table | — | **待补充** — do not invent |

Never cite Fablet/JAMES table numbers as our results.

---

## 1. Target venue & verified literature (DOIs)

| Role | Citation | DOI (WebSearch-verified 2026-09-15) |
|------|----------|-------------------------------------|
| Primary multimodal SSC baseline | Fablet et al. 2024 JAMES | https://doi.org/10.1029/2023MS003609 |
| 4DVarNet-SSH methods / OSSE style | Beauchamp et al. 2023 GMD | https://doi.org/10.5194/gmd-16-2119-2023 |
| Dynamical SSH–SST counterpart | Le Guillou, Chapron & Rio 2025 JAMES (VarDyn) | https://doi.org/10.1029/2024MS004689 |
| eSQG motivation | Lapeyre & Klein 2006 JPO | https://doi.org/10.1175/JPO2840.1 |
| SST heat-budget / advection inversion | Rio et al. 2016 JTECH | https://doi.org/10.1175/JTECH-D-16-0017.1 |
| Multimodal SSH+SST DL narrative | Martin et al. 2023 JAMES | https://doi.org/10.1029/2022MS003589 |
| Spectral SST–SSH transfer language | González-Haro / Isern-Fontanet related | https://doi.org/10.1029/2019JC015958 |

Style: JAMES IMRaD + **Key Points** (3) + Plain Language Summary; geostrophy co-reported; open code; explicit “what is not claimed.”

---

## 2. Innovation framing (vs Fablet / vs VarDyn)

**In scope**
1. Soft differentiable physics **inside** the learned unrolled 4DVarNet cost (SQG + advection + optional strain reweighting)—not a byte-faithful Fablet clone.
2. Fair ablation matrix B1/B2/M1–M4 isolating SST synergy vs each physics knob under one protocol.
3. Stage E honesty: standalone SQG τ_uv ≈ 0 on this crop → treat λ_sqg as a soft prior, not a hard current map.
4. Contrast to VarDyn: VarDyn uses reduced dynamical models in a variational SSH–SST mapper; we learn SSC through an unrolled neural solver with optional soft physics residuals.

**Out of scope**
- Inventing 4DVarNet; quoting Fablet Table numbers as ours; claiming full 3D SQG; claiming full-grid JAMES Table scores without runs.

---

## 3. Section architecture (IMRaD + JAMES extras)

| Section | File | Notes |
|---------|------|-------|
| Key Points / PLS / Abstract | `assemble_paper.py` FRONT | Calibrated to post_p0 + Stage E |
| 1 Introduction | `01_introduction.md` | Gap, contributions, boundary |
| 2 Methods | `02_methods.md` | Compact solver + soft residuals in detail |
| 3 Experiments + Results | `03_experiments.md` | Only our measured tables |
| 4 Discussion | `04_discussion.md` | When physics helps/fails; VarDyn contrast |
| 5 Conclusions + Open Research + Refs | assembled | Bounded claims |

Outputs: `PAPER.md`, `paper.html`, `paper.pdf`.

---

## 4. Measured post_p0 ranking (crop96/15ep, mean of 3 seeds)

From `results/post_p0/ablation_summary.json` (τ_uv mean):

| ID | τ_uv | rmse_uv | rmse_ssh |
|----|-----:|--------:|---------:|
| B1 | 0.861 | 0.178 | 0.059 |
| B2 | 0.878 | 0.166 | 0.059 |
| M1 | 0.916 | 0.139 | 0.049 |
| M2 | 0.880 | 0.165 | 0.056 |
| M3 | 0.914 | 0.140 | 0.050 |
| M4 | 0.917 | 0.137 | 0.051 |
| R0 | 0.850 | 0.185 | 0.059 |
| geo (OI-only, seed0 B2) | 0.846 | 0.188 | — |

**Formal JAMES Table (full-grid long train): 待补充.**

---

## 5. Figure plan

| Fig | Content | Path |
|-----|---------|------|
| Stage E | SQG skill / adv residual sanity | `results/physics_ops/fig_*` |
| post_p0 | τ_uv / RMSE bars ±std | `results/figures/fig_post_p0_*` |
| Quarantined | legacy GPU96 bars | `fig_GPU96_*` (labelled pre_p0) |
| Synth | directional 8ep | `fig_synth_*` |

Mirrored to `docs/paper/figures/` and `docs/report/figures/`.

---

## 6. Terminology ledger

- **4DVarNet** — unrolled variational DA neural solver  
- **SSC** — sea surface currents (u, v)  
- **eSQG** — effective surface quasi-geostrophy (*style* operator here)  
- **τ_uv** — explained variance of UV  
- **λ_x** — resolved scale (PSD error/signal threshold)  
- **B1/B2/M1–M4/R0** — ablation IDs (see experiments)  
- **post_p0 / legacy_pre_review2** — protocol quarantine tags  
