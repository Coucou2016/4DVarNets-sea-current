# Paper outline — Physics-constrained 4DVarNet for SST–SSH current inversion

**Axes (nature-writing):** `task=manuscript`, `paper_type=methods`, `journal=generic` (target venue: **JAMES / GMD-style** methods paper; Nature-family *style* for clarity, not flagship Nature format), `language=en`.

**One-sentence argument (updated 2026-08-16 cont):**  
We embed explicit eSQG-style SQG, SST-advection, and optional strain-aware UV uncertainty into a Fablet-like 4DVarNet cost and show, under a fair ablation, that on short cropped NATL60 OSSE **SST synergy (B2) can outperform physics extras (M3/M4)** while all beat geostrophy — framing physics residuals as useful but **regime-/hyperparameter-dependent**, with full Table claims gated on uncropped long runs.

**Evidence strength today:** synthetic 8-ep directional ablation + B2/M3/M4-GPU96 crop96/20ep train+eval (local only). Do **not** claim JAMES Table numbers from Fablet 2024 as our results. Crop96/20ep != paper table.

**Draft sections (this session):** `01_introduction.md`, `02_methods.md`, `03_experiments.md`, `04_discussion.md`.

---

## 1. Target venue & style models to imitate

| Role | Paper | Why imitate | DOI / URL |
|------|-------|-------------|-----------|
| Primary baseline (methods + OSSE) | Fablet et al. 2024 JAMES | IMRaD, Key Points, multimodal 4DVarNet UV | https://doi.org/10.1029/2023MS003609 |
| Solver / SSH mapping style | Beauchamp et al. 2023 GMD | Clear methods, NATL60 OSSE, metrics, open code | https://doi.org/10.5194/gmd-16-2119-2023 |
| Physics prior (SQG) | Lapeyre & Klein 2006 JPO | eSQG motivation for SST→ψ | https://doi.org/10.1175/JPO2840.1 |
| SST advection / heat budget | Rio et al. 2016 JTECH | Heat-equation inversion for ageostrophy | https://doi.org/10.1175/JTECH-D-16-0017.1 |
| Spectral SST–SSH transfer | González-Haro & Isern-Fontanet / related JC | Transfer-function language | https://doi.org/10.1029/2019JC015958 |
| Multimodal SSH+SST DL | Martin et al. 2023 JAMES | Synergistic SST–SSH learning narrative | https://doi.org/10.1029/2022MS003589 |
| Dynamical SST–SSH joint mapping | VarDyn / Ballarotta et al. 2024+ JAMES | Explicit advection–diffusion + variational SSH–SST (contrast to learned synergy) | https://doi.org/10.1029/2024MS004689 |

Style cues from JAMES/GMD methods papers: Key Points (3 bullets); Plain Language Summary; explicit OSSE protocol; geostrophic baseline everywhere; open code/data; clear “what is *not* claimed.”

**Advisor framing accepted (Cursor WebSearch–backed, 2026-08-16 dual-agent turn):** keep contribution as *ablatable physics residuals + honest regime-dependence* rather than “physics always wins”; Methods/Results skeleton stays IMRaD with geostrophy co-reported; cite VarDyn as dynamical counterpart.

---

## 2. Defensible innovation points (vs Fablet 2024)

**In scope (ours):**
1. **Physics residuals inside the unrolled cost** — effective eSQG operator + SST advection residual + optional strain-heteroscedastic UV term — without replacing the ConvLSTM 4DVarNet solver.
2. **Ablation matrix B1/B2/M1–M4** isolating SQG, advection, and uncertainty contributions under identical training protocol.
3. **M4 collapse / scale diagnosis & mitigation** — σ from truth strain + clamp + MSE mix + σ-normalized UV term (engineering contribution with physical motivation).
4. **Honest partial result** — B2 > M3 ≳ M4 on GPU96 crop96/20ep; physics extras not universally additive.
5. **Reproducible OSSE→OSE path** — NATL60 paper splits documented; OSE/drifter eval hooks reserved.

**Out of scope / not claimed as novelty:**
- Inventing 4DVarNet itself (Fablet et al.).
- Claiming full 3D SQG inversion (we use *effective* eSQG-style mixing).
- Quoting crop/max_samples/short-epoch GPU runs as Table 1 NATL60 scores.
- Claiming M3/M4 beat B2 on NATL60 (evidence so far says otherwise on crop96/20ep).

---

## 3. Section architecture (IMRaD + JAMES extras)

| # | Section | Draft file | Status |
|---|---------|------------|--------|
| — | Title / Authors / Affiliations | — | TODO |
| — | Key Points (3 bullets) | — | TODO (calibrate to B2>M3) |
| — | Abstract / Plain Language Summary | — | TODO last |
| 1 | Introduction | `01_introduction.md` | **Drafted** |
| 2 | Related work | (merge into intro for now) | light |
| 3 | Methods | `02_methods.md` | **Drafted** |
| 4 | Experiments + Results | `03_experiments.md` | **Drafted** (prelim numbers) |
| 5 | Discussion | `04_discussion.md` | **Drafted** |
| 6 | Conclusions | in `PAPER.md` | Drafted (bounded) |
| — | Open Research / References | in `PAPER.md` | Key DOIs listed |
| — | Assembled MD / HTML | `PAPER.md`, `paper.html` | Built |
| — | Chinese full report | `docs/report/report.{html,md,pdf}` | Built (Base64 HTML) |

---

## 4. Experiment matrix (paper table criteria)

See `docs/PAPER_EXPERIMENTS.md`. Paper row requires: NATL60 source, paper splits, finished ckpt + `results/metrics_<ID>.json`, test split, no invented numbers.

| ID | Flags | Status (2026-08-16 cont) |
|----|-------|--------------------------|
| B2 synthetic 8ep | SST only | Done — directional |
| M3 synthetic 8ep | SQG+adv | Done — directional |
| M4 synthetic 8ep | +uncert | Done — M4 best among trio (directional) |
| B2-GPU96 | NATL60 crop96 20ep | Done — τ_uv 0.848, rmse_ssh 0.059 |
| M3-GPU96 | NATL60 crop96 20ep | Done — τ_uv 0.811, rmse_ssh 0.064 |
| M4-GPU96 | NATL60 crop96 20ep | **Retrained post-NLL-fix** — τ_uv 0.801, rmse_ssh 0.063 (best val 4.83); pre-fix archive `metrics_M4_GPU96_pre_nllfix.json` |
| M4 fix (σ-norm NLL) | code+config+test+**20ep retrain** | **Done** — SSH recovered; UV still B2>M3≳M4 |
| Full NATL60 200ep uncropped | — | **Not started** — only these qualify as Table rows |

---

## 5. Figure plan

| Fig | Content | Status |
|-----|---------|--------|
| Fig 1 | Method schematic (cost terms + unrolled solver) | TODO schematic |
| Fig 2 | Synthetic ablation τ_uv / RMSE | SciencePlots done |
| Fig 3 | B2/M3/M4-GPU96 loss + GPU96 tau/rmse bars | SciencePlots done (crop96 != Table) |
| Fig 4 | NATL60 maps (SSH/UV error) when metrics ready | Pending |
| Fig 5 | Ablation table graphic / λ_x | Pending full runs |

Paths: `results/figures/` and mirrored `docs/paper/figures/`. Caption drafts in `03_experiments.md` and `FIGURE_NOTES.md`.

---

## 6. Terminology ledger (canonical)

- **4DVarNet** — unrolled variational DA neural solver
- **SSC** — sea surface currents (u, v)
- **eSQG** — effective surface quasi-geostrophy (our operator is eSQG-*style*, not full 3D)
- **OSSE / OSE** — observing-system simulation / experiment
- **τ_uv** — explained variance of UV
- **λ_x** — resolved scale where error/signal PSD < 0.5
- **B2 / M3 / M4** — ablation IDs (SSH+SST; +SQG+adv; +strain uncert)

---

## 7. Assumptions / missing inputs

- Full-domain NATL60 long training not yet available → Table 1 empty of our numbers.
- ChatGPT browser MCP unavailable this dual-agent turn (no browser tool server) → Cursor WebSearch citations + refreshed paste (`PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_report.txt`). No conversation URL.
- GitHub sharing blocked: `gh` not installed / no auth; local `git init` optional only — no push.
- M4 GPU96 **post**-σ-normalization-fix retrain done (2026-08-16); SSH recovered (0.063). Still crop96/20ep ≠ paper table.
- Strain–uncertainty physical calibration (α, dx) still under review.
