# Physics-constrained 4DVarNet for SST–SSH sea-surface current inversion

**Assembled manuscript (working).** Crop96/20ep ≠ JAMES Table.



<!-- source: OUTLINE.md -->

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
- ChatGPT browser: tabs creatable but navigate to chatgpt.com failed (2+ attempts) → Cursor WebSearch citations + pastes with public GitHub URL (`PASTE_ChatA_GH_LIT_2026-08-16.txt`, `PASTE_ChatB_REPORT_REVIEW_2026-08-16.txt`). No conversation URL this turn.
- **Public GitHub:** https://github.com/Coucou2016/4DVarNets-sea-current (code+docs+metrics+figures; excludes `*.nc` / `*.pt` / wheels / secrets).
- M4 GPU96 **post**-σ-normalization-fix retrain done (2026-08-16); SSH recovered (0.063). Still crop96/20ep ≠ paper table.
- Strain–uncertainty physical calibration (α, dx) still under review.


<!-- source: 01_introduction.md -->

# 1. Introduction

**Status:** working draft (evidence-calibrated; crop96/20ep ≠ paper Table).  
**Nature-skills axes:** `task=manuscript`, `paper_type=methods`, `journal=generic` (JAMES/GMD-style), `language=en`.

---

Sea-surface current (SSC) estimation from satellite altimetry remains limited by the effective resolution of sea-surface height (SSH) and by the geostrophic approximation, which under-represents ageostrophic and small-scale contributions in energetic western-boundary regimes. Synergistic use of sea-surface temperature (SST) with SSH is a long-standing route to finer-scale surface velocities, via surface quasi-geostrophy (SQG / eSQG), heat-budget / advection inversions, and, more recently, multimodal deep learning (e.g. Lapeyre & Klein, 2006; Rio et al., 2016; Martin et al., 2023; Fablet et al., 2024).

Fablet et al. (2024, JAMES) established a multimodal 4DVarNet solver for SST–SSH → SSC in NATL60 Gulf Stream observing-system simulation experiments (OSSEs), with trainable observation/prior operators inside an unrolled variational loop. That baseline already demonstrates strong SST–SSH synergy relative to geostrophy. What it does **not** make explicit in the variational cost are physics residuals motivated by eSQG mixing and SST advection, nor a strain-aware treatment of UV uncertainty.

**Gap.** Without explicit physics residuals, it is hard to attribute gains to dynamical constraints versus learned multimodal synergy alone, and harder to diagnose when SQG/advection priors help or hurt. Conversely, naively adding physics terms can degrade skill if operators are misspecified, poorly scaled, or regime-dependent (e.g. when SST is a weak proxy for surface density, or when mixed-layer and unbalanced motions violate SQG assumptions; cf. Isern-Fontanet / González-Haro transfer-function literature; Miracca-Lage et al., 2022).

**This work.** We keep the Fablet-style ConvLSTM 4DVarNet solver and extend only the variational cost / supervised loss with (i) an *effective* eSQG-style SQG residual, (ii) an SST advection residual, and (iii) optional strain-heteroscedastic UV uncertainty. We evaluate a controlled ablation matrix (B2 / M3 / M4) under identical training protocols on synthetic and cropped NATL60 OSSEs.

**Contributions (honest).**

1. **Physics residuals inside the unrolled cost** — code-mapped operators for SQG, advection, and strain-aware UV uncertainty, without replacing the solver architecture.
2. **Ablation evidence under a fair protocol** — isolating SST synergy (B2) vs SQG+adv (M3) vs +uncertainty (M4).
3. **Negative / partial result on short cropped NATL60** — on GPU96 crop96/20ep, **B2 outperforms M3 ≳ M4** on τ_uv; all still beat geostrophy on currents. Physics extras are **regime- and hyperparameter-dependent**, not universally additive. (M4 SSH degradation from raw NLL was diagnosed and repaired; post-fix SSH ≈ B2/M3 while UV still trails B2.)
4. **M4 loss-scale diagnosis** — raw heteroscedastic NLL with σ0≪1 can swamp SSH terms; we document and mitigate the scale mismatch (σ-normalized UV term + retrain).

**Boundary.** Cropped / short-epoch GPU runs are **preliminary** and are **not** JAMES Table rows. Full-domain NATL60 long training and OSE/drifter evaluation remain future evidence gates.

### Key citations (verified DOI)

- Fablet et al. 2024 JAMES — https://doi.org/10.1029/2023MS003609  
- Beauchamp et al. 2023 GMD (4DVarNet-SSH) — https://doi.org/10.5194/gmd-16-2119-2023  
- Lapeyre & Klein 2006 JPO — https://doi.org/10.1175/JPO2840.1  
- Rio et al. 2016 JTECH — https://doi.org/10.1175/JTECH-D-16-0017.1  
- González-Haro & Isern-Fontanet / related — https://doi.org/10.1029/2019JC015958  
- Martin et al. 2023 JAMES — https://doi.org/10.1029/2022MS003589  


<!-- source: 02_methods.md -->

# 2. Methods

**Status:** working draft; equations mapped to repository modules.  
Solver architecture follows Fablet et al. (2024); novelty is confined to cost/loss physics terms.

---

## 2.1 State and unrolled variational solver

State at each analysis time is \(x = (\eta, u, v)\) (SSH and surface currents). Observations provide gappy SSH \(y\) and an SST window \(z\) of length \(dT\) (default 7 days). The solver performs \(K\) unrolled gradient steps on a variational cost \(U(x)\), with LSTM-parameterized updates (ConvLSTM), as in 4DVarNet.

| Piece | Code |
|-------|------|
| Model wrapper | `fourdvarnet/model.py` → `FourDVarNetUV` |
| Unrolled solver | `fourdvarnet/solver.py` → `Solver4DVarNet` |
| Prior \(\Phi\) | `fourdvarnet/prior.py` → `PhiPrior` |
| Obs operators \(G,H\) | `fourdvarnet/observation.py` → `ObservationOperator` |

Initial state uses observed/OI SSH plus geostrophic \((u_g, v_g)\).

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

---

## 2.3 Effective eSQG-style operator \(A_{\mathrm{SQG}}\)

We use an *effective* surface QG–style mixing of SST and SSH anomalies to predict a velocity field (not a full 3D SQG inversion). Code: `physics.sqg_velocity` (`fourdvarnet/physics.py`), with deformation radius `Ld` (`Ld_km` in config), Coriolis `f0`, gravity `g`, and grid spacings `dx, dy` from `physics.dx_deg`.

**Assumptions / limits.** SST is treated as a usable surface-density proxy; phase/amplitude relations may fail in mixed-layer–dominated or strongly unbalanced regimes. This motivates reporting cases where SQG residuals do **not** improve over pure SST synergy.

---

## 2.4 SST advection residual

Heat-budget style residual:

\[
r_{\mathrm{adv}} = \partial_t T + u\,\partial_x T + v\,\partial_y T - \kappa \nabla^2 T,
\]

with \(\partial_t T\) from consecutive SST frames in the \(dT\) window. Code: `physics.sst_advection_residual`. Requires \(dT \ge 2\) SST frames.

---

## 2.5 Supervised training loss and strain uncertainty (M4)

Outside the inner cost, training minimizes a weighted supervised loss on SSH, \(\nabla\)SSH, UV, divergence, and prior consistency (`fourdvarnet/losses.py` → `TrainingLoss`; weights under `loss:` in config).

**M4 — strain-aware UV term.** Let \(\sigma = \sigma_0 (1 + \alpha\,\mathrm{strain}(u,v))\), clamped to \(\sigma \le \sigma_0 \cdot m_{\max}\). By default σ is computed from **truth** UV (`uncert_from_truth`) to block the collapse mode where predicted strain is inflated to shrink a heteroscedastic NLL.

Normalized UV uncertainty term (MSE units when \(\sigma \approx \sigma_0\)):

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
| B2 | SSH+SST (Fablet-like synergy) | sst=1, sqg=0, adv=0, uncert=0 |
| M3 | + SQG + advection | sst=1, sqg=1, adv=1, uncert=0 |
| M4 | + strain UV uncertainty | sst=1, sqg=1, adv=1, uncert=1 |

Full matrix B1/M1/M2 documented in `docs/PAPER_EXPERIMENTS.md`.

---

## 2.7 Metrics

Primary: \(\tau_{uv}\) (explained variance), `rmse_uv`, `rmse_ssh`, resolved scale \(\lambda_x\) (when defined). Always co-report **geostrophic** baseline on the same split (`fourdvarnet/metrics.py`, `scripts/evaluate.py`).


<!-- source: 03_experiments.md -->

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


<!-- source: 04_discussion.md -->

# 4. Discussion

**Status:** working draft. Interprets measured crop96/20ep and synthetic evidence only.

---

## 4.1 Main interpretation

On the **short cropped NATL60 OSSE**, explicit SQG + advection residuals (M3) and strain uncertainty (M4) **do not improve** surface-current skill over the pure SST–SSH synergy baseline (B2). All three learned configurations still **strongly outperform geostrophy** on τ_uv and rmse_uv. The honest claim is therefore:

> Physics-constrained cost terms are implementable inside 4DVarNet and remain competitive with geostrophy, but **SST synergy alone can dominate** under this cropped/short-epoch protocol; SQG/advection/uncertainty are **not universally additive**.

This is a useful **partial / negative result** for methods papers: it bounds when “more physics in the cost” helps.

---

## 4.2 Why B2 > M3 is plausible

Rival explanations (not mutually exclusive):

1. **Operator misspecification.** Our \(A_{\mathrm{SQG}}\) is eSQG-*style*, not full interior+surface QG. When SST poorly tracks surface density (mixed layer, unbalanced motions), the SQG residual may pull velocities toward a biased attractor (literature: eSQG success depends on SST–density proxy quality; seasonal/ML depth dependence — e.g. discussions in Miracca-Lage et al. 2022; isQG motivations in Wang et al. 2013).
2. **Redundant information.** Multimodal \(G/H\) SST–SSH terms (B2) may already capture much of the transferable SST structure; adding λ_sqg / λ_adv with fixed weights can over-constrain the unrolled trajectory on short training.
3. **Scale / crop artefacts.** Crop96 windows truncate mesoscale context that SQG/advection residuals assume; short 20-ep schedules may favor the simpler B2 loss landscape.
4. **Hyperparameter regime.** Default `lam_sqg=lam_adv=0.1` was not re-tuned on NATL60 GPU96; under- or over-weighted physics can erase gains.

Synthetic 8-ep ranking (M4 best) vs GPU96 ranking (B2 best) supports **regime dependence**: directional wiring success ≠ NATL60 cropped transfer.

---

## 4.3 M4 SSH degradation (diagnosed) and post-fix retrain

**Pre-fix GPU96** matched M3 on tau_uv but roughly **doubled** rmse_ssh (0.122 vs ~0.06). Best-val ~682 vs ~5 for B2/M3 was a **loss-scale bug**, not a deep dynamical failure:

- Raw heteroscedastic NLL `||e||^2/sigma^2` with `sigma_0≈0.05` is `~1/sigma_0^2` larger than MSE.
- UV gradients then dominate; SSH terms become relatively weak — SSH fit suffers while UV remains OK via the uncertainty weighting.

**Mitigation + retrain (2026-08-16):** sigma-normalized UV term + `uncert_mse_mix: 0.5`; full M4-GPU96 **20ep retrain** on faceswap CUDA. Post-fix: best val **4.83**, rmse_ssh **0.063**, tau_uv **0.801**, rmse_uv **0.211**. SSH recovered near B2/M3; currents still trail B2 (ranking unchanged: B2 > M3 ≳ M4). Pre-fix metrics archived as `results/metrics_M4_GPU96_pre_nllfix.json`.

## 4.4 Framing innovation without overclaim

Defensible:

- Explicit, ablatable physics residuals inside a Fablet-like 4DVarNet cost.
- Engineering diagnosis of uncertainty collapse / scale mismatch.
- Evidence that multimodal SST synergy can outperform added SQG/adv on a constrained OSSE — a caution for “physics always helps” narratives.

Not claimed:

- Universal superiority of M3/M4 over B2.
- Full 3D SQG inversion.
- Paper-table NATL60 scores from crop96/20ep.

---

## 4.5 Next evidence gates

1. **Full-grid / longer NATL60** (uncropped, ≥ paper epoch budget) for B2/M3/M4 with fixed seeds.
2. **λ_sqg / λ_adv / uncertainty sweep** on a small grid before locking Table configs.
3. ~~Re-train M4 with normalized NLL~~ **Done** (SSH 0.063); next: multi-seed / lambda sweeps.
4. **OSE / drifters** (`scripts/evaluate_drifters.py`) once OSSE Table rows exist.
5. Optional: M1/M2 single-term ablations to separate SQG vs advection on full protocol.

---

## 4.6 Limitations checklist

- Single GPU class (4GB) forced crop96; may interact with physics residuals.
- No formal significance / multi-seed intervals yet.
- Strain α / dx calibration still under review for physical units.
- Literature survey for ChatGPT-assisted architecture framing was Cursor WebSearch–backed this session (browser navigate to chatgpt.com failed after tab create); pastes with public GitHub URL left for human/advisor paste: `docs/chatgpt_collaboration/PASTE_ChatA_GH_LIT_2026-08-16.txt`.
- Public code+docs: https://github.com/Coucou2016/4DVarNets-sea-current

# 5. Conclusions (draft)

1. Explicit eSQG-style SQG, SST-advection, and optional strain-aware UV uncertainty can be embedded in a Fablet-like 4DVarNet cost without changing the ConvLSTM solver.
2. On NATL60 **crop96/20ep**, ranking is **B2 > M3 ≳ M4** on τ_uv; all beat geostrophy. Physics extras are **not universally additive** under this protocol.
3. M4 SSH degradation was an NLL scale bug; post-fix retrain recovers rmse_ssh ≈ 0.063 while UV still trails B2.
4. Paper-table claims require uncropped, long-epoch NATL60 (+ optional OSE/drifters).

# References (key DOIs verified)

- Fablet et al. 2024 JAMES — https://doi.org/10.1029/2023MS003609
- Beauchamp et al. 2023 GMD — https://doi.org/10.5194/gmd-16-2119-2023
- Lapeyre & Klein 2006 JPO — https://doi.org/10.1175/JPO2840.1
- Rio et al. 2016 JTECH — https://doi.org/10.1175/JTECH-D-16-0017.1
- Martín et al. / Martin et al. 2023 JAMES — https://doi.org/10.1029/2022MS003589
- González-Haro & Isern-Fontanet related — https://doi.org/10.1029/2019JC015958
- Ballarotta / VarDyn-style dynamical SST–SSH (2024/2025 JAMES family) — https://doi.org/10.1029/2024MS004689
