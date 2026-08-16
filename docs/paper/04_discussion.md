# 4. Discussion

**Status:** matured draft. Interprets measured crop96/20ep and synthetic evidence only.

---

## 4.1 Main interpretation

On the **short cropped NATL60 OSSE**, explicit SQG + advection residuals (M3) and strain uncertainty (M4) **do not improve** surface-current skill over the pure SST–SSH synergy baseline (B2). All three learned configurations still **strongly outperform geostrophy** on τ_uv and rmse_uv. Expanded diagnostics reinforce the message: only B2 shows positive τ_div / τ_vort / τ_strain on this protocol. The honest claim is therefore:

> Physics-constrained cost terms are implementable inside 4DVarNet and remain competitive with geostrophy, but **SST synergy alone can dominate** under this cropped/short-epoch protocol; SQG/advection/uncertainty are **not universally additive**.

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
- UV gradients then dominate; SSH terms become relatively weak — SSH fit suffers while UV remains OK via the uncertainty weighting.

**Mitigation + retrain (2026-08-16):** sigma-normalized UV term + `uncert_mse_mix: 0.5`; full M4-GPU96 **20ep retrain** on faceswap CUDA. Post-fix: best val **4.83**, rmse_ssh **0.063**, tau_uv **0.801**, rmse_uv **0.211**. SSH recovered near B2/M3; currents still trail B2 (ranking unchanged: B2 > M3 ≳ M4). Pre-fix metrics archived as `results/metrics_M4_GPU96_pre_nllfix.json`.

---

## 4.4 Framing innovation without overclaim

Defensible:

- Explicit, ablatable physics residuals inside a Fablet-like 4DVarNet cost.
- Engineering diagnosis of uncertainty collapse / scale mismatch.
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
3. **λ_sqg / λ_adv / uncertainty sweep** on a small grid before locking Table configs.
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
