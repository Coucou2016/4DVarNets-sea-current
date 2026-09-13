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
