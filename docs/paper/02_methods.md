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
