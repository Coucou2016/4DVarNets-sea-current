# 2. Methods

We describe a compact 4DVarNet-inspired ConvLSTM unrolled solver (not a byte-faithful Fablet/IMT reproduction). Soft physics residuals enter the variational cost and the supervised loss. After Phase-1 correctness fixes, the solver uses full unrolled automatic differentiation without per-iteration `detach` (see `Solver4DVarNet`; a truncated variant remains available for ablation).

## 2.1 State and unrolled variational solver

State at each analysis time is \(x = (\eta, u, v)\) (SSH and surface currents). Observations provide gappy SSH \(y\) and an SST window \(z\) of length \(dT\) (default 7 days). The solver performs \(K\) unrolled gradient steps on a variational cost \(U(x)\), with LSTM-parameterized updates (ConvLSTM), in the spirit of 4DVarNet (Beauchamp et al., 2023; Fablet et al., 2024).

| Piece | Module |
|-------|--------|
| Model wrapper | `fourdvarnet/model.py` → `FourDVarNetUV` |
| Unrolled solver | `fourdvarnet/solver.py` → `Solver4DVarNet` |
| Prior \(\Phi\) | `fourdvarnet/prior.py` → `PhiPrior` |
| Obs operators \(G,H\) | `fourdvarnet/observation.py` → `ObservationOperator` |

The initial state uses observed or optimally interpolated (OI) SSH plus geostrophic \((u_g, v_g)\). The **geostrophic baseline** for metrics uses pure OI/DUACS SSH (not the hybrid OI+sparse model input). Grid spacings \(dx, dy\) for SQG, advection, and divergence come from dataset meter metrics when NATL60 provides latitude-aware scales; synthetic data fall back to an isotropic configuration scale.

## 2.2 Variational cost with soft physics residuals

Following the multimodal 4DVarNet cost and extending it with soft residuals,

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

Implementation: `VariationalCost` in `fourdvarnet/solver.py`. Ablation flags (`use_sst`, `use_sqg`, `use_adv`) and default λ weights live in `config/default.yaml`.

**Contrast to VarDyn.** VarDyn (Le Guillou et al., 2025) jointly maps SSH and SST with quasi-geostrophic and advection–diffusion constraints inside a classical variational scheme. Our formulation targets SSC \((u,v)\) through a learned unrolled solver and treats SQG/advection as *soft* residuals that can be switched off, rather than replacing the neural propagator with a reduced dynamical model.

## 2.3 Effective eSQG-style operator \(A_{\mathrm{SQG}}\)

We use an effective surface-QG-style mixing of SST and SSH anomalies to predict a velocity field (not a full three-dimensional SQG inversion). Code: `physics.sqg_velocity` in `fourdvarnet/physics.py`, with deformation radius \(L_d\), Coriolis \(f_0\), gravity \(g\), and grid spacings from the dataset.

**Assumptions and limits.** SST is treated as a usable surface-density proxy. Phase and amplitude relations may fail when mixed-layer or unbalanced motions dominate, or when interior potential vorticity contributes strongly to surface velocity. Stage E (Section 3) therefore reports standalone SQG skill against NATL60 truth before interpreting \(\lambda_{\mathrm{sqg}}\) as a hard constraint.

## 2.4 SST advection residual

Heat-budget style residual (final-time backward difference for a single-time state),

\[
r_{\mathrm{adv}} = \partial_t T + u\,\partial_x T + v\,\partial_y T - \kappa \nabla^2 T,
\]

with \(\partial_t T = (T_t - T_{t-1})/\Delta t\) from consecutive SST frames in the \(dT\) window (`physics.sst_advection_residual`). This requires \(dT \ge 2\).

## 2.5 Supervised training loss and strain reweighting (M4)

Outside the inner cost, training minimizes a weighted supervised loss on SSH, \(\nabla\)SSH, UV, divergence, and prior consistency (`fourdvarnet/losses.py` → `TrainingLoss`).

**M4 — strain-aware spatial UV reweighting (not uncertainty estimation).** Let \(\sigma = \sigma_0 (1 + \alpha\,\mathrm{strain}(u,v))\), clamped to \(\sigma \le \sigma_0 \cdot m_{\max}\). By default σ is computed from truth UV (`uncert_from_truth`) to avoid a collapse mode in which predicted strain is inflated to shrink a heteroscedastic term. There is no learned σ head.

Normalized UV reweight term (MSE units when \(\sigma \approx \sigma_0\)):

\[
L_{\mathrm{uv}}^{\sigma}
= \mathbb{E}\Big[ \|e_{uv}\|^2 \big(\sigma_0/\sigma\big)^2 \Big]
+ \sigma_0^2\,\mathbb{E}\big[\log(\sigma/\sigma_0)^2\big],
\]

mixed with MSE: \(L_{uv} = m\,L_{\mathrm{MSE}} + (1-m)\,L_{\mathrm{uv}}^{\sigma}\). Raw heteroscedastic NLL with \(\sigma_0 \ll 1\) can dominate SSH terms; the normalized form keeps UV and SSH on a comparable scale (documented historically on pre-P0 M4 runs).

## 2.6 Ablation identifiers

| ID | Meaning | Flags |
|----|---------|-------|
| B1 | SSH-only | sst=0, sqg=0, adv=0, uncert=0 |
| B2 | SSH+SST synergy | sst=1, sqg=0, adv=0, uncert=0 |
| M1 | + SQG residual | sst=1, sqg=1, adv=0, uncert=0 |
| M2 | + advection residual | sst=1, sqg=0, adv=1, uncert=0 |
| M3 | + SQG + advection | sst=1, sqg=1, adv=1, uncert=0 |
| M4 | + strain UV reweighting | sst=1, sqg=1, adv=1, uncert=1 |
| R0 | Larger compact SSH+SST capacity | not byte-faithful Fablet R0 |
| geo | OI-only geostrophy | evaluation baseline |

## 2.7 Metrics

Primary scores: \(\tau_{uv}\) (explained variance of surface currents), `rmse_uv`, `rmse_ssh`. Diagnostics from the same evaluation JSON include \(\tau_{\mathrm{div}}\), \(\tau_{\mathrm{vort}}\), \(\tau_{\mathrm{strain}}\), and resolved scales \(\lambda_x\) (SSH and UV). Every model score is co-reported with the OI-only geostrophic baseline on the same split (`fourdvarnet/metrics.py`, `scripts/evaluate.py`).
