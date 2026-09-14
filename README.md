# 4DVarNets — Sea Surface Current Inversion from SST–SSH Synergies

**Public repo:** [github.com/Coucou2016/4DVarNets-sea-current](https://github.com/Coucou2016/4DVarNets-sea-current)

**Compact 4DVarNet-inspired** implementation motivated by **Fablet et al. (2024)**, *Inversion of Sea Surface Currents From Satellite-Derived SST-SSH Synergies With 4DVarNets*, JAMES ([doi:10.1029/2023MS003609](https://doi.org/10.1029/2023MS003609)), plus physics-augmented variational cost terms for follow-on ablations.

Reference code (full IMT stack): [CIA-Oceanix/4dvarnet-james-uv-ssc](https://github.com/CIA-Oceanix/4dvarnet-james-uv-ssc). This repo is **not** a byte-faithful Fablet reproduction (R0); a faithful R0 port is a later phase.

> **P0 / review2 (2026-09):** Consistency Stage A–D: non-periodic `geometry.grad_*` in physics; faithful solver (no per-iter detach; tensor norm scale only); paper-mode NATL60 (`obs`+`oi`, no truth→`y_ssh`); full SST mask window; OI-only geo baseline; full-test pooled evaluate. See [`docs/REVIEW_RESPONSE_ROUND2.md`](docs/REVIEW_RESPONSE_ROUND2.md).
>
> **GPU96 B2/M3/M4 scores live under `results/legacy_pre_review2/` and are tagged `legacy_pre_review2` / `pre_p0_fix` — obsolete for formal claims.** Do not cite them as paper Table rows until a post-P0 retrain. Historical directional note only: under crop96/20ep, B2 beat M3 ≳ M4 and geostrophy; physics extras did **not** beat B2.
>
> Release tag: **`v0.2.0-review2-fixed`**.

## Innovation vs Fablet 2024 (honest)

This codebase uses a **compact** unrolled ConvLSTM solver (state `x = (SSH, u, v)`, learned `G`/`H`, prior `Φ`). It is **4DVarNet-inspired**, not an unchanged / faithful Fablet solver. The inner cost can add explicit physics residuals:

```
U = λ_ssh ||SSH − SSH_obs||²_Ω
  + λ_mm  ||G(x) − H(SST)||²
  + λ_sqg ||u − A_SQG(SST, SSH)||²
  + λ_adv ||∂t T + u·∇T − κ∇²T||²   # final-time backward for single-time state
  + λ_Φ   ||x − Φ(x)||²
```

Optional supervised term (**M4**): **strain-aware spatial reweighting** of UV error
`σ = σ0 (1 + α strain)` — **not** a learned uncertainty / σ head.

`A_SQG` is an **effective eSQG-style** operator (geostrophic SSH mixed with a spectral SST streamfunction `ψ̂(k) ∝ θ̂(k)/(k + 1/L_d)`). It is not a full 3D SQG inversion.

| Term | Module |
|------|--------|
| State **x** = (SSH, u, v) | `fourdvarnet/model.py` |
| Cost **U** (Eq. 8 + SQG/advection) | `fourdvarnet/solver.py` `VariationalCost` |
| **G**, **H** multimodal conv nets | `fourdvarnet/observation.py` |
| Prior **Φ** | `fourdvarnet/prior.py` |
| Unrolled ConvLSTM (Eq. 9) | `fourdvarnet/convlstm.py` + `Solver4DVarNet` |
| Truncated-BPTT ablation | `Solver4DVarNetTruncated` |
| Geometry (f, dx, dy, non-periodic grads) | `fourdvarnet/geometry.py` |
| eSQG / advection / strain reweight | `fourdvarnet/physics.py` |
| Supervised losses + UV reweight | `fourdvarnet/losses.py` |
| τ, RMSE, λ_x, Lagrangian | `fourdvarnet/metrics.py` |

## Install

```bash
cd E:\Projects\20260522-4DVarNets-sea-current
pip install -r requirements.txt
```

## Quick verify (smoke + unit tests)

```bash
python scripts/smoke_test.py
python scripts/generate_synthetic.py --out data/synthetic_osse.npz
python -m pytest tests/ -v -p pytest
```

On some Windows conda installs a broken `zarr`/`numcodecs` pytest plugin must be skipped (`tests/conftest.py` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`). `scripts/run_verify.py` sets that variable for you.

Full pipeline (smoke, tests, train, eval):

```bash
python scripts/run_verify.py
```

YAML is always opened with UTF-8. Evaluation does **not** wrap the inner 4DVar solver in `torch.no_grad()` (the unrolled loop needs autograd).

## Train (synthetic OSSE)

Default `config/default.yaml` uses `data.source: synthetic` and turns SST, SQG, advection, and strain reweighting **on**. CPU CI does not need NATL60.

```bash
python scripts/train.py --config config/default.yaml
python scripts/train.py --use-sst 0 --use-sqg 0 --use-adv 0 --use-uncert 0   # SSH-only ablation (B1)
```

Checkpoints: `checkpoints/4dvarnet-<tag>-best.pt`

## Ablation matrix (B1 / B2 / M1–M4)

| ID | Setup | Command |
|----|--------|---------|
| B1 | SSH-only | `python scripts/train.py --exp-name B1 --use-sst 0 --use-sqg 0 --use-adv 0 --use-uncert 0` |
| B2 | SSH+SST (Fablet-like cost terms) | `python scripts/train.py --exp-name B2 --use-sst 1 --use-sqg 0 --use-adv 0 --use-uncert 0` |
| M1 | + SQG residual | `python scripts/train.py --exp-name M1 --use-sst 1 --use-sqg 1 --use-adv 0 --use-uncert 0` |
| M2 | + advection residual | `python scripts/train.py --exp-name M2 --use-sst 1 --use-sqg 0 --use-adv 1 --use-uncert 0` |
| M3 | SQG + advection | `python scripts/train.py --exp-name M3 --use-sst 1 --use-sqg 1 --use-adv 1 --use-uncert 0` |
| M4 | + strain-aware spatial reweighting | `python scripts/train.py --exp-name M4 --use-sst 1 --use-sqg 1 --use-adv 1 --use-uncert 1` |

Run the whole matrix (2 epochs when `--quick`):

```bash
python scripts/run_ablation.py --quick
python scripts/run_ablation.py              # full epoch count from config
```

## Evaluate

```bash
python scripts/evaluate.py --ckpt checkpoints/4dvarnet-ssh-sst-best.pt
python scripts/evaluate_drifters.py         # synthetic-drifter smoke if no GDP file
```

`evaluate.py` prints RMSE, τ_uv / τ_div / τ_vort / τ_strain, and λ_x (error/signal PSD) against a geostrophic baseline.

## NATL60 real OSSE (paper table)

Files are multi-GB (~3.6–14 GB each). The loader does **not** run at import time. Downloads resume via `curl -C -` / HTTP Range on `*.part` files.

```bash
python scripts/download_natl60.py --dry-run
python scripts/download_natl60.py           # skip complete files; resume partials
python scripts/verify_natl60.py             # shapes, date range, train/val/test counts
```

URLs live in `data/natl60.py`. Paths default to `config/paths.yaml` (`data/natl60/*.nc`). After download:

```bash
python scripts/train.py --source natl60
```

**Paper/default mode** requires `ssh_ref`, `sst_ref`, `u_ref`, `v_ref`, **`obs`**, and **`oi`**. The loader never sets `y_ssh = ssh_truth`. Debug-only `data.allow_truth_background: true` is off by default.

Paper experiment checklist: [`docs/PAPER_EXPERIMENTS.md`](docs/PAPER_EXPERIMENTS.md). P0 fix map: [`docs/REVIEW_RESPONSE_P0.md`](docs/REVIEW_RESPONSE_P0.md).

Paper splits (§3.2), Gulf Stream box 33–43N, 65–55W:

- Train: 2013-02-04 → 2013-09-30
- Val: 2013-01-01 → 2013-02-04 (shared day assigned to train)
- Test: 2012-10-20 → 2012-12-04

Sliding windows use `dT = 7`. Ablation metrics JSON: `results/` (**pre_p0_fix** until retrain).

## Metrics

- `tau_*`: explained variance `1 − MSE/Var(truth)` for UV, divergence, vorticity, strain
- `lambda_x_*_km`: smallest spatial scale where error PSD / signal PSD < 0.5 (Le Guillou / radially averaged rfft2)
- `resolved_timescale`: 1-D analogue when a time series is available
- `lagrangian_separation`: RK2 particles; mean separation vs truth currents

## Real OSE stubs (no live download)

`data/real.py` documents SWOT L3, OSTIA, and GDP ERDDAP IDs/DOIs, and provides `collocate_uv_at_points`. `scripts/evaluate_drifters.py` falls back to synthetic particles when no GDP extract is present.

## Limitations (honest)

- Synthetic OSSE scores are **not** JAMES 2024 Table numbers.
- GPU96 crop96/20ep metrics are **pre-P0-fix** and **obsolete for claims**.
- Compact Φ / G/H port ≠ official IMT `lit_model_uv.py` (~1.4M params, 200 epochs, Lightning/Hydra).
- `A_SQG` is a discrete eSQG-style mix, not a 3D inversion.
- M4 is strain reweighting, not uncertainty estimation (learned σ head = future M5).
- Full NATL60 + SWOT/OSTIA/GDP OSE needs downloaded cubes and a GPU.
- Lagrangian path is RK2 on the analysis grid, not a full FSLE package.

## License

Research implementation; original OceaniX code is CeCILL-C. See paper and official repo for citation.
