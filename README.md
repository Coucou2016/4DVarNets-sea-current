# 4DVarNets — Sea Surface Current Inversion from SST–SSH Synergies

**Public repo:** [github.com/Coucou2016/4DVarNets-sea-current](https://github.com/Coucou2016/4DVarNets-sea-current)

Implementation of **Fablet et al. (2024)**, *Inversion of Sea Surface Currents From Satellite-Derived SST-SSH Synergies With 4DVarNets*, JAMES ([doi:10.1029/2023MS003609](https://doi.org/10.1029/2023MS003609)), plus a physics-augmented variational cost for follow-on experiments.

Reference code: [CIA-Oceanix/4dvarnet-james-uv-ssc](https://github.com/CIA-Oceanix/4dvarnet-james-uv-ssc).

**Honest GPU96 note (crop96 / 20ep, ≠ JAMES table):** measured ranking **B2 > M3 ≳ M4** (τ_uv 0.848 / 0.811 / 0.801). See `results/metrics_*_GPU96.json` and `docs/report/`.

## Innovation vs Fablet 2024

The unrolled ConvLSTM 4DVarNet solver is unchanged (state `x = (SSH, u, v)`, learned `G`/`H`, prior `Φ`). The inner cost is extended with explicit physics residuals:

```
U = λ_ssh ||SSH − SSH_obs||²_Ω
  + λ_mm  ||G(x) − H(SST)||²
  + λ_sqg ||u − A_SQG(SST, SSH)||²
  + λ_adv ||∂t T + u·∇T − κ∇²T||²
  + λ_Φ   ||x − Φ(x)||²
```

Optional supervised term: strain-aware UV uncertainty `||u − u_gt||² / σ²(strain) + log σ²` with `σ = σ0 (1 + α strain)`.

`A_SQG` is an **effective eSQG-style** operator (geostrophic SSH mixed with a spectral SST streamfunction `ψ̂(k) ∝ θ̂(k)/(k + 1/L_d)`). It is not a full 3D SQG inversion.

| Term | Module |
|------|--------|
| State **x** = (SSH, u, v) | `fourdvarnet/model.py` |
| Cost **U** (Eq. 8 + SQG/advection) | `fourdvarnet/solver.py` `VariationalCost` |
| **G**, **H** multimodal conv nets | `fourdvarnet/observation.py` |
| Prior **Φ** | `fourdvarnet/prior.py` |
| Unrolled ConvLSTM (Eq. 9) | `fourdvarnet/convlstm.py` |
| eSQG / advection / strain σ | `fourdvarnet/physics.py` |
| Supervised losses + UV NLL | `fourdvarnet/losses.py` |
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

Default `config/default.yaml` uses `data.source: synthetic` and turns SST, SQG, advection, and uncertainty **on**. CPU CI does not need NATL60.

```bash
python scripts/train.py --config config/default.yaml
python scripts/train.py --use-sst 0 --use-sqg 0 --use-adv 0 --use-uncert 0   # SSH-only ablation (B1)
```

Checkpoints: `checkpoints/4dvarnet-<tag>-best.pt`

## Ablation matrix (B1 / B2 / M1–M4)

| ID | Setup | Command |
|----|--------|---------|
| B1 | SSH-only | `python scripts/train.py --exp-name B1 --use-sst 0 --use-sqg 0 --use-adv 0 --use-uncert 0` |
| B2 | SSH+SST (Fablet-like) | `python scripts/train.py --exp-name B2 --use-sst 1 --use-sqg 0 --use-adv 0 --use-uncert 0` |
| M1 | + SQG residual | `python scripts/train.py --exp-name M1 --use-sst 1 --use-sqg 1 --use-adv 0 --use-uncert 0` |
| M2 | + advection residual | `python scripts/train.py --exp-name M2 --use-sst 1 --use-sqg 0 --use-adv 1 --use-uncert 0` |
| M3 | SQG + advection | `python scripts/train.py --exp-name M3 --use-sst 1 --use-sqg 1 --use-adv 1 --use-uncert 0` |
| M4 | + strain uncertainty | `python scripts/train.py --exp-name M4 --use-sst 1 --use-sqg 1 --use-adv 1 --use-uncert 1` |

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

If files are missing, `load_natl60` / `train.py --source natl60` exit with the download command and URLs (no traceback spam).

Paper experiment checklist, metric names, and valid Table-row criteria: [`docs/PAPER_EXPERIMENTS.md`](docs/PAPER_EXPERIMENTS.md).

Paper splits (§3.2), Gulf Stream box 33–43N, 65–55W:

- Train: 2013-02-04 → 2013-09-30
- Val: 2013-01-01 → 2013-02-04 (shared day assigned to train)
- Test: 2012-10-20 → 2012-12-04

Sliding windows use `dT = 7`, same as the synthetic dataset. Ablation metrics JSON: `results/`.

## Metrics

- `tau_*`: explained variance `1 − MSE/Var(truth)` for UV, divergence, vorticity, strain
- `lambda_x_*_km`: smallest spatial scale where error PSD / signal PSD < 0.5 (Le Guillou / radially averaged rfft2)
- `resolved_timescale`: 1-D analogue when a time series is available
- `lagrangian_separation`: RK2 particles; mean separation vs truth currents

## Real OSE stubs (no live download)

`data/real.py` documents SWOT L3, OSTIA, and GDP ERDDAP IDs/DOIs, and provides `collocate_uv_at_points`. `scripts/evaluate_drifters.py` falls back to synthetic particles when no GDP extract is present.

## Limitations (honest)

- Synthetic OSSE scores are **not** JAMES 2024 Table numbers. Quote NATL60 GPU runs for the paper table.
- `A_SQG` is a discrete eSQG-style mix, not a 3D inversion.
- Φ and G/H are a compact port of the official IMT `lit_model_uv.py` (~1.4M params, 200 epochs, Lightning/Hydra).
- Full NATL60 + SWOT/OSTIA/GDP OSE needs downloaded cubes and a GPU.
- Lagrangian path is RK2 on the analysis grid, not a full FSLE package.

## License

Research implementation; original OceaniX code is CeCILL-C. See paper and official repo for citation.
