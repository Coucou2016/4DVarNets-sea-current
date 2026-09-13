# Paper experiments (OSSE ablation)

Commands and criteria for a paper-credible B1/B2/M1–M4 table. Synthetic runs give **directional** evidence only; quote **NATL60** numbers for the JAMES-style table.

## Metric names (evaluate.py)

| Key | Meaning |
|-----|---------|
| `rmse_ssh`, `rmse_u`, `rmse_v`, `rmse_uv` | RMSE vs NATL60 / synthetic truth |
| `tau_uv` | Explained variance of surface currents |
| `tau_div`, `tau_vort`, `tau_strain` | Explained variance of derived fields |
| `lambda_x_uv_km` (and siblings) | Smallest scale where error/signal PSD &lt; 0.5 |
| `lambda_x_*_min_ratio` | Min err/signal over valid *k* (finite even when λ_x is null/NaN) |
| `lambda_x_*_resolved` | 1 if any bin meets the 0.5 threshold, else 0 |
| `lagrangian_separation` | Mean particle separation vs truth (when computed) |

Always report the **geostrophic baseline** on the same split (printed next to each metric and stored under `geostrophic` in JSON).

**λ_x NaN / null:** means *unresolved at all analysed scales* (no bin with err/signal &lt; 0.5). Use `min_ratio` to see distance to the cutoff — do not invent a scale.

## Valid Table row

A row is paper-ready only if **all** of the following hold:

1. Trained and evaluated on **NATL60** (`--source natl60`) with the Gulf Stream box and date splits below.
2. Checkpoint + `results/metrics_<ID>.json` exist for that ID.
3. Metrics are computed on the **test** split (2012-10-20 → 2012-12-04), not train/val.
4. Numbers are from a finished run (not invented, not copied from the JAMES paper without re-running).

Synthetic ablation JSON under `results/` may be cited as *synthetic OSSE directional evidence*, never as Table 1 NATL60 scores.

## NATL60 splits (paper §3.2)

Gulf Stream box: **33–43°N, 65–55°W**. Window length `dT = 7` days; analysis time = last day of the window.

| Split | Inclusive calendar range | Notes |
|-------|--------------------------|--------|
| Train | 2013-02-04 → 2013-09-30 | Shared day 2013-02-04 → train |
| Val | 2013-01-01 → 2013-02-04 | Val uses `[start, end)` so 02-04 is train-only |
| Test | 2012-10-20 → 2012-12-04 | |

Paths: `config/paths.yaml` → `data/natl60/*.nc`. Download:

```bash
python scripts/download_natl60.py --dry-run
python scripts/download_natl60.py
# resume safe: partial *.part files continue via curl -C - / HTTP Range
python scripts/verify_natl60.py
```

Required for **paper/default** loader: `ssh_ref`, `sst_ref`, `u_ref`, `v_ref`, **`obs`**, **`oi`**.
Debug-only `allow_truth_background: true` may omit obs/oi (never for Table claims).
The loader **never** sets `y_ssh = ssh_truth` in paper mode.

NetCDF `time` is stored as **seconds since 2012-10-01** (often without CF units). The loader maps that epoch so paper splits apply (coverage ≈ 2012-10-01 → 2013-09-30).

## Ablation flags

| ID | Setup | Flags |
|----|--------|-------|
| B1 | SSH-only | `--use-sst 0 --use-sqg 0 --use-adv 0 --use-uncert 0` |
| B2 | SSH+SST (Fablet-like) | `--use-sst 1 --use-sqg 0 --use-adv 0 --use-uncert 0` |
| M1 | + SQG residual | `--use-sst 1 --use-sqg 1 --use-adv 0 --use-uncert 0` |
| M2 | + advection residual | `--use-sst 1 --use-sqg 0 --use-adv 1 --use-uncert 0` |
| M3 | SQG + advection | `--use-sst 1 --use-sqg 1 --use-adv 1 --use-uncert 0` |
| M4 | + strain-aware spatial reweighting (not σ head) | `--use-sst 1 --use-sqg 1 --use-adv 1 --use-uncert 1` |

## Exact commands

### Wiring check (synthetic, 2 epochs)

```bash
python scripts/run_ablation.py --quick
```

### Directional CPU evidence (synthetic, longer)

```bash
python scripts/run_ablation.py --epochs 15 --only B2 M3 M4
# writes results/metrics_*.json and results/ablation_summary.json
```

### Paper table (NATL60 + GPU)

```bash
# config/default.yaml train.device: auto  (uses CUDA when available)
python scripts/run_ablation.py --source natl60 --epochs 200 --only B1 B2 M1 M2 M3 M4
python scripts/evaluate.py --source natl60 --ckpt checkpoints/4dvarnet-B2-best.pt --out results/metrics_B2.json
```

Single-experiment train:

```bash
python scripts/train.py --source natl60 --exp-name B2 --use-sst 1 --use-sqg 0 --use-adv 0 --use-uncert 0 --epochs 200
```

## Outputs

| Path | Content |
|------|---------|
| `checkpoints/4dvarnet-<ID>-best.pt` | Best val checkpoint |
| `checkpoints/history-<ID>.json` | Train/val loss curve |
| `results/metrics_<ID>.json` | Test metrics + geostrophic baseline |
| `results/ablation_summary.json` | Matrix summary |

## Device

`train.device: auto` selects CUDA when `torch.cuda.is_available()`, else CPU. On CPU, prefer `--epochs 15` directional runs rather than full 200-epoch NATL60.

**This workstation note (2026-08-15):** hardware is `GeForce GTX 950M` (4 GB). Default `E:\Miniconda3` ships `torch …+cpu`. Use the CUDA env:

```bash
C:\Users\Administrator\MiniConda3\envs\faceswap\python.exe -c "import torch; print(torch.cuda.is_available())"
# True after torch 1.12.1+cu113 install
C:\Users\Administrator\MiniConda3\envs\faceswap\python.exe scripts/train.py --source natl60 --exp-name B2-GPU96 \
  --epochs 20 --crop-size 96 --batch-size 1 --device cuda \
  --use-sst 1 --use-sqg 0 --use-adv 0 --use-uncert 0
```

OOM ladder on 4 GB: crop 128 → 96 → 80 → 64 → 48 (`--batch-size 1`). Prefer ≥64 when possible. Cropped/short-epoch NATL60 metrics are **not** paper Table rows.

## NATL60 CPU smoke (wiring only — not paper scores)

Full 200×200 NATL60 windows are heavy on CPU. For an end-to-end path check:

```bash
python scripts/verify_natl60.py
python scripts/train.py --source natl60 --exp-name NATL60smoke \
  --epochs 1 --crop-size 48 --max-samples 4 --batch-size 1 \
  --use-sst 1 --use-sqg 1 --use-adv 1 --use-uncert 0
python scripts/evaluate.py --source natl60 --ckpt checkpoints/4dvarnet-NATL60smoke-best.pt \
  --crop-size 48 --max-samples 4 --out results/metrics_NATL60smoke.json
```

`crop_size` / `max_samples` are smoke helpers only. Never quote those metrics as JAMES/NATL60 table rows. Evaluation must **not** wrap the inner 4DVar solver in `torch.no_grad()`.
