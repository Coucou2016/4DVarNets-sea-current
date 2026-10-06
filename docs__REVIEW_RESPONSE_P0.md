# Phase-1 P0 review response (correctness fixes)

Maps peer-review **P0 / Major Revision** blockers to code status after the Phase-1
implementation pass. Public repo: https://github.com/Coucou2016/4DVarNets-sea-current

**Verdict (this pass):** Closer to **严格完整** for Phase-1 P0 *correctness and
honesty*, but **not** paper-complete: GPU96 scores remain `pre_p0_fix`, and no
post-P0 NATL60 retrain was run. Do **not** invent JAMES table scores.

| # | P0 issue | Status | Where |
|---|----------|--------|--------|
| 1 | NATL60 truth fallback (`y_ssh = ssh_truth`) | **Fixed** | `data/natl60.py`, `data/dataset.py`, `config/default.yaml` (`allow_truth_background: false`). Paper mode requires `obs` + `oi`; missing → `FileNotFoundError` / `RuntimeError`. |
| 2 | Solver per-iter `x.detach()` | **Fixed** | `fourdvarnet/solver.py` `Solver4DVarNet` keeps full unrolled graph (`create_graph=True`, no detach). Ablation only: `Solver4DVarNetTruncated`. |
| 3 | Observation masked MSE denominator | **Fixed** | `_masked_mse`: `sum(r²·m)/sum(m)` over valid obs. |
| 4 | Lat-dependent f, dx, dy; non-periodic BCs | **Fixed** | `fourdvarnet/geometry.py`; `physics` grads alias non-periodic ops; dataset meter scales (`dx_m`/`dy_m`) propagate into `TrainingLoss`, variational SQG/adv cost, and `evaluate` metrics via batch `dx`/`dy` + `scales_from_dataset` (synthetic still uses config isotropic fallback when needed). |
| 5 | SST advection time alignment | **Fixed** | Final-time backward: `(T_t−T_{t−1})/Δt + u_t·∇T_t − κ∇²T_t` (documented in `physics.sst_advection_residual`). |
| 6 | SST mask full window | **Fixed** | Dataset returns `mask_sst[t−dT+1:t+1]`; advection zeros residual where time/spatial stencil invalid. |
| 7 | Strict xarray align | **Fixed** | `_align_like`: assert time length/Δt; no silent return of misaligned arrays on failure. Covered by `test_align_like_time_mismatch_raises`. |
| 8 | Docs / M4 naming honesty | **Fixed** | M4 = **strain-aware spatial reweighting**, not uncertainty estimation. Compact 4DVarNet-**inspired** (not faithful Fablet R0). Solver was **not** unchanged. GPU96 metrics marked obsolete. Generators (`assemble_paper.py`, `build_report.py`) + chapter sources aligned. |
| 9 | Unit tests | **Fixed** | `tests/test_p0_correctness.py` + updated physics/NATL60 tests. |
| 10 | DUACS/geo baseline fairness | **Fixed** | Geostrophic baseline currents from **pure OI/DUACS** (`y_oi`, `u_geo_oi`/`v_geo_oi`); hybrid OI+sparse remains model *input* only. |
| 11 | Train/val SST history purge | **Fixed** | Optional `purge_days` (default `dT-1` in NATL60 paper mode) drops train windows whose SST history overlaps val targets. |
| 12 | Temporal λ in evaluate | **Fixed** | `resolved_timescale` wired when ≥8 test windows; else evaluate prints/JSON-marks **待补充**. |
| 13 | Tag ablation summary | **Fixed** | `checkpoints/ablation_summary.json` carries `pre_p0_fix: true`. |

## Explicitly not done (Phase-2 / remaining)

- **G(full state) vs G(SSH only)** — observation operator still SSH-centric synergy; full-state \(G\) is Phase-2 architecture.
- **State channel normalization** — Phase-2.
- **Faithful R0 Fablet/IMT reproduction** (`lit_model_uv` parity) — Phase-2.
- **Full multi-seed NATL60 paper table retrain** — Phase-2 (no long retrain this turn).
- Learned uncertainty head (M5); VarDyn comparator.

## Old metrics

All `results/metrics_*.json` files and `checkpoints/ablation_summary.json` carry
`"pre_p0_fix": true` and a disclaimer string. GPU96 crop96/20ep ranking
(B2 > M3 ≳ M4) remains useful only as **historical / pre-fix** directional
context, not as evidence for claims after these correctness changes.
