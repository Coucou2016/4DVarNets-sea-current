# Phase-1 P0 review response (correctness fixes)

Maps peer-review **P0 / Major Revision** blockers to code status after the Phase-1
implementation pass. Public repo: https://github.com/Coucou2016/4DVarNets-sea-current

**Verdict:** B2 / M3 / M4 GPU96 numbers in `results/metrics_*_GPU96.json` are
**`pre_p0_fix`** and **must not** be used as formal paper claims until a
post-P0 retrain + eval. This pass does **not** invent JAMES table scores.

| # | P0 issue | Status | Where |
|---|----------|--------|--------|
| 1 | NATL60 truth fallback (`y_ssh = ssh_truth`) | **Fixed** | `data/natl60.py`, `data/dataset.py`, `config/default.yaml` (`allow_truth_background: false`). Paper mode requires `obs` + `oi`; missing → `FileNotFoundError` / `RuntimeError`. |
| 2 | Solver per-iter `x.detach()` | **Fixed** | `fourdvarnet/solver.py` `Solver4DVarNet` keeps full unrolled graph (`create_graph=True`, no detach). Ablation only: `Solver4DVarNetTruncated`. |
| 3 | Observation masked MSE denominator | **Fixed** | `_masked_mse`: `sum(r²·m)/sum(m)` over valid obs. |
| 4 | Lat-dependent f, dx, dy; non-periodic BCs | **Fixed** | `fourdvarnet/geometry.py`; `physics` grads alias non-periodic ops; `TrainingLoss` passes `dx`/`dy` into divergence; dataset geostrophy uses lat-aware metrics when lat/lon present. |
| 5 | SST advection time alignment | **Fixed** | Final-time backward: `(T_t−T_{t−1})/Δt + u_t·∇T_t − κ∇²T_t` (documented in `physics.sst_advection_residual`). |
| 6 | SST mask full window | **Fixed** | Dataset returns `mask_sst[t−dT+1:t+1]`; advection zeros residual where time/spatial stencil invalid. |
| 7 | Strict xarray align | **Fixed** | `_align_like`: assert time length/Δt; no silent return of misaligned arrays on failure. |
| 8 | Docs / M4 naming honesty | **Fixed** | M4 = **strain-aware spatial reweighting**, not uncertainty estimation. Compact 4DVarNet-**inspired** (not faithful Fablet R0). Solver was **not** unchanged. GPU96 metrics marked obsolete. |
| 9 | Unit tests | **Fixed** | `tests/test_p0_correctness.py` + updated physics/NATL60 tests. |

## Explicitly not done (next phases)

- Faithful R0 Fablet/IMT reproduction (full `lit_model_uv` parity)
- Full multi-seed NATL60 paper table retrain
- Learned uncertainty head (M5)
- VarDyn comparator

## Old metrics

All `results/metrics_*.json` files carry `"pre_p0_fix": true` and a disclaimer
string. GPU96 crop96/20ep ranking (B2 > M3 ≳ M4) remains useful only as
**historical / pre-fix** directional context, not as evidence for claims after
these correctness changes.
