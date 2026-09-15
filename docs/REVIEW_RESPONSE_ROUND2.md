# Round-2 review response — Stage A consistency (`v0.2.0-review2-fixed`)

Public repo: https://github.com/Coucou2016/4DVarNets-sea-current

**Acceptance target:** `clean clone → pip install → pytest -q → 100% pass`, with
physics / data / solver / tests / docs **internally consistent**.

**Verdict (this pass):** Stage A–D consistency is tagged `v0.2.0-review2-fixed`.
Stages E–H post-P0 measured runs live under `results/physics_ops/` and
`results/post_p0/` (crop96 multi-seed; **not** full-grid JAMES Table). Legacy
GPU96 remains `legacy_pre_review2` / `pre_p0_fix`. Do **not** invent JAMES
table scores.

| ID | Issue | Status | Where |
|----|-------|--------|--------|
| A1 | `physics` must use non-periodic `geometry.grad_*` | **Done** | `fourdvarnet/physics.py` aliases `grad_x`/`grad_y`/`laplacian_nonperiodic` |
| A2 | HxW / lat-row `dx`/`dy` broadcast in `grad_x`/`grad_y` | **Done** | `geometry._broadcast_metric` + sliced denominators |
| A3 | `allow_truth_background` / paper mode; no `y_ssh=ssh_truth` | **Done** | `data/natl60.py`, `scripts/verify_natl60.py`; strict length checks (no quiet truncate) |
| A4 | Full SST mask window; lat-aware scales; OI vs hybrid | **Done** | `data/dataset.py` (`mask_sst[t-dT+1:t+1]`, `y_oi`/`u_geo_oi`) |
| A5 | Default solver: no per-iter detach; no Python-float norm barrier | **Done** | `solver.Solver4DVarNet` + `GradUpdateLSTM` (tensor scale only); Truncated ablation only |
| A6 | Observation masked MSE ÷ `sum(mask)` | **Done** | `solver._masked_mse` |
| A7 | `L_adv` uses stencil / window valid mask | **Done** | `VariationalCost` + `stencil_valid_mask` |
| A8 | SST advection final-time backward scheme | **Done** | `physics.sst_advection_residual` (documented; not BDF2) |
| A9 | `TrainingLoss` passes real `dx`,`dy` into divergence | **Done** | `fourdvarnet/losses.py` |
| A10 | Geo baseline from pure OI only | **Done** | `scripts/evaluate.py` + dataset `u_geo_oi`; **full-test pooled** metrics primary; batch-mean kept as provisional |
| A11 | Honest docs; GPU96 legacy | **Done** | README / PAPER / OUTLINE; `results/legacy_pre_review2/`; LICENSE + NOTICE |
| A12 | Tests match implementation | **Done** | `tests/test_p0_correctness.py` (maps, density, early-iter grads, source guards, paper mode) |

## Stage E–H status

See `docs/STAGE_EFGH_LEDGER.md` and `results/post_p0/`. Stage E measured; F–H running/completing on crop96 multi-seed post-P0 protocol. Full-grid JAMES Table and byte-faithful R0 remain hardware-/scope-blocked (listed in ledger).

## Tag

**`v0.2.0-review2-fixed`** — Stage A–D consistency (geometry, solver graph, paper-mode NATL60, honest evaluate).

