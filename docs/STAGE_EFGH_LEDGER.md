# Stages E–H execution ledger (post-P0)

Public: https://github.com/Coucou2016/4DVarNets-sea-current

**Protocol (this machine):** GTX 950M 4GB, torch CUDA, NATL60 paper-mode (`obs`+`oi`),
**crop96 / batch1 / 15 epochs / seeds {0,1,2}**. These are **post_p0 directional** scores —
**not** full-grid ~200-ep JAMES Table rows.

| Stage | Item | Status | Artifact |
|-------|------|--------|----------|
| E | SQG skill vs truth UV | **Done** | `results/physics_ops/physics_ops_validation.json` |
| E | Advection residual sanity | **Done** | same + `fig_adv_residual_sanity.*` |
| E | λ_sqg caution (near-zero τ_uv) | **Done** | `lam_sqg_caution` in JSON |
| F–G | B1/B2/M1–M4/R0 multi-seed | **Done** | `results/post_p0/ablation_summary.json` |
| H | SST coarsening + sparsity + strain bins | **Done** | `results/post_p0/sensitivity/sensitivity_summary.json` |
| Docs | PAPER / report / AUDIT | **Done** | `docs/paper/`, `docs/report/`, `docs/AUDIT_EVIDENCE.md` |
| Repro | seed + git in ckpt/metrics; `environment.yml` | **Done** | |
| Gate | pytest | run at package time | |
| Gate | push + tag `v0.3.1-paper-report-pack` | pending | |

## post_p0 mean τ_uv (3 seeds)

| ID | τ_uv | rmse_uv | rmse_ssh |
|----|-----:|--------:|---------:|
| B1 | 0.861 | 0.178 | 0.059 |
| B2 | 0.878 | 0.166 | 0.059 |
| M1 | 0.916 | 0.139 | 0.049 |
| M2 | 0.880 | 0.165 | 0.056 |
| M3 | 0.914 | 0.140 | 0.050 |
| M4 | 0.917 | 0.137 | 0.051 |
| R0 | 0.850 | 0.185 | 0.059 |
| geo | 0.846 | — | — |

## Stage E (crop96, 23 test days)

| Quantity | Value |
|----------|------:|
| SQG τ_uv mean | **−0.011** |
| Adv RMS truth / scramble / zero | 6.06e−6 / 2.22e−5 / 4.97e−6 |

## Hardware-blocked

1. Uncropped ~200×200 × ~200-ep × multi-seed JAMES Table on 4GB VRAM.
2. Byte-faithful CIA-Oceanix R0 + official ckpt.
3. VarDyn comparator (separate stack).

## R0 honesty

R0 = compact larger `hidden_lstm` / `feat_dim` SSH+SST. **Not** byte-faithful.
