# Synthetic ablation note (CPU)

**Not** JAMES / NATL60 Table numbers. Directional evidence only (`data.source: synthetic`, 48×48 OSSE).

## Pre-fix (15 epochs, collapsed M4)

| ID | τ_uv (model) | τ_uv (geo) | rmse_uv | Notes |
|----|--------------|------------|---------|--------|
| B2 | 0.332 | 0.032 | 0.358 | SSH+SST beats geo |
| M3 | 0.356 | 0.032 | 0.351 | SQG+adv slightly above B2 |
| M4 | −1.07 | 0.032 | 0.630 | **BROKEN** — pred-strain NLL collapse |

## Post-fix (8 epochs, 2026-08-15)

| ID | τ_uv (model) | τ_uv (geo) | rmse_uv | Notes |
|----|--------------|------------|---------|--------|
| B2 | 0.242 | 0.032 | 0.381 | OK vs geo |
| M3 | 0.242 | 0.032 | 0.381 | OK vs geo |
| M4 | **0.322** | 0.032 | **0.360** | Fixed; competitive / best of trio at 8 ep |

Artifacts: `results/metrics_B2.json`, `metrics_M3.json`, `metrics_M4.json`, `ablation_summary.json`, `ablation_8ep_post_m4fix.txt`.

M4 fix: truth-based σ + clamp + MSE mix — `docs/chatgpt_collaboration/SESSION_2026-08-15.md` and `SESSION_2026-08-15_cont.md`.

Next for paper table: GPU, full NATL60 200-epoch matrix (do not use NATL60smoke 1-epoch numbers).
