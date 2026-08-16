# Engineering task: Fix M4 strain-uncertainty collapse + audit weak spots

**Role:** You are the external senior engineer. Cursor (local lead) will independently verify, apply, and test your patches. Deliver minimal, complete, testable fixes — not a paper draft.

**Baseline package:** `4dvarnets_m4_fix_src.zip`  
**Baseline SHA-256:** `F21DD7A2E18953E85C58C65BC223812384B0F01F155C5C8A52965D69F7442B0E`  
**Git:** no repository present → treat as `dirty` working tree. Do not assume remote access.

---

## 1. Background & goals

Physics-constrained 4DVarNet follow-on to Fablet et al. JAMES 2024 (SST–SSH → surface currents). Inner variational cost adds SQG + SST advection; supervised loss optionally uses strain-aware UV NLL:

\[
\sigma = \sigma_0 (1 + \alpha\,\mathrm{strain}),\quad
L_{\mathrm{uv}} = \mathbb{E}\big[\|u-u_{gt}\|^2/\sigma^2 + \log\sigma^2\big]
\]

**Synthetic 15-epoch ablation (CPU, directional only — NOT JAMES/NATL60 numbers):**

| ID | τ_uv (model) | τ_uv (geo) | Notes |
|----|--------------|------------|-------|
| B2 | ≈ 0.33 | ≈ 0.03 | OK |
| M3 | ≈ 0.36 | ≈ 0.03 | OK |
| **M4** | **≈ −1.07** | ≈ 0.03 | **BROKEN — worse than geo; collapse** |

Also: `lambda_x_*_km` often `NaN` in metrics JSON; NATL60 data is on disk (~35.7 GB) but paper table needs GPU.

**Primary goal (A):** Diagnose and fix M4 training collapse so strain-uncertainty improves (or at least does not destroy) UV skill vs M3/geo on synthetic short runs.

**Secondary goal (B):** Audit remaining bugs/weak spots vs paper plan (metrics λ_x, NATL60 train path, eval autograd, hyperparams). Propose minimal patches + tests.

**Out of scope for this chat:** Full paper writing; inventing NATL60/JAMES scores; GPU NATL60 200-epoch runs.

---

## 2. Architecture & non-negotiable boundaries

| Piece | Location |
|-------|----------|
| State x=(SSH,u,v), unrolled ConvLSTM | `fourdvarnet/model.py`, `convlstm.py` |
| Variational cost U | `fourdvarnet/solver.py` `VariationalCost` |
| G/H multimodal | `fourdvarnet/observation.py` |
| Prior Φ | `fourdvarnet/prior.py` |
| eSQG / adv / strain σ | `fourdvarnet/physics.py` |
| Supervised losses + UV NLL | `fourdvarnet/losses.py` |
| Metrics τ, λ_x, Lagrangian | `fourdvarnet/metrics.py` |
| Train / eval / ablation | `scripts/train.py`, `evaluate.py`, `run_ablation.py` |
| Config | `config/default.yaml` (`uncert_sigma0: 0.05`, `uncert_alpha: 1.0e4`) |

**Must preserve:**
- Unrolled 4DVarNet solver structure (do not replace with a different architecture).
- Honest synthetic vs NATL60 distinction; never invent table numbers.
- Eval must **not** wrap the inner solver in `torch.no_grad()` (autograd required for unrolled loop).
- CPU-runnable unit/smoke tests; do not require NATL60 `.nc` in CI.
- No secrets, no `.env`, no commits/push from your side (Cursor applies locally).

**Likely M4 failure modes to investigate (not exhaustive):**
1. `uncert_alpha: 1e4` + wrong `dx`/`dy` units → pathological σ (too large/small) → NLL dominates or gradients explode/vanish.
2. NLL **replaces** MSE UV term entirely (`l_uv_term = l_uv_nll`) while `loss.uv` weight stays 50 — scale mismatch vs other terms.
3. Strain computed from **predicted** u,v creates a trivial optimum (inflate strain → inflate σ → shrink NLL) that harms UV accuracy.
4. Missing detach / stop-grad on σ, or missing floor/clamp on log σ².
5. dx/dy not passed from physics grid (degrees vs meters) into `TrainingLoss`.

---

## 3. Research / change scope

**In scope:**
- `fourdvarnet/losses.py`, `physics.py` (strain_uncertainty / strain only as needed)
- `scripts/train.py` wiring of σ0, α, dx, dy, loss weights
- `config/default.yaml` hyperparameters for uncertainty (document rationale)
- `fourdvarnet/metrics.py` λ_x NaN handling if clearly buggy
- Related unit tests under `tests/`
- Short diagnostic notes in your reply

**Prefer minimal diffs.** Avoid drive-by refactors of G/H/Φ/ConvLSTM unless required for the bug.

**Audit checklist (report findings even if no code change):**
- [ ] M4 collapse root cause with evidence (math + code path)
- [ ] λ_x computation / NaN conditions
- [ ] NATL60 train path readiness (loader + flags; no need to run full train)
- [ ] `evaluate.py` autograd / no_grad correctness
- [ ] Default hyperparams vs synthetic grid spacing

---

## 4. Deliverables

1. **Root-cause analysis** (1–2 pages max in markdown): what broke M4, why B2/M3 were fine.
2. **Unified diff / patch files** applicable to the zip baseline (or clearly listed file replacements).
3. **New/updated tests** that would have caught the bug (e.g. NLL scale sanity, σ bounds, no-collapse on tiny train step, λ_x finite on synthetic).
4. **Config change notes** if α/σ0/weights change — with numerical justification.
5. **Reproduction commands** Cursor should run after apply.
6. Optional: short “known remaining risks” list.

Upload patches as files when possible; if paste-only, use complete `*** Begin Patch` / unified diff blocks per file.

---

## 5. Required tests (your design must enable Cursor to run)

```text
python scripts/smoke_test.py
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/ -v -p pytest
# Short M4 synthetic retrain (5–10 epochs) — Cursor will run:
python scripts/train.py --exp-name M4fix --use-sst 1 --use-sqg 1 --use-adv 1 --use-uncert 1 --epochs 8
python scripts/evaluate.py --ckpt checkpoints/4dvarnet-M4fix-best.pt --out results/metrics_M4fix.json
```

**Acceptance target (synthetic, directional):**
- After fix, M4 `tau_uv` must be **> geostrophic** (~0.03) and **not catastrophic** (≫ −1).
- Prefer M4 `tau_uv` ≥ M3-ish ballpark (≥ 0.2) on ~8–15 epoch CPU synthetic, OR document why uncertainty needs longer training / different schedule **and** show a measurable improvement vs the broken −1.07 baseline (e.g. τ_uv > 0 and rmse_uv < geo).
- All existing + new unit tests green; smoke green.
- Do **not** claim NATL60/JAMES scores.

---

## 6. Forbidden claims / ops

- Do not invent NATL60 or JAMES table metrics.
- Do not treat mock/unit tests as production validation of paper numbers.
- Do not request git commit/push/deploy.
- Do not include secrets or download huge `.nc` into the patch.
- Do not rewrite the entire model “for cleanliness.”

---

## 7. Acceptance criteria (Cursor will independently verify)

1. Patches apply cleanly to local tree.
2. Smoke + pytest pass.
3. Short M4 retrain shows clear recovery vs τ_uv ≈ −1.07 / rmse_uv ≈ 0.63.
4. Analysis matches code; no hand-wavy “maybe α is large” without a proposed fix.
5. Any remaining issues explicitly listed with severity.

---

## 8. Start now

Unpack the attached zip mentally from paths above. Prioritize **(A) M4 fix** then **(B) audit**. Return patches + analysis in one response when ready; if you need intermediate clarification, ask one focused question — otherwise proceed with best engineering judgment.
