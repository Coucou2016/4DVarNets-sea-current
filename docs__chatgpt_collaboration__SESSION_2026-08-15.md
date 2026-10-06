# ChatGPT collaboration session — 2026-08-15

## Goal
Fix M4 strain-uncertainty training collapse (τ_uv ≈ −1.07) and audit weak spots; dual-agent workflow with ChatGPT as external senior engineer.

## ChatGPT links
- **Blocked:** Cursor `cursor-ide-browser` MCP could create tabs (`about:blank`) but `browser_navigate` immediately failed with either `No browser tab available` or `Browser view not found` (tabs vanished between create and navigate). Retried after freeing C: disk space; same failure. No ChatGPT conversation URL was obtained.
- User has ChatGPT Pro/Plus available, but this session could not reach the ChatGPT UI via the built-in browser automation channel.

## Baseline package
| Field | Value |
|-------|-------|
| Zip (post-fix tree) | `tmp/4dvarnets_m4_fix_src.zip` |
| Size | 62481 bytes |
| SHA-256 | `6957FC774A4720E9AB6831E182ED01263B983A17D84289A856F1FA6A4216AF83` |
| Git | **no `.git` repository** → baseline recorded as `dirty` |
| Secret scan | clean (no private keys / AWS keys / sk- tokens in staged source) |
| Excluded | `.git`, `__pycache__`, `.pytest_cache`, `checkpoints/*.pt`, `data/natl60/*.nc`, `.env` |

Task brief prepared at: `docs/chatgpt_collaboration/TASK_BRIEF_M4_FIX.md` (intended for ChatGPT upload).

## What was asked of ChatGPT (not delivered — browser blocker)
A. Diagnose/fix M4 strain-uncertainty collapse  
B. Audit λ_x NaN, NATL60 path, eval autograd, hyperparams  
C. Minimal patches + tests  

## What Cursor lead did instead (accountable local fix)
**Root cause:** `TrainingLoss` computed `σ = σ0(1+α·strain(u_pred,v_pred))` and **replaced** the UV MSE with Gaussian NLL. The optimizer could inflate predicted strain → inflate σ → shrink NLL without learning accurate UV (classic heteroscedastic collapse). Evidence: broken M4 `results/metrics_M4.json` τ_uv≈−1.07 vs B2/M3 ≈0.33–0.36.

**Fix (local):**
1. `fourdvarnet/losses.py` — σ from **truth** UV by default (`uncert_from_truth`); clamp σ ≤ `σ0·uncert_max_mult`; mix MSE floor `uncert_mse_mix=0.25`.
2. `config/default.yaml` — document/new knobs.
3. `scripts/train.py` — wire knobs.
4. `fourdvarnet/physics.py` — docstring warning on detach/truth.
5. `tests/test_losses.py` — regression: wild pred strain must not beat calm NLL when σ from truth.

## Independent verification
| Gate | Result |
|------|--------|
| `python scripts/smoke_test.py` | PASSED |
| `pytest tests/ -v -p pytest` (PYTEST_DISABLE_PLUGIN_AUTOLOAD=1) | **37 passed** (was 35) |
| M4fix train 8 epochs CPU synthetic | val loss ↓ 1500→1231 |
| `evaluate.py` on `4dvarnet-M4fix-best.pt` | **τ_uv=0.2807** vs geo **0.0316**; rmse_uv **0.371** vs geo **0.431** |

Compare to broken M4: τ_uv **−1.07** → **+0.28**; rmse_uv **0.63** → **0.37**.

Artifacts: `results/metrics_M4fix.json`, `results/m4fix_train.txt`, `results/m4fix_eval.txt`, `checkpoints/4dvarnet-M4fix-best.pt`.

## Audit notes (B) — no ChatGPT, local review
| Item | Status |
|------|--------|
| Eval `no_grad` | OK — `evaluate.py` documents and avoids wrapping solver |
| NATL60 loader | Present; data on disk; train path works with `--source natl60` (GPU still needed for paper table) |
| λ_x NaN | Expected when no PSD bin has err/signal &lt; 0.5; not a crash bug |
| Hyperparams α=1e4 | Reasonable with dx≈5.5 km once σ not from live pred |

## Infra incidents this session
1. **C: disk full** (~0.16 GB free) → `SQLITE_FULL`, hung shells, flaky browser. Freed by clearing `%LOCALAPPDATA%\Temp` → ~8.5 GB free.
2. Browser MCP still cannot open ChatGPT after disk recovery.

## Code status
- Changes are **local only** (no git repo; no commit/push/PR).
- Not production/NATL60 paper validation; synthetic directional evidence only.

## Next recommended
1. User opens ChatGPT manually (or fix Cursor browser) and pastes `TASK_BRIEF_M4_FIX.md` + zip for senior review of the local fix.
2. Longer synthetic ablation (15 epochs) B2/M3/M4fix for updated notes.
3. GPU NATL60 200-epoch matrix when hardware available — do not invent JAMES scores.
