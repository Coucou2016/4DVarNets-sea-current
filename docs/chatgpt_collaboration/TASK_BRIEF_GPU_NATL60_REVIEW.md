# Engineering task: Review M4 fix + GPU NATL60 next steps

**Role:** External senior engineer. Cursor applies/verifies locally. Do **not** invent JAMES/NATL60 table scores.

**Baseline package:** `4dvarnets_gpu_collab_src.zip`  
**SHA-256:** `9FA88476228CC198EB9AF99B4BAFC8397B186EDDAF05B59E624C3633142412A6`  
**Git:** no `.git` → treat as dirty local tree. No commit/push from your side.

---

## 1. Background (measured, local)

Physics-constrained 4DVarNet (SQG + adv + optional strain uncertainty) on SST–SSH → UV.

**M4 collapse (fixed locally):** σ was built from **predicted** UV strain → heteroscedastic NLL collapse. Fix: σ from **truth** UV (`uncert_from_truth`), clamp `σ ≤ σ0·uncert_max_mult`, MSE mix floor `uncert_mse_mix=0.25`.

**Post-fix synthetic 8-epoch directional (CPU) — NOT paper table:**

| ID | τ_uv | geo τ_uv | rmse_uv |
|----|------|----------|---------|
| B2 | 0.242 | 0.032 | 0.381 |
| M3 | 0.242 | 0.032 | 0.381 |
| M4 | **0.322** | 0.032 | **0.360** |

NATL60 on disk; loader OK; 1-epoch crop-48 smoke was wiring-only. User machine has **GTX 950M 4GB** (driver 460.89); default env was `torch 2.12.0+cpu` — Cursor is bringing up a CUDA-capable env separately.

**Primary goal:** Critique the M4 fix; propose **minimal** next engineering for GPU NATL60 ablation **B2 → M3 → M4**; refine `docs/PAPER_EXPERIMENTS.md` runbook for a 4GB Maxwell GPU (crop/batch/epochs realism). Optional tiny patches only.

**Out of scope:** Full paper prose with invented numbers; 200-epoch claims without runs; replacing ConvLSTM 4DVarNet architecture.

---

## 2. Architecture boundaries (must preserve)

| Piece | Location |
|-------|----------|
| State / unrolled solver | `fourdvarnet/model.py`, `convlstm.py`, `solver.py` |
| Physics (eSQG/adv/strain) | `fourdvarnet/physics.py` |
| Losses (incl. M4 NLL) | `fourdvarnet/losses.py` |
| Train/eval/ablation | `scripts/train.py`, `evaluate.py`, `run_ablation.py` |
| Paper runbook | `docs/PAPER_EXPERIMENTS.md` |

- Eval must **not** wrap inner solver in `torch.no_grad()`.
- Never quote crop/max_samples smoke metrics as Table 1.
- Prefer minimal diffs; no drive-by refactors of G/H/Φ.

---

## 3. Deliverables (acceptably testable)

1. **Critique** of M4 truth-σ / clamp / mse-mix fix: remaining failure modes, hyperparam risks (α=1e4, dx≈5.5 km).
2. **GPU NATL60 engineering plan** for 4GB VRAM: recommended `--crop-size`, `--batch-size`, `--max-samples` (if any), epoch schedule for B2 then M3/M4, OOM fallback ladder (e.g. 128→96→64→48) — still prefer >48 when possible.
3. **Runbook refinements** (markdown bullets or patch to `PAPER_EXPERIMENTS.md`) for small-GPU reality without weakening paper-row definition.
4. Optional **unified diffs** only if a clear bug/gap remains (tests preferred).
5. **Forbidden claims list** restated: no invented τ/RMSE/λ_x for NATL60.

---

## 4. Required verification Cursor will run

After your reply (patches applied if any):

```bash
python scripts/smoke_test.py
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/ -v -p pytest
```

If GPU env ready:

```bash
python -c "import torch; assert torch.cuda.is_available()"
python scripts/train.py --source natl60 --exp-name B2 --device cuda --epochs <your_suggest> --crop-size <your_suggest> --batch-size 1 --use-sst 1 --use-sqg 0 --use-adv 0 --use-uncert 0
```

Do not claim Cursor’s smoke as paper validation.

---

## 5. Acceptance criteria

- [ ] Explicit agree/disagree on M4 root cause + whether local fix is sufficient for next GPU runs
- [ ] Concrete B2→M3→M4 GPU schedule sized for ~4GB VRAM
- [ ] Clear distinction: synthetic directional vs NATL60 paper-ready rows
- [ ] Any patch applies cleanly to the zip tree and is minimal
- [ ] No invented metrics

Upload a short markdown reply + optional patch files.

