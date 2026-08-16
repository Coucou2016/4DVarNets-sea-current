# ChatGPT collaboration session — 2026-08-16 (expanded: lit + paper + SciencePlots)

User: continue plan + Chinese asks (progress, ChatGPT lit/web search, paper framework, SciencePlots, nature-skills, GPU).

## Dual-agent rules (this session)

- ChatGPT = advisor only; Cursor = sole implementer / acceptance
- **No file uploads** to ChatGPT (text paste only); no zip packages requested
- No git commit/push/PR
- Browser MCP for ChatGPT: **unavailable** (`cursor-ide-browser` not in available MCP servers this turn; only `cursor-app-control`). Paste briefs left on disk for user.

## Progress snapshot

| Item | Status |
|------|--------|
| Repo | **No `.git`** - local-only dirty tree |
| Tests | 39 pytest + smoke |
| GPU | GTX 950M 4GB; **faceswap** `torch 1.12.1+cu113` |
| B2-GPU96 train | **COMPLETE** (20/20); best val **4.5316**; ckpt `checkpoints/4dvarnet-B2-GPU96-best.pt` |
| B2-GPU96 eval | **DONE** -> `results/metrics_B2_GPU96.json` (tau_uv~0.848; crop96/20ep != paper Table) |
| M3-GPU96 train | **COMPLETE** (20/20); best val **5.1902**; ckpt `checkpoints/4dvarnet-M3-GPU96-best.pt` |
| M3-GPU96 eval | **DONE** -> `results/metrics_M3_GPU96.json` (tau_uv~0.811; sst+sqg+adv) |
| M4-GPU96 train | **COMPLETE** (20/20); best val **682.36** @ep11 (uncert NLL scale); ckpt `checkpoints/4dvarnet-M4-GPU96-best.pt` |
| M4-GPU96 eval | **DONE** -> `results/metrics_M4_GPU96.json` (tau_uv~0.806; +uncert; late-epoch val spike after ep11) |
| Synthetic ablation | B2/M3/M4 metrics JSON on disk (directional) |
| SciencePlots | Regenerated incl. M4 loss + GPU96 bars with B2/M3/M4 (Times New Roman) |
| ChatGPT | Not used this follow-up (no spam) |

## ChatGPT collab

| Chat | Purpose | URL | Status |
|------|---------|-----|--------|
| A | M4 loss + 4GB recipe | — | Paste ready: `docs/chatgpt_collaboration/PASTE_ChatA_M4_GPU_2026-08-16.txt` — **not pasted** (browser MCP missing) |
| B | Lit survey + outline (ask ChatGPT to use web search) | — | Paste ready: `docs/chatgpt_collaboration/PASTE_ChatB_LIT_FRAMEWORK_2026-08-16.txt` — **not pasted** |

**Independent lit (Cursor WebSearch, verified DOIs):**

- Fablet et al. 2024 JAMES — https://doi.org/10.1029/2023MS003609
- Beauchamp et al. 2023 GMD 4DVarNet-SSH — https://doi.org/10.5194/gmd-16-2119-2023
- Lapeyre & Klein 2006 JPO SQG — https://doi.org/10.1175/JPO2840.1
- Rio et al. 2016 JTECH heat-budget currents — https://doi.org/10.1175/JTECH-D-16-0017.1
- González-Haro / Isern-Fontanet transfer fn — https://doi.org/10.1029/2019JC015958
- Martin et al. 2023 JAMES SST–SSH DL — https://doi.org/10.1029/2022MS003589

Advice accepted without ChatGPT reply: built local outline from verified sources + nature-skills methods playbook.

## Local paper docs

- `docs/paper/OUTLINE.md`
- `docs/paper/DRAFT_FRAMEWORK.md`
- Figures: `docs/paper/figures/` (mirrored from `results/figures/`)

## SciencePlots

```
faceswap: pip install SciencePlots matplotlib
import scienceplots OK; Times New Roman present
text.usetex forced False (no system LaTeX)
```

Helper: `fourdvarnet/plotting.py`  
Regenerator: `scripts/plot_science.py`  
Outputs:

- `results/figures/fig_synth_ablation_tau_uv.{png,pdf}`
- `results/figures/fig_synth_ablation_rmse_uv.{png,pdf}`
- `results/figures/fig_B2_GPU96_loss.{png,pdf}`
- mirrored under `docs/paper/figures/`

`requirements.txt` updated with `matplotlib`, `SciencePlots`.

## nature-skills

- **Already installed** as Cursor skill pack at `C:\Users\Administrator\.cursor\skills\nature-skills\` (Yuan1z0825/nature-skills style; not pip/npm).
- Followed: `nature-writing` (methods paper_type, one-sentence argument, terminology ledger, no invented claims) and `nature-figure` Python path with **user override** to SciencePlots + Times New Roman.
- No additional clone required this session.

## Code changes (local)

1. `scripts/evaluate.py` / `evaluate_drifters.py` — drop `weights_only=` for torch 1.12
2. `fourdvarnet/plotting.py` + `scripts/plot_science.py` — SciencePlots figures
3. Paper docs + ChatGPT paste briefs
4. `requirements.txt` — matplotlib, SciencePlots

## Tests

- smoke: **PASSED** (`faceswap`)
- pytest: **39 passed** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, faceswap); `results/pytest_last.txt`

## Git

**仅本地修改** — no `.git` / no commit / no push.

## Unverified risks

- Citation completeness (TGRS SWOT list not exhaustive); ChatGPT web survey not executed in-browser
- B2-GPU96 / M3-GPU96 metrics not paper Table rows (crop96, 20ep)
- α=1e4 / dx units for strain NLL still risk for M4 GPU
- Concurrent CPU eval + GPU train may slow machine
- Paper claims must stay behind completed experiments

## Follow-up (same day, GPU/paper)

- Monitored M4-GPU96 to TRAIN_EXIT 0 (~34 min); best val 682.36 @ep11; val unstable after ep12.
- Evaluated M4 with `--source natl60 --crop-size 96` -> `results/metrics_M4_GPU96.json`.
- Regenerated SciencePlots (`scripts/plot_science.py`) incl. `fig_M4_GPU96_loss` and B2/M3/M4 GPU96 tau/rmse bars.
- crop96/20ep metrics are **not** paper Table rows.
- ChatGPT: not used. No git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>".

### GPU96 crop96/20ep metrics (local; != paper table)

| ID | best val | tau_uv | rmse_uv | rmse_ssh | lambda_x_uv_km |
|----|----------|--------|---------|----------|----------------|
| B2 | 4.5316 | 0.8481 | 0.1843 | 0.0590 | 90.3 |
| M3 | 5.1902 | 0.8106 | 0.2063 | 0.0643 | 79.6 |
| M4 | **4.83 (post-NLL-fix)** | 0.8015 | 0.2112 | **0.0627** | 86.1 |
| M4 pre-fix | 682.36 | 0.8056 | 0.2093 | 0.1224 | 91.1 |
| geo | - | -3.740 | 1.029 | 0.0586 | 101.6 |

**Update:** M4-GPU96 retrained post-NLL-fix (see `SESSION_2026-08-16_cont.md`). Metrics JSON overwritten; pre-fix backed up.
