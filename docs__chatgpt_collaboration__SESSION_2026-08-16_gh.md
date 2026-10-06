# ChatGPT collaboration session — 2026-08-16 GitHub public share

User mission: dual-agent + **public GitHub** authorization exception + report/paper polish.

## Dual-agent rules

- ChatGPT = advisor only; Cursor = sole implementer & acceptance
- Text paste only — no zip uploads to ChatGPT
- This turn: user authorized public `gh repo create` + push of code+docs (no data/nc, no huge ckpts, no secrets)

## A. GitHub public share — DONE

| Item | Result |
|------|--------|
| `.gitignore` | Excludes `data/natl60/*.nc`, `*.pt`, `tmp/`, wheels, `.env`, `__pycache__`, `.pytest_cache` |
| Initial commit | `44da37b` — 153 files (code, docs, metrics JSON, figures, report) |
| Public repo | **https://github.com/Coucou2016/4DVarNets-sea-current** |
| Visibility | PUBLIC (`gh repo view` confirmed) |
| Browse check | HTTP 200 on README, PAPER.md, raw `metrics_B2_GPU96.json` |
| Excluded | NATL60 `*.nc` (~36 GB), checkpoints `*.pt`, `tmp/wheels`, `synthetic_osse.npz`, secrets |

Follow-up commit this session: README/OUTLINE/report/discussion updates + refreshed figures + session pastes + tests logs.

## B. ChatGPT browser

| Attempt | Result |
|---------|--------|
| `browser_tabs` list | Empty |
| `browser_tabs` new | Created viewId, then vanished |
| `browser_navigate` chatgpt.com (with/without newTab / viewId) | **Fail ×2+:** “No browser tab available” / “Browser view not found” |
| Paste to chatgpt.com | **Not completed** |

**Fallback:** professional pastes left on disk for human/advisor paste (include public GitHub URL):

- `docs/chatgpt_collaboration/PASTE_ChatA_GH_LIT_2026-08-16.txt` (lit + framing + ask ChatGPT to read GitHub)
- `docs/chatgpt_collaboration/PASTE_ChatB_REPORT_REVIEW_2026-08-16.txt` (Chinese report / Methods review)

**Conversation URL:** none (automation failed).

### Independently verified citations (Cursor WebSearch)

| Paper | DOI |
|-------|-----|
| Fablet et al. 2024 JAMES SST–SSH 4DVarNets | https://doi.org/10.1029/2023MS003609 |
| Beauchamp et al. 2023 GMD 4DVarNet-SSH | https://doi.org/10.5194/gmd-16-2119-2023 |
| VarDyn dynamical SST–SSH (Le Guillou et al.; JAMES e2024MS004689) | https://doi.org/10.1029/2024MS004689 |

**Advice stance (Cursor-accepted without ChatGPT reply):** keep *ablatable physics residuals + honest B2>M3≳M4 regime dependence*; do not claim JAMES table scores; cite VarDyn as dynamical counterpart. No fabricated metrics.

## C. SciencePlots + report/paper

- Env: `faceswap` — SciencePlots OK; regenerated via `scripts/plot_science.py` (300 dpi pipeline in `fourdvarnet/plotting.py`)
- `scripts/build_report.py` → `docs/report/report.{html,md,pdf}` + root `report.html`
- Paper drafts: `docs/paper/01–04`, `PAPER.md`, `OUTLINE.md` updated with public GitHub URL

Measured GPU96 (post-NLL-fix M4) unchanged:

| ID | tau_uv | rmse_uv | rmse_ssh |
|----|--------|---------|----------|
| B2 | 0.848 | 0.184 | 0.059 |
| M3 | 0.811 | 0.206 | 0.064 |
| M4 | 0.801 | 0.211 | 0.063 |
| geo | −3.74 | 1.029 | 0.059 |

## D. Tests (faceswap)

- `scripts/smoke_test.py`: **PASSED** (`results/smoke_last.txt`)
- `python -m pytest tests/ -v` with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: **40 passed** (`results/pytest_last.txt`)

## E. Risks

- ChatGPT did not actually read the GitHub URL this turn (browser automation broken) — human must paste briefs
- crop96/20ep ≠ JAMES paper table; full-domain long runs still 待补充
- No PR / no deploy (by design)

## F. Git status intent

- Committed + pushed to **public** share repo
- No PR to unrelated repos; no deploy
