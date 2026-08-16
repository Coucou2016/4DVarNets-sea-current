# ChatGPT collaboration session — 2026-08-16 report turn

User mission: dual-agent + paper/report (full self-contained `report.html`).

## Dual-agent rules

- ChatGPT = advisor only; Cursor = sole implementer & acceptance
- Text paste only — no zip/md uploads to ChatGPT
- Login/captcha → pause (not reached)
- GitHub: try if `gh` works — **blocked** (`gh` not on PATH / not authenticated); no password prompts; fall back to text paste

## A. ChatGPT browser

| Attempt | Result |
|---------|--------|
| Discover browser MCP tools | **None available** (no browser/chatgpt MCP server in this agent environment) |
| Paste to chatgpt.com | **Skipped** — cannot lock/navigate tabs without browser MCP |

**Fallback:** Cursor `WebSearch` verified DOIs; refreshed paste left for human/advisor:

- `docs/chatgpt_collaboration/PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_report.txt`

**Conversation URL:** none (browser channel unavailable).

### Independently verified citations (WebSearch)

| Paper | DOI |
|-------|-----|
| Fablet et al. 2024 JAMES SST–SSH 4DVarNets | https://doi.org/10.1029/2023MS003609 |
| Beauchamp et al. 2023 GMD 4DVarNet-SSH | https://doi.org/10.5194/gmd-16-2119-2023 |
| VarDyn dynamical SST–SSH (2024 MS) | https://doi.org/10.1029/2024MS004689 |
| Prior outline DOIs (Lapeyre & Klein; Rio; Martin; González-Haro related) | retained in `docs/paper/OUTLINE.md` |

Accepted framing: ablatable physics residuals + honest B2>M3≳M4 regime dependence; VarDyn as dynamical counterpart. `OUTLINE.md` updated.

## B. Figures (SciencePlots)

- Env: `faceswap` — SciencePlots OK; Times New Roman OK; CJK: SimSun / Microsoft YaHei
- Helper: `fourdvarnet/plotting.py` — CJK fallback documented; DPI 300
- Regenerated via `scripts/plot_science.py` → `results/figures/` + mirror `docs/paper/figures/`

## C. Paper outputs

| Path | Status |
|------|--------|
| `docs/paper/01–04_*.md` | Polished/kept evidence-calibrated |
| `docs/paper/OUTLINE.md` | Updated (VarDyn; report paths; GitHub/ChatGPT notes) |
| `docs/paper/PAPER.md` | Assembled |
| `docs/paper/paper.html` | Self-contained (Base64 figures) |

## D. Full research report (CRITICAL)

| Path | Status |
|------|--------|
| `docs/report/report.html` | **Done** — inline CSS, Base64 PNGs, HTML tables, no CDN |
| `docs/report/report.md` | Done |
| `docs/report/report.pdf` | Done via Microsoft Edge headless |
| `report.html` (repo root) | Mirror copy |
| `scripts/build_report.py` | Generator |

Measured GPU96 table embedded (post-NLL-fix M4):

| ID | tau_uv | rmse_uv | rmse_ssh |
|----|--------|---------|----------|
| B2 | 0.848 | 0.184 | 0.059 |
| M3 | 0.811 | 0.206 | 0.064 |
| M4 | 0.801 | 0.211 | 0.063 |
| geo | −3.74 | 1.029 | 0.059 |

## E. GitHub / git

- `gh`: not installed → **GitHub blocked** (noted in report)
- Local `git init` + commit of code+docs: attempted in same turn if feasible; **no push**

## F. Tests

- faceswap `python -m pytest tests/ -v` with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: **40 passed**
- Log: `results/pytest_last.txt`

## Item-19 summary (for parent)

1. ChatGPT: browser MCP absent → no conversation URL; paste `PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_report.txt`; DOIs via WebSearch.
2. GitHub: `gh` missing; `git init` created but `git status` blocked by dubious ownership (no `git config` changes per policy) → **no commit/push**.
3. Figures: SciencePlots regenerated under `results/figures/` + `docs/paper/figures/`.
4. Paper: `docs/paper/PAPER.md`, `docs/paper/paper.html`.
5. Report: `docs/report/report.html` (+ root `report.html`), `report.md`, `report.pdf` (Edge).
6. Metrics: B2>M3≳M4 post-NLL-fix; crop96/20ep ≠ table.
7. pytest: 40 passed.
