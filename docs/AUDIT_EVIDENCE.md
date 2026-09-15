# Audit evidence — data / code / results provenance

**Public repo:** https://github.com/Coucou2016/4DVarNets-sea-current  
**Purpose:** Demonstrate that reported numbers are computed in *this* repository, quarantine legacy protocols, and list what was **not** run. This document is *not* a manuscript section.

**Audit date:** 2026-09-15  

---

## 1. Scientific honesty rules (enforced)

1. Use **only** metrics JSON under `results/` (including `results/legacy_pre_review2/` when labelled as quarantine).
2. **Never** copy Fablet/JAMES (or any external) table numbers as “our results.”
3. Label protocol tags: `post_p0`, `legacy_pre_review2` / `pre_p0_fix`, synthetic short-train.
4. Full-grid ~200-epoch JAMES Table claims: **待补充** (not invented).

---

## 2. Primary measured artefacts (post_p0)

| Artefact | Role | SHA256 (first 16 hex) |
|----------|------|------------------------|
| `results/post_p0/ablation_summary.json` | Multi-seed mean±std for B1–M4, R0 | `c2ddc875a018a1ae` |
| `results/post_p0/metrics_{ID}-s{0,1,2}.json` | Per-seed evaluation | (see §2.1) |
| `results/physics_ops/physics_ops_validation.json` | Stage E SQG / advection | `c01f1054d6627029` |
| `results/legacy_pre_review2/metrics_B2_GPU96.json` | Quarantined historical | `4c9344624604903e` |

**Protocol (ablation_summary):** NATL60, `crop_size=96`, `epochs=15`, `batch_size=1`, `seeds=[0,1,2]`, CUDA. Note in JSON: *not* full-grid JAMES Table rows. Git commit recorded in summary: `89f61b6`.

### 2.1 Per-seed metric files

Present under `results/post_p0/`:

- B1-s0/s1/s2, B2-s0/s1/s2, M1-s0/s1/s2, M2-s0/s1/s2, M3-s0/s1/s2, M4-s0/s1/s2, R0-s0/s1/s2

Each JSON includes `ckpt`, `crop_size`, `seed`, `git_commit_ckpt`, `git_commit_eval`, `model.*`, `geostrophic.*`, and `protocol_note`.

### 2.2 Checkpoints (local only; not pushed)

Corresponding best weights: `checkpoints/4dvarnet-{ID}-s{seed}-best.pt` (excluded from git as `*.pt`). Training histories: `checkpoints/history-{ID}-s{seed}.json` (+ `-meta.json`).

---

## 3. How numbers were computed

| Stage | Script | Inputs | Outputs |
|-------|--------|--------|---------|
| E | `scripts/validate_physics_operators.py` | NATL60 truth crop96 | `results/physics_ops/*` |
| F–G train/eval | `scripts/run_post_p0_pipeline.py` → `scripts/train.py`, `scripts/evaluate.py` | `config/default.yaml`, `config/r0_compact_faithful.yaml` | ckpts + `results/post_p0/metrics_*.json` |
| F–G summarize | pipeline / `finalize_post_p0.py` | metrics JSONs | `ablation_summary.json` |
| H | `scripts/run_sensitivity.py` | B2-s0 / M3-s0 ckpts | `results/post_p0/sensitivity/` (if completed) |
| Figures | `scripts/plot_science.py` + `fourdvarnet/plotting.py` | metrics JSON | `results/figures/`, mirrored to `docs/paper/figures/`, `docs/report/figures/` |
| Paper | `scripts/assemble_paper.py` | `docs/paper/0*.md` | `PAPER.md`, `paper.html`, `paper.pdf` |
| Report | `scripts/build_report.py` | metrics + PNGs | `docs/report/report.{html,md,pdf}`, root `report.html` |

Core libraries: `fourdvarnet/model.py`, `solver.py`, `physics.py`, `losses.py`, `metrics.py`; data `data/natl60.py`, `data/dataset.py`.

---

## 4. Quarantine: legacy vs current

| Tag | Path | Use |
|-----|------|-----|
| **post_p0** | `results/post_p0/` | Primary scientific ranking for this package |
| **legacy_pre_review2** / **pre_p0_fix** | `results/legacy_pre_review2/` | Historical crop96/20ep only; do not cite as current claims |
| Synthetic short | `results/metrics_B2.json` etc. | Directional code-path only |

Known historical issue (documented): pre-P0 geostrophic τ_uv could be pathological (~−3.7); post_p0 OI-only geo τ_uv ≈ **0.846** (sane).

---

## 5. What was NOT run / 待补充

1. Full-grid uncropped NATL60 (~200×200) × ~200 epochs × multi-seed JAMES Table (4GB VRAM / wall-time).
2. Byte-faithful CIA-Oceanix R0 + official pretrained checkpoint evaluation.
3. VarDyn comparator experiments (separate codebase).
4. Formal OSE / drifter score tables (hooks may exist; scores 待补充).
5. Copying any external paper table as ours — **not done**.

5. Stage H sensitivity: **Done** — `results/post_p0/sensitivity/sensitivity_summary.json` (SST factors 1/4/8; altimetry keep_every=3; strain bins). M3 τ_uv drops under strong SST coarsening (factor 8: ~0.876 vs ~0.909 at factor 1); sparsity impact mild.

---

## 6. Pytest / tooling gate

Command (faceswap env, 2026-09-15 package build):

```text
python -m pytest -q --tb=line
..........................................................               [100%]
```

Status: **green** (58 tests).

---

## 7. Plagiarism / table hygiene

- Manuscript and report tables are built from **local** JSON only.
- Literature DOIs were verified via WebSearch (2026-09-15): Fablet 2024 `10.1029/2023MS003609`; Beauchamp 2023 `10.5194/gmd-16-2119-2023`; VarDyn 2025 `10.1029/2024MS004689`; Lapeyre & Klein 2006 `10.1175/JPO2840.1`.
- Paper body avoids local absolute paths and git SHAs (audit/report may include them).

---

## 8. Deliverable map

| Deliverable | Paths |
|-------------|-------|
| Outline | `docs/paper/OUTLINE.md` |
| Paper | `docs/paper/PAPER.md`, `paper.html`, `paper.pdf` |
| Report | `docs/report/report.md`, `report.html`, `report.pdf` (+ root `report.html`) |
| Audit | `docs/AUDIT_EVIDENCE.md` (+ optional html/pdf) |
| Figures | `results/figures/`, `docs/paper/figures/`, `docs/report/figures/` |

---

## 9. Recompute hashes (maintainer)

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('results/post_p0/ablation_summary.json'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```
