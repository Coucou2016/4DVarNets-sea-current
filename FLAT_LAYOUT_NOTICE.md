# FLAT LAYOUT NOTICE — the top-level, folder-free layout is INTENTIONAL

> **TL;DR — This is deliberate.** Every code, documentation, paper, report and
> core-result file in this repository lives **directly at the repository root**
> (the top level). There are **no sub-directories** in the published tree.
> This is not an accident, a broken export, or a mistaken `git mv`; it is the
> requested layout for this repository. See `FLAT_FILE_INDEX.md` for the full
> map, `flat_unflatten.py` to rebuild the runnable tree, and the `structured`
> branch / tag `v0.3.1-paper-report-pack` for the canonical structured release.

---

## 1. Why the layout is flat (explicit rationale)

The repository is published as a **read / review mirror** so that an LLM agent
(ChatGPT, or any other agent/tool) and a human can retrieve the *complete*
project — code, docs, paper, report, figures and core result data — from a
**single top-level directory listing**, with **no recursion required**.

Concretely, this makes these workflows trivial:

- **One-shot context loading.** A single `GET /contents/` (GitHub API) or one
  `ls` at the root returns *every* file name in the project. An agent does not
  have to walk `docs/paper/figures/...`, `results/post_p0/...`, etc. to discover
  that those files exist.
- **Cross-agent / cross-audit review.** Several agents (or one agent reviewing
  another's work) can each fetch the same flat listing and reason about the
  same, complete file set, which is exactly what the cross-review / cross-audit
  workflow requires. Renamed-but-related artifacts stay adjacent in the listing.
- **Robust to path-assembly bugs.** Deep paths are a common failure point for
  automated tooling (URL-encoding, case, trailing slashes). A flat namespace
  removes that class of error.
- **Cheap archival.** The whole payload is small (single-digit MB, see §4), so
  a flat listing is a practical way to hand over everything at once.

The trade-off is explicit and accepted: flattening breaks Python package
structure, so `import fourdvarnet` does **not** work in the flat tree. That is
why the runnable, structured tree is preserved (see §5) and why an inverse
helper is shipped (see §6).

---

## 2. Naming convention

The separator is a **double underscore** `__`. Every `/` of the original
relative path becomes `__`, and the file extension is preserved.

| Original path | Flat filename |
|---|---|
| `fourdvarnet/solver.py` | `fourdvarnet__solver.py` |
| `docs/paper/PAPER.md` | `docs__paper__PAPER.md` |
| `docs/report/figures/fig_sqg_skill.png` | `docs__report__figures__fig_sqg_skill.png` |
| `results/post_p0/metrics_B2-s0.json` | `results__post_p0__metrics_B2-s0.json` |
| `config/default.yaml` | `config__default.yaml` |

Notes:

- **Top-level files keep their names unchanged** (`README.md`, `LICENSE`,
  `NOTICE`, `report.html`, `pytest.ini`, `requirements.txt`, `environment.yml`,
  `.gitignore`, and the flat helper docs/scripts themselves).
- Pre-existing double underscores in a name are preserved, and the leading
  separator still applies. Example: `data/__init__.py` → `data____init__.py`
  (the first `__` is the separator, the second `__` is the original file name).
  `flat_unflatten.py` reverses this exactly and unambiguously.
- The mapping is **complete and machine-readable**: `FLAT_FILE_INDEX.md` contains
  a JSON block (`<!-- FLAT_MAP_JSON_BEGIN --> … <!-- FLAT_MAP_JSON_END -->`)
  plus human-readable tables (flat name → original path → size in bytes),
  grouped by area (root / code / config / docs / paper / report / results /
  checkpoints).

---

## 3. Reproducibility of the layout itself

The flattening is itself a committed, reproducible artifact:

- `flat_flatten.py` — performs the flatten (source tree → flat root). Supports
  `--dry-run`, `--index-only`, `--no-git-mv`; it is idempotent (re-running on an
  already-flat tree is a no-op) and it **regenerates `FLAT_FILE_INDEX.md`** so
  the map can never drift from the tree.
- `flat_unflatten.py` — the inverse. Reads `FLAT_FILE_INDEX.md` and rebuilds the
  structured tree.
- `FLAT_FILE_INDEX.md` — the authoritative map (also the unflatten input).

---

## 4. What is intentionally NOT here, and why

The flat listing is limited to source, docs and *lightweight* results. Excluded
on purpose:

| Excluded | Reason |
|---|---|
| `data/natl60/*.nc` (≈ 35 GB, 6 NetCDF cubes) | Far beyond GitHub's per-repo and per-file limits; re-downloadable via `scripts/download_natl60.py` (URLs in `data__natl60.py`). |
| `data/synthetic_osse.npz` (≈ 1.9 MB) | Regenerable synthetic OSSE artifact; produced by `scripts/generate_synthetic.py`. |
| `*.pt` / `*.pth` / `*.ckpt` model weights | Large binaries; not needed to read/review the science; retrainable. |
| `tmp/**` (incl. `tmp/wheels/*.whl`, ≈ 2 GB) | Local scratch, wheelhouse and staging; not project content. |
| `.pytest_cache/**`, `__pycache__/**` | Tool caches. |
| `.env`, `*.pem`, `*.key`, `secrets/`, `.netrc`, credentials | Secrets must never be published. |
| `.git/**` | Git internals, not content. |

All of these are also listed in `.gitignore`. Nothing excluded here was ever
tracked in the published history except the empty placeholder files
(`data/natl60/.gitkeep`, `data/real/.gitkeep`), which **are** included in flat
form as `data__natl60__.gitkeep` and `data__real__.gitkeep` so the intended data
directories remain documented.

Sizing check: the flat payload is ~8 MB across ~333 files, with the largest
single file ≈ 0.96 MB (`docs__report__report.pdf`). Well within every GitHub
limit, so no artifact had to be dropped for size, and no Git LFS is used.

---

## 5. How to get the runnable, structured tree back

There are two independent ways to recover the exact original tree. Both are
preserved permanently.

**Option A — reconstruct locally from the flat files (no network needed):**

```bash
# Rebuild into a fresh sibling directory (recommended; leaves this repo intact):
python flat_unflatten.py --out ../4dvar-structured

# Rebuild in place next to the flat files (flat files are COPIED, not deleted):
python flat_unflatten.py --out .

# Inspect the plan first, or move instead of copy:
python flat_unflatten.py --out . --dry-run
python flat_unflatten.py --out . --move
```

Running `--out .` is safe: flat filenames contain `__` and sit at the root,
while reconstructed files go into sub-directories (`fourdvarnet/`, `docs/`, …),
so the two never collide. After reconstruction:

```bash
python -m pytest tests/ -q
```

**Option B — check out the preserved structured branch / tag:**

```bash
git fetch origin
git checkout structured                 # runnable, structured tree (branch)
# or the canonical structured release:
git checkout v0.3.1-paper-report-pack
```

`structured` is the runnable source of truth and is described in
`docs__STAGE_EFGH_LEDGER.md` / `README.md` history. Tag
`v0.3.1-paper-report-pack` is the canonical structured release package.

---

## 6. Provenance and scientific integrity

- **Flattening copies bytes; it does not rewrite them.** Each flat file is the
  *same content* as its structured counterpart (sizes recorded in
  `FLAT_FILE_INDEX.md`). Evidence files are byte-identical; only the *name*
  changed.
- **Scientific results provenance is unchanged.** Post-P0 measured runs and the
  audit trail live in the same files as before, now flat:
  `results__post_p0__*.json` (Stages F–H, crop96 / 15 epochs / seeds 0–2),
  `results__physics_ops__physics_ops_validation.json` (Stage E),
  `results__legacy_pre_review2__*` (GPU96 pre-P0, obsolete for formal claims),
  and the audit narrative in `docs__AUDIT_EVIDENCE.md` / `.html` / `.pdf`
  plus `docs__REVIEW_RESPONSE_P0.md` and `docs__REVIEW_RESPONSE_ROUND2.md`.
- **Paper and report remain self-contained.** `docs__report__report.html`
  inlines its figures as Base64 `data:image` payloads (verified: 7 inline
  images), so it opens in a browser with no sibling files. The PDFs
  (`docs__paper__paper.pdf`, `docs__report__report.pdf`,
  `docs__AUDIT_EVIDENCE.pdf`) are included as-is.
- **Upstream reference.** This is an independent, compact 4DVarNet-*inspired*
  codebase motivated by Fablet et al. (2024), *JAMES*,
  [doi:10.1029/2023MS003609](https://doi.org/10.1029/2023MS003609), and the
  official OceaniX stack
  [CIA-Oceanix/4dvarnet-james-uv-ssc](https://github.com/CIA-Oceanix/4dvarnet-james-uv-ssc)
  (CeCILL-C, CEA/CNRS/INRIA). See `LICENSE` (MIT for original contributions)
  and `NOTICE` for the full dual-license / attribution statement. Any reuse of
  upstream-derived code must respect CeCILL-C.

---

## 7. Quick reference

| Question | Answer |
|---|---|
| Is the flat layout intentional? | **Yes.** This document is the statement of intent. |
| Where is the file map? | `FLAT_FILE_INDEX.md` (tables + embedded JSON). |
| How do I run the code? | `python flat_unflatten.py --out .` → `pytest tests/ -q`, **or** `git checkout structured`. |
| How do I re-flatten? | `python flat_flatten.py --dry-run` then `python flat_flatten.py`. |
| Where are the paper/report? | `docs__paper__PAPER.md` / `docs__paper__paper.pdf` / `docs__report__report.md` / `docs__report__report.html` / `docs__report__report.pdf`. |
| Where are the core results? | `results__post_p0__*`, `results__physics_ops__*`, `results__legacy_pre_review2__*`. |
| Why is `data/natl60/*.nc` missing? | ~35 GB; re-download with `scripts/download_natl60.py` (see §4). |
