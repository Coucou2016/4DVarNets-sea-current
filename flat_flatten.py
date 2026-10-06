#!/usr/bin/env python3
"""Flatten this repository to the TOP LEVEL (no folders).

Every tracked (and selected untracked) file is moved to the repository root and
renamed with ``__`` (double underscore) as the directory separator, e.g.::

    fourdvarnet/solver.py              -> fourdvarnet__solver.py
    docs/paper/figures/fig_sqg_skill.png
                                       -> docs__paper__figures__fig_sqg_skill.png
    results/post_p0/metrics_B2-s0.json -> results__post_p0__metrics_B2-s0.json

The flat layout is **intentional**; see ``FLAT_LAYOUT_NOTICE.md``.

This script also regenerates ``FLAT_FILE_INDEX.md`` (human tables + a
machine-readable JSON map) which is the authoritative inverse map used by
``flat_unflatten.py``.

Usage
-----
    python flat_flatten.py --dry-run          # show the plan, change nothing
    python flat_flatten.py                    # perform the flatten
    python flat_flatten.py --no-git-mv        # plain filesystem moves
    python flat_flatten.py --index-only       # only rewrite FLAT_FILE_INDEX.md

Re-running on an already-flat tree is a no-op (it detects zero sources and
keeps the existing index).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration of what gets flattened
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent

# Root-level files keep their name unchanged (no separator needed).
ROOT_FILES = [
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "README.md",
    "environment.yml",
    "pytest.ini",
    "requirements.txt",
    "report.html",
]

# Files created by the flattening itself: never treated as flatten sources.
SELF_FILES = {
    "flat_flatten.py",
    "flat_unflatten.py",
    "FLAT_FILE_INDEX.md",
    "FLAT_LAYOUT_NOTICE.md",
}

# area -> allowed extensions (None means "every file not otherwise excluded")
AREAS: dict[str, set[str] | None] = {
    "checkpoints": {".json"},
    "config": {".yaml", ".yml"},
    "data": {".py"},
    "docs": None,
    "fourdvarnet": {".py"},
    "results": None,
    "scripts": {".py"},
    "tests": {".py"},
}

# basenames always allowed inside an area even if the extension is not listed
ALLOWED_BASENAMES = {".gitkeep"}

# Directories never walked.
EXCLUDE_DIR_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".git",
}

# Directories never walked, relative to the repo root.
EXCLUDE_DIR_RELPATHS = {"tmp"}

# Extensions never flattened (heavy data / weights / archives).
EXCLUDE_EXTENSIONS = {
    ".nc",
    ".nc4",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".whl",
    ".zip",
    ".part",
    ".tar",
    ".gz",
}

# Extra safety: filename patterns never flattened.
EXCLUDE_NAME_PATTERNS = [
    re.compile(r"(?i)^\.env($|\.)"),
    re.compile(r"(?i)\.pem$"),
    re.compile(r"(?i)\.key$"),
    re.compile(r"(?i)^id_rsa"),
    re.compile(r"(?i)\.netrc$"),
    re.compile(r"(?i)credential"),
    re.compile(r"(?i)secret"),
]

SEPARATOR = "__"


def group_for(rel: str) -> str:
    """Human grouping used in the index."""
    r = rel.replace("\\", "/")
    if r.startswith("fourdvarnet/") or r.startswith("scripts/") or r.startswith("tests/"):
        return "code"
    if r.startswith("data/"):
        return "code"
    if r.startswith("docs/paper/"):
        return "paper"
    if r.startswith("docs/report/"):
        return "report"
    if r.startswith("docs/"):
        return "docs"
    if r.startswith("results/"):
        return "results"
    if r.startswith("config/"):
        return "config"
    if r.startswith("checkpoints/"):
        return "checkpoints"
    return "root"


def flat_name(rel: str) -> str:
    return rel.replace("\\", "/").replace("/", SEPARATOR)


def is_excluded_dir(parts: tuple[str, ...]) -> bool:
    if any(p in EXCLUDE_DIR_PARTS for p in parts):
        return True
    rel = "/".join(parts)
    return any(rel == e or rel.startswith(e + "/") for e in EXCLUDE_DIR_RELPATHS)


def is_excluded_file(path: Path) -> bool:
    name = path.name
    if any(p.search(name) for p in EXCLUDE_NAME_PATTERNS):
        return True
    if name in SELF_FILES:
        return True
    if path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True
    return False


def build_plan(root: Path) -> list[dict]:
    """Return the list of {flat, original, bytes, group} entries."""
    entries: list[dict] = []

    for name in ROOT_FILES:
        p = root / name
        if p.is_file():
            entries.append(
                {
                    "flat": name,
                    "original": name,
                    "bytes": p.stat().st_size,
                    "group": "root",
                }
            )

    for area in sorted(AREAS):
        area_dir = root / area
        if not area_dir.is_dir():
            continue
        allowed_exts = AREAS[area]
        for dirpath, dirnames, filenames in os.walk(area_dir):
            here = Path(dirpath)
            rel_dir = here.relative_to(root)
            parts = tuple(rel_dir.parts)
            if is_excluded_dir(parts):
                dirnames[:] = []
                continue
            dirnames[:] = sorted(
                d
                for d in dirnames
                if not is_excluded_dir(parts + (d,))
            )
            for fn in sorted(filenames):
                fp = here / fn
                if is_excluded_file(fp):
                    continue
                if fn not in ALLOWED_BASENAMES:
                    if allowed_exts is not None and fp.suffix.lower() not in allowed_exts:
                        continue
                rel = fp.relative_to(root).as_posix()
                entries.append(
                    {
                        "flat": flat_name(rel),
                        "original": rel,
                        "bytes": fp.stat().st_size,
                        "group": group_for(rel),
                    }
                )

    entries.sort(key=lambda e: e["original"])

    # collision check
    seen: dict[str, str] = {}
    for e in entries:
        prev = seen.get(e["flat"])
        if prev is not None:
            raise SystemExit(
                f"FLAT NAME COLLISION: {e['flat']!r} would be produced by both "
                f"{prev!r} and {e['original']!r}. Resolve before flattening."
            )
        seen[e["flat"]] = e["original"]

    return entries


# --------------------------------------------------------------------------- #
# Index generation
# --------------------------------------------------------------------------- #

GROUP_TITLES = {
    "root": "Repository root files",
    "code": "Source code (fourdvarnet / data / scripts / tests)",
    "config": "Configuration (config/)",
    "docs": "Documentation (docs/)",
    "paper": "Paper (docs/paper/)",
    "report": "Report (docs/report/)",
    "results": "Results (results/)",
    "checkpoints": "Training history (checkpoints/)",
}

GROUP_ORDER = ["root", "code", "config", "docs", "paper", "report", "results", "checkpoints"]


def write_index(root: Path, entries: list[dict], out: Path) -> None:
    total_bytes = sum(e["bytes"] for e in entries)
    by_group: dict[str, list[dict]] = {}
    for e in entries:
        by_group.setdefault(e["group"], []).append(e)

    lines: list[str] = []
    lines.append("# FLAT_FILE_INDEX.md")
    lines.append("")
    lines.append(
        "Authoritative map between the **flat** (top-level, no folders) filenames in "
        "this repository and their **original** relative paths in the structured tree."
    )
    lines.append("")
    lines.append(
        "Separator convention: every `/` in the original path becomes `__` "
        "(double underscore). Example: `docs/paper/figures/fig_sqg_skill.png` -> "
        "`docs__paper__figures__fig_sqg_skill.png`."
    )
    lines.append("")
    lines.append("See [`FLAT_LAYOUT_NOTICE.md`](FLAT_LAYOUT_NOTICE.md) for the rationale.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Flat files indexed: **{len(entries)}**")
    lines.append(f"- Total payload: **{total_bytes} bytes** ({total_bytes / 1e6:.2f} MB)")
    lines.append(f"- Flat separator: `{SEPARATOR}`")
    lines.append("- Restore the structured, runnable tree with:")
    lines.append("  `python flat_unflatten.py --out .`")
    lines.append("")
    lines.append("Counts by group:")
    lines.append("")
    lines.append("| Group | Files | Bytes |")
    lines.append("|-------|-------|-------|")
    for g in GROUP_ORDER:
        if g not in by_group:
            continue
        n = len(by_group[g])
        b = sum(e["bytes"] for e in by_group[g])
        lines.append(f"| {GROUP_TITLES.get(g, g)} | {n} | {b} |")
    lines.append("")

    for g in GROUP_ORDER:
        if g not in by_group:
            continue
        lines.append(f"## {GROUP_TITLES.get(g, g)}")
        lines.append("")
        lines.append("| Flat filename | Original path | Bytes |")
        lines.append("|---------------|---------------|-------|")
        for e in by_group[g]:
            lines.append(f"| `{e['flat']}` | `{e['original']}` | {e['bytes']} |")
        lines.append("")

    lines.append("## Machine-readable map (JSON)")
    lines.append("")
    lines.append("<!-- FLAT_MAP_JSON_BEGIN -->")
    lines.append("```json")
    payload = {
        "separator": SEPARATOR,
        "total_files": len(entries),
        "total_bytes": total_bytes,
        "files": [
            {
                "flat": e["flat"],
                "original": e["original"],
                "bytes": e["bytes"],
                "group": e["group"],
            }
            for e in entries
        ],
    }
    lines.append(json.dumps(payload, indent=2))
    lines.append("```")
    lines.append("<!-- FLAT_MAP_JSON_END -->")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def tracked_set(root: Path) -> set[str]:
    try:
        res = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=str(root),
            capture_output=True,
            check=True,
        )
    except Exception:
        return set()
    out = res.stdout.decode("utf-8", "replace")
    return {p for p in out.split("\0") if p}


def move_file(root: Path, rel_src: str, flat: str, tracked: set[str], use_git_mv: bool) -> str:
    src = root / rel_src
    dst = root / flat
    if rel_src.replace("\\", "/") == flat:
        # top-level file that already lives at its flat name: nothing to do
        return "already-flat" if src.exists() else "missing"
    if not src.exists():
        if dst.exists():
            return "already-flat"
        return "missing"
    if dst.exists():
        raise SystemExit(f"REFUSING to overwrite existing {dst}")
    if use_git_mv and rel_src in tracked:
        try:
            subprocess.run(
                ["git", "mv", "--", rel_src, flat],
                cwd=str(root),
                capture_output=True,
                check=True,
            )
            return "git-mv"
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr.decode("utf-8", "replace").strip()
            print(f"  ! git mv failed for {rel_src}: {msg}; falling back to move")
    shutil.move(str(src), str(dst))
    return "move"


def remove_empty_dirs(root: Path, entries: list[dict]) -> list[str]:
    candidates = {Path(e["original"]).parts[0] for e in entries if "/" in e["original"]}
    removed: list[str] = []
    for area in sorted(candidates):
        base = root / area
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base, topdown=False):
            p = Path(dirpath)
            for d in list(dirnames):
                sub = p / d
                if sub.is_dir() and not any(sub.iterdir()):
                    try:
                        sub.rmdir()
                        removed.append(sub.relative_to(root).as_posix())
                    except OSError:
                        pass
            if p != base and p.is_dir() and not any(p.iterdir()):
                try:
                    p.rmdir()
                    removed.append(p.relative_to(root).as_posix())
                except OSError:
                    pass
        if base.is_dir() and not any(base.iterdir()):
            try:
                base.rmdir()
                removed.append(base.relative_to(root).as_posix())
            except OSError:
                pass
    return removed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT), help="repository root (default: script location)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without touching files")
    ap.add_argument("--index-only", action="store_true", help="only regenerate FLAT_FILE_INDEX.md")
    ap.add_argument("--no-git-mv", action="store_true", help="use plain filesystem moves, not `git mv`")
    ap.add_argument("--index", default=None, help="index output path (default: <root>/FLAT_FILE_INDEX.md)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    index_path = Path(args.index) if args.index else root / "FLAT_FILE_INDEX.md"

    entries = build_plan(root)
    total = sum(e["bytes"] for e in entries)
    print(f"planned flat files: {len(entries)}  ({total} bytes / {total / 1e6:.2f} MB)")

    if args.index_only:
        if not entries:
            print("no source files found; refusing to overwrite index in --index-only mode")
            return 1
        write_index(root, entries, index_path)
        print(f"wrote {index_path}")
        return 0

    if args.dry_run:
        for e in entries:
            print(f"  {e['original']}  ->  {e['flat']}  ({e['bytes']} B)")
        print("dry run: no files moved, index not rewritten")
        return 0

    if not entries:
        if index_path.exists():
            print("tree already flat (no sources found); existing index kept unchanged")
            return 0
        print("nothing to flatten and no index present")
        return 1

    tracked = tracked_set(root)
    counts: dict[str, int] = {}
    for e in entries:
        kind = move_file(root, e["original"], e["flat"], tracked, not args.no_git_mv)
        counts[kind] = counts.get(kind, 0) + 1

    removed = remove_empty_dirs(root, entries)

    write_index(root, entries, index_path)

    print("moves:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if removed:
        print("removed empty dirs:", ", ".join(removed))
    print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
