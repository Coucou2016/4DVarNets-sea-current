#!/usr/bin/env python3
"""Reconstruct the original structured (runnable) tree from the flat layout.

This is the inverse of ``flat_flatten.py``. It reads ``FLAT_FILE_INDEX.md``
(the machine-readable JSON map embedded in it, with a markdown-table fallback)
and copies/moves every flat file back to its original relative path, recreating
the directories.

The flat layout is **intentional** (see ``FLAT_LAYOUT_NOTICE.md``); this helper
exists so the code can be run again, e.g. for cross-checking by other agents.

Usage
-----
    # Rebuild the structured tree in a *separate* directory (recommended):
    python flat_unflatten.py --out ../4dvar-structured

    # Rebuild in the current directory (flat files are copied, not deleted):
    python flat_unflatten.py --out .

    # Keep no duplicates: move instead of copy (destroys the flat layout):
    python flat_unflatten.py --out . --move

    python flat_unflatten.py --out . --dry-run
    python flat_unflatten.py --out . --list

Notes
-----
* Default behaviour is **copy**, so the flat repository stays complete.
* Running with ``--out .`` does NOT collide with the flat files: flat names
  contain ``__`` and live at the root, while the reconstructed files live in
  subdirectories (``fourdvarnet/``, ``docs/``, ...).
* After reconstruction, verify with:
      python -m pytest tests/ -q
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

JSON_BEGIN = "<!-- FLAT_MAP_JSON_BEGIN -->"
JSON_END = "<!-- FLAT_MAP_JSON_END -->"

# | `flat__name.py` | `orig/path.py` | 1234 |
ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*$"
)


def parse_index(index_path: Path) -> list[dict]:
    """Return [{flat, original, bytes}] from FLAT_FILE_INDEX.md."""
    text = index_path.read_text(encoding="utf-8")

    if JSON_BEGIN in text and JSON_END in text:
        block = text.split(JSON_BEGIN, 1)[1].split(JSON_END, 1)[0]
        block = block.strip()
        block = re.sub(r"^```(?:json)?\s*", "", block)
        block = re.sub(r"\s*```$", "", block)
        data = json.loads(block)
        return [
            {
                "flat": f["flat"],
                "original": f["original"],
                "bytes": int(f.get("bytes", 0)),
            }
            for f in data["files"]
        ]

    # fallback: markdown tables
    out: list[dict] = []
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            flat, original, nbytes = m.group(1), m.group(2), int(m.group(3))
            if flat.lower() in ("flat filename", "flat name"):
                continue
            out.append({"flat": flat, "original": original, "bytes": nbytes})
    if not out:
        raise SystemExit(f"could not parse any file map from {index_path}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="destination root for the structured tree (default: .)")
    ap.add_argument("--index", default=None, help="path to FLAT_FILE_INDEX.md (default: next to this script)")
    ap.add_argument("--src", default=None, help="directory holding the flat files (default: index directory)")
    ap.add_argument("--move", action="store_true", help="move instead of copy (removes the flat files)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan without writing anything")
    ap.add_argument("--list", action="store_true", help="just list the map and exit")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing destination files")
    args = ap.parse_args(argv)

    index_path = Path(args.index) if args.index else HERE / "FLAT_FILE_INDEX.md"
    if not index_path.is_file():
        print(f"index not found: {index_path}", file=sys.stderr)
        return 2
    src_root = Path(args.src).resolve() if args.src else index_path.resolve().parent
    out_root = Path(args.out).resolve()

    entries = parse_index(index_path)

    if args.list:
        for e in entries:
            print(f"{e['flat']}  ->  {e['original']}")
        print(f"total: {len(entries)} files")
        return 0

    n_done = n_same = n_missing = 0
    for e in entries:
        src = src_root / e["flat"]
        dst = out_root / e["original"]
        if args.dry_run:
            print(f"[dry-run] {src.name}  ->  {dst}")
            continue
        if not src.is_file():
            n_missing += 1
            print(f"  ! missing flat file: {src.name}")
            continue
        if dst.exists() and not args.overwrite:
            if dst.stat().st_size == e["bytes"]:
                n_same += 1
                continue
            print(f"  ! exists (use --overwrite): {e['original']}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        n_done += 1

    if args.dry_run:
        print(f"dry run: {len(entries)} entries, nothing written")
        return 0

    print(
        f"reconstructed {n_done} files into {out_root}"
        f" ({n_same} already present, {n_missing} missing flat sources)"
    )
    print("verify with:  python -m pytest tests/ -q")
    return 0 if n_missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
