#!/usr/bin/env python3
"""Verify NATL60 paths, open datasets, print shapes / date ranges / splits."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.natl60 import (
    GULF_STREAM_BOX,
    NATL60_SPLITS,
    check_natl60_paths,
    indices_for_split,
    load_natl60,
    missing_files_message,
)


def main() -> None:
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    paths = cfg.get("natl60") or {}
    if not paths:
        raise SystemExit("config/paths.yaml has no natl60 mapping")

    resolved = {}
    for k, rel in paths.items():
        p = Path(rel)
        if not p.is_absolute():
            p = ROOT / p
        resolved[k] = str(p)
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        part = p.with_suffix(p.suffix + ".part")
        part_s = f", partial {part.stat().st_size} B" if part.exists() else ""
        print(f"{k}: {'OK' if exists else 'MISSING'} {p} ({size} B{part_s})")

    missing = check_natl60_paths(resolved)
    if missing:
        print(missing_files_message(resolved, missing))
        raise SystemExit(2)

    optional = []
    for key in ("obs", "oi"):
        p = resolved.get(key)
        if p and not Path(p).exists():
            optional.append(key)
    if optional:
        print(f"note: optional files missing (loader will use SSH ref / full mask): {optional}")

    print(f"box={GULF_STREAM_BOX}")
    data = load_natl60(resolved, root=ROOT)
    for key in ("ssh", "sst", "u", "v", "y_ssh", "mask_ssh", "mask_sst"):
        arr = data[key]
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")
    times = data["time"]
    tmin, tmax = times[0], times[-1]
    print(f"  time: n={len(times)} range={tmin} → {tmax}")
    print(f"  lat shape={data['lat'].shape} lon shape={data['lon'].shape}")
    dT = 7
    for split in ("train", "val", "test"):
        idx = indices_for_split(times, split, dT)
        print(f"  split {split} {NATL60_SPLITS[split]}: n_windows={len(idx)}")
    print("NATL60 loader OK")


if __name__ == "__main__":
    main()
