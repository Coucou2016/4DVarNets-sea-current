#!/usr/bin/env python3
"""Verify NATL60 paths, open datasets, print shapes / date ranges / splits.

Paper mode (default): obs + oi + refs are all required. There is no silent
SSH-truth background fallback. Debug-only truth background requires an
explicit ``--allow-truth-background`` flag.
"""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--allow-truth-background",
        action="store_true",
        help="DEBUG ONLY: allow missing obs/oi and y_ssh=ssh_truth (never for paper claims).",
    )
    args = ap.parse_args()
    allow_tb = bool(args.allow_truth_background)

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

    mode = "DEBUG allow_truth_background" if allow_tb else "paper mode (obs+oi+refs required)"
    print(f"mode: {mode}")

    missing = check_natl60_paths(resolved, allow_truth_background=allow_tb)
    if missing:
        print(missing_files_message(resolved, missing, allow_truth_background=allow_tb))
        raise SystemExit(2)

    print(f"box={GULF_STREAM_BOX}")
    data = load_natl60(resolved, root=ROOT, allow_truth_background=allow_tb)
    for key in ("ssh", "sst", "u", "v", "y_ssh", "y_oi", "mask_ssh", "mask_sst"):
        if key not in data:
            continue
        arr = data[key]
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")
    # Sanity: paper mode must not set y_ssh identical to truth everywhere
    if not allow_tb and "y_ssh" in data and "ssh" in data:
        same = (data["y_ssh"] == data["ssh"]).all()
        if same:
            raise SystemExit(
                "FAIL: y_ssh identical to ssh truth — truth background leakage (paper mode)."
            )
        print("  check: y_ssh is not identical to ssh truth (OK)")
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
