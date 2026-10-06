#!/usr/bin/env python3
"""Generate synthetic OSSE NetCDF/NPZ for training and tests."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic import SyntheticOSSEConfig, generate_synthetic_osse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic_osse.npz")
    p.add_argument("--n-time", type=int, default=40)
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = SyntheticOSSEConfig(n_time=args.n_time, height=args.size, width=args.size, seed=args.seed)
    generate_synthetic_osse(cfg, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
