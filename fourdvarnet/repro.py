"""Reproducibility helpers: RNG seeding and git commit capture."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path


def git_commit(root: Path | None = None) -> str:
    """Return short HEAD SHA, or 'unknown' if not a git checkout."""
    cwd = str(root) if root is not None else None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs (CPU + CUDA when present)."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN is slow on GTX 950M; keep default nondeterministic kernels.
