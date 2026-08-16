#!/usr/bin/env python3
"""One-shot verification: smoke + pytest + mini-train + eval."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    print(">>", " ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    py = sys.executable
    run([py, "scripts/smoke_test.py"])
    run(
        [py, "-m", "pytest", "tests/", "-v", "-p", "pytest"],
        extra_env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    run([py, "scripts/generate_synthetic.py"])
    run([py, "scripts/train.py", "--epochs", "6", "--exp-name", "verify"])
    run([py, "scripts/evaluate.py", "--ckpt", "checkpoints/4dvarnet-verify-best.pt"])
    print("ALL VERIFY STEPS PASSED")


if __name__ == "__main__":
    main()
