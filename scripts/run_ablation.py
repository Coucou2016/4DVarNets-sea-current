#!/usr/bin/env python3
"""Run the B1/B2/M1–M4 ablation matrix (synthetic or NATL60).

B1 SSH-only, B2 SSH-SST, M1 +sqg, M2 +adv, M3 both, M4 +uncert.
Use --quick for 2 epochs (wiring check). Prefer --epochs N for paper-prep
directional runs. Metrics JSON lands under --results-dir when eval succeeds.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "B1": dict(use_sst=0, use_sqg=0, use_adv=0, use_uncert=0),
    "B2": dict(use_sst=1, use_sqg=0, use_adv=0, use_uncert=0),
    "M1": dict(use_sst=1, use_sqg=1, use_adv=0, use_uncert=0),
    "M2": dict(use_sst=1, use_sqg=0, use_adv=1, use_uncert=0),
    "M3": dict(use_sst=1, use_sqg=1, use_adv=1, use_uncert=0),
    "M4": dict(use_sst=1, use_sqg=1, use_adv=1, use_uncert=1),
}


def _parse_eval_stdout(text: str) -> dict[str, float]:
    """Parse ``key: value   (geostrophic ...)`` lines from evaluate.py."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9_]+):\s*([-+eE0-9.nan]+)", line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        try:
            out[key] = float(raw)
        except ValueError:
            continue
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="2 epochs (synthetic wiring check)")
    p.add_argument("--epochs", type=int, default=None, help="override train epochs (paper-prep)")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--only", nargs="*", default=None, help="subset of experiment names")
    p.add_argument("--source", default=None, help="synthetic | natl60 (passed to train/eval)")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--results-dir", default="results", help="directory for metrics JSON")
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--name-suffix", default="", help="appended to exp-name / metrics stem")
    args = p.parse_args()

    names = args.only or list(EXPERIMENTS)
    py = sys.executable
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name}; choose from {list(EXPERIMENTS)}")
        flags = EXPERIMENTS[name]
        exp_name = f"{name}{args.name_suffix}"
        cmd = [
            py,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            args.config,
            "--exp-name",
            exp_name,
            "--use-sst",
            str(flags["use_sst"]),
            "--use-sqg",
            str(flags["use_sqg"]),
            "--use-adv",
            str(flags["use_adv"]),
            "--use-uncert",
            str(flags["use_uncert"]),
        ]
        if args.source:
            cmd += ["--source", args.source]
        if args.quick:
            cmd += ["--epochs", "2"]
        elif args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.crop_size is not None:
            cmd += ["--crop-size", str(args.crop_size)]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.device:
            cmd += ["--device", args.device]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        print(">>", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(r.returncode)
        ckpt = ROOT / "checkpoints" / f"4dvarnet-{exp_name}-best.pt"
        row: dict = {"name": name, "exp_name": exp_name, **flags, "ckpt": str(ckpt), "seed": args.seed}
        if not args.skip_eval and ckpt.exists():
            metrics_path = results_dir / f"metrics_{exp_name}.json"
            ev_cmd = [
                py,
                str(ROOT / "scripts" / "evaluate.py"),
                "--config",
                args.config,
                "--ckpt",
                str(ckpt),
                "--out",
                str(metrics_path),
                "--batch-size",
                "1",
            ]
            if args.source:
                ev_cmd += ["--source", args.source]
            if args.crop_size is not None:
                ev_cmd += ["--crop-size", str(args.crop_size)]
            if args.max_samples is not None:
                ev_cmd += ["--max-samples", str(args.max_samples)]
            if args.device:
                ev_cmd += ["--device", args.device]
            ev = subprocess.run(ev_cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(ev.stdout, end="" if ev.stdout.endswith("\n") or not ev.stdout else "\n")
            if ev.stderr:
                print(ev.stderr, file=sys.stderr, end="")
            row["eval_ok"] = ev.returncode == 0
            if metrics_path.exists():
                try:
                    row["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    row["metrics"] = _parse_eval_stdout(ev.stdout)
            else:
                row["metrics"] = _parse_eval_stdout(ev.stdout)
        summary.append(row)

    out = results_dir / "ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Keep legacy path for older docs / smoke
    legacy = ROOT / "checkpoints" / "ablation_summary.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {legacy}")


if __name__ == "__main__":
    main()
