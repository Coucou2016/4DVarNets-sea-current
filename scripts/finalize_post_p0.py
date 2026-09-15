#!/usr/bin/env python3
"""After F/G ablation_summary exists: run Stage H, plots, report stubs, final ledger."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fourdvarnet.repro import git_commit  # noqa: E402


def _run(cmd: list[str]) -> int:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> None:
    py = sys.executable
    summary_path = ROOT / "results" / "post_p0" / "ablation_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}; wait for run_post_p0_pipeline.py")

    ledger = json.loads(summary_path.read_text(encoding="utf-8"))
    # Prefer seed-0 ckpts for sensitivity
    b2 = ROOT / "checkpoints" / "4dvarnet-B2-s0-best.pt"
    m3 = ROOT / "checkpoints" / "4dvarnet-M3-s0-best.pt"
    if not b2.is_file() or not m3.is_file():
        raise SystemExit(f"need {b2} and {m3} for Stage H")

    rc = _run(
        [
            py,
            str(ROOT / "scripts" / "run_sensitivity.py"),
            "--crop-size",
            "96",
            "--device",
            "cuda",
            "--b2-ckpt",
            "checkpoints/4dvarnet-B2-s0-best.pt",
            "--m3-ckpt",
            "checkpoints/4dvarnet-M3-s0-best.pt",
            "--out-dir",
            "results/post_p0/sensitivity",
            "--sst-factors",
            "1",
            "4",
            "8",
            "--mask-keep-every",
            "3",
        ]
    )
    if rc != 0:
        raise SystemExit(rc)

    _run([py, str(ROOT / "scripts" / "plot_science.py")])
    _run([py, str(ROOT / "scripts" / "assemble_paper.py")])
    _run([py, str(ROOT / "scripts" / "build_report.py")])

    phys = ROOT / "results" / "physics_ops" / "physics_ops_validation.json"
    sens = ROOT / "results" / "post_p0" / "sensitivity" / "sensitivity_summary.json"
    final = {
        "git_commit": git_commit(ROOT),
        "stage_E": json.loads(phys.read_text(encoding="utf-8")) if phys.is_file() else None,
        "stage_FG_acceptance": ledger.get("acceptance"),
        "stage_FG_summary": ledger.get("summary"),
        "stage_H": json.loads(sens.read_text(encoding="utf-8")) if sens.is_file() else None,
        "hardware_blocked": [
            "Full-grid uncropped NATL60 (~200x200) x ~200 epochs x multi-seed on GTX 950M 4GB (OOM / wall-time).",
            "Byte-faithful CIA-Oceanix R0 Lightning stack + official pretrained ckpt eval (not available locally).",
            "Optional M5 learned sigma head skipped (not clean on 4GB schedule).",
            "VarDyn comparator experiments not run (out of scope / separate codebase).",
        ],
        "protocol": ledger.get("protocol"),
    }
    out = ROOT / "results" / "post_p0" / "STAGES_EFGH_FINAL.json"
    out.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
