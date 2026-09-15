#!/usr/bin/env python3
"""Stage F–G orchestrator: multi-seed post-P0 NATL60 crop96 train+eval.

Writes under results/post_p0/. Not full-grid JAMES Table rows (VRAM-limited).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fourdvarnet.repro import git_commit  # noqa: E402

EXPERIMENTS = {
    "B1": dict(use_sst=0, use_sqg=0, use_adv=0, use_uncert=0),
    "B2": dict(use_sst=1, use_sqg=0, use_adv=0, use_uncert=0),
    "M1": dict(use_sst=1, use_sqg=1, use_adv=0, use_uncert=0),
    "M2": dict(use_sst=1, use_sqg=0, use_adv=1, use_uncert=0),
    "M3": dict(use_sst=1, use_sqg=1, use_adv=1, use_uncert=0),
    "M4": dict(use_sst=1, use_sqg=1, use_adv=1, use_uncert=1),
}

METRIC_KEYS = ["tau_uv", "rmse_uv", "rmse_ssh", "tau_div", "tau_vort"]


def _run(cmd: list[str]) -> int:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def _mean_std(vals: list[float]) -> dict:
    clean = [v for v in vals if v == v]
    if not clean:
        return {"mean": None, "std": None, "n": 0, "values": vals}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0, "n": 1, "values": vals}
    return {
        "mean": float(statistics.mean(clean)),
        "std": float(statistics.stdev(clean)),
        "n": len(clean),
        "values": vals,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="subset e.g. B1 B2 M1 M2 M3 M4 R0",
    )
    p.add_argument("--results-dir", default="results/post_p0")
    p.add_argument("--skip-r0", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="skip seed runs with metrics JSON already present")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--r0-config", default="config/r0_compact_faithful.yaml")
    args = p.parse_args()

    py = sys.executable
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    names = args.only or (list(EXPERIMENTS) + ([] if args.skip_r0 else ["R0"]))
    commit = git_commit(ROOT)
    ledger: dict = {
        "stage": "F_G",
        "git_commit": commit,
        "protocol": {
            "source": "natl60",
            "crop_size": args.crop_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
            "device": args.device,
            "note": (
                "post_p0 directional evidence on GTX 950M 4GB (crop96). "
                "Not full-grid JAMES Table rows."
            ),
        },
        "runs": [],
        "summary": {},
        "acceptance": {},
    }

    for name in names:
        if name == "R0":
            if args.skip_r0:
                continue
            config = args.r0_config
            flags = dict(use_sst=1, use_sqg=0, use_adv=0, use_uncert=0)
            r0_note = (
                "R0 = compact larger-capacity SSH+SST (hidden_lstm/feat closer to Fablet). "
                "Not a byte-faithful CIA-Oceanix transplant; no official ckpt eval on this machine."
            )
        elif name in EXPERIMENTS:
            config = args.config
            flags = EXPERIMENTS[name]
            r0_note = None
        else:
            raise SystemExit(f"unknown experiment {name}")

        seed_metrics = []
        for seed in args.seeds:
            # Prefer 5 seeds for B2/M3 when user passed ≥5; otherwise use all provided.
            exp_name = f"{name}-s{seed}"
            metrics_path = results_dir / f"metrics_{exp_name}.json"
            if args.skip_existing and metrics_path.is_file():
                print(f"SKIP existing {metrics_path}", flush=True)
                try:
                    seed_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                    ledger["runs"].append(
                        {
                            "name": name,
                            "exp_name": exp_name,
                            "seed": seed,
                            "train_ok": True,
                            "eval_ok": True,
                            "skipped_existing": True,
                            "ckpt": str(ROOT / "checkpoints" / f"4dvarnet-{exp_name}-best.pt"),
                            "r0_note": r0_note,
                            "metrics": seed_metrics[-1],
                        }
                    )
                except json.JSONDecodeError:
                    print(f"WARN: corrupt metrics, will re-run {exp_name}", flush=True)
                else:
                    continue
            cmd = [
                py,
                str(ROOT / "scripts" / "train.py"),
                "--config",
                config,
                "--source",
                "natl60",
                "--exp-name",
                exp_name,
                "--epochs",
                str(args.epochs),
                "--crop-size",
                str(args.crop_size),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--use-sst",
                str(flags["use_sst"]),
                "--use-sqg",
                str(flags["use_sqg"]),
                "--use-adv",
                str(flags["use_adv"]),
                "--use-uncert",
                str(flags["use_uncert"]),
            ]
            rc = _run(cmd)
            ckpt = ROOT / "checkpoints" / f"4dvarnet-{exp_name}-best.pt"
            row = {
                "name": name,
                "exp_name": exp_name,
                "seed": seed,
                "train_ok": rc == 0,
                "ckpt": str(ckpt),
                "r0_note": r0_note,
            }
            if rc != 0:
                row["eval_ok"] = False
                ledger["runs"].append(row)
                continue
            metrics_path = results_dir / f"metrics_{exp_name}.json"
            ev = [
                py,
                str(ROOT / "scripts" / "evaluate.py"),
                "--config",
                config,
                "--source",
                "natl60",
                "--ckpt",
                str(ckpt),
                "--crop-size",
                str(args.crop_size),
                "--batch-size",
                "1",
                "--device",
                args.device,
                "--out",
                str(metrics_path),
            ]
            erc = _run(ev)
            row["eval_ok"] = erc == 0
            if metrics_path.exists():
                blob = json.loads(metrics_path.read_text(encoding="utf-8"))
                row["metrics"] = blob
                seed_metrics.append(blob)
            ledger["runs"].append(row)

        # Aggregate mean±std
        agg = {"n_seeds": len(seed_metrics), "metrics": {}}
        geo_taus = []
        for key in METRIC_KEYS:
            vals = []
            for blob in seed_metrics:
                m = blob.get("model") or {}
                v = m.get(key)
                vals.append(float(v) if v is not None else float("nan"))
            agg["metrics"][key] = _mean_std(vals)
        for blob in seed_metrics:
            g = (blob.get("geostrophic") or {}).get("tau_uv")
            if g is not None:
                geo_taus.append(float(g))
        agg["geostrophic_tau_uv"] = _mean_std(geo_taus)
        if r0_note:
            agg["faithfulness"] = "not_byte_faithful_compact_larger_capacity"
            agg["note"] = r0_note
        ledger["summary"][name] = agg

    # Acceptance checks (Stage F)
    b1 = ledger["summary"].get("B1", {}).get("metrics", {})
    b2 = ledger["summary"].get("B2", {}).get("metrics", {})
    geo = None
    for name in ledger["summary"]:
        g = ledger["summary"][name].get("geostrophic_tau_uv", {}).get("mean")
        if g is not None:
            geo = g
            break
    b2_tau = (b2.get("tau_uv") or {}).get("mean")
    b1_tau = (b1.get("tau_uv") or {}).get("mean")
    ledger["acceptance"] = {
        "geo_tau_uv": geo,
        "geo_sane": geo is not None and -0.5 <= geo <= 1.0,
        "b2_beats_b1_tau_uv": (
            b2_tau is not None and b1_tau is not None and b2_tau > b1_tau
        ),
        "b2_tau_uv": b2_tau,
        "b1_tau_uv": b1_tau,
    }

    out = results_dir / "ablation_summary.json"
    out.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print("Acceptance:", json.dumps(ledger["acceptance"], indent=2))


if __name__ == "__main__":
    main()
