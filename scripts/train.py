#!/usr/bin/env python3
"""Train 4DVarNet-SSH-SST on synthetic or NATL60 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import make_datasets, scales_from_dataset
from fourdvarnet.losses import LossWeights, TrainingLoss
from fourdvarnet.model import build_fourdvarnet
from fourdvarnet.repro import git_commit, seed_all


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths(root: Path) -> dict:
    p = root / "config" / "paths.yaml"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _batch_scales(batch: dict) -> tuple[float | None, float | None]:
    dx = batch.get("dx")
    dy = batch.get("dy")
    if dx is None or dy is None:
        return None, None
    if isinstance(dx, torch.Tensor):
        dx = float(dx.float().mean().item())
    if isinstance(dy, torch.Tensor):
        dy = float(dy.float().mean().item())
    return float(dx), float(dy)


def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        dx_b, dy_b = _batch_scales(batch)
        pred = model(
            batch["y_ssh"],
            batch["z_sst"],
            batch["mask_ssh"],
            batch["mask_sst"],
            batch["u_geo"],
            batch["v_geo"],
            dx=dx_b,
            dy=dy_b,
        )
        loss, _ = criterion(pred, batch["truth"], model.phi, dx=dx_b, dy=dy_b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def eval_epoch(model, loader, criterion, device):
    """Validation loss; 4DVar inner loop still uses autograd."""
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        dx_b, dy_b = _batch_scales(batch)
        pred = model(
            batch["y_ssh"],
            batch["z_sst"],
            batch["mask_ssh"],
            batch["mask_sst"],
            batch["u_geo"],
            batch["v_geo"],
            dx=dx_b,
            dy=dy_b,
        )
        loss, _ = criterion(pred, batch["truth"], model.phi, dx=dx_b, dy=dy_b)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def _flag(cli_value: int | None, cfg_value: bool) -> bool:
    if cli_value is None:
        return bool(cfg_value)
    return bool(cli_value)


def checkpoint_tag(use_sst: bool, use_sqg: bool, use_adv: bool, use_uncert: bool, exp_name: str | None) -> str:
    if exp_name:
        return exp_name
    tag = "ssh-sst" if use_sst else "ssh-only"
    if use_sqg:
        tag += "-sqg"
    if use_adv:
        tag += "-adv"
    if use_uncert:
        tag += "-uncert"
    return tag


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--use-sst", type=int, default=None, help="1=SSH+SST, 0=SSH only")
    p.add_argument("--use-sqg", type=int, default=None, help="1=SQG residual in inner cost")
    p.add_argument("--use-adv", type=int, default=None, help="1=SST advection residual in inner cost")
    p.add_argument("--use-uncert", type=int, default=None, help="1=strain-weighted UV NLL")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--exp-name", default=None, help="checkpoint tag (B1, M4, ...)")
    p.add_argument("--source", default=None, help="synthetic | natl60")
    p.add_argument("--crop-size", type=int, default=None, help="NATL60 center crop (CPU smoke)")
    p.add_argument("--max-samples", type=int, default=None, help="cap windows per split (CPU smoke)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: train.seed or 0)")
    p.add_argument("--lam-sqg", type=float, default=None, help="override model.lam_sqg")
    p.add_argument("--lam-adv", type=float, default=None, help="override model.lam_adv")
    p.add_argument("--lr", type=float, default=None, help="override train.lr")
    p.add_argument(
        "--device",
        default=None,
        help="auto | cpu | cuda (default: config train.device)",
    )
    args = p.parse_args()

    cfg = load_config(ROOT / args.config)
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    if args.source:
        cfg.setdefault("data", {})["source"] = args.source
    if args.crop_size is not None:
        cfg.setdefault("data", {})["crop_size"] = args.crop_size
    if args.max_samples is not None:
        cfg.setdefault("data", {})["max_samples"] = args.max_samples
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size
    if args.device:
        tcfg["device"] = args.device
    if args.lam_sqg is not None:
        mcfg["lam_sqg"] = float(args.lam_sqg)
    if args.lam_adv is not None:
        mcfg["lam_adv"] = float(args.lam_adv)
    if args.lr is not None:
        tcfg["lr"] = float(args.lr)

    seed = args.seed if args.seed is not None else int(tcfg.get("seed", 0))
    seed_all(seed)
    commit = git_commit(ROOT)

    use_sst = _flag(args.use_sst, mcfg.get("use_sst", True))
    use_sqg = _flag(args.use_sqg, mcfg.get("use_sqg", False))
    use_adv = _flag(args.use_adv, mcfg.get("use_adv", False))
    use_uncert = _flag(args.use_uncert, mcfg.get("use_uncert", False))
    epochs = args.epochs or tcfg["epochs"]
    want = str(tcfg.get("device", "auto")).lower()
    if want in ("auto", ""):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif want.startswith("cuda") and not torch.cuda.is_available():
        print("WARN: config asked for CUDA but torch.cuda.is_available() is False; using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(want)
    dcfg = cfg.get("data") or {}
    print(
        f"device={device}  epochs={epochs}  source={dcfg.get('source', 'synthetic')}"
        f"  crop_size={dcfg.get('crop_size')}  max_samples={dcfg.get('max_samples')}"
        f"  batch_size={tcfg['batch_size']}  seed={seed}  commit={commit}"
    )

    paths_all = load_paths(ROOT)
    natl_paths = paths_all.get("natl60") if isinstance(paths_all.get("natl60"), dict) else paths_all
    try:
        train_ds, val_ds, _ = make_datasets(cfg, ROOT, paths=natl_paths)
    except FileNotFoundError as exc:
        raise SystemExit(f"Data load failed:\n{exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(f"Dataset split error:\n{exc}") from exc
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"])

    # Prefer NATL60/synthetic dataset meter spacings over isotropic config fallback.
    scales = scales_from_dataset(train_ds, cfg)
    model = build_fourdvarnet(
        cfg,
        use_sst=use_sst,
        use_sqg=use_sqg,
        use_adv=use_adv,
        use_uncert=use_uncert,
        dx=scales["dx"],
        dy=scales["dy"],
        f0=scales["f0"],
    ).to(device)

    lw = LossWeights(**cfg["loss"])
    criterion = TrainingLoss(
        lw,
        use_uncert=use_uncert,
        sigma0=float(mcfg.get("uncert_sigma0", 0.05)),
        alpha=float(mcfg.get("uncert_alpha", 1.0e4)),
        dx=scales["dx"],
        dy=scales["dy"],
        uncert_from_truth=bool(mcfg.get("uncert_from_truth", True)),
        uncert_max_mult=float(mcfg.get("uncert_max_mult", 10.0)),
        uncert_mse_mix=float(mcfg.get("uncert_mse_mix", 0.25)),
    ).to(device)
    print(f"physics scales: dx={scales['dx']:.1f} m  dy={scales['dy']:.1f} m  f0={scales['f0']:.3e}")
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    ckpt_dir = ROOT / tcfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    tag = checkpoint_tag(use_sst, use_sqg, use_adv, use_uncert, args.exp_name)

    flags = {
        "use_sst": use_sst,
        "use_sqg": use_sqg,
        "use_adv": use_adv,
        "use_uncert": use_uncert,
        "exp_name": tag,
        "seed": seed,
        "git_commit": commit,
    }

    for ep in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, opt, criterion, device)
        va = eval_epoch(model, val_loader, criterion, device)
        history.append({"epoch": ep, "train": tr, "val": va})
        print(f"epoch {ep}: train={tr:.4f} val={va:.4f}")
        if va < best:
            best = va
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": ep,
                    "best_val": best,
                    "config": cfg,
                    **flags,
                },
                ckpt_dir / f"4dvarnet-{tag}-best.pt",
            )

    hist_name = f"history-{tag}.json" if args.exp_name else "history.json"
    with open(ckpt_dir / hist_name, "w", encoding="utf-8") as f:
        # Flat list kept for plot_science / legacy; meta nested alongside when readers support it.
        json.dump(history, f, indent=2)
    meta_path = ckpt_dir / (f"history-{tag}-meta.json" if args.exp_name else "history-meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "git_commit": commit, "epochs": epochs, "best_val": best}, f, indent=2)
    print(f"Best val loss: {best:.4f}  ckpt=4dvarnet-{tag}-best.pt  seed={seed}")


if __name__ == "__main__":
    main()
