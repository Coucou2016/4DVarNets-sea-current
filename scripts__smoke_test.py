#!/usr/bin/env python3
"""Quick import, forward-pass, and mini-train smoke test."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch
    from torch.utils.data import DataLoader

    from data.dataset import make_synthetic_datasets
    from data.synthetic import SyntheticOSSEConfig
    from fourdvarnet.losses import TrainingLoss
    from fourdvarnet.model import FourDVarNetUV

    print("1. imports OK")
    cfg = SyntheticOSSEConfig(n_time=20, height=32, width=32, dT=5)
    train_ds, _, _ = make_synthetic_datasets(cfg, None)
    batch = train_ds[0]
    print(f"2. dataset sample shapes: truth={batch['truth'].shape}")

    model = FourDVarNetUV(dT_sst=5, n_iter=3, hidden_lstm=32, feat_dim=8, use_sst=True)
    pred = model(
        batch["y_ssh"].unsqueeze(0),
        batch["z_sst"].unsqueeze(0),
        batch["mask_ssh"].unsqueeze(0),
        batch["mask_sst"].unsqueeze(0),
        batch["u_geo"].unsqueeze(0),
        batch["v_geo"].unsqueeze(0),
    )
    assert pred.shape == (1, 3, 32, 32), pred.shape
    print(f"3. forward OK: {pred.shape}")

    loader = DataLoader(make_synthetic_datasets(cfg, None)[0], batch_size=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = TrainingLoss()
    model.train()
    for i, b in enumerate(loader):
        if i >= 2:
            break
        p = model(b["y_ssh"], b["z_sst"], b["mask_ssh"], b["mask_sst"], b["u_geo"], b["v_geo"])
        loss, _ = crit(p, b["truth"], model.phi)
        loss.backward()
        opt.step()
        opt.zero_grad()
    print(f"4. mini-train loss={loss.item():.4f}")

    model_sst_off = FourDVarNetUV(dT_sst=5, n_iter=2, hidden_lstm=16, feat_dim=8, use_sst=False)
    p2 = model_sst_off(
        batch["y_ssh"].unsqueeze(0),
        batch["z_sst"].unsqueeze(0),
        batch["mask_ssh"].unsqueeze(0),
        batch["mask_sst"].unsqueeze(0),
        batch["u_geo"].unsqueeze(0),
        batch["v_geo"].unsqueeze(0),
    )
    assert p2.shape[1] == 3
    print("5. SSH-only model OK")

    dx = 0.05 * 111e3
    model_phys = FourDVarNetUV(
        dT_sst=5,
        n_iter=2,
        hidden_lstm=16,
        feat_dim=8,
        use_sst=True,
        use_sqg=True,
        use_adv=True,
        use_uncert=True,
        dx=dx,
        dy=dx,
        Ld=30e3,
        f0=7e-5,
    )
    p3 = model_phys(
        batch["y_ssh"].unsqueeze(0),
        batch["z_sst"].unsqueeze(0),
        batch["mask_ssh"].unsqueeze(0),
        batch["mask_sst"].unsqueeze(0),
        batch["u_geo"].unsqueeze(0),
        batch["v_geo"].unsqueeze(0),
    )
    assert p3.shape == (1, 3, 32, 32)
    crit_u = TrainingLoss(use_uncert=True, dx=dx, dy=dx)
    loss_u, logs = crit_u(p3, batch["truth"].unsqueeze(0), model_phys.phi)
    loss_u.backward()
    assert "l_uv_nll" in logs
    print("6. SQG+adv+uncert forward/backward OK")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
