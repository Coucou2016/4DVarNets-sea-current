"""Training improves over geostrophic baseline on synthetic data."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import make_synthetic_datasets
from data.synthetic import SyntheticOSSEConfig
from fourdvarnet.losses import TrainingLoss
from fourdvarnet.model import FourDVarNetUV
from fourdvarnet.physics import explained_variance


def test_training_reduces_loss():
    cfg = SyntheticOSSEConfig(n_time=24, height=32, width=32, dT=5)
    train_ds, _, test_ds = make_synthetic_datasets(cfg, None)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    model = FourDVarNetUV(dT_sst=5, n_iter=3, hidden_lstm=32, feat_dim=8, use_sst=True)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    crit = TrainingLoss()

    losses = []
    for epoch in range(6):
        model.train()
        ep_loss = 0.0
        n = 0
        for batch in loader:
            pred = model(
                batch["y_ssh"], batch["z_sst"], batch["mask_ssh"], batch["mask_sst"],
                batch["u_geo"], batch["v_geo"],
            )
            loss, _ = crit(pred, batch["truth"], model.phi)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n += 1
        losses.append(ep_loss / n)

    assert losses[-1] < losses[0] * 0.85, f"loss did not decrease: {losses}"


def test_sst_model_beats_geo_baseline():
    cfg = SyntheticOSSEConfig(n_time=28, height=32, width=32, dT=5, seed=7)
    train_ds, _, test_ds = make_synthetic_datasets(cfg, None)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4)

    model = FourDVarNetUV(dT_sst=5, n_iter=4, hidden_lstm=48, feat_dim=12, use_sst=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = TrainingLoss()

    for _ in range(12):
        model.train()
        for batch in train_loader:
            pred = model(
                batch["y_ssh"], batch["z_sst"], batch["mask_ssh"], batch["mask_sst"],
                batch["u_geo"], batch["v_geo"],
            )
            loss, _ = crit(pred, batch["truth"], model.phi)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    tau_model, tau_geo, n = 0.0, 0.0, 0
    # Inner 4DVar loop needs autograd even at inference (paper Eq. 9).
    for batch in test_loader:
        pred = model(
            batch["y_ssh"], batch["z_sst"], batch["mask_ssh"], batch["mask_sst"],
            batch["u_geo"], batch["v_geo"],
        )
        truth = batch["truth"].detach()
        pred = pred.detach()
        tau_model += explained_variance(
            torch.stack([pred[:, 1], pred[:, 2]], dim=1),
            torch.stack([truth[:, 1], truth[:, 2]], dim=1),
        )
        tau_geo += explained_variance(
            torch.stack([batch["u_geo"].squeeze(1), batch["v_geo"].squeeze(1)], dim=1),
            torch.stack([truth[:, 1], truth[:, 2]], dim=1),
        )
        n += 1
    tau_model /= n
    tau_geo /= n
    assert tau_model > tau_geo + 0.02, f"model tau={tau_model:.3f} geo={tau_geo:.3f}"
