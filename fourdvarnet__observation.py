"""Observation operator H and multimodal G (paper Eq. 8)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFeatureNet(nn.Module):
    """G(x) and H(z): 3×tanh conv + linear + batch norm (20-d features in paper)."""

    def __init__(self, in_channels: int, feat_dim: int = 20) -> None:
        super().__init__()
        h = max(feat_dim, 8)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, h, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(h, h, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(h, h, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(h, feat_dim, 3, padding=1),
        )
        self.bn = nn.BatchNorm2d(feat_dim, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.body(x))


class ObservationOperator(nn.Module):
    """
    Variational observation terms:
      - SSH: masked (y_ssh - x_ssh)
      - SST synergy: G(x_ssh) - H(z_sst)
    """

    def __init__(
        self,
        n_state: int = 3,
        dT_sst: int = 7,
        feat_dim: int = 20,
        ssh_index: int = 0,
    ) -> None:
        super().__init__()
        self.n_state = n_state
        self.dT_sst = dT_sst
        self.ssh_index = ssh_index
        self.G = MultimodalFeatureNet(1, feat_dim)
        self.H = MultimodalFeatureNet(dT_sst, feat_dim)
        self.sst_gate = nn.Conv2d(dT_sst, feat_dim, 3, padding=1)

    def ssh_residual(self, x: torch.Tensor, y_ssh: torch.Tensor, mask_ssh: torch.Tensor) -> torch.Tensor:
        x_ssh = x[:, self.ssh_index : self.ssh_index + 1]
        return (y_ssh - x_ssh) * mask_ssh

    def synergy_residual(
        self, x: torch.Tensor, z_sst: torch.Tensor, mask_sst: torch.Tensor
    ) -> torch.Tensor:
        x_ssh = x[:, self.ssh_index : self.ssh_index + 1]
        gx = self.G(x_ssh)
        hz = self.H(z_sst * mask_sst)
        res = gx - hz
        gate = torch.sigmoid(self.sst_gate(mask_sst))
        return res * gate

    def forward(
        self,
        x: torch.Tensor,
        y_ssh: torch.Tensor,
        z_sst: torch.Tensor,
        mask_ssh: torch.Tensor,
        mask_sst: torch.Tensor,
        use_sst: bool = True,
    ) -> list[torch.Tensor]:
        dy = [self.ssh_residual(x, y_ssh, mask_ssh)]
        if use_sst:
            dy.append(self.synergy_residual(x, z_sst, mask_sst))
        return dy
