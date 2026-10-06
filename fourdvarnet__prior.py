"""Trainable prior Φ: two-scale encoder (paper §4.1, Fablet et al. 2023)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(cout, cout, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhiPrior(nn.Module):
    """Projection prior x -> Φ(x) with HR + pooled LR pathways."""

    def __init__(self, n_channels: int, hidden: int = 32, pool: int = 2) -> None:
        super().__init__()
        self.pool = pool
        self.hr = ConvBlock(n_channels, hidden)
        self.hr_out = nn.Conv2d(hidden, n_channels, 1)
        self.lr = ConvBlock(n_channels, hidden)
        self.lr_out = nn.Conv2d(hidden, n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hr = self.hr_out(self.hr(x))
        xl = F.avg_pool2d(x, self.pool)
        lr = self.lr_out(self.lr(xl))
        lr = F.interpolate(lr, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return hr + lr
