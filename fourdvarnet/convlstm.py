"""2D ConvLSTM gradient update (paper Eq. 9, official solver.py)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTM2d(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        pad = (kernel_size - 1) // 2
        self.gates = nn.Conv2d(
            input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad, bias=True
        )

    def forward(
        self, inp: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = inp.shape[0]
        if state is None:
            h = torch.zeros(b, self.hidden_size, *inp.shape[2:], device=inp.device, dtype=inp.dtype)
            c = torch.zeros_like(h)
        else:
            h, c = state
        stacked = torch.cat([inp, h], dim=1)
        i, f, o, g = self.gates(stacked).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class GradUpdateLSTM(nn.Module):
    """Maps cost gradient -> state update (Eq. 9)."""

    def __init__(self, n_state_channels: int, hidden_dim: int = 150) -> None:
        super().__init__()
        self.lstm = ConvLSTM2d(n_state_channels, hidden_dim)
        self.proj = nn.Conv2d(hidden_dim, n_state_channels, kernel_size=1, bias=False)

    def forward(
        self,
        grad: torch.Tensor,
        hidden: torch.Tensor | None,
        cell: torch.Tensor | None,
        norm: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = float(norm) if isinstance(norm, torch.Tensor) else norm
        if norm > 0:
            grad = grad / norm
        state = (hidden, cell) if hidden is not None else None
        h, c = self.lstm(grad, state)
        step = self.proj(h)
        return step, h, c
