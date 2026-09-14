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
        # Always divide by an on-device tensor scale — never a Python-float barrier
        # via casting the scale tensor — so the faithful unrolled create_graph
        # path stays fully differentiable.
        if isinstance(norm, torch.Tensor):
            scale = norm.to(device=grad.device, dtype=grad.dtype)
        else:
            # Python scalar → tensor (never cast a Tensor scale through Python float).
            scale = grad.new_tensor(1.0 if not norm else norm)
        scale = scale.clamp_min(1e-12)
        grad = grad / scale
        state = (hidden, cell) if hidden is not None else None
        h, c = self.lstm(grad, state)
        step = self.proj(h)
        return step, h, c
