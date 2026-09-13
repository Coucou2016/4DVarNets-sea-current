"""Supervised training losses (paper Eqs 10–14) plus optional strain reweighting.

When ``use_uncert`` is on, the UV term mixes MSE with a σ-normalized
**strain-aware spatial reweighting** (not a learned uncertainty head):
σ = σ0 (1 + α strain). σ≈σ0 recovers MSE scale (avoids raw err²/σ²
swamping SSH when σ0≪1). M4 in this repo means reweighting, not σ-estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fourdvarnet.physics import FixedDivergence2d, FixedGradient2d, strain_uncertainty


@dataclass
class LossWeights:
    ssh: float = 50.0
    grad_ssh: float = 100.0
    uv: float = 50.0
    div: float = 100.0
    prior: float = 10.0


class TrainingLoss(nn.Module):
    def __init__(
        self,
        weights: LossWeights | None = None,
        use_uncert: bool = False,
        sigma0: float = 0.05,
        alpha: float = 1.0e4,
        dx: float = 1.0,
        dy: float = 1.0,
        uncert_from_truth: bool = True,
        uncert_max_mult: float = 10.0,
        uncert_mse_mix: float = 0.25,
    ) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.use_uncert = use_uncert
        self.sigma0 = sigma0
        self.alpha = alpha
        self.dx = dx
        self.dy = dy
        # σ from truth (or detached pred) blocks the strain-inflation collapse that
        # drove M4 τ_uv ≈ −1 when σ was computed from live predictions.
        self.uncert_from_truth = uncert_from_truth
        self.uncert_max_mult = float(uncert_max_mult)
        # Keep a MSE floor so NLL cannot fully abandon UV fit early in training.
        self.uncert_mse_mix = float(uncert_mse_mix)
        self.grad_op = FixedGradient2d()
        self.div_op = FixedDivergence2d()

    def forward(
        self,
        pred: torch.Tensor,
        truth: torch.Tensor,
        phi: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ssh_p, ssh_t = pred[:, 0:1], truth[:, 0:1]
        u_p, v_p = pred[:, 1:2], pred[:, 2:3]
        u_t, v_t = truth[:, 1:2], truth[:, 2:3]

        l_ssh = torch.mean((ssh_p - ssh_t) ** 2)
        gx_p, gy_p = self.grad_op(ssh_p)
        gx_t, gy_t = self.grad_op(ssh_t)
        l_gssh = torch.mean((gx_p - gx_t) ** 2 + (gy_p - gy_t) ** 2)
        l_uv = torch.mean((u_p - u_t) ** 2 + (v_p - v_t) ** 2)
        # Same metric spacing as train/eval physics operators.
        div_p = self.div_op(u_p, v_p, self.dx, self.dy)
        div_t = self.div_op(u_t, v_t, self.dx, self.dy)
        l_div = torch.mean((div_p - div_t) ** 2)
        l_phi = torch.mean((truth - phi(truth)) ** 2 + (pred - phi(pred)) ** 2)

        l_uv_term = l_uv
        l_uv_nll = pred.new_tensor(0.0)
        if self.use_uncert:
            if self.uncert_from_truth:
                u_sig, v_sig = u_t, v_t
            else:
                u_sig, v_sig = u_p.detach(), v_p.detach()
            sig = strain_uncertainty(u_sig, v_sig, self.sigma0, self.alpha, self.dx, self.dy)
            if self.uncert_max_mult > 1.0:
                sig = torch.clamp(sig, max=float(self.sigma0) * self.uncert_max_mult)
            uv_err2 = (u_p - u_t) ** 2 + (v_p - v_t) ** 2
            # Heteroscedastic UV term in MSE units: when σ=σ0 this equals MSE;
            # high-σ (high-strain) cells are down-weighted. Avoid raw err²/σ² which
            # is ~1/σ0² (~400×) larger than MSE and swamped SSH on GPU96 M4.
            s0 = float(self.sigma0)
            l_uv_nll = torch.mean(uv_err2 * (s0 / (sig + 1e-12)) ** 2) + (s0**2) * torch.mean(
                torch.log((sig / (s0 + 1e-12)) ** 2 + 1e-12)
            )
            mix = min(max(self.uncert_mse_mix, 0.0), 1.0)
            l_uv_term = mix * l_uv + (1.0 - mix) * l_uv_nll

        total = (
            self.w.ssh * l_ssh
            + self.w.grad_ssh * l_gssh
            + self.w.uv * l_uv_term
            + self.w.div * l_div
            + self.w.prior * l_phi
        )
        logs = {
            "loss": total.item(),
            "l_ssh": l_ssh.item(),
            "l_uv": l_uv.item(),
            "l_div": l_div.item(),
        }
        if self.use_uncert:
            logs["l_uv_nll"] = l_uv_nll.item()
        return total, logs
