"""Unrolled gradient descent on variational cost (paper Eq. 9).

Default ``Solver4DVarNet`` keeps a faithful unrolled autodiff graph across
iterations (``create_graph=True``, no per-iter ``detach``).
``Solver4DVarNetTruncated`` is an explicit ablation that detaches each iterate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fourdvarnet.convlstm import GradUpdateLSTM
from fourdvarnet.geometry import stencil_valid_mask
from fourdvarnet.observation import ObservationOperator
from fourdvarnet.physics import sqg_velocity, sst_advection_residual
from fourdvarnet.prior import PhiPrior


def _masked_mse(residual: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Average squared residual only over valid observation locations.

    Computes ``sum(r² * m) / sum(m)``. When ``residual`` is already zeroed by a
    binary mask (``r = err * m``), this equals the mean of ``err²`` on valid cells.
    Density-invariant: padding zeros outside the mask do not dilute the mean.
    """
    if mask is None:
        m = torch.isfinite(residual).to(dtype=residual.dtype)
    else:
        m = mask.to(device=residual.device, dtype=residual.dtype)
        # Invalid / NaN residual cells do not contribute.
        m = m * torch.isfinite(residual).to(dtype=residual.dtype)
    denom = m.sum()
    # Compare without .item() so autograd tracing stays on tensors when needed.
    if denom <= 0:
        return residual.new_tensor(0.0)
    r = torch.where(torch.isfinite(residual), residual, residual.new_zeros(()))
    return (r.pow(2) * m).sum() / denom


class VariationalCost(nn.Module):
    """Extended 4DVar cost (Fablet Eq. 8 + physics residuals).

    U = λ_ssh ||SSH - SSH_obs||²_Ω
      + λ_mm  ||G(x) - H(SST)||²
      + λ_sqg ||u - A_SQG(SST, SSH)||²
      + λ_adv ||∂t T + u·∇T - κ∇²T||²
      + λ_Φ   ||x - Φ(x)||²
    """

    def __init__(
        self,
        phi: PhiPrior,
        obs: ObservationOperator,
        lam_obs: float = 1.0,
        lam_sst: float = 1.0,
        lam_prior: float = 0.1,
        use_sst: bool = True,
        use_sqg: bool = False,
        use_adv: bool = False,
        lam_sqg: float = 0.1,
        lam_adv: float = 0.1,
        Ld: float = 30e3,
        kappa: float = 50.0,
        dt: float = 86400.0,
        dx: float = 0.05 * 111e3,
        dy: float | None = None,
        f0: float = 7.0e-5,
        g: float = 9.81,
    ) -> None:
        super().__init__()
        self.phi = phi
        self.obs = obs
        self.lam_obs = lam_obs
        self.lam_sst = lam_sst
        self.lam_prior = lam_prior
        self.use_sst = use_sst
        self.use_sqg = use_sqg
        self.use_adv = use_adv
        self.lam_sqg = lam_sqg
        self.lam_adv = lam_adv
        self.Ld = Ld
        self.kappa = kappa
        self.dt = dt
        self.dx = dx
        self.dy = dx if dy is None else dy
        self.f0 = f0
        self.g = g

    def forward(
        self,
        x: torch.Tensor,
        y_ssh: torch.Tensor,
        z_sst: torch.Tensor,
        mask_ssh: torch.Tensor,
        mask_sst: torch.Tensor,
        dx: float | torch.Tensor | None = None,
        dy: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        dx_v = self.dx if dx is None else dx
        dy_v = self.dy if dy is None else dy
        if isinstance(dx_v, torch.Tensor) and dx_v.ndim > 0:
            dx_v = float(dx_v.float().mean().item())
        if isinstance(dy_v, torch.Tensor) and dy_v.ndim > 0:
            dy_v = float(dy_v.float().mean().item())

        dy_obs = self.obs(x, y_ssh, z_sst, mask_ssh, mask_sst, use_sst=self.use_sst)
        # SSH residual is already (y-x)*mask; denom uses mask so zeros do not dilute.
        loss = self.lam_obs * _masked_mse(dy_obs[0], mask_ssh)
        if self.use_sst and len(dy_obs) > 1:
            # Synergy residual is soft-gated; weight by SST validity over the window.
            m_sst = mask_sst
            if m_sst.ndim == 4 and m_sst.shape[1] > 1:
                m_sst = m_sst[:, -1:]  # spatial weight at analysis time
            loss = loss + self.lam_sst * _masked_mse(dy_obs[1], m_sst)
        dx_prior = x - self.phi(x)
        loss = loss + self.lam_prior * torch.mean(dx_prior**2)

        ssh = x[:, 0:1]
        u = x[:, 1:2]
        v = x[:, 2:3]
        sst_last = z_sst[:, -1:] if z_sst.ndim == 4 else z_sst.unsqueeze(1)

        if self.use_sqg and self.lam_sqg != 0.0:
            u_sqg, v_sqg = sqg_velocity(
                sst_last, ssh, Ld=self.Ld, f=self.f0, g=self.g, dx=dx_v, dy=dy_v
            )
            loss = loss + self.lam_sqg * (torch.mean((u - u_sqg) ** 2) + torch.mean((v - v_sqg) ** 2))

        if self.use_adv and self.lam_adv != 0.0 and z_sst.ndim == 4 and z_sst.shape[1] >= 2:
            adv = sst_advection_residual(
                z_sst,
                u,
                v,
                kappa=self.kappa,
                dt=self.dt,
                dx=dx_v,
                dy=dy_v,
                mask_sst=mask_sst,
            )
            # L_adv: mean over stencil-valid cells (t, t−1, spatial FD), not global mean.
            if mask_sst is not None:
                if mask_sst.ndim == 4:
                    m_t = mask_sst[:, -1] > 0.5
                    m_tm1 = mask_sst[:, -2] > 0.5
                else:
                    m_t = mask_sst > 0.5
                    m_tm1 = m_t
                m_adv = (m_t & m_tm1 & stencil_valid_mask(m_t)).to(dtype=adv.dtype)
                adv_t = adv if adv.ndim == 4 else adv.unsqueeze(1)
                m_use = m_adv if m_adv.ndim == adv_t.ndim else m_adv.unsqueeze(1)
                loss = loss + self.lam_adv * _masked_mse(adv_t, m_use)
            else:
                loss = loss + self.lam_adv * torch.mean(adv**2)
        return loss


class Solver4DVarNet(nn.Module):
    """K iterations of LSTM gradient updates with a full unrolled graph.

    Each iteration computes ``grad = ∂U/∂x`` with ``create_graph=True`` and
    applies the ConvLSTM step **without** detaching ``x``, so supervised
    losses back-propagate through all K inner steps.
    """

    def __init__(
        self,
        n_channels: int,
        n_iter: int = 5,
        hidden_lstm: int = 150,
        feat_dim: int = 20,
        dT_sst: int = 7,
        use_sst: bool = True,
        lam_obs: float = 1.0,
        lam_sst: float = 1.0,
        lam_prior: float = 0.1,
        use_sqg: bool = False,
        use_adv: bool = False,
        lam_sqg: float = 0.1,
        lam_adv: float = 0.1,
        Ld: float = 30e3,
        kappa: float = 50.0,
        dt: float = 86400.0,
        dx: float = 0.05 * 111e3,
        dy: float | None = None,
        f0: float = 7.0e-5,
        g: float = 9.81,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.use_sst = use_sst
        self.use_sqg = use_sqg
        self.use_adv = use_adv
        self.phi = PhiPrior(n_channels)
        self.obs = ObservationOperator(n_state=n_channels, dT_sst=dT_sst, feat_dim=feat_dim)
        self.cost = VariationalCost(
            self.phi,
            self.obs,
            lam_obs=lam_obs,
            lam_sst=lam_sst,
            lam_prior=lam_prior,
            use_sst=use_sst,
            use_sqg=use_sqg,
            use_adv=use_adv,
            lam_sqg=lam_sqg,
            lam_adv=lam_adv,
            Ld=Ld,
            kappa=kappa,
            dt=dt,
            dx=dx,
            dy=dy,
            f0=f0,
            g=g,
        )
        self.grad_step = GradUpdateLSTM(n_channels, hidden_lstm)

    def forward(
        self,
        x0: torch.Tensor,
        y_ssh: torch.Tensor,
        z_sst: torch.Tensor,
        mask_ssh: torch.Tensor,
        mask_sst: torch.Tensor,
        dx: float | torch.Tensor | None = None,
        dy: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x0
        h = c = None
        norm: torch.Tensor | None = None
        for _ in range(self.n_iter):
            x = x.requires_grad_(True)
            loss = self.cost(x, y_ssh, z_sst, mask_ssh, mask_sst, dx=dx, dy=dy)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            if norm is None:
                # Detached tensor scale (no .item()/float) — constant across iters.
                norm = torch.sqrt(torch.mean(grad**2) + 1e-12).detach()
            step, h, c = self.grad_step(grad, h, c, norm)
            step = step / self.n_iter
            x = x - step
        return x


class Solver4DVarNetTruncated(Solver4DVarNet):
    """Ablation: detach state each iteration (truncated BPTT through the solver).

    Not the default — use only when comparing against the full unrolled graph.
    """

    def forward(
        self,
        x0: torch.Tensor,
        y_ssh: torch.Tensor,
        z_sst: torch.Tensor,
        mask_ssh: torch.Tensor,
        mask_sst: torch.Tensor,
        dx: float | torch.Tensor | None = None,
        dy: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x0
        h = c = None
        norm: torch.Tensor | None = None
        for _ in range(self.n_iter):
            x = x.detach().requires_grad_(True)
            loss = self.cost(x, y_ssh, z_sst, mask_ssh, mask_sst, dx=dx, dy=dy)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            if norm is None:
                norm = torch.sqrt(torch.mean(grad**2) + 1e-12).detach()
            step, h, c = self.grad_step(grad, h, c, norm)
            step = step / self.n_iter
            x = x - step
        return x
