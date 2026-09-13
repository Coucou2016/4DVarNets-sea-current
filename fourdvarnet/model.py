"""End-to-end 4DVarNet for SSH + SSC from SST-SSH observations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fourdvarnet.solver import Solver4DVarNet

DEG_TO_M = 111e3


class FourDVarNetUV(nn.Module):
    """
    Inputs per batch:
      y_ssh: (B, 1, H, W) — DUACS / OI SSH + gappy obs stacked as channels optional
      z_sst: (B, dT, H, W) — SST time window
      mask_ssh, mask_sst: observation masks
    Output:
      state (B, 3, H, W): [SSH, u, v]

    Compact 4DVarNet-inspired unrolled ConvLSTM solver (not a byte-faithful
    Fablet/IMT reproduction). Optional SQG / advection residuals live in
    VariationalCost only. ``use_uncert`` is strain-aware spatial reweighting
    in the supervised loss (not a learned uncertainty head).
    """

    def __init__(
        self,
        dT_sst: int = 7,
        n_iter: int = 5,
        hidden_lstm: int = 150,
        feat_dim: int = 20,
        use_sst: bool = True,
        lam_obs: float = 1.0,
        lam_sst: float = 1.0,
        lam_prior: float = 0.1,
        use_sqg: bool = False,
        use_adv: bool = False,
        use_uncert: bool = False,
        lam_sqg: float = 0.1,
        lam_adv: float = 0.1,
        Ld: float = 30e3,
        kappa: float = 50.0,
        dt: float = 86400.0,
        dx: float = 0.05 * DEG_TO_M,
        dy: float | None = None,
        f0: float = 7.0e-5,
        g: float = 9.81,
    ) -> None:
        super().__init__()
        self.dT_sst = dT_sst
        self.use_sst = use_sst
        self.use_sqg = use_sqg
        self.use_adv = use_adv
        self.use_uncert = use_uncert
        self.n_state = 3
        self.dx = dx
        self.dy = dx if dy is None else dy
        self.f0 = f0
        self.solver = Solver4DVarNet(
            n_channels=self.n_state,
            n_iter=n_iter,
            hidden_lstm=hidden_lstm,
            feat_dim=feat_dim,
            dT_sst=dT_sst,
            use_sst=use_sst,
            lam_obs=lam_obs,
            lam_sst=lam_sst,
            lam_prior=lam_prior,
            use_sqg=use_sqg,
            use_adv=use_adv,
            lam_sqg=lam_sqg,
            lam_adv=lam_adv,
            Ld=Ld,
            kappa=kappa,
            dt=dt,
            dx=dx,
            dy=self.dy,
            f0=f0,
            g=g,
        )

    @property
    def phi(self) -> nn.Module:
        return self.solver.phi

    def build_initial_state(self, y_ssh: torch.Tensor, u_geo: torch.Tensor, v_geo: torch.Tensor) -> torch.Tensor:
        """Initialize x^(0) from observations + geostrophic currents."""
        return torch.cat([y_ssh, u_geo, v_geo], dim=1)

    def forward(
        self,
        y_ssh: torch.Tensor,
        z_sst: torch.Tensor,
        mask_ssh: torch.Tensor,
        mask_sst: torch.Tensor,
        u_geo: torch.Tensor,
        v_geo: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.build_initial_state(y_ssh, u_geo, v_geo)
        return self.solver(x0, y_ssh, z_sst, mask_ssh, mask_sst)


def physics_scales_from_config(cfg: dict[str, Any]) -> dict[str, float]:
    phys = cfg.get("physics") or {}
    model = cfg.get("model") or {}
    dx_m = float(phys.get("dx_deg", 0.05)) * DEG_TO_M
    return {
        "dx": dx_m,
        "dy": dx_m,
        "f0": float(phys.get("f0", 7.0e-5)),
        "Ld": float(model.get("Ld_km", 30.0)) * 1e3,
        "kappa": float(model.get("kappa", 50.0)),
        "dt": float(model.get("dt_seconds", 86400.0)),
    }


def build_fourdvarnet(cfg: dict[str, Any], **overrides: Any) -> FourDVarNetUV:
    """Construct ``FourDVarNetUV`` from ``config/default.yaml`` plus flag overrides."""
    m = cfg.get("model") or {}
    scales = physics_scales_from_config(cfg)
    kwargs: dict[str, Any] = dict(
        dT_sst=int(m.get("dT_sst", 7)),
        n_iter=int(m.get("n_iter", 5)),
        hidden_lstm=int(m.get("hidden_lstm", 64)),
        feat_dim=int(m.get("feat_dim", 12)),
        use_sst=bool(m.get("use_sst", True)),
        lam_obs=float(m.get("lam_obs", 1.0)),
        lam_sst=float(m.get("lam_sst", 1.0)),
        lam_prior=float(m.get("lam_prior", 0.1)),
        use_sqg=bool(m.get("use_sqg", False)),
        use_adv=bool(m.get("use_adv", False)),
        use_uncert=bool(m.get("use_uncert", False)),
        lam_sqg=float(m.get("lam_sqg", 0.1)),
        lam_adv=float(m.get("lam_adv", 0.1)),
        **scales,
    )
    kwargs.update(overrides)
    return FourDVarNetUV(**kwargs)
