from __future__ import annotations
from typing import Optional

import torch

from .base import Obs

Tensor = torch.Tensor


class DenoisingProblem:
    """
    Pure denoising inverse problem: y = x_gt + noise
    Operator form: A = I, A^T = I, A^T A = I.
    For RED-DEQ: (A^T A / sigma2 + lam I) x = rhs  ->  x = rhs / (1/sigma2 + lam)
    """
    name = "denoising"

    def __init__(self, sigma: float = 0.1):
        self.sigma = float(sigma)

    def make_observation(self, x_gt: Tensor, image_idx: int = 0) -> Obs:
        noise = self.sigma * torch.randn_like(x_gt)
        y = x_gt + noise
        return Obs(y=y, aux={})

    def grad_data(self, x: Tensor, obs: Obs, data_weight: float = 1.0) -> Tensor:
        residual = self.forward(x, obs) - obs.y
        return data_weight * self.adjoint(residual, obs)

    def forward(self, x: Tensor, obs: Obs) -> Tensor:
        return x

    def adjoint(self, y: Tensor, obs: Obs) -> Tensor:
        return y

    def broadcast_to_x(self, s: Tensor, x: Tensor) -> Tensor:
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        return s

    def normal(self, x: Tensor, obs: Obs) -> Tensor:
        return x

    def solve_normal_plus_lambda(
        self,
        rhs: Tensor,
        lam: Tensor,
        obs: Obs,
        x0: Optional[Tensor] = None,
        sigma2: Optional[float] = None,
    ) -> Tensor:
        """
        Solve: (I / sigma2 + lam I) x = rhs  ->  x = rhs / (1/sigma2 + lam)
        """
        while lam.ndim < rhs.ndim:
            lam = lam.unsqueeze(-1)
        denom = (1.0 if sigma2 is None else 1.0 / sigma2) + lam
        return rhs / denom.clamp_min(1e-6)

    def solve_prox(self, x_hat, obs, lam, x0=None):
        sigma_n = self.sigma
        lambda_eff = (sigma_n ** 2) / lam
        lambda_eff_x = self.broadcast_to_x(lambda_eff, x_hat)
        rhs = self.adjoint(obs.y, obs) + lambda_eff_x * x_hat
        return self.solve_normal_plus_lambda(rhs=rhs, lam=lambda_eff, obs=obs, x0=x0 if x0 is not None else x_hat)

    def prox_data(self, z: Tensor, obs: Obs, mu, sigma: float = None) -> Tensor:
        sigma2 = max((sigma if sigma is not None else self.sigma) ** 2, 1e-8)
        At_y = self.adjoint(obs.y, obs)
        rhs = (1.0 / sigma2) * At_y + mu * z
        if torch.is_tensor(mu):
            lam = mu.to(device=z.device, dtype=z.dtype).expand(z.shape[0])
        else:
            lam = torch.full((z.shape[0],), mu, device=z.device, dtype=z.dtype)
        return self.solve_normal_plus_lambda(rhs=rhs, lam=lam, obs=obs, sigma2=sigma2)

    def prox_data_rho(self, z: Tensor, obs: Obs, rho: Tensor) -> Tensor:
        rho = rho.to(device=z.device, dtype=z.dtype).expand(z.shape[0])
        At_y = self.adjoint(obs.y, obs)
        while rho.ndim < z.ndim:
            rho = rho.unsqueeze(-1)
        rhs = At_y + rho * z
        lam = rho.flatten()[:z.shape[0]]
        return self.solve_normal_plus_lambda(rhs=rhs, lam=lam, obs=obs, sigma2=None)

    # --- Mean-Flow measurement-space source support ---
    def solve_gram_plus_lambda(self, rhs: Tensor, lam, obs: Obs) -> Tensor:
        """Measurement-space (A A^T + lam I)^{-1} rhs.  A = I  =>  rhs / (1 + lam)."""
        return rhs / (1.0 + float(lam))

    def coverage_map(self, obs: Obs) -> Tensor:
        """1-channel per-pixel measurement-coverage map for network conditioning.
        Denoising observes every pixel -> all ones."""
        y = obs.y
        return torch.ones(y.shape[0], 1, y.shape[2], y.shape[3],
                          device=y.device, dtype=y.dtype)

    def project(self, x: Tensor, obs: Obs) -> Tensor:
        return x
