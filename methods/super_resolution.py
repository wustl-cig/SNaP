from __future__ import annotations
from typing import Optional

import torch

from .base import Obs

Tensor = torch.Tensor


class SuperResolutionProblem:
    """
    Single-image super-resolution via box decimation (no anti-alias prefilter),
    the standard box-decimation (no prefilter) operator:

        A(x)   = x[..., ::sf, ::sf]               (keep every sf-th pixel)
        A^T(y) = zero-fill upsample of y by sf

    Because decimation just selects a fixed grid of pixels, A^T A is a
    diagonal (binary mask) operator -- structurally identical to inpainting
    with a fixed, periodic mask -- so the same closed-form solve applies:

        (A^T A / sigma2 + lam I) x = rhs  ->  x = rhs / (mask/sigma2 + lam)
    """
    name = "sr"

    def __init__(
        self,
        sf: int = 4,
        dim_image: int = 128,
        sigma: float = 0.05,
    ):
        self.sf = int(sf)
        self.dim_image = int(dim_image)
        self.sigma = float(sigma)
        assert self.dim_image % self.sf == 0, "dim_image must be divisible by sf"

        mask = torch.zeros(1, 1, self.dim_image, self.dim_image)
        mask[..., :: self.sf, :: self.sf] = 1.0
        self.mask = mask

    def _mask(self, x: Tensor) -> Tensor:
        return self.mask.to(device=x.device, dtype=x.dtype)

    def downsample(self, x: Tensor) -> Tensor:
        return x[..., :: self.sf, :: self.sf]

    def upsample(self, y: Tensor, like: Tensor) -> Tensor:
        z = torch.zeros_like(like)
        z[..., :: self.sf, :: self.sf] = y
        return z

    def make_observation(self, x_gt: Tensor, image_idx: int = 0) -> Obs:
        y_clean = self.downsample(x_gt)
        noise = self.sigma * torch.randn_like(y_clean)
        y = y_clean + noise
        return Obs(y=y, aux={"shape": x_gt.shape})

    def grad_data(self, x: Tensor, obs: Obs, data_weight: float = 1.0) -> Tensor:
        residual = self.forward(x, obs) - obs.y
        return data_weight * self.adjoint(residual, obs)

    def forward(self, x: Tensor, obs: Obs) -> Tensor:
        return self.downsample(x)

    def adjoint(self, y: Tensor, obs: Obs) -> Tensor:
        # Batch/channels come from `y`, only H,W from the recorded shape: callers may
        # reuse a single obs for a LARGER batch (the launch source check and the M-draw
        # diversity metric both expand obs.y), so obs.aux["shape"][0] is not the batch.
        H, W = obs.aux["shape"][-2], obs.aux["shape"][-1]
        like = torch.zeros(y.shape[0], y.shape[1], H, W, device=y.device, dtype=y.dtype)
        return self.upsample(y, like)

    def broadcast_to_x(self, s: Tensor, x: Tensor) -> Tensor:
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        return s

    def normal(self, x: Tensor, obs: Obs) -> Tensor:
        return self._mask(x) * x

    def solve_normal_plus_lambda(
        self,
        rhs: Tensor,
        lam: Tensor,
        obs: Obs,
        x0: Optional[Tensor] = None,
        sigma2: Optional[float] = None,
    ) -> Tensor:
        """
        Solve: (A^T A / sigma2 + lam I) x = rhs  ->  (mask/sigma2 + lam) ⊙ x = rhs
        """
        while lam.ndim < rhs.ndim:
            lam = lam.unsqueeze(-1)
        mask = self._mask(rhs)
        mask_scaled = mask if sigma2 is None else mask / sigma2
        denom = mask_scaled + lam
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
        """Measurement-space (A A^T + lam I)^{-1} rhs, in low-res space.
        Decimation gives A A^T = I on the LR grid  =>  rhs / (1 + lam)."""
        return rhs / (1.0 + float(lam))

    def coverage_map(self, obs: Obs) -> Tensor:
        """1-channel HR coverage map = the periodic sampling mask (observed HR pixels)."""
        B = obs.y.shape[0]          # not aux["shape"][0]: obs.y may be an expanded batch
        m = self.mask.to(device=obs.y.device, dtype=obs.y.dtype)
        return m.expand(B, 1, m.shape[2], m.shape[3])

    def project(self, x: Tensor, obs: Obs) -> Tensor:
        """
        Hard projection: enforce observed low-res samples exactly.
        """
        mask = self._mask(x)
        y_full = self.adjoint(obs.y, obs)
        return (1.0 - mask) * x + mask * y_full
