from __future__ import annotations
from typing import Optional, Tuple

import torch

from .base import Obs

Tensor = torch.Tensor


class InpaintingProblem:
    """
    Inpainting inverse problem: y = M ⊙ x_gt + n,   n ~ N(0, sigma^2 I)
    Operator form: A(x)      = M ⊙ x A^T(y)    = M ⊙ y A^T A(x)  = M ⊙ x
    For RED-DEQ we also need to solve: (A^T A + λ I) x = rhs which becomes elementwise: (M + λ) ⊙ x = rhs
    so x = rhs / (M + λ)
    """
    name = "inpainting"
    def __init__(
        self,
        mask_type: str = "random",   # "random" or "box"
        drop_prob: float = 0.6,      # for random mask: fraction missing
        block_frac: float = 0.5,     # for block mask: block size = frac*H, frac*W
        block_size: Optional[int] = None,  # for block mask: exact square block size in pixels
        block_center: bool = False,  # box inpainting: a centered square hole
        same_mask_across_batch: bool = True,
        mask_channels: str = "shared",  # "shared" or "per_channel"
        sigma: float = 0.05,
        mask_seed: Optional[int] = None,  # fixed seed → same mask per image across runs
    ):
        self.mask_type = mask_type
        self.drop_prob = float(drop_prob)
        self.block_frac = float(block_frac)
        self.block_size = None if block_size is None else int(block_size)
        self.block_center = bool(block_center)
        self.same_mask_across_batch = bool(same_mask_across_batch)
        self.mask_channels = mask_channels
        self.sigma = sigma
        self.mask_seed = mask_seed

    def _mask_shape(self, x: Tensor) -> Tuple[int, int, int, int]:
        B, C, H, W = x.shape
        if self.mask_channels == "shared":
            return (B, 1, H, W)
        elif self.mask_channels == "per_channel":
            return (B, C, H, W)
        else:
            raise ValueError(f"mask_channels must be 'shared' or 'per_channel', got {self.mask_channels}")

    def _make_generator(self, image_idx: int) -> Optional[torch.Generator]:
        if self.mask_seed is None:
            return None
        gen = torch.Generator()
        gen.manual_seed(self.mask_seed + image_idx)
        return gen

    @torch.no_grad()
    def _sample_random_mask(self, x: Tensor, gen: Optional[torch.Generator] = None) -> Tensor:
        B, C, H, W = x.shape
        shape = self._mask_shape(x)

        if self.same_mask_across_batch:
            base = (torch.rand(1, shape[1], H, W, generator=gen) > self.drop_prob).float().to(x.device)
            mask = base.repeat(B, 1, 1, 1)
        else:
            mask = (torch.rand(*shape, generator=gen) > self.drop_prob).float().to(x.device)
        return mask

    @torch.no_grad()
    def _sample_block_mask(self, x: Tensor, gen: Optional[torch.Generator] = None) -> Tensor:
        B, C, H, W = x.shape
        shape = self._mask_shape(x)

        if self.block_size is None:
            bh = max(1, int(round(self.block_frac * H)))
            bw = max(1, int(round(self.block_frac * W)))
        else:
            bh = bw = max(1, self.block_size)
        bh = min(bh, H)
        bw = min(bw, W)

        mask = torch.ones(*shape, device=x.device)

        if self.block_center:
            top = (H - bh) // 2
            left = (W - bw) // 2
            mask[:, :, top:top+bh, left:left+bw] = 0.0
        elif self.same_mask_across_batch:
            top = torch.randint(0, H - bh + 1, (1,), generator=gen).item()
            left = torch.randint(0, W - bw + 1, (1,), generator=gen).item()
            mask[:, :, top:top+bh, left:left+bw] = 0.0
        else:
            tops = torch.randint(0, H - bh + 1, (B,), generator=gen)
            lefts = torch.randint(0, W - bw + 1, (B,), generator=gen)
            for i in range(B):
                mask[i, :, tops[i]:tops[i]+bh, lefts[i]:lefts[i]+bw] = 0.0

        return mask

    @torch.no_grad()
    def sample_mask(self, x: Tensor, gen: Optional[torch.Generator] = None) -> Tensor:
        if self.mask_type == "random":
            return self._sample_random_mask(x, gen)
        if self.mask_type in ("block", "box"):
            return self._sample_block_mask(x, gen)
        raise ValueError(f"Unknown mask_type={self.mask_type!r}; valid: 'random', 'box'")

    def make_observation(self, x_gt: Tensor, image_idx: int = 0) -> Obs:
        """
        x_gt: (B,C,H,W) in [-1,1]
        image_idx: starting index of this batch in the dataset; used for reproducible mask seeding.
        """
        gen = self._make_generator(image_idx)
        mask = self.sample_mask(x_gt, gen)
        noise = self.sigma * torch.randn_like(x_gt)
        y = mask * x_gt + noise                       # y = M x + n
        return Obs(y=y, aux={"mask": mask})

    def grad_data(self, x: Tensor, obs: Obs, data_weight: float = 1.0) -> Tensor:
        """
        Gradient of:  1 / (2 sigma_n^2) ||A x - y||^2
        For inpainting:  A x = M * x   ----->     A^T(Ax - y) = M * (M*x - y) = M*x - y
        """
        residual = self.forward(x, obs) - obs.y
        return data_weight * self.adjoint(residual, obs)

    def forward(self, x: Tensor, obs: Obs) -> Tensor:
        """
        Apply A(x) = M ⊙ x
        """
        mask = obs.aux["mask"]
        return mask * x

    def adjoint(self, y: Tensor, obs: Obs) -> Tensor:
        """
        Apply A^T(y) = M ⊙ y
        """
        mask = obs.aux["mask"]
        return mask * y

    def broadcast_to_x(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        return s

    def normal(self, x: Tensor, obs: Obs) -> Tensor:
        """
        Apply A^T A(x) = M ⊙ x
        """
        mask = obs.aux["mask"]
        return mask * x

    def solve_normal_plus_lambda(
            self,
            rhs: Tensor,
            lam: Tensor,
            obs: Obs,
            x0: Optional[Tensor] = None,
            sigma2: Optional[float] = None,
    ) -> Tensor:
        """
        Solve:
            (A^T A / sigma2 + λ I) x = rhs

        For inpainting:
            (M / sigma2 + λ) ⊙ x = rhs
            x = rhs / (M / sigma2 + λ)

        lam:
            either shape (B,) or broadcastable to rhs
        x0:
            unused here, included for API compatibility
        sigma2:
            noise variance sigma^2; when None falls back to sigma2=1 (legacy behavior)
        """
        mask = obs.aux["mask"]

        # reshape lam to broadcast with rhs
        while lam.ndim < rhs.ndim:
            lam = lam.unsqueeze(-1)

        mask_scaled = mask if sigma2 is None else mask / sigma2
        denom = mask_scaled + lam
        return rhs / denom.clamp_min(1e-6)

    def solve_prox(self, x_hat, obs, lam, x0=None):
        """
        Solves:
            min_x ||A x - y||^2 / (2 sigma_n^2)
                 + ||x - x_hat||^2 / (2 lam)

        where lam = nu_t^2.
        """
        sigma_n = self.sigma
        lambda_eff = (sigma_n ** 2) / lam
        lambda_eff_x = self.broadcast_to_x(lambda_eff, x_hat)

        rhs = self.adjoint(obs.y, obs) + lambda_eff_x * x_hat
        return self.solve_normal_plus_lambda( rhs=rhs, lam=lambda_eff,
                                              obs=obs, x0=x0 if x0 is not None else x_hat,)

    def prox_data(self, z: Tensor, obs: Obs, mu: float | Tensor, sigma: float = None) -> Tensor:
        """
        HQS x-update: argmin_x (1/(2*sigma^2))||Ax - y||^2 + mu*||x - z||^2
        Closed-form for inpainting:
          x[mask=1] = ((1/sigma^2)*y + mu*z) / (1/sigma^2 + mu)
          x[mask=0] = z
        """
        sigma2 = max((sigma if sigma is not None else self.sigma) ** 2, 1e-8)
        At_y = self.adjoint(obs.y, obs)
        rhs = (1.0 / sigma2) * At_y + mu * z
        if torch.is_tensor(mu):
            lam = mu.to(device=z.device, dtype=z.dtype).expand(z.shape[0])
        else:
            lam = torch.full((z.shape[0],), mu, device=z.device, dtype=z.dtype)
        return self.solve_normal_plus_lambda(rhs=rhs, lam=lam, obs=obs, sigma2=sigma2)

    def prox_data_rho(self, z: Tensor, obs: Obs, rho: Tensor) -> Tensor:
        """
        HQS x-update directly parameterized by rho.

        For inpainting:
            (A^T A + rho I) x = A^T y + rho z
        """
        rho = rho.to(device=z.device, dtype=z.dtype).expand(z.shape[0])
        At_y = self.adjoint(obs.y, obs)
        while rho.ndim < z.ndim:
            rho = rho.unsqueeze(-1)
        rhs = At_y + rho * z
        lam = rho.flatten()[:z.shape[0]]
        return self.solve_normal_plus_lambda(rhs=rhs, lam=lam, obs=obs, sigma2=None)

    # --- Mean-Flow measurement-space source support ---
    def solve_gram_plus_lambda(self, rhs: Tensor, lam, obs: Obs) -> Tensor:
        """Measurement-space (A A^T + lam I)^{-1} rhs.  A A^T = M (0/1 mask)  =>
        elementwise rhs / (M + lam)."""
        mask = obs.aux["mask"]
        return rhs / (mask + float(lam)).clamp_min(1e-8)

    def coverage_map(self, obs: Obs) -> Tensor:
        """1-channel coverage map = the inpainting mask (reduced to one channel)."""
        mask = obs.aux["mask"]
        if mask.shape[1] > 1:
            mask = mask.mean(dim=1, keepdim=True)
        return mask

    def project(self, x: Tensor, obs: Obs) -> Tensor:
        """
        Hard projection: enforce observed pixels exactly.
        """
        mask = obs.aux["mask"]
        y = obs.y
        return (1.0 - mask) * x + mask * y
