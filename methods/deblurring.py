from __future__ import annotations
from typing import Optional

import torch

from .base import Obs

Tensor = torch.Tensor


def _gaussian_2d_kernel(sigma: float, size: int) -> Tensor:
    x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
    y = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


class GaussianDeblurringProblem:
    """
    Gaussian deblurring inverse problem with circular (FFT) boundary handling,
    the standard circular (FFT) Gaussian blur operator.

    y = A(x_gt) + noise,  A(x) = k * x  (circular convolution)

    Circular convolution is diagonalized by the 2D FFT, so:
        A^T A x = ifft2( |fft2(k)|^2 * fft2(x) )
        (A^T A / sigma2 + lam I) x = rhs
            <=>  fft2(x) = fft2(rhs) / (|fft2(k)|^2 / sigma2 + lam)
    """
    name = "blur_gauss"

    def __init__(
        self,
        sigma_blur: float = 1.0,
        kernel_size: int = 25,
        dim_image: int = 128,
        num_channels: int = 3,
        sigma: float = 0.05,
    ):
        self.sigma_blur = float(sigma_blur)
        self.kernel_size = int(kernel_size)
        self.dim_image = int(dim_image)
        self.num_channels = int(num_channels)
        self.sigma = float(sigma)

        kernel = _gaussian_2d_kernel(self.sigma_blur, self.kernel_size)
        filt = torch.zeros((1, self.num_channels, self.dim_image, self.dim_image))
        filt[..., : self.kernel_size, : self.kernel_size] = kernel
        filt = torch.roll(
            filt,
            shifts=(-(self.kernel_size - 1) // 2, -(self.kernel_size - 1) // 2),
            dims=(2, 3),
        )
        self.filter = filt
        self._fft_filter_cache = {}

    def _fft_filter(self, device) -> Tensor:
        if device not in self._fft_filter_cache:
            self._fft_filter_cache[device] = torch.fft.fft2(self.filter.to(device=device))
        return self._fft_filter_cache[device]

    def make_observation(self, x_gt: Tensor, image_idx: int = 0) -> Obs:
        noise = self.sigma * torch.randn_like(x_gt)
        y = self.forward(x_gt, Obs(y=None, aux={})) + noise
        return Obs(y=y, aux={})

    def grad_data(self, x: Tensor, obs: Obs, data_weight: float = 1.0) -> Tensor:
        residual = self.forward(x, obs) - obs.y
        return data_weight * self.adjoint(residual, obs)

    def forward(self, x: Tensor, obs: Obs) -> Tensor:
        H = self._fft_filter(x.device)
        return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * H))

    def adjoint(self, y: Tensor, obs: Obs) -> Tensor:
        H = self._fft_filter(y.device)
        return torch.real(torch.fft.ifft2(torch.fft.fft2(y) * torch.conj(H)))

    def broadcast_to_x(self, s: Tensor, x: Tensor) -> Tensor:
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        return s

    def normal(self, x: Tensor, obs: Obs) -> Tensor:
        H = self._fft_filter(x.device)
        return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * (H * torch.conj(H))))

    def solve_normal_plus_lambda(
        self,
        rhs: Tensor,
        lam: Tensor,
        obs: Obs,
        x0: Optional[Tensor] = None,
        sigma2: Optional[float] = None,
    ) -> Tensor:
        """
        Solve: (A^T A / sigma2 + lam I) x = rhs via FFT diagonalization.
        """
        while lam.ndim < rhs.ndim:
            lam = lam.unsqueeze(-1)
        H = self._fft_filter(rhs.device)
        H2 = (H * torch.conj(H)).real
        H2_scaled = H2 if sigma2 is None else H2 / sigma2
        denom = (H2_scaled + lam).clamp_min(1e-6)
        x_fft = torch.fft.fft2(rhs) / denom
        return torch.real(torch.fft.ifft2(x_fft))

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
        """Measurement-space (A A^T + lam I)^{-1} rhs via FFT.
        Circular blur => A A^T = |fft2(k)|^2, so
            fft2(x) = fft2(rhs) / (|H|^2 + lam)."""
        H = self._fft_filter(rhs.device)
        H2 = (H * torch.conj(H)).real
        denom = (H2 + float(lam)).clamp_min(1e-8)
        x_fft = torch.fft.fft2(rhs) / denom
        return torch.real(torch.fft.ifft2(x_fft))

    def ata_spectrum(self, device=None) -> Tensor:
        """eig(A^H A) = |H(w)|^2 on the fft2 grid, shape (H, W). Exact -- circular blur is
        Fourier-diagonal. Used to print the Eq.14 w_i histogram at launch."""
        H = self._fft_filter(device if device is not None else self.filter.device)
        return (H * torch.conj(H)).real[0, 0]

    def solve_normal_eq(self, rhs: Tensor, reg, obs: Obs) -> Tensor:
        """(A^H A + reg)^-1 rhs, exactly and elementwise.

        Circular blur makes A^H A = |H(w)|^2 diagonal in Fourier; a Fourier-diagonal prior
        makes reg = sigma_n^2 Sigma^-1(w) diagonal in the SAME basis, so the solve is one
        division per mode -- no CG. `reg` may be a scalar or broadcastable (.,1,H,W)."""
        H = self._fft_filter(rhs.device)
        H2 = (H * torch.conj(H)).real
        r = reg if torch.is_tensor(reg) else torch.as_tensor(float(reg), device=rhs.device,
                                                             dtype=H2.dtype)
        denom = (H2 + r.to(H2.dtype)).clamp_min(1e-12)
        return torch.real(torch.fft.ifft2(torch.fft.fft2(rhs) / denom))

    def coverage_map(self, obs: Obs) -> Tensor:
        """1-channel coverage map. Blur observes every pixel -> all ones."""
        y = obs.y
        return torch.ones(y.shape[0], 1, y.shape[2], y.shape[3],
                          device=y.device, dtype=y.dtype)

    def project(self, x: Tensor, obs: Obs) -> Tensor:
        return x
