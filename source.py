"""Measurement-dependent Gaussian-posterior source.

Replaces the white-noise base measure x1 ~ N(0, I) of a standard conditional flow with
the exact posterior under a Gaussian working prior x0 ~ N(0, tau^2 I):

    x1 ~ N(m_y, tau^2 W),
    m_y = A^T (A A^T + lambda I)^{-1} y,   W = I - A^T (A A^T + lambda I)^{-1} A,
    lambda = sigma_n^2 / tau^2.

Sampling avoids W^{1/2} via a square-root-free randomize-then-optimize (RTO) sampler:

    x_p ~ N(0, tau^2 I),   n' ~ N(0, sigma_n^2 I),
    x1  = x_p + A^T (A A^T + lambda I)^{-1} (y - A x_p - n').

Operator-agnostic: it needs only the InverseProblem's forward (A), adjoint (A^T), and the
regularized pseudo-inverse A^T (A A^T + lambda I)^{-1}. By default that is assembled as
`adjoint(problem.solve_gram_plus_lambda(rhs, lam, obs))`, which every problem in methods/
implements in closed form (inpainting: rhs/(M+lam); denoising & SR: rhs/(1+lam); deblur:
FFT). A problem may instead expose `problem.apply_pinv_reg(rhs, lam, obs)` to compute the
same quantity by a better-conditioned route -- multi-coil MRI does, see
`MRIProblem.apply_pinv_reg`.
"""

from __future__ import annotations

import torch
from torch import Tensor


class StandardGaussianSource:
    """Ablation source: measurement-independent white noise x1 ~ N(0, tau^2 I).

    The measurement never enters the *source* here -- it is injected only through the
    network conditioning [anchor, coverage] in the pipeline. This turns the method into a
    standard conditional flow (Gaussian -> image, conditioned on y as extra channels), and
    isolates the contribution of the measurement-dependent posterior source.
    """

    def __init__(self, tau: float = 1.0):
        self.tau = float(tau)

    @torch.no_grad()
    def m_y(self, problem, obs) -> Tensor:
        # No measurement-dependent mean; the source is zero-mean.
        return torch.zeros_like(problem.adjoint(obs.y, obs))

    @torch.no_grad()
    def sample(self, problem, obs, generator: torch.Generator | None = None,
               return_anchor: bool = False) -> Tensor:
        # Draw in IMAGE space (A^T y gives the right image-space shape for all operators,
        # including resolution-changing ones like SR).
        # `return_anchor` is accepted for signature parity with GaussianPosteriorSource but
        # IGNORED: the ablation source carries no measurement information. The pipeline
        # supplies the m_y conditioning anchor via its dedicated posterior helper, so it
        # must NOT be routed through this source (whose m_y is all zeros).
        img_like = problem.adjoint(obs.y, obs)
        return self.tau * torch.randn(img_like.shape, device=img_like.device,
                                      dtype=img_like.dtype, generator=generator)


class GaussianPosteriorSource:
    """x1 ~ N(m_y, tau^2 W) for the Gaussian working prior x0 ~ N(0, tau^2 I)."""

    def __init__(self, tau: float = 1.0):
        self.tau = float(tau)

    @staticmethod
    def _sigma_of(problem, obs):
        """Measurement noise for THIS observation: the per-sample (B,) tensor published by
        `input_snr_db` mode when present, else the problem's scalar sigma."""
        if isinstance(getattr(obs, "aux", None), dict) and obs.aux.get("sigma") is not None:
            return obs.aux["sigma"]
        return float(problem.sigma)

    def _lam(self, sigma_n):
        """lambda = sigma^2/tau^2, floored for Gram stability. Scalar in, scalar out;
        per-sample tensor in, per-sample tensor out."""
        if torch.is_tensor(sigma_n):
            return (sigma_n ** 2 / self.tau ** 2).clamp_min(1e-8)
        return max(sigma_n ** 2 / self.tau ** 2, 1e-8)

    def _gram_solve(self, problem, obs, rhs: Tensor, lam) -> Tensor:
        """(A A^T + lam I)^{-1} rhs in measurement space."""
        if hasattr(problem, "solve_gram_plus_lambda"):
            return problem.solve_gram_plus_lambda(rhs, lam, obs)
        raise NotImplementedError(
            f"No measurement-space Gram solve for problem {getattr(problem,'name','?')!r}. "
            "Implement `solve_gram_plus_lambda(rhs, lam, obs)` on the problem."
        )

    def _pinv_reg(self, problem, obs, rhs: Tensor, lam) -> Tensor:
        """A^T (A A^T + lam I)^{-1} rhs, in IMAGE space.

        Every use of the Gram solve in this class is immediately followed by an adjoint,
        so this is the real primitive. Problems may override it with a numerically better
        route for the SAME quantity: multi-coil MRI uses the equivalent image-space form
        (A^T A + lam I)^{-1} A^T, because CG on its measurement-space Gram is far too
        ill-conditioned to converge in a handful of iterations (see
        MRIProblem.apply_pinv_reg). Otherwise fall back to adjoint(gram_solve(.)), which is
        exact and closed-form for inpainting / denoising / SR / deblurring.
        """
        if hasattr(problem, "apply_pinv_reg"):
            return problem.apply_pinv_reg(rhs, lam, obs)
        return problem.adjoint(self._gram_solve(problem, obs, rhs, lam), obs)

    @torch.no_grad()
    def m_y(self, problem, obs) -> Tensor:
        """Posterior-source mean m_y = A^T (A A^T + lambda I)^{-1} y."""
        lam = self._lam(self._sigma_of(problem, obs))
        return self._pinv_reg(problem, obs, obs.y, lam)

    @torch.no_grad()
    def sample(self, problem, obs, generator: torch.Generator | None = None,
               return_anchor: bool = False) -> Tensor:
        """Square-root-free RTO sample x1 ~ N(m_y, tau^2 W).

        Default path: exactly ONE Gram solve, K(y - A x_p - n'), because the whole
        argument is assembled BEFORE applying K = (A A^T + lam I)^-1.

        return_anchor=True exposes the anchor m_y = A^T K y by splitting that single solve
        into TWO -- m_y and the noise term A^T K(A x_p + n') -- and caches m_y on
        obs.aux['m_y'] so the pipeline reuses it instead of issuing a third solve. x1 is
        algebraically identical by linearity of K: K(y - A x_p - n') = K y - K(A x_p + n').

        Cost of that extra solve: free for SR / inpainting / denoising (all closed-form,
        O(1) elementwise), real only for multi-coil MRI, where each solve is a CG -- so
        cond_anchor='m_y' costs two CG solves per step there against one for 'aty'.
        """
        y = obs.y
        sigma_n = self._sigma_of(problem, obs)
        lam = self._lam(sigma_n)
        # x_p lives in IMAGE space (x0 space); A(x_p) then lands in measurement space.
        # For operators that change resolution (e.g. SR) image != measurement shape, so
        # derive the image shape from A^T y rather than assuming it equals y's shape.
        img_like = problem.adjoint(y, obs)
        x_p = self.tau * torch.randn(img_like.shape, device=y.device, dtype=y.dtype,
                                     generator=generator)
        _sn = (sigma_n.view(-1, *([1] * (y.dim() - 1))) if torch.is_tensor(sigma_n)
               else sigma_n)
        n_prime = _sn * torch.randn(y.shape, device=y.device, dtype=y.dtype,
                                    generator=generator)
        if return_anchor:
            m_y = self._pinv_reg(problem, obs, y, lam)
            noise_term = self._pinv_reg(problem, obs, problem.forward(x_p, obs) + n_prime, lam)
            x1 = x_p + m_y - noise_term
            if isinstance(obs.aux, dict):
                obs.aux["m_y"] = m_y                 # cache for pipeline._anchor_m_y reuse
            return x1
        resid = y - problem.forward(x_p, obs) - n_prime
        return x_p + self._pinv_reg(problem, obs, resid, lam)
