"""Multi-coil compressed-sensing MRI problem for the Mean-Flow framework.

Implements the same `InverseProblem` interface as `InpaintingProblem`, but for the
standard parallel-imaging operator used by CS-MRI benchmarks, so Mean-Flow trains/tests
on *identical* measurements:

    A(x) = M ⊙ F( S ⊙ x ),    y = A(x) (+ noise)

with x the complex MVUE image (carried as 2 real channels [real, imag], (B,2,H,W)),
S the per-slice ESPIRiT sensitivity maps (B, C, H, W) complex, F the orthonormal 2-D DFT
(with the usual fftshift convention), and
M a Cartesian phase-encode mask. Measurements y are carried as real-view multi-coil
k-space, (B, C, H, W, 2), matching `torch.view_as_real`.

Unlike inpainting/SR, parallel MRI's measurement-space Gram A·Aᵀ is NOT diagonal (coils
couple), so `solve_gram_plus_lambda` — the (A Aᵀ + λ I)⁻¹ needed by the posterior source —
is solved with conjugate gradient using only forward/adjoint. This makes the operator work
with BOTH Mean-Flow sources: `posterior` (GaussianPosteriorSource, uses the CG solve) and
`gaussian` (StandardGaussianSource, needs only adjoint).

Per-slice maps are data, not fixed: the training loop sets them each batch via `set_maps`
(the MRI dataset yields {"target": mvue2ch, "maps": maps}).

MASKS ARE FIXED BY DEFAULT (`random_mask=False`). One sampling pattern, selected entirely
by `mask_seed`, is used for training, validation and test -- matching the baselines and the
earlier `..._fixedmask_...` brain runs. Change WHICH mask by changing `mask_seed`; that is
the only intended control. `random_mask=True` is the opt-in variant that draws a fresh
pattern per example, which trains a model that generalises across masks but is a different
experiment: it widens the conditioning distribution and creates a train/test mismatch,
since evaluation always pins the mask regardless.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from .base import Obs

Tensor = torch.Tensor


class MRIProblem:
    name = "cs_mri"
    # ||A||_F^2/n is NOT the mask fraction here. The 0/1-mask shortcut in
    # pipeline._trace_ratio is exact only when the singular values are in {0,1}; for
    # multi-coil A = M F S they are not, because sum_c|S_c|^2 is 0 off the object.
    # Measured on a real brain scan: Hutchinson 0.2043 vs mask fraction 0.2500, and
    # mask_frac x mean(SoS^2) = 0.2500 x 0.8172 = 0.2043 reproduces it exactly.
    exact_mask_trace = False

    def __init__(
        self,
        sigma: float = 0.01,              # ABSOLUTE noise std (source λ = sigma²/τ²); >0 keeps CG well-posed
        input_snr_db: Optional[float] = None,  # if set, OVERRIDES sigma: per-sample noise at this SNR
        total_lines: int = 320,
        acceleration_ratio: int = 4,
        pattern: str = "random",          # "random" | "equispaced"
        orientation: str = "vertical",
        image_size: int = 320,
        random_mask: bool = False,        # DEFAULT FIXED: one mask chosen by mask_seed.
                                          # True = fresh pattern per example (opt-in; see the
                                          # module docstring -- a different experiment).
        mask_seed: Optional[int] = 0,     # THE mask selector. Fixed everywhere by default;
                                          # only random_mask=True adds image_idx/counter.
        cg_iters: int = 12,
        cg_tol: float = 1e-6,
        compute_dtype: Optional[str] = None,  # None -> keep input dtype; "float64" -> match vendored exactly
        device: str = "cuda",
    ):
        self.sigma = float(sigma)
        # Two ways to specify the measurement noise, and they are NOT equivalent:
        #   sigma        - absolute std. Simple, but the realised SNR then varies with each
        #                  slice's k-space energy (measured ~1.4 dB sd, ~5 dB spread), and
        #                  the SAME sigma means different SNRs on different datasets
        #                  (brain 0.0073 -> 19.1 dB, knee 0.02 -> 16.7 dB).
        #   input_snr_db - PER-SAMPLE noise scaled so every slice lands on the requested
        #                  SNR exactly. This is the convention used by the
        #                  self-supervised CS-MRI literature, and it is what
        #                  makes train and test agree: the rule is applied to whatever
        #                  sample is in front of it, so no calibration can go stale.
        # When input_snr_db is set, `sigma` becomes a per-sample tensor published on
        # obs.aux["sigma"]; the source and the network's sigma-conditioning read it there.
        self.input_snr_db = None if input_snr_db is None else float(input_snr_db)
        self.total_lines = int(total_lines)
        self.acceleration_ratio = int(acceleration_ratio)
        self.pattern = pattern
        self.orientation = orientation
        self.image_size = int(image_size)
        self.random_mask = bool(random_mask)
        self.mask_seed = mask_seed
        self.cg_iters = int(cg_iters)
        self.cg_tol = float(cg_tol)
        self.compute_dtype = {None: None, "float64": torch.float64,
                              "float32": torch.float32}[compute_dtype]
        self.device = device
        self._maps: Optional[Tensor] = None   # (B, C, H, W) complex, set per batch
        self._mask_calls = 0                  # advances the mask seed when random_mask

    # ---- per-batch state ---------------------------------------------------
    def set_maps(self, maps: Tensor) -> None:
        """maps: (B, C, H, W) complex sensitivity maps for the current batch."""
        if not torch.is_complex(maps):
            maps = torch.view_as_complex(maps.contiguous()) if maps.shape[-1] == 2 else maps.to(torch.complex64)
        self._maps = maps

    # ---- Cartesian phase-encode mask (identical to vendored get_mask) -------
    @staticmethod
    def _get_mask_1d(acs_lines: int, total_lines: int, R: int, pattern: str, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        num_sampled = int(np.floor(total_lines / R))
        center = np.arange((total_lines - acs_lines) // 2, (total_lines + acs_lines) // 2)
        outer = np.setdiff1d(np.arange(total_lines), center)
        if pattern == "random":
            picked = rng.choice(outer, size=int(num_sampled - acs_lines), replace=False)
        elif pattern == "equispaced":
            picked = outer[:: int(R)]
        else:
            raise NotImplementedError(f"mask pattern {pattern!r}")
        m = np.zeros(total_lines)
        m[center] = 1.0
        m[picked] = 1.0
        return m

    def _mask_for(self, seed: int, device, W: int) -> Tensor:
        R = self.acceleration_ratio
        acs = int(np.floor((0.08 if 1 < R <= 6 else 0.04) * self.total_lines))
        m = self._get_mask_1d(acs, self.total_lines, R, self.pattern, seed).astype(bool)
        if self.orientation == "vertical":
            t = torch.from_numpy(m[None, None, None, :].copy())   # (1,1,1,W)
        elif self.orientation == "horizontal":
            t = torch.from_numpy(m[None, None, :, None].copy())   # (1,1,H,1)
        else:
            raise NotImplementedError(self.orientation)
        return t.to(device)

    # ---- FFT (fftshift convention identical to vendored MultiCoilMRI) -------
    @staticmethod
    def _fft(x: Tensor) -> Tensor:
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        return torch.fft.ifftshift(x, dim=(-2, -1))

    @staticmethod
    def _ifft(x: Tensor) -> Tensor:
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")
        return torch.fft.fftshift(x, dim=(-2, -1))

    # ---- 2ch-real image <-> complex ----------------------------------------
    @staticmethod
    def _img_c(x: Tensor) -> Tensor:      # (B,2,H,W) real -> (B,H,W) complex
        return torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())

    @staticmethod
    def _img_r(xc: Tensor) -> Tensor:     # (B,H,W) complex -> (B,2,H,W) real
        return torch.view_as_real(xc).permute(0, 3, 1, 2).contiguous()

    def _maps_for(self, obs: Optional[Obs]) -> Tensor:
        maps = (obs.aux.get("maps") if (obs is not None and isinstance(obs.aux, dict)) else None)
        if maps is None:
            maps = self._maps
        if maps is None:
            raise RuntimeError("MRIProblem: sensitivity maps not set. Call set_maps(...) "
                               "each batch or pass them in obs.aux['maps'].")
        return maps

    # ---- forward / adjoint -------------------------------------------------
    def forward(self, x: Tensor, obs: Obs) -> Tensor:
        """A(x): (B,2,H,W) image -> (B,C,H,W,2) masked multi-coil k-space (real-view)."""
        maps = self._maps_for(obs)
        mask = obs.aux["mask"]
        cdt = self.compute_dtype
        xc = self._img_c(x if cdt is None else x.to(cdt))        # (B,H,W) complex
        coils = maps * xc.unsqueeze(1)                           # (B,C,H,W)
        ksp = mask * self._fft(coils)                            # broadcast mask over coils
        return torch.view_as_real(ksp)

    def adjoint(self, y: Tensor, obs: Obs) -> Tensor:
        """Aᵀ(y): (B,C,H,W,2) real-view k-space -> (B,2,H,W) image.
        Aᵀ y = Σ_c conj(S_c) · F⁻¹(M ⊙ y_c)."""
        maps = self._maps_for(obs)
        mask = obs.aux["mask"]
        yc = y if torch.is_complex(y) else torch.view_as_complex(y.contiguous())  # (B,C,H,W)
        img = self._ifft(mask * yc) * torch.conj(maps)          # (B,C,H,W)
        xc = img.sum(dim=1)                                     # coil-combine -> (B,H,W)
        return self._img_r(xc)

    def normal(self, x: Tensor, obs: Obs) -> Tensor:
        return self.adjoint(self.forward(x, obs), obs)

    # ---- observation -------------------------------------------------------
    def make_observation(self, x_gt: Tensor, image_idx: int = 0) -> Obs:
        """x_gt: (B,2,H,W) MVUE. Uses self._maps (set per batch)."""
        B, _, H, W = x_gt.shape
        maps = self._maps_for(None).to(x_gt.device)
        # random_mask -> a fresh mask each call. An explicit image_idx pins which one;
        # with no index (the pipeline's train/val/sample calls all pass none) advance an
        # internal counter instead. Without the counter `seed` collapsed to `mask_seed`
        # and EVERY training step reused one single mask pattern, making random_mask a
        # no-op. Evaluation pins the mask by construction: the stage builds its val/test
        # problem with random_mask=False, so eval keeps seed == mask_seed exactly.
        if self.random_mask:
            seed = (self.mask_seed or 0) + (image_idx or self._mask_calls)
            self._mask_calls += 1
        else:
            seed = (self.mask_seed or 0)
        mask = self._mask_for(seed, x_gt.device, W).to(x_gt.real.dtype if torch.is_complex(x_gt) else x_gt.dtype)
        obs = Obs(y=None, aux={"maps": maps, "mask": mask})
        y = self.forward(x_gt, obs)
        if self.input_snr_db is not None:
            y, sig = self._add_noise_at_snr(y, obs, self.input_snr_db)
            obs.aux["sigma"] = sig                      # (B,) per-sample, read downstream
            obs.y = y
            return obs
        if self.sigma > 0:
            # Noise ONLY on sampled lines. Unsampled lines were never measured, so noise
            # there is not physical -- and it is discarded anyway (adjoint re-applies the
            # mask), so this is bookkeeping, not a behaviour change: the noise reaching
            # A^H is identical. What it fixes is the MEANING of sigma. Previously
            # `randn_like(y)` filled the whole tensor, so ||n|| computed over y was
            # sqrt(1/R) larger than the noise that actually enters -- 6.02 dB at R=4 --
            # which is exactly the convention error that made a "20 dB" setting really
            # 26 dB. With the mask applied, sigma reads against the sampled support, the
            # same per-sample input-SNR convention.
            y = y + self.sigma * torch.randn_like(y) * obs.aux["mask"].unsqueeze(-1)
        obs.y = y
        return obs

    def _live_mask(self, obs: Obs) -> Tensor:
        """(B,C,H,W,1) 1 where a measurement exists: sampled line AND non-padded coil.
        Zero-padded coils carry S_c = 0, so noise there is inert (A^H kills it) -- but it
        would still inflate ||n|| and corrupt the SNR bookkeeping, so exclude it."""
        maps = self._maps_for(obs)
        coil_live = (maps.abs().amax(dim=(-2, -1)) > 0).to(maps.real.dtype)   # (B,C)
        return (obs.aux["mask"].to(maps.real.dtype)
                * coil_live[:, :, None, None]).unsqueeze(-1)

    def _add_noise_at_snr(self, y: Tensor, obs: Obs, snr_db: float):
        """y <- y + n with 20 log10(||y||/||n||) == snr_db per sample. Returns (y, sigma)
        where sigma is the (B,) equivalent std on live entries, needed for lambda."""
        live = self._live_mask(obs)
        n = torch.randn_like(y) * live
        flat = lambda t: t.flatten(1)
        y_norm = flat(y).norm(dim=1)
        n_norm = flat(n).norm(dim=1).clamp_min(1e-12)
        scale = (y_norm / n_norm) / (10.0 ** (snr_db / 20.0))                 # (B,)
        n = n * scale.view(-1, *([1] * (y.dim() - 1)))
        n_live = live.expand_as(y).flatten(1).sum(dim=1).clamp_min(1.0)
        sigma = flat(n).norm(dim=1) / n_live.sqrt()                           # (B,)
        return y + n, sigma

    @torch.no_grad()
    def measure_snr_db(self, x_gt: Tensor, n_draws: int = 4) -> float:
        """Realised input SNR = 20 log10(||A x|| / ||n||) at the CURRENT sigma, in dB.

        Reported rather than assumed: a fixed absolute sigma does NOT give a fixed SNR --
        it varies with each slice's k-space energy (measured sd ~1.4 dB, ~5 dB spread)."""
        # input_snr_db mode is per-sample and hits the target by construction, so report
        # it directly rather than probing the (unused) scalar fallback.
        if self.input_snr_db is not None:
            return float(self.input_snr_db)
        if self.sigma <= 0:
            return float("inf")
        s, sig = self.sigma, self.sigma
        self.sigma = 0.0
        try:
            obs = self.make_observation(x_gt)
        finally:
            self.sigma = s
        yc = obs.y
        msk = obs.aux["mask"].unsqueeze(-1)
        out = []
        for _ in range(n_draws):
            n = sig * torch.randn_like(yc) * msk
            out.append(20.0 * torch.log10(yc.norm() / n.norm().clamp_min(1e-12)))
        return float(torch.stack(out).mean())

    # ---- image-domain sensitivity coverage (cond_coverage="sens") ---------
    def sens_map(self, obs: Obs) -> Tensor:
        """sum_c |S_c|^2 as a (B,1,H,W) image-domain coverage map.

        Unlike `coverage_map` this actually lives in the image domain: it is 1 where the
        coil array sees the object and 0 where it is blind (real ESPIRiT maps are exactly
        0 off-support), so it tells the network where A carries no information at all.
        The k-space sampling mask cannot express that -- it is the same everywhere in
        image space."""
        maps = self._maps_for(obs)
        return (maps.abs() ** 2).sum(dim=1, keepdim=True).to(torch.float32)

    # ---- conditioning coverage map ----------------------------------------
    def coverage_map(self, obs: Obs) -> Tensor:
        """1-channel k-space sampling coverage, broadcast to (B,1,H,W)."""
        mask = obs.aux["mask"].to(torch.float32)                # (1,1,1,W) or (1,1,H,1)
        aty = self.adjoint(obs.y, obs)
        B, _, H, W = aty.shape
        return mask.expand(B, 1, H, W).contiguous()

    # ---- regularized pseudo-inverse (posterior source) ---------------------
    @torch.no_grad()
    def apply_pinv_reg(self, rhs: Tensor, lam, obs: Obs) -> Tensor:
        """Aᵀ (A Aᵀ + λ I)⁻¹ rhs, returned in IMAGE space (B,2,H,W).

        This is the ONLY thing the posterior source ever asks for -- it always follows
        `solve_gram_plus_lambda` with an `adjoint`. Computed via the algebraically
        identical IMAGE-space form

            Aᵀ (A Aᵀ + λ I)⁻¹  =  (Aᵀ A + λ I)⁻¹ Aᵀ                            (*)

        which for multi-coil MRI is vastly better conditioned than the measurement-space
        route, and therefore the one to use. Why the measurement-space route fails here:
        A Aᵀ acts on C·H·W multi-coil k-space but factors through the H·W image, so its
        rank is at most H·W -- it is rank-deficient by the coil count (15x for this data),
        on top of the (1 - 1/R) of k-space the mask zeroes out. With λ = σ²/τ² = 1e-4 its
        condition number is ~1e4+ AND the right-hand side y has large components outside
        range(A), which the solve must map to y_null/λ. CG needs far more than `cg_iters`
        steps to get there, so the iterate is still enormous when the loop exits: measured
        on the knee test split, the anchor m_y came out ~5.7x over-scaled at 6.5 dB PSNR,
        WORSE than the plain adjoint Aᵀy (24.4 dB), and the source sample x1 at 3.6 dB.
        Form (*) instead applies CG to Aᵀ A + λ I, whose spectrum is in [λ, ~1] and whose
        right-hand side Aᵀ rhs already lies in range(Aᵀ) -- no null-space blow-up. Same
        cost (one forward+adjoint per iteration), same 12 iterations: 26.0 dB.

        Reductions are PER-SAMPLE (unlike `solve_gram_plus_lambda` below), so a batch of
        several slices does not share one CG step size.
        """
        # lam may be a float OR a per-sample (B,1,1,1) tensor (input_snr_db mode). The CG
        # reductions below are already per-sample, so a tensor lam just broadcasts.
        if torch.is_tensor(lam):
            lam = lam.view(-1, 1, 1, 1).to(rhs.real.dtype if torch.is_complex(rhs) else rhs.dtype)
        else:
            lam = float(lam)
        b = self.adjoint(rhs, obs)                      # image space, in range(Aᵀ)

        def AtA_plus(v: Tensor) -> Tensor:
            return self.adjoint(self.forward(v, obs), obs) + lam * v

        def dot(u: Tensor, v: Tensor) -> Tensor:        # (B,1,1,1) per-sample inner product
            return (u * v).flatten(1).sum(1).view(-1, 1, 1, 1)

        x = torch.zeros_like(b)
        r = b.clone()                                   # r = b - (AᵀA + λ)·0
        p = r.clone()
        rs = dot(r, r)
        b_norm = dot(b, b).clamp_min(1e-30)
        for _ in range(self.cg_iters):
            Ap = AtA_plus(p)
            alpha = rs / dot(p, Ap).clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = dot(r, r)
            if bool(((rs_new / b_norm) < self.cg_tol ** 2).all()):
                break
            p = r + (rs_new / rs.clamp_min(1e-30)) * p
            rs = rs_new
        return x

    # ---- measurement-space Gram solve via CG ------------------------------
    @torch.no_grad()
    def solve_gram_plus_lambda(self, rhs: Tensor, lam, obs: Obs) -> Tensor:
        """(A Aᵀ + λ I)⁻¹ rhs in measurement space (real-view k-space), by conjugate gradient.
        A Aᵀ v = A( Aᵀ v ); Hermitian PSD, so CG converges.

        NOT used by the posterior source -- see `apply_pinv_reg` above, which the source
        prefers because this operator is too ill-conditioned for a short CG run here.
        Kept for interface parity with the other problems and for direct callers."""
        lam = float(lam)
        rc = rhs if torch.is_complex(rhs) else torch.view_as_complex(rhs.contiguous())  # (B,C,H,W)

        def AAt(vc: Tensor) -> Tensor:
            vr = torch.view_as_real(vc)
            out = self.forward(self.adjoint(vr, obs), obs)       # real-view
            return torch.view_as_complex(out.contiguous()) + lam * vc

        x = torch.zeros_like(rc)
        r = rc - AAt(x)
        p = r.clone()
        rs = torch.real(torch.vdot(r.flatten(), r.flatten()))
        rhs_norm = torch.real(torch.vdot(rc.flatten(), rc.flatten())).clamp_min(1e-30)
        for _ in range(self.cg_iters):
            Ap = AAt(p)
            denom = torch.real(torch.vdot(p.flatten(), Ap.flatten())).clamp_min(1e-30)
            alpha = rs / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.real(torch.vdot(r.flatten(), r.flatten()))
            if (rs_new / rhs_norm) < self.cg_tol ** 2:
                break
            p = r + (rs_new / rs) * p
            rs = rs_new
        return torch.view_as_real(x)

    def project(self, x: Tensor, obs: Obs) -> Tensor:
        return x
