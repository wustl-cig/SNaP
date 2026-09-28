"""SNaP: one-step posterior sampling for image inverse problems.

Convention:
  x0 = clean image (target), x1 = measurement-dependent source.
  path  z_t = (1 - t) x0 + t x1,   t in [0,1]   (t=0 clean, t=1 source)
  vel   v   = x1 - x0
  x-prediction:  u_{r,t} = (z - net(z, r, t, cond, sigma_n)) / t   ; net predicts x0.

Training: the MeanFlow identity target V = u + (t - r) sg[d/dt u] is regressed against v,
with d/dt u a forward-mode JVP with tangent (v, 0, 1) on (z, r, t).
One-step sampling: x0_hat = net(x1, r=0, t=1, cond, sigma_n).
Few-step: compose u over a partition of [0,1]; no retraining.

The network is conditioned on the measurement through an image-shaped `cond` tensor
[anchor, coverage] concatenated onto z (held constant under the JVP). `coverage` is a
1-channel per-pixel measurement-coverage map supplied by the problem (mask for
inpainting/SR, ones for denoising/deblur), so the net input is 2C+1 channels regardless
of the inverse problem (2C for MRI, where the k-space mask is not an image-domain map).

Exposes `solver_block`, `train_step`, `val_step`, and `sample` with the signatures the
SNaPStage train/eval loop expects.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.func import jvp

from source import GaussianPosteriorSource, StandardGaussianSource


def verify_ckpt_metadata(saved: Optional[dict], current: dict, source: str = "") -> None:
    """Compare a checkpoint's metadata against the current config.

    RAISE on the fields that change what the network was trained to consume:
      cond_anchor    - the conditioning statistic (A^T y vs m_y)
      source_type    - the base measure x1 was drawn from (posterior vs gaussian)
      cond_coverage  - the number of conditioning channels, i.e. the net's INPUT WIDTH
      noise_mode     - absolute sigma vs per-sample input_snr_db; scalar vs per-sample
                       lambda AND scalar vs per-sample sigma-conditioning
    WARN on tau / beta / gamma_max / input_snr_db. No auto-migration."""
    where = f" ({source})" if source else ""
    if not saved:
        warnings.warn(
            f"checkpoint has no metadata{where}; cannot verify cond_anchor / "
            "cond_coverage / noise_mode / tau / beta / gamma_max. If it predates "
            "metadata, confirm them manually before trusting the numbers.",
            RuntimeWarning, stacklevel=2)
        return
    sc, cc = saved.get("cond_anchor"), current.get("cond_anchor")
    if sc != cc:
        raise RuntimeError(
            f"cond_anchor mismatch{where}: checkpoint was trained with cond_anchor={sc!r} but "
            f"the current config is {cc!r}. This changes the conditioning statistic (A^T y vs "
            f"m_y); loading would feed out-of-distribution conditioning and silently degrade "
            f"results. Set snap.cond_anchor={sc!r} to match the checkpoint, or retrain.")
    scov, ccov = saved.get("cond_coverage"), current.get("cond_coverage")
    if scov is not None and scov != ccov:
        raise RuntimeError(
            f"cond_coverage mismatch{where}: checkpoint={scov!r} vs current={ccov!r}. This "
            f"changes the number of conditioning channels, so the net's input width differs "
            f"(2C vs 2C+1) -- the load would either fail on shape or silently mis-feed a "
            f"channel. Set snap.cond_coverage={scov!r} to match, or retrain.")
    ssrc, csrc = saved.get("source_type"), current.get("source_type")
    if ssrc is not None and ssrc != csrc:
        raise RuntimeError(
            f"source_type mismatch{where}: checkpoint was trained with source_type={ssrc!r} "
            f"but the current config is {csrc!r}. The model learns transport FROM its base "
            f"measure -- 'posterior' starts at the RTO sample N(m_y, tau^2 W), 'gaussian' at "
            f"white tau*N(0, I) -- so sampling from the other one initialises the ODE "
            f"off-distribution and the numbers are not comparable. Set snap.source={ssrc!r} "
            f"to match the checkpoint, or retrain.")
    smode, cmode = saved.get("noise_mode"), current.get("noise_mode")
    if smode is not None and smode != cmode:
        raise RuntimeError(
            f"noise_mode mismatch{where}: checkpoint={smode!r} vs current={cmode!r}. "
            f"'input_snr_db' gives PER-SAMPLE sigma (hence per-sample lambda and per-sample "
            f"sigma-conditioning); 'sigma' gives one scalar. The network is conditioned on "
            f"sigma, so swapping the mode feeds it a statistic it was not trained on. Match "
            f"the mode, or retrain.")
    ssnr, csnr = saved.get("input_snr_db"), current.get("input_snr_db")
    if ssnr != csnr:
        warnings.warn(
            f"input_snr_db differs{where}: checkpoint={ssnr!r} vs current={csnr!r}. The model "
            "was trained at a different measurement SNR; it is sigma-conditioned so it may "
            "cope, but this is a train/test distribution shift.", RuntimeWarning, stacklevel=2)
    for k in ("tau", "beta", "gamma_max"):
        if saved.get(k) != current.get(k):
            warnings.warn(
                f"{k} differs{where}: checkpoint={saved.get(k)!r} vs current={current.get(k)!r}. "
                "Proceeding (warn-only).", RuntimeWarning, stacklevel=2)


def cond_extra_channels(coverage: str) -> int:
    """Number of conditioning channels BEYOND the image-shaped anchor.

    The net sees [z (C), anchor (C), <extra>], so its input width is 2C + this. Kept as a
    module function because `stage.build_models` and `eval.build_net` must size the net
    identically to what `_build_cond` will hand it -- a mismatch is a silent shape error
    at best and a silently mis-fed channel at worst."""
    return 0 if str(coverage).lower() == "none" else 1


class SNaPPipeline(torch.nn.Module):

    def __init__(
        self,
        solver_block: torch.nn.Module,   # the SNaPUNet (name kept for stage/EMA compat)
        problem,
        tau: float = 1.0,
        *,
        source_type: str = "posterior",   # "posterior" (measurement-dependent) | "gaussian" (ablation)
        cond_anchor: str = "m_y",          # conditioning anchor: "m_y" (regularized) | "aty" (A^T y)
        cond_coverage: str = "auto",       # "auto" | "none" | "sens" | "kspace_mask"
        p_ratio: float = 0.75,
        corner_frac: float = 0.0,
        logit_mu: float = 0.0,
        logit_sigma: float = 1.0,
        weight_p: float = 1.0,
        weight_c: float = 1e-3,
        weight_norm: str = "fixed",        # "fixed" (constant c) | "ema" (running-mean of m)
        weight_norm_decay: float = 0.99,
        beta: float = 0.0,
        gamma_max: float = 1e4,
        t_eps: float = 1e-3,
        sample_k: int = 1,
    ):
        super().__init__()
        self.solver_block = solver_block
        self.problem = problem
        self.tau = float(tau)
        self.source_type = str(source_type).lower()
        if self.source_type in ("gaussian", "standard", "white"):
            self.source_type = "gaussian"
            self.source = StandardGaussianSource(tau=tau)
        elif self.source_type in ("posterior", "rto", "meanflow"):
            self.source_type = "posterior"
            self.source = GaussianPosteriorSource(tau=tau)
        else:
            raise ValueError(f"unknown source_type {source_type!r} "
                             "(expected 'posterior' or 'gaussian')")
        # Dedicated posterior anchor for conditioning, ALWAYS the GaussianPosteriorSource
        # formula regardless of source_type. Kept independent of self.source so the
        # 'gaussian' ablation still injects measurement information via m_y instead of the
        # all-zero StandardGaussianSource.m_y (which would invalidate the ablation).
        self._anchor = GaussianPosteriorSource(tau=tau)
        self.cond_anchor = str(cond_anchor).lower()
        if self.cond_anchor not in ("m_y", "aty"):
            raise ValueError(f"cond_anchor must be 'm_y' or 'aty', got {cond_anchor!r}")
        # Second conditioning channel. For INPAINTING the mask is image-shaped and
        # spatially meaningful, so feeding it as a spatial channel is coherent. For MRI it
        # is not: the sampling mask lives in k-space, so as an image channel it asks the
        # network to treat one input as obeying a different geometry from the others.
        #   none        - drop it (the anchor already carries the measurement information)
        #   sens        - sum_c |S_c|^2, a genuinely image-domain coverage map
        #   kspace_mask - the image-domain 0/1 mask
        #   auto        - "none" for MRI, "kspace_mask" everywhere else
        cc = str(cond_coverage).lower()
        if cc not in ("auto", "none", "sens", "kspace_mask"):
            raise ValueError("cond_coverage must be auto|none|sens|kspace_mask, "
                             f"got {cond_coverage!r}")
        if cc == "auto":
            cc = "none" if str(getattr(problem, "name", "")).lower() in ("cs_mri", "mri") \
                 else "kspace_mask"
        self.cond_coverage = cc
        self.p_ratio = float(p_ratio)
        self.corner_frac = float(corner_frac)
        if not 0.0 <= self.corner_frac <= 1.0:
            raise ValueError(f"corner_frac must be in [0,1], got {corner_frac!r}")
        self.logit_mu = float(logit_mu)
        self.logit_sigma = float(logit_sigma)
        self.weight_p = float(weight_p)
        self.weight_c = float(weight_c)
        self.weight_norm = str(weight_norm).lower()
        if self.weight_norm not in ("fixed", "ema"):
            raise ValueError(f"weight_norm must be 'fixed' or 'ema', got {weight_norm!r}")
        self.weight_norm_decay = float(weight_norm_decay)
        self._m_ema = None                  # running mean of m; None until the first step
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)
        # NOTE: no trace-norm cache -- _trace_ratio is deliberately recomputed per call
        # because inpainting and MRI masks vary per sample; do not reintroduce caching.
        # `_sigma_base` is the nominal (config) sigma that all eval paths pin to.
        self._sigma_base = float(self.problem.sigma)
        self.t_eps = float(t_eps)
        self.sample_k = int(sample_k)

    # ----------------------------- helpers -----------------------------
    def _net(self):
        # The raw JVP-safe module (never DDP-wrapped -- multi-GPU sync is done with manual
        # gradient all-reduce hooks in the stage). During sampling the stage swaps in the
        # raw EMA module, which is also called directly here.
        return self.solver_block

    def _anchor_m_y(self, obs) -> Tensor:
        """Regularized measurement anchor m_y = A^T(AA^T + lam I)^-1 y, via the posterior
        formula ALWAYS (self._anchor, independent of self.source). Reuses obs.aux['m_y']
        if the posterior source already solved for it during sample(return_anchor=True);
        otherwise (e.g. the 'gaussian' ablation source, which does not) it does the one
        Gram solve here and caches the result. For MRI that solve is a CG, so the cache
        prevents a duplicate."""
        if isinstance(obs.aux, dict) and obs.aux.get("m_y") is not None:
            return obs.aux["m_y"]
        m = self._anchor.m_y(self.problem, obs)
        if isinstance(obs.aux, dict):
            obs.aux["m_y"] = m
        return m

    def _build_cond(self, obs) -> Tensor:
        """Image-shaped conditioning [anchor, coverage] with coverage reduced to 1 channel.
        `anchor` is the regularized m_y (default) or the plain adjoint A^T y (cond_anchor
        flag). Coverage comes from problem.coverage_map (mask / ones)."""
        if self.cond_anchor == "m_y":
            anchor = self._anchor_m_y(obs)
        else:  # "aty"
            anchor = self.problem.adjoint(obs.y, obs)
        if self.cond_coverage == "none":
            return anchor
        if self.cond_coverage == "sens" and hasattr(self.problem, "sens_map"):
            cov = self.problem.sens_map(obs)
        elif hasattr(self.problem, "coverage_map"):
            cov = self.problem.coverage_map(obs)
        else:
            mask = obs.aux.get("mask", None) if isinstance(obs.aux, dict) else None
            cov = mask if mask is not None else torch.ones_like(anchor[:, :1])
        if cov.shape[1] > 1:
            cov = cov.mean(dim=1, keepdim=True)
        cov = cov.expand(anchor.shape[0], 1, anchor.shape[2], anchor.shape[3])
        return torch.cat([anchor, cov], dim=1)

    def _sigma_vec(self, B: int, device, obs=None) -> Tensor:
        """(B,) sigma for the network's sigma-conditioning. Under `input_snr_db` the noise
        is per-sample, so the net must be told the ACTUAL sigma of each sample rather than
        one broadcast scalar -- otherwise its conditioning disagrees with its input."""
        if obs is not None and isinstance(getattr(obs, "aux", None), dict) \
                and obs.aux.get("sigma") is not None:
            return obs.aux["sigma"].to(device).reshape(-1)[:B]
        return torch.full((B,), float(self.problem.sigma), device=device)

    @torch.no_grad()
    def _update_m_ema(self, m: Tensor) -> Tensor:
        """EMA of mean(m), all-reduced so every rank shares ONE normalizer.

        Without the all-reduce each rank would build its own EMA from its own batches and
        scale its gradient slightly differently, which then gets averaged -- a silent
        per-rank loss reweighting. Cheap: one scalar per step."""
        bm = m.detach().mean()
        if dist.is_available() and dist.is_initialized():
            bm = bm.clone()
            dist.all_reduce(bm, op=dist.ReduceOp.SUM)
            bm /= dist.get_world_size()
        if self._m_ema is None:
            self._m_ema = bm.clone()
        else:
            d = self.weight_norm_decay
            self._m_ema.mul_(d).add_(bm, alpha=1.0 - d)
        return self._m_ema.clamp_min(1e-12)

    def _gamma(self, sigma: Tensor) -> Tensor:
        """beta/lambda, capped so the metric's condition number stays <= 1 + gamma_max.

        The cap is a safety valve for the sigma -> 0 tail ONLY (where lambda -> 0 makes
        beta/lambda blow up), NOT the normal operating point: at sigma=0.05, tau=1 the
        full-whitening ratio beta/lambda is 400, so a cap near that value would clamp
        every beta above ~0.25 to the same metric and make a beta sweep degenerate."""
        lam = (sigma ** 2) / (self.tau ** 2)
        return (self.beta / lam.clamp_min(1e-12)).clamp_max(self.gamma_max)

    def metadata(self) -> Dict[str, Any]:
        """Config-identity fields recorded in checkpoints (see verify_ckpt_metadata) so a
        load under a different conditioning statistic is caught, not silently mis-scored."""
        return {
            "cond_anchor": self.cond_anchor,
            "cond_coverage": self.cond_coverage,
            # Which base measure x1 was drawn from during training: the measurement-
            # dependent RTO posterior N(m_y, tau^2 W) or the ablation's white tau*N(0, I).
            # The learned map is a transport FROM this measure, so sampling a checkpoint
            # from the other one starts the ODE off-distribution.
            "source_type": self.source_type,
            # How the measurement noise is specified. Recorded because it changes the
            # SHAPE of sigma (scalar vs per-sample) and therefore of lambda and of the
            # net's sigma-conditioning -- not just its value.
            "noise_mode": ("input_snr_db"
                           if getattr(self.problem, "input_snr_db", None) is not None
                           else "sigma"),
            "input_snr_db": getattr(self.problem, "input_snr_db", None),
            "tau": self.tau,
            "beta": self.beta,
            "gamma_max": self.gamma_max,
        }

    def config_report(self) -> str:
        """One-line startup diagnostic: tau, sigma, the resulting lambda, and the implied
        RAW (uncapped) beta/lambda -- so a binding gamma_max cap is obvious BEFORE a run
        rather than discovered from a flat beta sweep after."""
        snr = getattr(self.problem, "input_snr_db", None)
        if snr is not None:
            # sigma (hence lambda) is PER-SAMPLE here; report the SNR being enforced.
            return (f"[config] tau={self.tau:g} beta={self.beta:g} source={self.source_type} "
                    f"anchor={self.cond_anchor} noise=input_snr_db={snr:g} dB "
                    f"(per-sample sigma -> per-sample lambda) gamma_max={self.gamma_max:g} "
                    f"| p_ratio={self.p_ratio:g} corner_frac={self.corner_frac:g}")
        s = self._sigma_base
        lam = (s ** 2) / (self.tau ** 2)
        raw = self.beta / max(lam, 1e-12)
        binds = (self.beta > 0.0) and (raw > self.gamma_max)
        return (f"[config] tau={self.tau:g} beta={self.beta:g} source={self.source_type} "
                f"anchor={self.cond_anchor} sigma={s:g} lambda={lam:.3g} "
                f"raw beta/lambda={raw:.3g} gamma_max={self.gamma_max:g} "
                f"| p_ratio={self.p_ratio:g} corner_frac={self.corner_frac:g}"
                + ("  <<< CAP BINDS (raw beta/lambda exceeds gamma_max; a beta sweep will "
                   "saturate -- raise gamma_max or lower beta)" if binds else ""))

    def _trace_ratio(self, obs, ref: Tensor) -> Tensor:
        """||A||_F^2 / n, per-sample. Exact when a 0/1 mask is available, else a
        Hutchinson probe. NOT cached: inpainting and MRI masks vary per sample.

        The mask shortcut requires singular values in {0,1}, i.e. A a plain selection --
        true for inpainting, FALSE for multi-coil MRI, where A = M F S and sum_c|S_c|^2
        vanishes off the object. Problems set `exact_mask_trace = False` to force the
        Hutchinson branch; MRIProblem does.

        Only reached when beta > 0 -- `_whitened_sq` returns plain MSE otherwise."""
        if (getattr(self.problem, "exact_mask_trace", True)
                and isinstance(obs.aux, dict) and "mask" in obs.aux
                and obs.aux["mask"] is not None):
            m = obs.aux["mask"]
            return m.flatten(1).float().mean(dim=1)        # (B,)
        with torch.no_grad():
            z = torch.randn_like(ref)
            Az = self.problem.forward(z, obs)
            r = Az.pow(2).flatten(1).sum(1) / z.pow(2).flatten(1).sum(1)
        return r

    def _whitened_sq(self, delta: Tensor, obs, sigma: Tensor) -> Tensor:
        """tau^2 ||delta||^2_{W^-1}, per-sample, rescaled to plain-MSE magnitude.

        W^-1 = (1/tau^2) I + A^H A / sigma_n^2, so scaling by the scalar tau^2 gives

            ||delta||^2  +  (tau^2/sigma_n^2) ||A delta||^2
                            \___ gamma = beta/lambda ___/

        beta = 0 (the default) turns this back into plain MSE."""
        if self.beta <= 0.0:
            return delta.pow(2).flatten(1).mean(dim=1)
        g = self._gamma(sigma)
        n = delta[0].numel()
        Ad = self.problem.forward(delta, obs)
        q = (delta.pow(2).flatten(1).sum(1) + g * Ad.pow(2).flatten(1).sum(1)) / n
        return q / (1.0 + g * self._trace_ratio(obs, delta))

    def _sample_t_r(self, B: int, device, corner_frac: Optional[float] = None):
        """Sample (t, r) for the MeanFlow objective.

        Base schedule: t ~ logit-normal; r = t for a fraction p_ratio of samples, else
        r ~ U(0, t) -- the r < t "average-velocity" branch.

        Corner oversampling (ADDITIVE, on top of p_ratio): reserve `corner_frac` of the
        r != t samples for the (r~0, t~1) corner -- r ~ U(0, 0.1), t ~ U(0.9, 1.0). That
        corner is the exact (r=0, t=1) slice one-step sampling evaluates at, which the base
        logit-normal schedule (logit_mu=0, logit_sigma=1) reaches only ~once per few-hundred
        steps. Train-only: val_step passes corner_frac=0.0 so the monitoring residual keeps
        the base t distribution and stays comparable."""
        cf = self.corner_frac if corner_frac is None else float(corner_frac)
        t = torch.sigmoid(self.logit_mu + self.logit_sigma * torch.randn(B, device=device))
        r = t.clone()
        neq = torch.rand(B, device=device) > self.p_ratio      # (1 - p_ratio) get r < t
        r = torch.where(neq, torch.rand(B, device=device) * t, r)
        if cf > 0.0:
            corner = neq & (torch.rand(B, device=device) < cf)  # cf of the r != t samples
            if corner.any():
                t = torch.where(corner, 0.9 + 0.1 * torch.rand(B, device=device), t)
                r = torch.where(corner, 0.1 * torch.rand(B, device=device), r)
        return t, r

    @staticmethod
    def _map(s: Tensor, ref: Tensor) -> Tensor:
        return s.view(-1, 1, 1, 1).expand(ref.shape[0], 1, ref.shape[2], ref.shape[3])

    def _u_fn(self, z, r, t, cond, sigma):
        """x-prediction velocity u = (z - net([z,cond], r, t, sigma)) / t."""
        x_in = torch.cat([z, cond], dim=1)
        x0_pred = self._net()(x_in, t, r, sigma)
        t_map = self._map(t, z).clamp_min(self.t_eps)
        return (z - x0_pred) / t_map

    # ----------------------------- training -----------------------------
    def train_step(self, batch: Tensor, device: torch.device) -> Dict[str, Any]:
        x0 = batch.to(device, non_blocking=True)   # clean image
        B = x0.shape[0]
        obs = self.problem.make_observation(x0)
        # return_anchor makes the posterior source expose+cache m_y so _build_cond reuses
        # it rather than issuing its own solve.
        x1 = self.source.sample(self.problem, obs,
                                return_anchor=(self.cond_anchor == "m_y"))
        cond = self._build_cond(obs)
        sigma = self._sigma_vec(B, device, obs)     # per-sample under input_snr_db, else scalar

        t, r = self._sample_t_r(B, device)
        t_map = self._map(t, x0)
        z = (1.0 - t_map) * x0 + t_map * x1         # z_t
        v = x1 - x0                                 # velocity

        # JVP is not autocast-safe; force fp32 forward-mode AD.
        with torch.autocast(device_type=("cuda" if x0.is_cuda else "cpu"), enabled=False):
            def fn(z_, r_, t_):
                return self._u_fn(z_, r_, t_, cond, sigma)

            u, dudt = jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
            tr = (t - r).view(-1, 1, 1, 1)
            V = u + tr * dudt.detach()              # MeanFlow identity (stopgrad on dudt)
            delta = V - v
            m = self._whitened_sq(delta, obs, sigma)   # plain MSE when beta=0
            # Adaptive weight. "fixed": w = (m + c)^-p. Its problem is that m is not
            # scale-free, so a fixed c silently anneals the objective from adaptive
            # (m>>c, w~1/m) to plain MSE (m<<c, w~1/c) partway through training, and means
            # something different again at another tau or on another task.
            # "ema": normalise m by its own running mean first, so c is dimensionless and
            # the loss shape is stable across tau, task, and training time.
            m_ref = self._update_m_ema(m) if self.weight_norm == "ema" else None
            m_w = m.detach() if m_ref is None else (m.detach() / m_ref)
            w = (m_w + self.weight_c) ** (-self.weight_p)
            loss = (w * m).mean()

        # No endpoint ||x0_hat - x0||^2 term: it has the posterior MEAN as its population
        # minimizer, so it would bias the one-step map toward MMSE and away from the
        # posterior sampling this model exists to do. x0_hat below is for LOGGING only.
        with torch.no_grad():
            x0_hat_log = self._one_step(x1, cond, sigma)

        with torch.no_grad():
            # gamma-cap saturation monitor: fraction of samples where beta/lambda would
            # exceed gamma_max (i.e. the clamp_max in _gamma binds). With beta=0 (the
            # default) gamma=0 everywhere -> frac=0 (no-op).
            lam_g = (sigma ** 2) / (self.tau ** 2)
            gamma_capped_frac = ((self.beta / lam_g.clamp_min(1e-12)) > self.gamma_max).float().mean()

        return {
            "loss": loss,
            "gamma_capped_frac": gamma_capped_frac.detach(),
            "m_ema": (self._m_ema.detach() if self._m_ema is not None
                      else m.new_tensor(float("nan"))),
            "velocity_mse": m.mean().detach(),
            "x_hat": x0_hat_log,
            "x_gt": x0,
            "y": obs.y,
        }

    @torch.no_grad()
    def val_step(self, batch: Tensor, device: torch.device) -> Dict[str, Any]:
        x0 = batch.to(device, non_blocking=True)
        B = x0.shape[0]
        self.problem.sigma = self._sigma_base       # val uses the FIXED nominal sigma
        obs = self.problem.make_observation(x0)
        x1 = self.source.sample(self.problem, obs,
                                return_anchor=(self.cond_anchor == "m_y"))
        cond = self._build_cond(obs)
        sigma = self._sigma_vec(B, device, obs)

        # Plain flow-matching residual at r = t (no JVP needed for monitoring).
        # corner_frac=0.0: keep val on the BASE t distribution so the metric stays
        # comparable and is not skewed toward the oversampled t~1 corner.
        t, _ = self._sample_t_r(B, device, corner_frac=0.0)
        t_map = self._map(t, x0)
        z = (1.0 - t_map) * x0 + t_map * x1
        v = x1 - x0
        u = self._u_fn(z, t, t, cond, sigma)
        return {
            "velocity_mse": F.mse_loss(u, v).detach(),
            "x_hat": self._one_step(x1, cond, sigma),
            "x_gt": x0,
            "y": obs.y,
        }

    # ----------------------------- sampling -----------------------------
    def _one_step(self, x1, cond, sigma) -> Tensor:
        B = x1.shape[0]
        r0 = torch.zeros(B, device=x1.device)
        t1 = torch.ones(B, device=x1.device)
        x_in = torch.cat([x1, cond], dim=1)
        return self._net()(x_in, t1, r0, sigma)   # = x1 - u_{0,1}(x1)

    def _k_step(self, x1, cond, sigma, k: int) -> Tensor:
        z = x1
        ts = torch.linspace(1.0, 0.0, k + 1, device=x1.device)  # source(1) -> clean(0)
        B = x1.shape[0]
        for i in range(k):
            t_cur = torch.full((B,), float(ts[i]), device=x1.device)
            r_next = torch.full((B,), float(ts[i + 1]), device=x1.device)
            u = self._u_fn(z, r_next, t_cur, cond, sigma)
            z = z - (float(ts[i]) - float(ts[i + 1])) * u        # z_r = z_t - (t-r) u
        return z

    def _display_y(self, obs, x_gt: Tensor):
        """Image-shaped measurement for metrics/visualization (the "degraded input").

        For image-space operators this is y itself; for operators whose measurement is not
        image-shaped (MRI multi-coil k-space) fall back to the zero-filled adjoint A^T y.
        Operators that change resolution (e.g. SR) produce a smaller y, so nearest-upsample
        it to the GT size, otherwise PSNR/SSIM against x_gt are undefined."""
        y_disp = obs.y
        # Inpainting adds its noise over the WHOLE image (`y = M*x + n`), so raw y holds
        # pure noise where the mask is 0. Every consumer of y applies A^T = M*, which
        # discards it -- but this display/metric path does not, so it would score the
        # measurement against noise-filled holes instead of the zeros a reader sees.
        # A^T y IS the degraded image for a projector operator, so take it.
        if isinstance(obs.aux, dict) and obs.aux.get("mask") is not None:
            y_disp = self.problem.adjoint(obs.y, obs)
        elif (y_disp.ndim != x_gt.ndim or y_disp.shape[1] != x_gt.shape[1]
                or y_disp.shape[-2:] != x_gt.shape[-2:]):
            try:
                y_disp = self.problem.adjoint(obs.y, obs)
            except Exception:
                y_disp = None
        if y_disp is not None and y_disp.shape[-2:] != x_gt.shape[-2:]:
            y_disp = F.interpolate(y_disp, size=x_gt.shape[-2:], mode="nearest")
        return y_disp

    @torch.no_grad()
    def sample(self, x_gt: Tensor, steps: int = 1) -> Dict[str, Tensor]:
        """`steps` = number of SNaP steps k (1 = one-step)."""
        self.problem.sigma = self._sigma_base       # inference uses the FIXED nominal sigma
        obs = self.problem.make_observation(x_gt)
        x1 = self.source.sample(self.problem, obs,
                                return_anchor=(self.cond_anchor == "m_y"))
        cond = self._build_cond(obs)
        sigma = self._sigma_vec(x_gt.shape[0], x_gt.device, obs)
        k = max(1, int(steps))
        x_hat = self._one_step(x1, cond, sigma) if k == 1 else self._k_step(x1, cond, sigma, k)
        out = {"x_hat": x_hat, "x_gt": x_gt, "y": self._display_y(obs, x_gt)}
        if isinstance(obs.aux, dict) and ("mask" in obs.aux):
            out["mask"] = obs.aux["mask"]
        return out
