"""Training / evaluation stage for SNaP.

Self-contained: it owns its build, training loop, EMA, cosine LR schedule, checkpointing
and test loop. The network is a SNaPUNet used directly as the `solver_block` -- there
is no iterative solver (a one-step model has no trajectory to enforce data consistency
along; the measurement-dependent source replaces subspace confinement). The pipeline is
SNaPPipeline (JVP training + one/few-step sampling).

Multi-GPU uses MANUAL data parallelism (broadcast weights from rank 0, all-reduce grads
via post-accumulate hooks) rather than DDP, because forward-mode AD (JVP) inside
DDP.forward trips DDP's bucket rebuild.
"""

from __future__ import annotations

import copy
import glob
import os

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from core.config import SNAP_META_KEY, ckpt_meta, snap_cfg
from dataset.build import build_loader
from methods.registry import get_problem
from utils.image_io import save_images, save_images_mri
from utils.metrics import save_metric, write_metrics_csv, psnr_function, mri_psnr_ssim

from snap_unet import SNaPUNet
from pipeline import SNaPPipeline, verify_ckpt_metadata, cond_extra_channels

import builtins as _builtins
from datetime import datetime as _datetime


def _log_print(*args, **kwargs):
    """Timestamp + flush every status line. Shadows the builtin `print` in this module
    so all `print(...)` status output is prefixed with [YYYY-MM-DD HH:MM:SS]; tqdm bars
    (which write via their own stream, not print) are unaffected."""
    kwargs.setdefault("flush", True)
    _builtins.print(f"[{_datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)


print = _log_print

LOG_TAG = "snap"
CKPT_PREFIX = "snap"


class SNaPStage:

    def __init__(self, config, device, rank, world_size, is_main, physical_gpu=None):
        self.cfg = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main
        self.physical_gpu = physical_gpu
        self.save_root = config["_runtime"]["save_root"]

        self.problem = None
        self.train_problem = None
        self.val_problem = None
        self.solver_block = None
        self.pipeline = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.ema_solver_block = None
        self.use_ema = False
        self.ema_decay = 0.999
        self.ema_start_epoch = 0
        self.resume_ckpt_path = None
        self.resume_ema_state = None
        self._grad_hooks = []
        self.initial_lr = float(config["training"].get("lr", 1e-4))

    # --------------------------------------------------------------- utils
    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def _prep_batch(self, batch):
        """Return the clean-image target and, for dict batches (MRI), set the current
        problem's per-sample operator state (sensitivity maps) before the step."""
        if isinstance(batch, dict):
            x = batch["target"].to(self.device, non_blocking=True)
            maps = batch.get("maps", None)
            if maps is not None and hasattr(self.pipeline.problem, "set_maps"):
                self.pipeline.problem.set_maps(maps.to(self.device, non_blocking=True))
            return x
        return batch.to(self.device, non_blocking=True)

    # --------------------------------------------------------------- build
    def build(self):
        self.build_problem()
        self.build_models()
        self.build_dataloader()
        self.build_pipeline()
        if self.is_main:
            print(self.pipeline.config_report())
            self._log_data_scale()
        self.build_optimizers()
        self._init_ema()

    def _log_data_scale(self):
        """Print tau against the EMPIRICAL data scale under this loader's normalization.

        tau sets the null-space variance of the source, so a tau that does not match the
        data scale corrupts most of the transport. MRI images are mostly background, so
        their scale is far below the [-1,1] convention natural images use -- tau=1 against
        a std of 0.15 is a ~44x variance over-dispersion. Printing it makes that visible at
        launch for every operator instead of being discovered from a bad curve."""
        loader = self.trainloader if self.trainloader is not None else self.valloader
        if loader is None:
            return
        try:
            rms, sd, n = [], [], 0
            for b in loader:
                x = b["target"] if isinstance(b, dict) else b
                rms.append(float(x.pow(2).sum(dim=1).mean().sqrt()))
                sd.append(float(x.std()))
                n += 1
                if n >= 16:
                    break
            r, v = float(np.mean(rms)), float(np.mean(sd))
            tau = float(self.pipeline.tau)
            msg = (f"[scale] tau={tau:g} | data sqrt(E|x|^2)={r:.4f} std={v:.4f} | "
                   f"tau/rms={tau/max(r,1e-12):.3f} (var x{(tau/max(r,1e-12))**2:.3f}) "
                   f"| cond_coverage={self.pipeline.cond_coverage}")
            # Realised measurement SNR, so `noise` is never taken on faith: sigma is an
            # ABSOLUTE std, so the same sigma lands at different SNRs on different data.
            prob = self.pipeline.problem
            if hasattr(prob, "measure_snr_db"):
                try:
                    b = next(iter(loader))
                    x = b["target"] if isinstance(b, dict) else b
                    if isinstance(b, dict) and b.get("maps") is not None:
                        prob.set_maps(b["maps"].to(self.device))
                    snr = prob.measure_snr_db(x.to(self.device))
                    if getattr(prob, "input_snr_db", None) is not None:
                        msg += f" | input_snr_db={snr:.1f} dB (per-sample sigma)"
                    else:
                        msg += f" | sigma={prob.sigma:g} -> input SNR {snr:.1f} dB"
                except Exception as e:
                    msg += f" | SNR probe failed ({type(e).__name__})"
            print(msg)
        except Exception as e:                       # diagnostics must never kill a run
            print(f"[scale] skipped ({type(e).__name__}: {e})")

    def build_problem(self):
        problem_name = self.cfg["experiment"]["stage"]
        pcfg = dict(self.cfg["methods"][problem_name])
        noise_val = pcfg.pop("noise", None)
        if noise_val is not None:
            pcfg["sigma"] = float(noise_val)
        ProblemCls = get_problem(problem_name)
        self.train_problem = ProblemCls(**pcfg)
        self.val_problem = ProblemCls(**pcfg)
        # Training may draw a fresh mask per step (MRIProblem.make_observation); evaluation
        # must pin ONE mask so the val curve and the test numbers are comparable across
        # epochs.
        if hasattr(self.val_problem, "random_mask"):
            self.val_problem.random_mask = False
        self.problem = self.train_problem
        # MRI targets are complex (2-ch real/imag): metrics/images must use the
        # dynamic-range magnitude convention, not the natural-image code path.
        self._is_mri = str(problem_name).lower() in ("cs_mri", "mri")

    def build_models(self):
        mcfg = self.cfg["model"]
        C = int(mcfg.get("in_ch", 3))
        out_chans = int(mcfg.get("out_ch", 3))
        # net input = [z (C), anchor (C), <coverage>]. The coverage channel is dropped
        # when snap.cond_coverage == "none" (the MRI default -- the k-space sampling
        # mask is not an image-domain quantity), so the width must follow the flag.
        mfc = snap_cfg(self.cfg)
        cov = str(mfc.get("cond_coverage", "auto")).lower()
        if cov == "auto":
            cov = "none" if self._is_mri else "kspace_mask"
        self._cond_coverage = cov
        in_chans = 2 * C + cond_extra_channels(cov)
        self.solver_block = SNaPUNet(
            input_channels=in_chans,
            output_channels=out_chans,
            input_height=int(mcfg.get("input_height", 128)),
            ch=int(mcfg.get("ch", 32)),
            ch_mult=mcfg.get("ch_mult", [1, 2, 4, 8]),
            num_res_blocks=int(mcfg.get("num_res_blocks", 6)),
            attn_resolutions=mcfg.get("attn_resolutions", [16, 8]),
            dropout=float(mcfg.get("dropout", 0.0)),
            resamp_with_conv=bool(mcfg.get("resamp_with_conv", True)),
            cond_r=bool(mcfg.get("cond_r", True)),
            cond_sigma=bool(mcfg.get("cond_sigma", True)),
            sigma_emb_scale=float(mcfg.get("sigma_emb_scale", 10.0)),
        ).to(self.device)
        if self.is_main:
            n_par = sum(p.numel() for p in self.solver_block.parameters())
            print(f"[model] SNaPUNet in={in_chans} out={out_chans} | {n_par/1e6:.1f}M params")

    def build_dataloader(self):
        use_ddp = bool(self.cfg["distributed"].get("use_ddp", False))
        r = self.rank if use_ddp else 0
        ws = self.world_size if use_ddp else 1
        steps_per_epoch = self.cfg["training"].get("steps_per_epoch", None)
        if steps_per_epoch is not None:
            steps_per_epoch = int(steps_per_epoch)

        phase = self.cfg["experiment"].get("phase", "train")
        self.trainloader = None
        if phase == "train":
            self.trainloader, _ = build_loader(
                cfg_dict=self.cfg, split="train", rank=r, world_size=ws,
                steps_per_epoch=steps_per_epoch,
            )
        # phase=test evaluates the held-out TEST split; training monitors `validation`.
        eval_split = "test" if phase == "test" else "validation"
        self.valloader, _ = build_loader(cfg_dict=self.cfg, split=eval_split,
                                         rank=r, world_size=ws)

    def build_pipeline(self):
        mfcfg = snap_cfg(self.cfg)
        scfg = self.cfg.get("sample", {})
        self.pipeline = SNaPPipeline(
            solver_block=self.solver_block,
            problem=self.problem,
            tau=float(mfcfg.get("tau", 1.0)),
            source_type=str(mfcfg.get("source", "posterior")),
            cond_anchor=str(mfcfg.get("cond_anchor", "m_y")),
            cond_coverage=str(mfcfg.get("cond_coverage", "auto")),
            p_ratio=float(mfcfg.get("p_ratio", 0.75)),
            corner_frac=float(mfcfg.get("corner_frac", 0.0)),
            logit_mu=float(mfcfg.get("logit_mu", 0.0)),
            logit_sigma=float(mfcfg.get("logit_sigma", 1.0)),
            weight_p=float(mfcfg.get("weight_p", 1.0)),
            weight_c=float(mfcfg.get("weight_c", 1e-3)),
            weight_norm=str(mfcfg.get("weight_norm", "fixed")),
            weight_norm_decay=float(mfcfg.get("weight_norm_decay", 0.99)),
            beta=float(mfcfg.get("beta", 0.0)),
            gamma_max=float(mfcfg.get("gamma_max", 1e4)),
            t_eps=float(mfcfg.get("t_eps", 1e-3)),
            sample_k=int(scfg.get("steps", 1)),
        )

        resume = self.cfg["model"].get("resume_path", None)
        if resume and os.path.exists(resume):
            map_location = self.device if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(resume, map_location=map_location, weights_only=True)
            verify_ckpt_metadata(ckpt_meta(ckpt), self.pipeline.metadata(),
                                 source=f"resume {resume}")
            self.solver_block.load_state_dict(ckpt["model"], strict=True)
            self.resume_ema_state = ckpt.get("ema_model", None)
            self.start_epoch = int(ckpt.get("epoch", 0))
            self.resume_ckpt_path = resume
            if self.is_main:
                print(f"[snap] Resumed from {resume}, start_epoch={self.start_epoch}")

        self._setup_manual_ddp()

    def build_optimizers(self):
        tcfg = self.cfg["training"]
        lr = float(tcfg.get("lr", 1e-4))
        wd = float(tcfg.get("weight_decay", 0.0))
        self.initial_lr = lr
        self.optimizer = torch.optim.AdamW(self.solver_block.parameters(), lr=lr,
                                           weight_decay=wd)
        self.scheduler = None

    # ------------------------------------------------------- manual DDP
    def _setup_manual_ddp(self):
        if not (bool(self.cfg["distributed"].get("use_ddp", False)) and dist.is_initialized()):
            return
        ws = dist.get_world_size()
        net = self._unwrap(self.solver_block)

        with torch.no_grad():
            for p in net.parameters():
                dist.broadcast(p.data, src=0)
            for b in net.buffers():
                if torch.is_tensor(b) and b.is_floating_point():
                    dist.broadcast(b.data, src=0)

        # `_sync_grads` is the no_sync switch for gradient accumulation: only the LAST
        # micro-batch of an accumulation window all-reduces. That is exact -- each rank
        # accumulates its own micro-batch gradients locally and the single reduction at
        # the end averages the local sums -- and it costs ONE all-reduce per optimizer
        # step instead of one per micro-batch.
        self._sync_grads = True

        def _make_hook():
            def _hook(param):
                if param.grad is None or not getattr(self, "_sync_grads", True):
                    return
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.mul_(1.0 / ws)
            return _hook

        self._grad_hooks = [
            p.register_post_accumulate_grad_hook(_make_hook())
            for p in net.parameters() if p.requires_grad
        ]
        if self.is_main:
            print(f"[snap] Manual data-parallel: gradient all-reduce over {ws} ranks "
                  f"({len(self._grad_hooks)} params hooked).")

    # --------------------------------------------------------------- EMA
    def _enable_ema(self):
        model = self._unwrap(self.solver_block)
        self.ema_solver_block = copy.deepcopy(model).to(self.device)
        self.ema_solver_block.eval()
        for p in self.ema_solver_block.parameters():
            p.requires_grad_(False)

    def _init_ema(self):
        tcfg = self.cfg["training"]
        self.use_ema = bool(tcfg.get("use_ema", True))
        self.ema_decay = float(tcfg.get("ema_decay", 0.999))
        self.ema_start_epoch = int(tcfg.get("ema_start_epoch", 0))
        if not self.use_ema:
            return
        if self.resume_ema_state is not None:
            self._enable_ema()
            self.ema_solver_block.load_state_dict(self.resume_ema_state, strict=True)
            self.resume_ema_state = None
            if self.is_main:
                print("[EMA] Restored EMA weights from checkpoint")
        elif self.start_epoch >= self.ema_start_epoch:
            self._enable_ema()
        if self.is_main:
            print(f"[EMA] Enabled with decay={self.ema_decay}; start_epoch={self.ema_start_epoch}")

    @torch.no_grad()
    def _update_ema(self, epoch: int):
        if not self.use_ema or epoch < self.ema_start_epoch:
            return
        if self.ema_solver_block is None:
            self._enable_ema()
            if self.is_main:
                print(f"[EMA] Initialized at epoch {epoch + 1}")
            return
        model = self._unwrap(self.solver_block)
        ema_state = self.ema_solver_block.state_dict()
        model_state = model.state_dict()
        for k in ema_state.keys():
            if torch.is_floating_point(ema_state[k]):
                ema_state[k].mul_(self.ema_decay).add_(
                    model_state[k].detach(), alpha=1.0 - self.ema_decay)
            else:
                ema_state[k].copy_(model_state[k])

    # ----------------------------------------------------- problem swap
    def _set_train_problem(self):
        self.pipeline.problem = self.train_problem

    def _set_val_problem(self):
        self.pipeline.problem = self.val_problem

    # ------------------------------------------------------- scheduler
    def _restore_optimizer_state(self):
        if not self.resume_ckpt_path or self.cfg["experiment"].get("phase") != "train":
            return
        map_location = self.device if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(self.resume_ckpt_path, map_location=map_location, weights_only=True)
        opt_state = ckpt.get("optimizer")
        if opt_state is None:
            return
        self.optimizer.load_state_dict(opt_state)
        if self.is_main:
            print("[Resume] Restored optimizer state from checkpoint")

    def _build_scheduler(self, num_epochs: int):
        tcfg = self.cfg["training"]
        if str(tcfg.get("lr_schedule", "none")).lower() != "cosine":
            self.scheduler = None
            return
        lr = float(tcfg.get("lr", self.initial_lr))
        lr_min = float(tcfg.get("lr_min", 1e-6))
        for group in self.optimizer.param_groups:
            group["initial_lr"] = lr
        last_epoch = self.start_epoch - 1 if self.start_epoch > 0 else -1
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, num_epochs), eta_min=lr_min, last_epoch=last_epoch)
        if self.is_main:
            print(f"[LR] CosineAnnealing lr={lr:.2e} -> {lr_min:.2e} over "
                  f"{num_epochs} epochs; start_epoch={self.start_epoch}")

    # --------------------------------------------------------- ckpt io
    def _save_ckpt(self, ckpt_path: str, epoch: int, include_optimizer: bool = True):
        if self.rank != 0:
            return
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        model = self._unwrap(self.solver_block)
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            # Config identity, so a later load under a different conditioning statistic
            # (esp. cond_anchor) is caught rather than silently mis-scored.
            SNAP_META_KEY: self.pipeline.metadata(),
        }
        # Optimizer + scheduler roughly triple the file size (AdamW carries two moment
        # buffers per param). They're only needed to *resume training*, so keep them out
        # of inference-only checkpoints (e.g. the "best" snapshot used for eval).
        if include_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                payload["scheduler"] = self.scheduler.state_dict()
        if self.use_ema and self.ema_solver_block is not None:
            payload["ema_model"] = self.ema_solver_block.state_dict()
        torch.save(payload, ckpt_path)
        if self.is_main:
            print(f"[Checkpoint] Saved: {ckpt_path}")

    def _rotate_checkpoints(self, ckpt_dir: str, keep_last_n: int):
        """Keep only the most recent `keep_last_n` periodic epoch checkpoints so a long
        run can't fill the filesystem. Never touches snap_best.pt.
        keep_last_n <= 0 disables rotation (keep everything)."""
        if self.rank != 0 or keep_last_n is None or keep_last_n <= 0:
            return
        cks = sorted(glob.glob(os.path.join(ckpt_dir, f"{CKPT_PREFIX}_epoch_*.pt")))
        for old in cks[:-keep_last_n]:
            try:
                os.remove(old)
                if self.is_main:
                    print(f"[Checkpoint] Rotated out old checkpoint: {old}")
            except OSError as e:
                print(f"[Checkpoint] Could not remove {old}: {e}")

    def _update_best_checkpoint(self, best_so_far, score, epoch):
        """Keep ONE best checkpoint, by the LOWEST TRAINING MSE (lower is always better).

        Selecting on a TRAINING score avoids a failure mode of val-based selection here:
        `val` is computed on the EMA weights, and at a long EMA horizon the shadow is still
        dominated by its initialisation for many epochs, so best-selection sticks to early
        epochs. CONSEQUENCE TO EXPECT: training MSE falls close to monotonically, so `best`
        is usually the most recent epoch. That is intended, not a bug -- for careful
        checkpoint selection, read the logged validation curve and load the corresponding
        snap_epoch_*.pt."""
        if score is None or not np.isfinite(score) or score >= best_so_far:
            return best_so_far
        path = os.path.join(self.save_root, "checkpoints", f"{CKPT_PREFIX}_best.pt")
        self._save_ckpt(path, epoch, include_optimizer=False)
        prev = "--" if best_so_far == float("inf") else f"{best_so_far:.6e}"
        print(f"[Best] train_mse={score:.6e} at epoch {epoch} (prev {prev}) -> {path}")
        return score

    # --------------------------------------------------------- training
    def run_train(self):
        cfg = self.cfg
        tcfg = cfg["training"]
        lcfg = cfg["logging"]
        scfg = cfg.get("sample", {})

        num_epochs = int(tcfg.get("epochs", 100))
        use_ddp = bool(cfg["distributed"].get("use_ddp", False))
        grad_clip = float(tcfg.get("grad_clip", 1.0))
        # Gradient accumulation: `accum` micro-batches per optimizer step. The effective
        # batch becomes batch_size * accum * world_size, which is how you keep a usable
        # effective batch at 256/320 px where memory caps batch_size per GPU.
        accum = max(1, int(tcfg.get("grad_accum_steps", 1)))
        if accum > 1:
            # pipeline._update_m_ema runs once per train_step, i.e. `accum` times per
            # optimizer step, so the adaptive-weight EMA would decay accum-fold faster in
            # optimizer-step terms. Take the accum-th root to hold the per-STEP decay.
            d0 = float(self.pipeline.weight_norm_decay)
            self.pipeline.weight_norm_decay = d0 ** (1.0 / accum)
            if self.is_main:
                bs_cfg = int(self.cfg["dataloader"]["batch_size"])
                n_gpu = max(1, len(self.cfg["distributed"].get("gpus", [0])))
                print(f"[accum] grad_accum_steps={accum} | effective batch = "
                      f"{bs_cfg * accum * n_gpu} ({bs_cfg} x {accum} x {n_gpu} gpus) | "
                      f"weight_norm_decay {d0:g} -> {self.pipeline.weight_norm_decay:.6f}")
        log_every = int(lcfg.get("log_every", 5))
        save_every = int(lcfg.get("save_every", 5))
        sample_every = int(lcfg.get("save_img_every", 5))
        keep_last_n = int(lcfg.get("keep_last_n", 0))
        do_val = bool(lcfg.get("validate", False))

        if self.is_main:
            print(f"[seed] experiment.seed = {cfg['experiment'].get('seed')}")

        self._restore_optimizer_state()
        self._build_scheduler(num_epochs)

        # ONE best checkpoint, lowest TRAINING mse. See _update_best_checkpoint.
        best_train_mse = float("inf")

        for epoch in range(self.start_epoch, num_epochs):
            if self.is_main:
                print(f"[Epoch {epoch+1}/{num_epochs}]")
            if use_ddp:
                sampler = getattr(self.trainloader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            self.solver_block.train()
            self._set_train_problem()
            train_losses, train_vel_mse, train_mema = [], [], []

            iterator = self.trainloader
            if self.is_main:
                iterator = tqdm(self.trainloader, desc=f"Train Epoch {epoch+1}", leave=False)

            micro = 0                      # micro-batches accumulated since the last step
            n_micro = len(self.trainloader)
            for step, batch in enumerate(iterator):
                x0 = self._prep_batch(batch)  # clean image; sets MRI maps if present
                if micro == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                # all-reduce only on the micro-batch that will trigger the step
                is_last = (micro + 1 >= accum) or (step + 1 == n_micro)
                self._sync_grads = is_last
                out = self.pipeline.train_step(x0, self.device)
                loss = out["loss"]
                loss_val = float(loss.detach().item())
                if not torch.isfinite(loss):
                    # drop the whole accumulation window rather than step on partial grads
                    self.optimizer.zero_grad(set_to_none=True)
                    micro = 0
                    self._sync_grads = True
                    if self.is_main:
                        print(f"[WARNING] non-finite loss ({loss_val:.4g}) at "
                              f"epoch {epoch+1} step {step}, skipping.")
                    train_losses.append(float("nan"))
                    continue
                (loss / accum).backward()      # /accum so the effective LR is unchanged
                micro += 1
                if is_last:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.solver_block.parameters(),
                                                       max_norm=grad_clip)
                    self.optimizer.step()
                    self._update_ema(epoch)
                    micro = 0
                    self._sync_grads = True

                train_losses.append(loss_val)
                train_vel_mse.append(float(out["velocity_mse"].item()))
                if self.pipeline.weight_norm == "ema":
                    train_mema.append(float(out["m_ema"].item()))

            # periodic reconstruction preview
            if self.is_main and (epoch + 1) % sample_every == 0:
                self._log_sample_images(epoch, scfg)

            # periodic metrics + best checkpoint. `logging.validate` (default False)
            # decides whether the held-out validation pass runs at all; with it off the
            # single-draw PSNR/LPIPS below come from a FIXED batch of TRAINING images and
            # nothing touches the validation set during training.
            if self.is_main and (epoch + 1) % log_every == 0:
                mset = "val" if do_val else "train"
                train_loss = float(np.nanmean(train_losses)) if train_losses else 0.0
                tv = float(np.nanmean(train_vel_mse)) if train_vel_mse else 0.0
                mtxt = (f"m_ema={float(np.nanmean(train_mema)):.6e} "
                        if train_mema else "")
                vtxt = f"val_psnr={self._validate():.4f} " if do_val else ""
                # single-draw PSNR/LPIPS on a fixed 16-image batch (of the validation
                # split when `validate` is on, of the training split otherwise)
                p1, lp1 = self._val_draws(split=mset)
                print(f"[{LOG_TAG}][{epoch+1}/{num_epochs}] train={train_loss:.6f} "
                      f"mse={tv:.6f} {mtxt}{vtxt}"
                      f"{mset}16_psnr={p1:.3f} {mset}16_lpips={lp1:.4f}")
                # best checkpoint: ALWAYS lowest training mse, single file, no eviction
                best_train_mse = self._update_best_checkpoint(best_train_mse, tv, epoch + 1)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.is_main and (epoch + 1) % save_every == 0:
                ckpt_dir = os.path.join(self.save_root, "checkpoints")
                ckpt_path = os.path.join(ckpt_dir, f"{CKPT_PREFIX}_epoch_{epoch+1:04d}.pt")
                self._save_ckpt(ckpt_path, epoch + 1)
                self._rotate_checkpoints(ckpt_dir, keep_last_n)

    # ------------------------------------------------------ monitoring
    def _fixed_batch(self, n: int, split: str = "val"):
        """First `n` items of `split`, deterministic and cached.

        split="train" is the source for the epoch metrics when `logging.validate` is off.
        The train loader is a RandomSampler(replacement=True), so "first n" is not a fixed
        set across epochs -- we cache the first batch ever seen and reuse it every epoch,
        which is what makes the epoch-to-epoch numbers comparable."""
        attr = "_fixed_vb" if split == "val" else "_fixed_tb"
        loader = self.valloader if split == "val" else self.trainloader
        if getattr(self, attr, None) is None:
            xs, maps = [], []
            for b in loader:
                x = b["target"] if isinstance(b, dict) else b
                xs.append(x)
                if isinstance(b, dict) and b.get("maps") is not None:
                    maps.append(b["maps"])
                if sum(t.shape[0] for t in xs) >= n:
                    break
            setattr(self, attr, (torch.cat(xs)[:n],
                                 torch.cat(maps)[:n] if maps else None))
        return getattr(self, attr)

    def _lpips_fn(self):
        if getattr(self, "_lpips", None) is None:
            import lpips as _l
            self._lpips = _l.LPIPS(net="alex").to(self.device).eval()
            for q in self._lpips.parameters():
                q.requires_grad_(False)
        return self._lpips

    @torch.no_grad()
    def _val_draws(self, n_img: int = 16, seed: int = 1234, split: str = "train"):
        """Single-draw one-step PSNR / LPIPS on a fixed batch, with y drawn from a fixed
        seed so the number is comparable across epochs.

        Single-draw PSNR is what a collapsed sampler scores well on -- use eval_avg.py
        (sample-averaged PSNR + inter-draw spread) to tell sampling from regression."""
        pipe = self.pipeline
        self.solver_block.eval()
        self._set_val_problem()
        saved = pipe.solver_block
        if self.use_ema and self.ema_solver_block is not None:
            self.ema_solver_block.eval()
            pipe.solver_block = self.ema_solver_block
        x_gt, maps = self._fixed_batch(n_img, split=split)
        x_gt = x_gt.to(self.device)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        psnrs, lps = [], []
        try:
            for i in range(x_gt.shape[0]):
                xg = x_gt[i:i+1]
                if maps is not None and hasattr(pipe.problem, "set_maps"):
                    pipe.problem.set_maps(maps[i:i+1].to(self.device))
                pipe.problem.sigma = pipe._sigma_base
                obs = pipe.problem.make_observation(xg)
                cond = pipe._build_cond(obs)
                x1 = pipe.source.sample(pipe.problem, obs, generator=gen)
                sig = pipe._sigma_vec(1, self.device, obs)
                xh = pipe._one_step(x1, cond, sig)
                # MRI targets are 2-channel complex: score the MAGNITUDE with a per-slice
                # data range, never the natural-image psnr_function -- that averages over
                # the real/imag channels and is ~3 dB optimistic.
                if self._is_mri:
                    psnrs.append(float(mri_psnr_ssim(xh, xg)["psnr"]))
                else:
                    psnrs.append(float(psnr_function(xh, xg)))
                    # LPIPS is an RGB perceptual metric; it cannot take 2-channel complex
                    # MRI. Reported as nan there.
                    lps.append(float(self._lpips_fn()(xh.clamp(-1, 1), xg.clamp(-1, 1))))
        finally:
            pipe.solver_block = saved
            self._set_train_problem()
            self.solver_block.train()
        return (float(np.mean(psnrs)),
                float(np.mean(lps)) if lps else float("nan"))

    def _log_sample_images(self, epoch, scfg):
        self.solver_block.eval()
        self._set_val_problem()
        _b = next(iter(self.valloader))
        if isinstance(_b, dict):
            if _b.get("maps", None) is not None and hasattr(self.pipeline.problem, "set_maps"):
                self.pipeline.problem.set_maps(_b["maps"][:4].to(self.device, non_blocking=True))
            x_gt = _b["target"][:4].to(self.device, non_blocking=True)
        else:
            x_gt = _b.to(self.device, non_blocking=True)[:4]
        old_solver = self.pipeline.solver_block
        if self.use_ema and self.ema_solver_block is not None:
            self.pipeline.solver_block = self.ema_solver_block
        with torch.no_grad():
            out = self.pipeline.sample(x_gt=x_gt, steps=int(scfg.get("steps", 1)))
        self.pipeline.solver_block = old_solver
        self._set_train_problem()
        self.solver_block.train()
        if self._is_mri:
            save_images_mri(x_gt=x_gt, x_hat=out["x_hat"], y=out["y"],
                            save_dir=self.save_root, prefix=f"epoch_{epoch+1:04d}")
            psnr_in = mri_psnr_ssim(out["y"], out["x_gt"])["psnr"]
            psnr_out = mri_psnr_ssim(out["x_hat"], out["x_gt"])["psnr"]
        else:
            save_images(x_gt=x_gt, x_hat=out["x_hat"], y=out["y"],
                        save_dir=self.save_root, prefix=f"epoch_{epoch+1:04d}")
            psnr_in = psnr_function(out["y"], out["x_gt"])
            psnr_out = psnr_function(out["x_hat"], out["x_gt"])
        print(f"[val] epoch {epoch+1} | Input PSNR: {psnr_in:.2f} dB "
              f"| Output PSNR: {psnr_out:.2f} dB")

    @torch.no_grad()
    def _validate(self) -> float:
        """Deterministic one-step PSNR over the validation split (higher is better).

        A FIXED seed is set for the whole pass (and the global RNG restored after), so the
        masks, measurement noise, source draw and data order are identical every epoch --
        the metric then reflects only the model, not fresh sampling noise re-rolled each
        validation. It is SINGLE-DRAW: fine as a curve for one run, but it cannot see
        sampler collapse -- use eval_avg.py for that."""
        self.solver_block.eval()
        self._set_val_problem()
        saved = self.pipeline.solver_block
        if self.ema_solver_block is not None:
            self.ema_solver_block.eval()
            self.pipeline.solver_block = self.ema_solver_block

        val_seed = int(self.cfg["logging"].get("val_seed", 1234))
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(val_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(val_seed)

        psnr_sum, n_img = 0.0, 0
        try:
            for batch in self.valloader:
                x0 = self._prep_batch(batch)
                out = self.pipeline.val_step(x0, self.device)
                nb = int(out["x_hat"].shape[0])
                if self._is_mri:
                    # mri_psnr_ssim averages PER SLICE within the batch, so weighting by
                    # batch size recovers the exact per-slice mean over the split.
                    psnr_sum += mri_psnr_ssim(out["x_hat"], out["x_gt"])["psnr"] * nb
                else:
                    psnr_sum += float(psnr_function(out["x_hat"], out["x_gt"])) * nb
                n_img += nb
        finally:
            self.pipeline.solver_block = saved
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        self._set_train_problem()
        self.solver_block.train()
        return (psnr_sum / n_img) if n_img else float("nan")

    # ------------------------------------------------------------- test
    @torch.no_grad()
    def run_test(self):
        cfg = self.cfg
        scfg = cfg.get("sample", {})
        self.solver_block.eval()
        self._set_val_problem()
        save_root = self.save_root
        os.makedirs(save_root, exist_ok=True)

        iterator = self.valloader
        if self.is_main:
            iterator = tqdm(self.valloader, desc="Test", leave=False)

        old_solver = self.pipeline.solver_block
        if self.use_ema and self.ema_solver_block is not None:
            self.pipeline.solver_block = self.ema_solver_block

        records = []
        global_idx = 0
        for batch in iterator:
            x_gt = self._prep_batch(batch)   # dict (MRI) -> target + set maps; else tensor
            out = self.pipeline.sample(x_gt=x_gt, steps=int(scfg.get("steps", 1)))
            if self.is_main:
                B = x_gt.shape[0]
                if self._is_mri:
                    # dynamic-range magnitude PSNR/SSIM
                    for i in range(B):
                        m = mri_psnr_ssim(out["x_hat"][i:i+1], out["x_gt"][i:i+1])
                        m_in = mri_psnr_ssim(out["y"][i:i+1], out["x_gt"][i:i+1])
                        records.append({"index": global_idx + i,
                                        "psnr": m["psnr"], "ssim": m["ssim"], "lpips": float("nan"),
                                        "psnr_in": m_in["psnr"], "ssim_in": m_in["ssim"],
                                        "lpips_in": float("nan")})
                        save_images_mri(x_gt=out["x_gt"][i:i+1], x_hat=out["x_hat"][i:i+1],
                                        y=out["y"][i:i+1], save_dir=save_root,
                                        prefix=f"test_img_{global_idx+i:04d}")
                else:
                    save_metric(pred=out["x_hat"], gt=out["x_gt"], y=out["y"],
                                save_root=save_root, records=records, global_offset=global_idx)
                    for i in range(B):
                        save_images(x_gt=out["x_gt"][i:i+1], x_hat=out["x_hat"][i:i+1],
                                    y=out["y"][i:i+1], save_dir=save_root,
                                    prefix=f"test_img_{global_idx+i:04d}")
                global_idx += B

        self.pipeline.solver_block = old_solver
        if self.is_main and records:
            write_metrics_csv(records, os.path.join(save_root, "test_metrics.csv"))
