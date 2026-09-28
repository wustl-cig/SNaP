"""Evaluate a trained checkpoint on held-out test images (any problem).

Loads a checkpoint (EMA weights when present), builds the inverse problem named by the
config's `experiment.stage` (inpainting / box_inpainting / denoising / blur_gauss / sr /
cs_mri), runs one-step (or few-step) sampling on the `test` split, saves reconstructions
and a per-image metrics CSV (PSNR/SSIM/LPIPS), and prints the averages. Inference only
(no JVP/backward), so it is safe to run on a GPU that training is already using.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/celeba.yaml \
        --run_id my_run --n 100
    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/afhq.yaml \
        --ckpt /path/ckpt.pt --stage denoising --n 100 --k 4
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from core.config import ckpt_meta, load_config, snap_cfg
from dataset.build import _build_dataset
from methods.registry import get_problem
from utils.metrics import psnr_ssim, write_metrics_csv, to_01

from snap_unet import SNaPUNet
from pipeline import SNaPPipeline, verify_ckpt_metadata, cond_extra_channels


def build_problem(cfg):
    stage = cfg["experiment"]["stage"]
    pcfg = dict(cfg["methods"][stage])
    noise_val = pcfg.pop("noise", None)
    if noise_val is not None:
        pcfg["sigma"] = float(noise_val)
    prob = get_problem(stage)(**pcfg)
    # PIN THE MASK. A config may carry `random_mask: true` because TRAINING wants a fresh
    # mask per example; evaluation must use the single mask chosen by mask_seed, or every
    # test slice is scored against a different operator and the run is irreproducible.
    if getattr(prob, "random_mask", False):
        prob.random_mask = False
        print(f"[eval] random_mask pinned to False (mask_seed={getattr(prob,'mask_seed','?')})")
    return prob


def _coverage(cfg):
    cov = str(snap_cfg(cfg).get("cond_coverage", "auto")).lower()
    if cov == "auto":
        cov = "none" if str(cfg["experiment"]["stage"]).lower() in ("cs_mri", "mri") \
              else "kspace_mask"
    return cov


def build_net(cfg, device):
    """Build the network with the SAME input width the pipeline will hand it."""
    m = cfg["model"]
    C = int(m.get("in_ch", 3))
    net = SNaPUNet(
        input_channels=2 * C + cond_extra_channels(_coverage(cfg)),
        output_channels=int(m.get("out_ch", 3)),
        input_height=int(m.get("input_height", 128)),
        ch=int(m.get("ch", 32)),
        ch_mult=m.get("ch_mult", [1, 2, 4, 8]),
        num_res_blocks=int(m.get("num_res_blocks", 6)),
        attn_resolutions=m.get("attn_resolutions", [16, 8]),
        dropout=float(m.get("dropout", 0.0)),   # inert at eval (net.eval() below), kept
                                                # so the module tree matches training's
        resamp_with_conv=bool(m.get("resamp_with_conv", True)),
        cond_r=bool(m.get("cond_r", True)),
        cond_sigma=bool(m.get("cond_sigma", True)),
        sigma_emb_scale=float(m.get("sigma_emb_scale", 10.0)),
    ).to(device)
    return net


def build_pipeline(cfg, net, problem, device):
    m = snap_cfg(cfg)
    pipe = SNaPPipeline(net, problem,
                            tau=float(m.get("tau", 1.0)),
                            source_type=str(m.get("source", "posterior")),
                            cond_anchor=str(m.get("cond_anchor", "m_y")),
                            cond_coverage=str(m.get("cond_coverage", "auto"))).to(device)
    pipe.solver_block = net
    return pipe


def find_latest_ckpt(run_dir):
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*.pt")))
    if not cks:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/checkpoints")
    return cks[-1]


def _mri_panels(x_gt, y, x_hat):
    """Magnitude panels for a complex MRI triplet, windowed to the GT's own max.

    Replaces `to_01` for MRI, which is written for [-1,1] image datasets and would keep
    the tensor 2-channel (the imaginary part then becomes per-pixel alpha in the saved
    PNG) and clamp away the bright end. Windowing to |x_gt|.max() is the same dynamic
    range `dynamic_range_metrics` uses, so what you see matches what is scored.
    """
    mag = lambda t: (torch.view_as_complex(t.permute(1, 2, 0).contiguous().to(torch.float64)).abs()
                     if t.shape[0] == 2 else t.to(torch.float64).squeeze(0))
    g, o, h = mag(x_gt), mag(y), mag(x_hat)
    dr = g.max().clamp_min(1e-12)
    return torch.stack([(t / dr).clamp(0, 1).float().unsqueeze(0) for t in (g, o, h)], 0)


def dynamic_range_metrics(xh, xg):
    """Magnitude PSNR/SSIM with a per-image data_range -- the MRI reporting convention.

    It differs from `psnr_ssim` in two ways that BOTH inflate the latter on complex MRI:
      * it scores the MAGNITUDE image, not the 2-channel real/imag one -- phase error is
        not counted, and averaging over 2 channels halves the MSE (a free +3 dB);
      * data_range is per-image max|x_gt| rather than the fixed [0,1] range that
        to_01()'s shift-and-clamp implies.
    `psnr_dr`/`ssim_dr` are the columns to quote for MRI.
    """
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    to_mag = lambda t: (torch.view_as_complex(t.permute(1, 2, 0).contiguous().to(torch.float64))
                        .abs().unsqueeze(0) if t.shape[0] == 2 else t.to(torch.float64))
    mh, mg = to_mag(xh), to_mag(xg)
    dr = mg.max()
    mh = mh.clip(0, dr)
    mse = torch.mean((mh - mg) ** 2)
    psnr = 10.0 * torch.log10(dr ** 2 / (mse + 1e-20))
    s = ssim_fn(mh.unsqueeze(0).float(), mg.unsqueeze(0).float(), data_range=float(dr))
    return {"psnr_dr": psnr.item(), "ssim_dr": s.item()}


def robust_metrics(xh, xg, y, use_lpips=True):
    if use_lpips:
        try:
            return psnr_ssim(xh, xg, y), True
        except Exception as e:
            print(f"[eval] LPIPS unavailable ({type(e).__name__}); PSNR/SSIM only.")
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    xh1, xg1, y1 = to_01(xh), to_01(xg), to_01(y)
    psnr = 10.0 * torch.log10(1.0 / (torch.mean((xh1 - xg1) ** 2) + 1e-8))
    psnr_in = 10.0 * torch.log10(1.0 / (torch.mean((y1 - xg1) ** 2) + 1e-8))
    s = ssim(xh1.unsqueeze(0), xg1.unsqueeze(0), data_range=1.0)
    s_in = ssim(y1.unsqueeze(0), xg1.unsqueeze(0), data_range=1.0)
    return {"psnr": psnr.item(), "ssim": s.item(), "lpips": float("nan"),
            "psnr_in": psnr_in.item(), "ssim_in": s_in.item(),
            "lpips_in": float("nan")}, False


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "celeba.yaml"))
    ap.add_argument("--run_id", default=None, help="run dir under <output_root>/<stage>/train")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (overrides run_id)")
    ap.add_argument("--stage", default=None, help="override experiment.stage (problem)")
    ap.add_argument("--n", type=int, default=100, help="number of test images")
    ap.add_argument("--k", type=int, default=1, help="SNaP steps (1 = one-step)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    if args.stage:
        cfg["experiment"]["stage"] = args.stage
    stage = cfg["experiment"]["stage"]

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        if not args.run_id:
            raise ValueError("Provide --ckpt or --run_id.")
        run_dir = os.path.join(cfg["paths"]["output_root"], stage, "train", args.run_id)
        ckpt_path = find_latest_ckpt(run_dir)
    epoch_tag = os.path.splitext(os.path.basename(ckpt_path))[0].split("_")[-1]

    rid = args.run_id or os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    out_dir = args.out or os.path.join(
        cfg["paths"]["output_root"], stage, "test", f"{rid}_ep{epoch_tag}_k{args.k}")
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = build_net(cfg, device)
    use_ema = (not args.no_ema) and (ckpt.get("ema_model") is not None)
    net.load_state_dict(ckpt["ema_model"] if use_ema else ckpt["model"])
    net.eval()
    print(f"[eval] stage={stage} ckpt={ckpt_path} (epoch {ckpt.get('epoch','?')}) | "
          f"weights={'EMA' if use_ema else 'model'} | k={args.k} | n={args.n}")

    problem = build_problem(cfg)
    pipe = build_pipeline(cfg, net, problem, device)
    # Guard: a checkpoint trained under a different conditioning statistic (esp.
    # cond_anchor A^T y vs m_y) would receive OOD conditioning here and silently degrade.
    verify_ckpt_metadata(ckpt_meta(ckpt), pipe.metadata(), source=ckpt_path)

    cfg["data"]["max_len"] = args.n
    ds = _build_dataset(cfg, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)

    # `recons/*.pt` exists for MRI only: magnitude PNGs cannot be un-rendered back to
    # complex, and re-running the sampler over a whole split to fix a rendering choice is
    # expensive. RGB loses nothing to 8-bit PNG, so image-domain runs skip them.
    rec_dir = os.path.join(out_dir, "recons")
    single_dirs = {k: os.path.join(img_dir, k) for k in ("gt", "obs", "recon")}
    torch.manual_seed(args.seed)
    records = []
    lpips_ok = True
    idx = 0
    for batch in loader:
        # MRI loaders yield {"target", "maps"}: the sensitivity maps ARE part of the
        # forward operator and must be installed before sampling. Image datasets yield a
        # plain tensor.
        if isinstance(batch, dict):
            x_gt = batch["target"].to(device, non_blocking=True)
            maps = batch.get("maps", None)
            if maps is not None and hasattr(pipe.problem, "set_maps"):
                pipe.problem.set_maps(maps.to(device, non_blocking=True))
        else:
            x_gt = batch.to(device, non_blocking=True)
        out = pipe.sample(x_gt, steps=args.k)
        x_hat, y = out["x_hat"], out["y"]
        for b in range(x_gt.shape[0]):
            met, lpips_ok = robust_metrics(x_hat[b], x_gt[b], y[b], use_lpips=lpips_ok)
            met.update(dynamic_range_metrics(x_hat[b], x_gt[b]))
            met.update({k + "_in": v for k, v in
                        dynamic_range_metrics(y[b], x_gt[b]).items()})
            met["index"] = idx
            records.append(met)
            # MRI is complex -> render magnitudes, windowed to the GT max (see _mri_panels).
            is_mri = (x_gt[b].shape[0] == 2)
            trip = _mri_panels(x_gt[b], y[b], x_hat[b]) if is_mri else \
                   torch.stack([to_01(x_gt[b]), to_01(y[b]), to_01(x_hat[b])], 0)
            # The three panels as separate files, for figures and for anything that wants
            # a directory of images (e.g. FID) rather than a strip.
            for name, panel in zip(("gt", "obs", "recon"), trip):
                d = single_dirs[name]
                os.makedirs(d, exist_ok=True)
                vutils.save_image(panel, os.path.join(d, f"{idx:04d}.png"))
            if is_mri:
                os.makedirs(rec_dir, exist_ok=True)
                torch.save({"recon": x_hat[b].cpu(), "target": x_gt[b].cpu(),
                            "observation": y[b].cpu()},
                           os.path.join(rec_dir, f"result_{idx:04d}.pt"))
            idx += 1

    write_metrics_csv(records, os.path.join(out_dir, "test_metrics.csv"))

    import numpy as np
    def mean(k): return float(np.mean([r[k] for r in records]))
    print(f"\n[eval] N={len(records)} {args.split} images | stage={stage}")
    if all("psnr_dr" in r for r in records):
        print(f"  [magnitude / dynamic-range metric]")
        print(f"  input : PSNR {mean('psnr_dr_in'):.2f}  SSIM {mean('ssim_dr_in'):.3f}")
        print(f"  recon : PSNR {mean('psnr_dr'):.2f}  SSIM {mean('ssim_dr'):.3f}")
    print(f"  input : PSNR {mean('psnr_in'):.2f}  SSIM {mean('ssim_in'):.3f}"
          + ("" if not lpips_ok else f"  LPIPS {mean('lpips_in'):.3f}"))
    print(f"  recon : PSNR {mean('psnr'):.2f}  SSIM {mean('ssim'):.3f}"
          + ("" if not lpips_ok else f"  LPIPS {mean('lpips'):.3f}"))
    print(f"\n[eval] wrote: {os.path.join(out_dir, 'test_metrics.csv')}")


if __name__ == "__main__":
    main()
