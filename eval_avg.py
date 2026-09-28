"""Sample-averaged (posterior-mean) SNaP evaluation -- generic, non-MRI.

    CUDA_VISIBLE_DEVICES=0 python eval_avg.py \
        --config <run_dir>/config_used.yaml \
        --ckpt   <run_dir>/checkpoints/snap_epoch_0040.pt \
        --n 100 --split test --M 1 4 16 --out <dir>

Counterpart of eval_avg_mri.py for image-domain problems (deblur / inpainting / SR /
denoising). Same protocol, and the same trap to avoid:

WHAT IS AVERAGED. For each image the measurement y is drawn ONCE and held fixed; only the
SOURCE randomness is resampled across the M draws. Averaging those estimates E[x|y], the
posterior mean. Averaging repeated `pipe.sample()` calls instead would be WRONG -- sample()
calls make_observation() internally, so every call redraws the measurement noise, and
averaging over that averages away the noise itself. That is a strictly easier problem and
inflates PSNR for reasons unrelated to the model.

M is evaluated as PREFIXES of one pool of draws, so the curve is monotone by construction
and M=4 literally reuses M=1's draw plus three more -- no seed confounding across M.

Also reports the inter-draw spread ||x_i - xbar|| / ||xbar||: single-draw PSNR rewards a
collapsed sampler, so the spread is what says whether the model is sampling or regressing.

Metrics on [0,1] with data_range 1 (RGB): PSNR and SSIM per image, LPIPS(alex) on [-1,1].

OUTPUT LAYOUT. One self-contained directory per M, so each averaging level can be read,
scored and turned into a figure on its own:

    <out>/avg_metrics.csv          every M side by side, one row per image (the curve)
    <out>/k1/test_metrics.csv      that M alone, same columns as eval.py's
    <out>/k1/images/0000_gt.png    ground truth
                   0000_obs.png    degraded input (A^H y, the image-domain reference)
                   0000_output.png the M-averaged reconstruction
    <out>/k4/ ... k16/ ... k100/

gt and obs are identical across the M dirs by construction -- y is drawn once per image
and held fixed -- and are duplicated so each dir stands alone.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torchvision.utils as vutils

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core.config import ckpt_meta, load_config            # noqa: E402
from dataset.build import _build_dataset                  # noqa: E402
from methods.base import Obs                              # noqa: E402
from pipeline import verify_ckpt_metadata                 # noqa: E402
from eval import build_net, build_problem, build_pipeline  # noqa: E402


def to01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100, help="first n images of the split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--M", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--net_batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1,
                    help="SNaP steps k along the trajectory (1 = the one-step jump "
                         "r=0,t=1). k>1 walks t=1 -> 0 on a uniform grid via _k_step. "
                         "TOTAL NFE per image is steps * M, so a k-step M-draw eval costs "
                         "k times the k=1 eval at the same M.")
    ap.add_argument("--save_imgs", type=int, default=-1,
                    help="how many images to write PNGs for; -1 (default) = all of them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_ema", action="store_true",
                    help="score the raw training weights instead of the EMA (same flag as "
                         "eval.py). Matters mid-run: a long EMA horizon leaves the shadow "
                         "weights well behind the raw ones for many epochs.")
    a = ap.parse_args()
    M_list = sorted(set(a.M))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    cfg = load_config(a.config)
    net = build_net(cfg, dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    use_ema = (ck.get("ema_model") is not None) and not a.no_ema
    net.load_state_dict(ck["ema_model"] if use_ema else ck["model"])
    net.eval()
    problem = build_problem(cfg)
    pipe = build_pipeline(cfg, net, problem, dev)
    # raises on source_type / cond_anchor / cond_coverage mismatch
    verify_ckpt_metadata(ckpt_meta(ck), pipe.metadata(), source=a.ckpt)

    cfg["data"]["max_len"] = a.n
    ds = _build_dataset(cfg, split=a.split)
    print(f"[avg] {len(ds)} {a.split} images | M = {M_list} | "
          f"weights={'EMA' if use_ema else 'model'} | epoch {ck.get('epoch')}")
    print(f"[avg] source={pipe.source_type} | tau={pipe.tau} | sigma_n={problem.sigma}")

    import lpips as _l
    lp = _l.LPIPS(net="alex").to(dev).eval()
    for q in lp.parameters():
        q.requires_grad_(False)
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

    # one directory per M: <out>/k<M>/{test_metrics.csv, images/}
    m_dirs = {M: os.path.join(a.out, f"k{M}") for M in M_list}
    for M, d in m_dirs.items():
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
    n_imgs = len(ds) if a.save_imgs < 0 else a.save_imgs

    rows = []
    N = max(M_list)
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            xg = (b["target"] if isinstance(b, dict) else b).unsqueeze(0).to(dev)
            gen = torch.Generator(device=dev).manual_seed(a.seed * 100003 + i)
            # make_observation() draws its noise from the GLOBAL rng, not `gen`, so seed
            # that too -- otherwise every checkpoint is scored on a different y.
            # The offset is ESSENTIAL: seeding the global stream with the SAME value as
            # `gen` makes n'[0] bit-identical to the noise added to y (same seed, same
            # generator), so draw 0 cancels the measurement noise exactly and scores ~1.7x
            # better than draws 1..15 -- silently inflating M=1 by ~2.5 dB.
            _oseed = a.seed * 100003 + i + 777767
            torch.manual_seed(_oseed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_oseed)
            problem.sigma = pipe._sigma_base
            obs1 = problem.make_observation(xg)          # y drawn ONCE, fixed from here on
            cond1 = pipe._build_cond(obs1)
            obsN = Obs(y=obs1.y.expand(N, *obs1.y.shape[1:]).contiguous(),
                       aux=dict(obs1.aux))
            x1 = pipe.source.sample(problem, obsN, generator=gen)
            cond = cond1.expand(N, *cond1.shape[1:])
            sig = pipe._sigma_vec(N, dev, obsN)
            # k=1 routes to _one_step so the default path stays bit-identical to
            # before this flag existed (_k_step(k=1) is algebraically the same thing, but
            # goes through _u_fn's t_eps clamp -- keep the exact old call for k=1).
            def _run(sl_z, sl_c, sl_s):
                return (pipe._one_step(sl_z, sl_c, sl_s) if a.steps <= 1
                        else pipe._k_step(sl_z, sl_c, sl_s, int(a.steps)))
            outs = [_run(x1[s:s + a.net_batch], cond[s:s + a.net_batch],
                         sig[s:s + a.net_batch])
                    for s in range(0, N, a.net_batch)]
            draws = torch.cat(outs, 0)                   # (N,C,H,W)

            r = {"index": i}
            g01 = to01(xg)
            for M in M_list:
                xb = draws[:M].mean(0, keepdim=True)
                p01 = to01(xb)
                mse = torch.mean((p01 - g01) ** 2)
                r[f"psnr_M{M}"] = float(10.0 * torch.log10(1.0 / (mse + 1e-12)))
                r[f"ssim_M{M}"] = float(ssim_fn(p01, g01, data_range=1.0))
                r[f"lpips_M{M}"] = float(lp(xb.clamp(-1, 1), xg.clamp(-1, 1)))
                if M == max(M_list):
                    r["spread"] = float((draws - xb).pow(2).mean().sqrt()
                                        / xb.pow(2).mean().sqrt()) * 100.0
            # zero-filled / degraded reference for the panel. Use A^T y, NOT y: for SR
            # the measurement lives on the LR grid (64x64 at sf=2) and does not even have
            # the target's shape, so comparing y to the ground truth is a shape error.
            # A^T y is the image-domain degraded reference for every operator, and matches
            # what eval.py reports as `psnr_in`.
            y_img = problem.adjoint(obs1.y, obs1)
            r["psnr_in"] = float(10.0 * torch.log10(
                1.0 / (torch.mean((to01(y_img) - g01) ** 2) + 1e-12)))
            rows.append(r)
            if i < n_imgs:
                # gt/obs repeated per M dir so each one is self-contained; `output` is
                # that M's own average, which is the only panel that differs between them.
                for M in M_list:
                    idir = os.path.join(m_dirs[M], "images")
                    vutils.save_image(g01, os.path.join(idir, f"{i:04d}_gt.png"))
                    vutils.save_image(to01(y_img[:1]), os.path.join(idir, f"{i:04d}_obs.png"))
                    vutils.save_image(to01(draws[:M].mean(0, keepdim=True)),
                                      os.path.join(idir, f"{i:04d}_output.png"))
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(ds)}", flush=True)

    with open(os.path.join(a.out, "avg_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ...and one per M, with eval.py's column names and its trailing `avg` row, so the
    # two scripts' outputs are interchangeable downstream (plotting, aggregation).
    for M in M_list:
        sub = [{"psnr": r[f"psnr_M{M}"], "ssim": r[f"ssim_M{M}"],
                "lpips": r[f"lpips_M{M}"], "psnr_in": r["psnr_in"],
                "index": r["index"]} for r in rows]
        cols = ["psnr", "ssim", "lpips", "psnr_in", "index"]
        avg = {c: float(np.mean([r[c] for r in sub])) for c in cols if c != "index"}
        avg["index"] = "avg"
        with open(os.path.join(m_dirs[M], "test_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(sub)
            w.writerow(avg)

    print(f"\ndegraded input PSNR: {np.mean([r['psnr_in'] for r in rows]):.3f} dB\n")
    print(f"{'M':>4} {'PSNR':>9} {'sd':>7} {'SSIM':>8} {'LPIPS':>8}")
    for M in M_list:
        p = np.array([r[f"psnr_M{M}"] for r in rows])
        s = np.array([r[f"ssim_M{M}"] for r in rows])
        l = np.array([r[f"lpips_M{M}"] for r in rows])
        print(f"{M:>4} {p.mean():9.3f} {p.std():7.3f} {s.mean():8.4f} {l.mean():8.4f}")
    sp = np.array([r["spread"] for r in rows])
    print(f"\ninter-draw spread (N={N}): {sp.mean():.2f}% +- {sp.std():.2f}")
    print(f"[avg] wrote {a.out}/avg_metrics.csv and "
          f"{'/'.join('k%d' % M for M in M_list)}/{{test_metrics.csv, images/}}")


if __name__ == "__main__":
    main()
