"""Sample-averaged (posterior-mean) SNaP evaluation for multi-coil MRI.

    CUDA_VISIBLE_DEVICES=0 python eval_avg_mri.py \
        --config <run_dir>/config_used.yaml \
        --ckpt   <run_dir>/checkpoints/snap_best.pt \
        --n 100 --split test --M 1 4 16 --out <dir>

WHAT IS AVERAGED, AND WHY IT MATTERS. For each slice the measurement y is drawn ONCE and
held fixed; only the SOURCE randomness (x_p, n' in GaussianPosteriorSource.sample) is
resampled. Averaging those draws estimates E[x | y] -- the posterior mean.

Averaging `pipe.sample()` calls instead would be WRONG: sample() calls make_observation()
internally, so every call draws a fresh noise realisation. Averaging over those averages
away the measurement noise itself, which is a strictly easier problem and inflates PSNR
for reasons that have nothing to do with the model.

M is evaluated as PREFIXES of one pool of draws, so the curve is monotone by construction
and M=4 literally reuses M=1's sample plus three more -- no seed confounding across M.

Metrics are the MRI convention: magnitude, per slice, data_range = that slice's max.
"""
from __future__ import annotations
import argparse, csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch
import torchvision.utils as vutils
from core.config import ckpt_meta, load_config
from dataset.build import _build_dataset
from methods.base import Obs
from pipeline import verify_ckpt_metadata
from eval import (build_problem, build_net, build_pipeline, dynamic_range_metrics,
                  _mri_panels)


def draw_and_average(pipe, x_gt_1, maps_1, M_list, src_gen, net_batch=8):
    """x_gt_1/maps_1: (1,...). Returns {M: x_hat_avg (1,2,H,W)} plus the fixed obs."""
    problem = pipe.problem
    problem.sigma = pipe._sigma_base
    N = max(M_list)
    problem.set_maps(maps_1)
    obs_1 = problem.make_observation(x_gt_1)        # y drawn ONCE, fixed from here on
    cond_1 = pipe._build_cond(obs_1)

    # Draws are accumulated in chunks, never materialised together: at M=100 the source
    # solve alone holds 100 x (coils, 320, 320) complex k-space, which OOMs an 11 GB card.
    # Only a running SUM survives each chunk, so peak memory is set by net_batch, not N.
    # M values are still exact prefixes of ONE pool -- the running sum is snapshotted the
    # moment the draw count reaches each M -- so the curve stays monotone by construction
    # and M=4 is still M=1 plus three more. (The draws themselves differ from a
    # single-batch run: sampling b at a time consumes the generator differently. Same
    # distribution, and every M sees the same pool, which is what the comparison needs.)
    want = sorted(set(M_list))
    out, run, seen = {}, None, 0
    while seen < N:
        b = min(net_batch, N - seen)
        # batched obs sharing the SAME y; maps/mask/sigma must be expanded to match
        aux = {"maps": obs_1.aux["maps"].expand(b, *obs_1.aux["maps"].shape[1:]).contiguous(),
               "mask": obs_1.aux["mask"]}
        if "sigma" in obs_1.aux:
            sg = obs_1.aux["sigma"]
            aux["sigma"] = sg.expand(b) if sg.ndim else sg.repeat(b)
        obs_b = Obs(y=obs_1.y.expand(b, *obs_1.y.shape[1:]).contiguous(), aux=aux)
        problem.set_maps(aux["maps"])

        x1 = pipe.source.sample(problem, obs_b, generator=src_gen, return_anchor=False)
        sig = pipe._sigma_vec(b, x_gt_1.device, obs_b)
        d = pipe._one_step(x1, cond_1.expand(b, *cond_1.shape[1:]), sig)   # (b,2,H,W)
        del x1

        for j in range(b):
            one = d[j:j + 1]
            run = one.clone() if run is None else run + one
            seen += 1
            if seen in want:
                out[seen] = run / seen
        del d
    problem.set_maps(maps_1)                         # restore single-slice maps
    return out, obs_1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100, help="first n slices of the split")
    ap.add_argument("--ids", default=None,
                    help="optional fixed subset: a file of comma/newline-separated "
                         "dataset indices, or an inline comma list. Overrides --n.")
    ap.add_argument("--M", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--net_batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M_list = sorted(set(a.M))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    cfg = load_config(a.config)
    net = build_net(cfg, dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    use_ema = ck.get("ema_model") is not None
    net.load_state_dict(ck["ema_model"] if use_ema else ck["model"]); net.eval()
    problem = build_problem(cfg)                     # build_problem pins random_mask=False
    pipe = build_pipeline(cfg, net, problem, dev)
    verify_ckpt_metadata(ckpt_meta(ck), pipe.metadata(), source=a.ckpt)

    ds = _build_dataset(cfg, split=a.split)
    if a.ids:
        raw = open(a.ids).read() if os.path.exists(a.ids) else a.ids
        ids = [int(t) for t in raw.replace("\n", ",").split(",") if t.strip()]
    else:
        ids = list(range(min(a.n, len(ds))))
    ds = torch.utils.data.Subset(ds, ids)
    print(f"[avg] {len(ds)} {a.split} slices | M = {M_list} | weights={'EMA' if use_ema else 'model'}")

    m_dirs = {M: os.path.join(a.out, f"k{M}") for M in M_list}
    for md in m_dirs.values():
        os.makedirs(os.path.join(md, "images"), exist_ok=True)
        os.makedirs(os.path.join(md, "recons"), exist_ok=True)

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            xg = b["target"].unsqueeze(0).to(dev); mp = b["maps"].unsqueeze(0).to(dev)
            gen = torch.Generator(device=dev).manual_seed(a.seed * 100003 + i)
            avgs, obs1 = draw_and_average(pipe, xg, mp, M_list, gen, a.net_batch)
            zf = pipe.problem.adjoint(obs1.y, obs1)          # zero-filled, for the figure
            r = {"index": i}
            d_in = dynamic_range_metrics(zf[0], xg[0])
            r["psnr_in"] = d_in["psnr_dr"]; r["ssim_in"] = d_in["ssim_dr"]
            for M in M_list:
                d = dynamic_range_metrics(avgs[M][0], xg[0])
                r[f"psnr_M{M}"] = d["psnr_dr"]; r[f"ssim_M{M}"] = d["ssim_dr"]
            rows.append(r)
            # Same layout as the image-domain sweep (eval_avg.py): <out>/k<M>/ holding
            # images/NNNN_{gt,obs,output}.png. The three panels are sliced from ONE
            # _mri_panels() call so all three share that slice's windowing -- magnitudes
            # windowed independently would not be comparable side by side.
            # recons/*.pt stays for MRI only: these are complex, and a magnitude PNG
            # cannot be un-rendered back to them if the rendering is ever fixed.
            for M in M_list:
                md = m_dirs[M]
                torch.save({"recon": avgs[M][0].cpu(), "target": xg[0].cpu()},
                           os.path.join(md, "recons", f"result_{i:04d}.pt"))
                gt_p, obs_p, out_p = _mri_panels(xg[0], zf[0], avgs[M][0])
                idir = os.path.join(md, "images")
                vutils.save_image(gt_p, os.path.join(idir, f"{i:04d}_gt.png"))
                vutils.save_image(obs_p, os.path.join(idir, f"{i:04d}_obs.png"))
                vutils.save_image(out_p, os.path.join(idir, f"{i:04d}_output.png"))
            if (i + 1) % 20 == 0: print(f"  {i+1}/{len(ds)}", flush=True)

    with open(os.path.join(a.out, "avg_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ...and one per M. Columns are the psnr_dr/ssim_dr the MRI convention requires;
    # there is deliberately no plain `psnr` column, so a row from here can never be read
    # on the natural-image axis by mistake.
    for M in M_list:
        sub = [{"psnr_dr": r[f"psnr_M{M}"], "ssim_dr": r[f"ssim_M{M}"],
                "psnr_dr_in": r["psnr_in"], "ssim_dr_in": r["ssim_in"],
                "index": r["index"]} for r in rows]
        cols = ["psnr_dr", "ssim_dr", "psnr_dr_in", "ssim_dr_in", "index"]
        avg = {c: float(np.mean([q[c] for q in sub])) for c in cols if c != "index"}
        avg["index"] = "avg"
        with open(os.path.join(m_dirs[M], "test_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(sub); w.writerow(avg)
    print(f"\n{'M':>4} {'PSNR':>9} {'sd':>7} {'SSIM':>9}")
    for M in M_list:
        p = np.array([r[f"psnr_M{M}"] for r in rows]); s = np.array([r[f"ssim_M{M}"] for r in rows])
        print(f"{M:>4} {p.mean():9.3f} {p.std():7.3f} {s.mean():9.4f}")
    print(f"\nzero-filled input: {np.mean([r['psnr_in'] for r in rows]):.3f} dB "
          f"psnr_dr / {np.mean([r['ssim_in'] for r in rows]):.4f} ssim_dr")
    print(f"[avg] wrote {a.out}/avg_metrics.csv and "
          f"{'/'.join('k%d' % M for M in M_list)}/{{test_metrics.csv, images/, recons/}}")


if __name__ == "__main__":
    main()
