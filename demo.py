"""Run a pretrained SNaP model on a few images and look at the result.

    python demo.py --list                        # what you can run
    python demo.py --model celeba_sr             # 4 test images, one-step
    python demo.py --model celeba_sr --M 16      # average 16 posterior draws per image
    python demo.py --model afhq_inpaint --images my_photos/ --n 2
    python demo.py --model celeba_deblur --device cpu

Writes `outputs/demo_<model>/` with, per image, the ground truth, the degraded
measurement A^T y, and the reconstruction, plus a `grid.png` comparing them side by
side, and prints PSNR/SSIM/LPIPS against the numbers in the paper.

Needs: the checkpoint (scripts/download_checkpoints.py) and, unless you pass --images,
the dataset (scripts/download_data.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch                                            # noqa: E402
import torchvision.transforms as T                      # noqa: E402
import torchvision.utils as vutils                      # noqa: E402
from PIL import Image                                   # noqa: E402

from core.config import ckpt_meta, load_config          # noqa: E402
from eval import build_net, build_problem, build_pipeline   # noqa: E402
from pipeline import verify_ckpt_metadata               # noqa: E402

MANIFEST = os.path.join(HERE, "checkpoints", "manifest.json")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def to01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def load_images(cfg, args, n):
    """n images as one (n,C,H,W) tensor in [-1,1], with the config's own preprocessing."""
    dcfg = cfg["data"]
    if args.images:
        files = sorted(f for f in os.listdir(args.images)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)[:n]
        if not files:
            raise SystemExit(f"no images found in {args.images}")
        paths = [os.path.join(args.images, f) for f in files]
    else:
        split_dir = os.path.join(dcfg["root"], args.split)
        if not os.path.isdir(split_dir):
            raise SystemExit(
                f"dataset split not found: {split_dir}\n"
                f"  get it with:  python scripts/download_data.py "
                f"{'afhq' if 'afhq' in dcfg['root'] else 'celeba'}\n"
                f"  or point the demo at your own pictures:  --images <folder>")
        files = sorted(os.listdir(split_dir))[:n]
        paths = [os.path.join(split_dir, f) for f in files]
    tf = []
    if dcfg.get("center_crop", False):
        tf.append(T.CenterCrop(dcfg.get("crop_size", 178)))
    tf += [T.Resize((dcfg["size"], dcfg["size"]),
                    interpolation=T.InterpolationMode.BILINEAR, antialias=True),
           T.ToTensor(), T.Lambda(lambda t: t * 2.0 - 1.0)]
    tf = T.Compose(tf)
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]), files


def metrics(x_hat, x_gt, lpips_fn):
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    a, b = to01(x_hat), to01(x_gt)
    mse = torch.mean((a - b) ** 2)
    out = {"psnr": float(10.0 * torch.log10(1.0 / (mse + 1e-12))),
           "ssim": float(ssim(a, b, data_range=1.0))}
    out["lpips"] = (float(lpips_fn(x_hat.clamp(-1, 1), x_gt.clamp(-1, 1)))
                    if lpips_fn is not None else float("nan"))
    return out


def label_grid(path, titles, cell, pad=2):
    """Write column titles above an already-saved grid; silently skip if PIL has no font."""
    try:
        from PIL import ImageDraw
        img = Image.open(path)
        band = 18
        out = Image.new("RGB", (img.width, img.height + band), "white")
        out.paste(img, (0, band))
        d = ImageDraw.Draw(out)
        for i, t in enumerate(titles):
            d.text((pad + i * (cell + 2 * pad) + 4, 4), t, fill="black")
        out.save(path)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="a name from checkpoints/manifest.json")
    ap.add_argument("--list", action="store_true", help="list the available models and exit")
    ap.add_argument("--n", type=int, default=4, help="how many images")
    ap.add_argument("--M", type=int, default=1, help="posterior draws to average per image")
    ap.add_argument("--k", type=int, default=1, help="SNaP steps (1 = one-step)")
    ap.add_argument("--images", default=None, help="use this folder of images instead of the test split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-lpips", action="store_true", help="skip LPIPS (avoids the extra download)")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))["models"]
    if args.list or not args.model:
        print("available models (checkpoints/manifest.json):\n")
        for k, e in manifest.items():
            print(f"  {k:16s} {e['description']}")
        print("\n  python demo.py --model celeba_sr")
        return 0
    if args.model not in manifest:
        raise SystemExit(f"unknown model {args.model!r}; available: {', '.join(manifest)}")
    entry = manifest[args.model]

    if entry["stage"] == "cs_mri":
        raise SystemExit(
            "The MRI model needs credentialed fastMRI data, which cannot be downloaded\n"
            "automatically -- see docs/DATA.md. Once the data is in place, evaluate it with\n"
            f"  python eval_avg_mri.py --config {entry['config']} \\\n"
            f"      --ckpt checkpoints/{entry['file']} \\\n"
            f"      --ids data_splits/brain_test100.txt --M 1 16 --out outputs/mri")

    ckpt = os.path.join(HERE, "checkpoints", entry["file"])
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}\n"
                         f"  get it with:  python scripts/download_checkpoints.py {args.model}")

    cfg = load_config(os.path.join(HERE, entry["config"]))
    device = torch.device(args.device)
    print(f"[demo] {args.model}: {entry['description']}")
    print(f"[demo] device={device} | M={args.M} draw(s) | k={args.k} step(s)")

    net = build_net(cfg, device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    use_ema = ck.get("ema_model") is not None
    net.load_state_dict(ck["ema_model"] if use_ema else ck["model"])
    net.eval()
    problem = build_problem(cfg)
    pipe = build_pipeline(cfg, net, problem, device)
    verify_ckpt_metadata(ckpt_meta(ck), pipe.metadata(), source=ckpt)

    x_gt, names = load_images(cfg, args, args.n)
    x_gt = x_gt.to(device)
    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips as _l
            lpips_fn = _l.LPIPS(net="alex").to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
        except Exception as e:
            print(f"[demo] LPIPS unavailable ({type(e).__name__}); reporting PSNR/SSIM only")

    out_dir = args.out or os.path.join(HERE, "outputs", f"demo_{args.model}")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    rows, table = [], []
    with torch.no_grad():
        for i in range(x_gt.shape[0]):
            xg = x_gt[i:i + 1]
            gen = torch.Generator(device=device).manual_seed(args.seed * 1000 + i)
            # y is drawn ONCE and held fixed; only the source randomness is resampled
            # across the M draws, so their mean estimates E[x|y].
            obs = problem.make_observation(xg)
            cond = pipe._build_cond(obs)
            M = max(1, args.M)
            obsM = obs if M == 1 else type(obs)(
                y=obs.y.expand(M, *obs.y.shape[1:]).contiguous(), aux=dict(obs.aux))
            x1 = pipe.source.sample(problem, obsM, generator=gen)
            sig = pipe._sigma_vec(x1.shape[0], device, obsM)
            condM = cond.expand(x1.shape[0], *cond.shape[1:])
            draws = (pipe._one_step(x1, condM, sig) if args.k <= 1
                     else pipe._k_step(x1, condM, sig, args.k))
            x_hat = draws.mean(0, keepdim=True)
            y_img = pipe._display_y(obs, xg)

            m = metrics(x_hat, xg, lpips_fn)
            m_in = metrics(y_img, xg, lpips_fn)
            table.append((names[i], m_in, m))
            for tag, img in (("gt", xg), ("obs", y_img), ("recon", x_hat)):
                vutils.save_image(to01(img), os.path.join(out_dir, f"{i:04d}_{tag}.png"))
            rows += [to01(xg)[0], to01(y_img)[0], to01(x_hat)[0]]

    grid_path = os.path.join(out_dir, "grid.png")
    vutils.save_image(torch.stack(rows), grid_path, nrow=3, padding=2)
    label_grid(grid_path, ["ground truth", "measurement", "reconstruction"], x_gt.shape[-1])

    print(f"\n{'image':<24}{'PSNR in':>9}{'PSNR':>9}{'SSIM':>8}{'LPIPS':>9}")
    for name, mi, m in table:
        print(f"{name[:24]:<24}{mi['psnr']:9.2f}{m['psnr']:9.2f}{m['ssim']:8.4f}{m['lpips']:9.4f}")
    n = len(table)
    avg = {k: sum(m[k] for _, _, m in table) / n for k in ("psnr", "ssim", "lpips")}
    avg_in = sum(mi["psnr"] for _, mi, _ in table) / n
    print(f"{'mean of ' + str(n):<24}{avg_in:9.2f}{avg['psnr']:9.2f}{avg['ssim']:8.4f}{avg['lpips']:9.4f}")

    ref = entry.get("metrics", {}).get(f"M{args.M}")
    if ref and not args.images and args.k == 1:
        print(f"\npaper, {entry['metrics']['n']} test images, M={args.M}: "
              f"PSNR {ref[0]:.2f}  SSIM {ref[1]:.4f}" +
              (f"  LPIPS {ref[2]:.4f}" if ref[2] is not None else "") +
              f"\n(this demo used {n} image{'s' if n > 1 else ''}, so expect scatter of a dB or two)")
    print(f"\nwrote {out_dir}/  (grid.png + per-image PNGs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
