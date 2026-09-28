import torch
import torch.nn.functional as F
import os
import csv
import numpy as np

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

def to_01(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5

# Lazy-loaded LPIPS model (AlexNet backbone, loaded once per process)
_lpips_model = None

def _get_lpips(device: torch.device):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False)
    return _lpips_model.to(device)


@torch.no_grad()
def mri_magnitude(x: torch.Tensor) -> torch.Tensor:
    """(B,2,H,W) or (2,H,W) real/imag -> (B,1,H,W) complex magnitude |x|."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous()).abs().unsqueeze(1)


@torch.no_grad()
def mri_psnr_ssim(x_hat: torch.Tensor, x_gt: torch.Tensor) -> dict:
    """Dynamic-range PSNR/SSIM on the complex MAGNITUDE, matching the baselines'
    the dynamic-range PSNR/SSIM convention used by CS-MRI benchmarks:
      - collapse 2-ch (real,imag) -> magnitude |.|
      - per-image data_range = target.max(); prediction clipped to [0, data_range]
      - PSNR = 10 log10(data_range^2 / MSE)   (identical to piq.psnr used by baselines)
    x_hat, x_gt: (B,2,H,W) or (2,H,W)."""
    yh = mri_magnitude(x_hat).double()
    yg = mri_magnitude(x_gt).double()
    psnrs, ssims = [], []
    for a, b in zip(yh, yg):
        dr = b.max().clamp_min(1e-12)
        a1 = a.clamp(0, dr).unsqueeze(0)
        b1 = b.unsqueeze(0)
        mse = ((a1 - b1) ** 2).mean()
        psnrs.append(10.0 * torch.log10(dr ** 2 / (mse + 1e-12)))
        ssims.append(ssim(a1.float(), b1.float(), data_range=float(dr)))
    return {"psnr": float(torch.stack(psnrs).mean()),
            "ssim": float(torch.stack(ssims).mean())}


def psnr_function(x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
    """x_hat, x_gt in [-1,1], shape (B,C,H,W). Returns mean PSNR."""
    x_hat = to_01(x_hat)
    x_gt  = to_01(x_gt)
    mse   = torch.mean((x_hat - x_gt) ** 2, dim=[1, 2, 3])
    return (10.0 * torch.log10(1.0 / (mse + 1e-8))).mean()


@torch.no_grad()
def psnr_ssim(x_hat: torch.Tensor, x_gt: torch.Tensor, y: torch.Tensor) -> dict:
    """
    x_hat, x_gt, y: single-image tensors in [-1,1], shape (C,H,W).
    Returns dict with psnr, ssim, lpips (reconstruction) and
    psnr_in, ssim_in, lpips_in (degraded input).
    LPIPS: lower is better.
    """
    device = x_gt.device
    lpips_fn = _get_lpips(device)

    # upsample y to x_gt resolution if sizes differ (e.g. SR task)
    if y.shape[-2:] != x_gt.shape[-2:]:
        y = F.interpolate(y.unsqueeze(0), size=x_gt.shape[-2:], mode="bilinear",
                          align_corners=False).squeeze(0)

    # [0,1] versions for PSNR / SSIM
    xh  = to_01(x_hat)
    xg  = to_01(x_gt)
    yi  = to_01(y)

    # PSNR
    psnr    = 10.0 * torch.log10(1.0 / (torch.mean((xh - xg) ** 2) + 1e-8))
    psnr_in = 10.0 * torch.log10(1.0 / (torch.mean((yi - xg) ** 2) + 1e-8))

    # SSIM  (torchmetrics needs a batch dim)
    ssim_val = ssim(xh.unsqueeze(0), xg.unsqueeze(0), data_range=1.0, reduction="none")
    ssim_in  = ssim(yi.unsqueeze(0), xg.unsqueeze(0), data_range=1.0, reduction="none")

    # LPIPS (expects [-1,1], batch dim)
    lp     = lpips_fn(x_hat.unsqueeze(0).clamp(-1, 1),
                      x_gt.unsqueeze(0).clamp(-1, 1))
    lp_in  = lpips_fn(y.unsqueeze(0).clamp(-1, 1),
                      x_gt.unsqueeze(0).clamp(-1, 1))

    return {
        "psnr":     psnr.item(),
        "ssim":     ssim_val.item(),
        "lpips":    lp.item(),
        "psnr_in":  psnr_in.item(),
        "ssim_in":  ssim_in.item(),
        "lpips_in": lp_in.item(),
    }

def save_metric(pred, gt, y, save_root, records, global_offset=0):
    os.makedirs(save_root, exist_ok=True)
    B = gt.shape[0]
    for b in range(B):
        metrics = psnr_ssim(pred[b], gt[b], y[b])
        metrics["index"] = global_offset + b
        records.append(metrics)
    return records


def write_metrics_csv(records, csv_path):
    if not records:
        return
    # Average EVERY numeric column the records carry, not a hardcoded list: the
    # dynamic-range metrics (psnr_dr/ssim_dr/..) were added to the per-image records
    # later, and a fixed list left them blank in the avg row -- silently dropping the
    # summary of the only convention MRI is scored on.
    avg = {"index": "avg"}
    for k in records[0]:
        if k == "index":
            continue
        vals = [r[k] for r in records if isinstance(r.get(k), (int, float))]
        avg[k] = float(np.mean(vals)) if len(vals) == len(records) else ""
    rows = records + [avg]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
