from __future__ import annotations
from pathlib import Path

import torch
import torchvision.utils as vutils


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5

def save_images_mri(*, x_gt, x_hat, y, save_dir, prefix):
    """Save MRI reconstructions as complex-MAGNITUDE grayscale, not as raw 2-channel
    tensors clamped to [-1,1]. Each panel is scaled by the GT's 99.5th-percentile
    magnitude so gt/output/obs share one intensity scale."""
    save_dir = Path(save_dir) / "imgs"
    save_dir.mkdir(parents=True, exist_ok=True)

    def _mag(t):
        if t is None:
            return None
        t = t[0:1].detach().float().cpu()
        if t.shape[1] == 2:  # (1,2,H,W) real/imag -> (1,1,H,W) |.|
            t = torch.view_as_complex(t.permute(0, 2, 3, 1).contiguous()).abs().unsqueeze(1)
        return t

    g, h, yy = _mag(x_gt), _mag(x_hat), _mag(y)
    vmax = max(float(g.quantile(0.995)), 1e-8) if g is not None else 1.0
    for t, suf in [(g, "gt"), (h, "output"), (yy, "obs")]:
        if t is None:
            continue
        vutils.save_image((t / vmax).clamp(0, 1), save_dir / f"{prefix}_{suf}.png",
                          nrow=1, padding=0, normalize=False)


def save_images(
    *,
    x_gt: torch.Tensor,
    x_hat: torch.Tensor,
    y: torch.Tensor,
    save_dir: str | Path,
    prefix: str,
):

    save_dir = Path(save_dir)/"imgs"
    save_dir.mkdir(parents=True, exist_ok=True)

    if x_gt is not None:
        x_gt0 = _to_01(x_gt[0:1])
        vutils.save_image(
            x_gt0,
            save_dir / f"{prefix}_gt.png",
            nrow=1,
            padding=0,
            normalize=False,
        )

    if x_hat is not None:
        x_hat0 = _to_01(x_hat[0:1])
        vutils.save_image(
            x_hat0,
            save_dir / f"{prefix}_output.png",
            nrow=1,
            padding=0,
            normalize=False,
        )


    if y is not None:
        y0 = _to_01(y[0:1])
        vutils.save_image(y0,save_dir / f"{prefix}_obs.png",nrow=1,padding=0,normalize=False, )
        vutils.save_image(
            y0,
            save_dir / f"{prefix}_obs.png",
            nrow=1,
            padding=0,
            normalize=False,
        )

