"""UNet variant for SNaP.

Subclasses the DDPM-style `UNet` in networks/unet/Unet.py and adds conditioning on the
second MeanFlow time `r` and the measurement noise level `sigma_n`, on top of the usual
time `t`. Each extra scalar gets its own sinusoidal+MLP embedding; the embeddings are
summed into the time embedding that the UNet body consumes.

Because the base `UNet.forward` inlines its body (no temb hook), the body is re-stated
here as `forward_with_temb`. This is the only duplicated logic; keep it in sync if the
base UNet's forward pass ever changes.

The net is an x-prediction network: it predicts the clean image x0 from the interpolant
z (concatenated with measurement conditioning) and (t, r, sigma_n). It must be JVP-safe
(forward-mode AD w.r.t. z, r, t) -- the base UNet's hand-written attention is (unlike
fused kernels), so this works.
"""

from __future__ import annotations

import torch
from torch import Tensor

from networks.unet.Unet import UNet, TimestepEmbedding


class SNaPUNet(UNet):
    def __init__(
        self,
        *args,
        cond_r: bool = True,
        cond_sigma: bool = True,
        sigma_emb_scale: float = 10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        temb_ch = self.ch * 4
        self.cond_r = cond_r
        self.cond_sigma = cond_sigma
        self.sigma_emb_scale = float(sigma_emb_scale)
        if cond_r:
            self.r_temb_net = TimestepEmbedding(
                embedding_dim=self.ch, hidden_dim=temb_ch, output_dim=temb_ch, act=self.act
            )
        if cond_sigma:
            self.s_temb_net = TimestepEmbedding(
                embedding_dim=self.ch, hidden_dim=temb_ch, output_dim=temb_ch, act=self.act
            )

    def forward(self, x: Tensor, t: Tensor, r: Tensor | None = None,
                sigma: Tensor | None = None) -> Tensor:
        """x: (B, Cin, H, W); t, r, sigma: (B,). Returns x0-prediction (B, Cout, H, W)."""
        temb = self.temb_net(t)
        if self.cond_r and r is not None:
            temb = temb + self.r_temb_net(r)
        if self.cond_sigma and sigma is not None:
            temb = temb + self.s_temb_net(sigma * self.sigma_emb_scale)
        return self.forward_with_temb(x, temb)

    # -- vendored copy of UNet.forward's body, parameterized by a precomputed temb --
    def forward_with_temb(self, x: Tensor, temb: Tensor) -> Tensor:
        hs = [self.begin_conv(x)]

        # Downsampling
        for i_level in range(self.num_resolutions):
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                h = block_modules[f"{i_level}a_{i_block}a_block"](hs[-1], temb)
                if h.size(2) in self.attn_resolutions:
                    h = block_modules[f"{i_level}a_{i_block}b_attn"](h, temb)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(block_modules[f"{i_level}b_downsample"](hs[-1]))

        # Middle
        h = hs[-1]
        h = self._compute_cond_module(self.mid_modules, h, temb)

        # Upsampling
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block_modules[f"{i_level}a_{i_block}a_block"](h, temb)
                if h.size(2) in self.attn_resolutions:
                    h = block_modules[f"{i_level}a_{i_block}b_attn"](h, temb)
            if i_level != 0:
                h = block_modules[f"{i_level}b_upsample"](h)

        return self.end_conv(h)
