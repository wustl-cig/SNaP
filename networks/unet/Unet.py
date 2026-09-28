import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.nn.utils.parametrizations import spectral_norm

class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x) * x


def group_norm(out_ch):
    return nn.GroupNorm(
        num_groups=32,
        num_channels=out_ch,
        eps=1e-6,
        affine=True,
    )

def _calculate_correct_fan(tensor, mode):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        return fan_in
    if mode == "fan_out":
        return fan_out
    if mode == "fan_avg":
        return (fan_in + fan_out) / 2
    raise ValueError(f"Unsupported mode: {mode}")


def kaiming_uniform_(tensor, gain=1.0, mode="fan_in"):
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1.0, fan)
    bound = math.sqrt(3.0 * var)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(
        tensor,
        gain=1e-10 if scale == 0 else scale,
        mode="fan_avg",
    )


def dense(in_channels, out_channels, init_scale=1.0, use_spectral_norm = False):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    if use_spectral_norm:
        lin = spectral_norm(lin)
    return lin


def conv2d(
    in_planes,
    out_planes,
    kernel_size=(3, 3),
    stride=1,
    dilation=1,
    padding=1,
    bias=True,
    padding_mode="zeros",
    init_scale=1.0,
    use_spectral_norm=False,
):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        padding_mode=padding_mode,
    )
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    if use_spectral_norm:
        conv = spectral_norm(conv)
    return conv


def get_sinusoidal_positional_embedding(timesteps, embedding_dim):
    if len(timesteps.size()) == 0:
        timesteps = timesteps.unsqueeze(0)

    assert len(timesteps.size()) == 1

    timesteps = timesteps.to(torch.get_default_dtype())
    device = timesteps.device

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float, device=device) * -emb
    )
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), "constant", 0)

    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=Swish()):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, t):
        temb = get_sinusoidal_positional_embedding(t, self.embedding_dim)
        return self.main(temb)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        temb_ch,
        out_ch=None,
        conv_shortcut=False,
        dropout=0.0,
        normalize=group_norm,
        act=Swish(),
        use_spectral_norm=False,
    ):
        super().__init__()

        self.out_ch = out_ch if out_ch is not None else in_ch
        self.act = act

        self.temb_proj = dense(temb_ch, self.out_ch)

        self.norm1 = normalize(in_ch)
        self.conv1 = conv2d(in_ch, self.out_ch, use_spectral_norm=use_spectral_norm)

        self.norm2 = normalize(self.out_ch)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv2d(self.out_ch, self.out_ch, init_scale=0.0, use_spectral_norm=use_spectral_norm)

        if in_ch != self.out_ch:
            if conv_shortcut:
                self.shortcut = conv2d(in_ch, self.out_ch, use_spectral_norm=use_spectral_norm)
            else:
                self.shortcut = conv2d(
                    in_ch,
                    self.out_ch,
                    kernel_size=(1, 1),
                    padding=0,
                    use_spectral_norm=use_spectral_norm,
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))

        h = h + self.temb_proj(self.act(temb))[:, :, None, None]

        h = self.conv2(self.dropout(self.act(self.norm2(h))))

        return self.shortcut(x) + h


class SelfAttention(nn.Module):
    def __init__(self, in_channels, normalize=group_norm, use_spectral_norm=False):
        super().__init__()

        self.norm = normalize(in_channels)

        self.attn_q = conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0,
            use_spectral_norm = use_spectral_norm,
        )
        self.attn_k = conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0,
            use_spectral_norm=use_spectral_norm
        )
        self.attn_v = conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0,
            use_spectral_norm=use_spectral_norm
        )

        self.proj_out = conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            init_scale=0.0,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x, temb=None):
        B, C, H, W = x.shape

        h = self.norm(x)

        q = self.attn_q(h).view(B, C, H * W)
        k = self.attn_k(h).view(B, C, H * W)
        v = self.attn_v(h).view(B, C, H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k) * (C ** -0.5)
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(B, C, H, W)
        h = self.proj_out(h)

        return x + h


def upsample(in_ch, with_conv=True):
    layers = [nn.Upsample(scale_factor=2, mode="nearest")]
    if with_conv:
        layers.append(conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=1))
    return nn.Sequential(*layers)


def downsample(in_ch, with_conv=True):
    if with_conv:
        return conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=2)
    return nn.AvgPool2d(2, 2)


class UNet(nn.Module):
    def __init__(
        self,
        input_channels,
        input_height,
        ch=32,
        output_channels=3,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=6,
        attn_resolutions=(16, 8),
        dropout=0.0,
        resamp_with_conv=True,
        act=Swish(),
        normalize=group_norm,
        use_spectral_norm=False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.ch = ch
        self.output_channels = input_channels if output_channels is None else output_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(ch_mult)
        self.act = act
        self.use_spectral_norm = use_spectral_norm

        assert input_height % (2 ** (self.num_resolutions - 1)) == 0

        temb_ch = ch * 4

        self.temb_net = TimestepEmbedding(
            embedding_dim=ch,
            hidden_dim=temb_ch,
            output_dim=temb_ch,
            act=act,
        )

        self.begin_conv = conv2d(input_channels, ch)

        # Down path
        curr_res = input_height
        in_ch = ch
        unet_chs = [ch]
        self.down_modules = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block_modules = nn.ModuleDict()
            out_ch = ch * ch_mult[i_level]

            for i_block in range(num_res_blocks):
                block_modules[f"{i_level}a_{i_block}a_block"] = ResidualBlock(
                    in_ch=in_ch,
                    temb_ch=temb_ch,
                    out_ch=out_ch,
                    dropout=dropout,
                    act=act,
                    normalize=normalize,
                )

                if curr_res in attn_resolutions:
                    block_modules[f"{i_level}a_{i_block}b_attn"] = SelfAttention(
                        out_ch,
                        normalize=normalize,
                    )

                unet_chs.append(out_ch)
                in_ch = out_ch

            if i_level != self.num_resolutions - 1:
                block_modules[f"{i_level}b_downsample"] = downsample(
                    out_ch,
                    with_conv=resamp_with_conv,
                )
                curr_res //= 2
                unet_chs.append(out_ch)

            self.down_modules.append(block_modules)

        # Middle
        self.mid_modules = nn.ModuleList(
            [
                ResidualBlock(
                    in_ch,
                    temb_ch=temb_ch,
                    out_ch=in_ch,
                    dropout=dropout,
                    act=act,
                    normalize=normalize,
                    use_spectral_norm=use_spectral_norm,
                ),
                SelfAttention(in_ch, normalize=normalize, use_spectral_norm = use_spectral_norm),
                ResidualBlock(
                    in_ch,
                    temb_ch=temb_ch,
                    out_ch=in_ch,
                    dropout=dropout,
                    act=act,
                    normalize=normalize,use_spectral_norm = use_spectral_norm,
                ),
            ]
        )

        # Up path
        self.up_modules = nn.ModuleList()

        for i_level in reversed(range(self.num_resolutions)):
            block_modules = nn.ModuleDict()
            out_ch = ch * ch_mult[i_level]

            for i_block in range(num_res_blocks + 1):
                block_modules[f"{i_level}a_{i_block}a_block"] = ResidualBlock(
                    in_ch=in_ch + unet_chs.pop(),
                    temb_ch=temb_ch,
                    out_ch=out_ch,
                    dropout=dropout,
                    act=act,
                    normalize=normalize,
                )

                if curr_res in attn_resolutions:
                    block_modules[f"{i_level}a_{i_block}b_attn"] = SelfAttention(
                        out_ch,
                        normalize=normalize,
                    )

                in_ch = out_ch

            if i_level != 0:
                block_modules[f"{i_level}b_upsample"] = upsample(
                    out_ch,
                    with_conv=resamp_with_conv,
                )
                curr_res *= 2

            self.up_modules.append(block_modules)

        assert len(unet_chs) == 0

        self.end_conv = nn.Sequential(
            normalize(in_ch),
            self.act,
            conv2d(in_ch, self.output_channels, init_scale=0.0, use_spectral_norm = use_spectral_norm),
        )

    def _compute_cond_module(self, module, x, temb):
        for m in module:
            x = m(x, temb)
        return x

    def forward(self, x, t):
        B, C, H, W = x.shape

        temb = self.temb_net(t)

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