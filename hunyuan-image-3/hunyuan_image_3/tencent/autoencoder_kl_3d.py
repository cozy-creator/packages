"""
Reference code
[FLUX] https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
[DCAE] https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/efficientvit/dc_ae.py

Modified for Cozy (packages se-079): a plain nn.Module without diffusers' ModelMixin/ConfigMixin;
no tiling, slicing, distributed decoding or weight loading (Runtime fills the weights). Each op runs in
the dtype Tencent's float16 autocast gave it, with no autocast: a conv in its operands' stored dtype
(float16), the group norms and the posterior's exp in float32 (xs-134).
"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple
import math
import numpy as np
from einops import rearrange
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def randn_tensor(shape, generator=None, device=None, dtype=None):
    """diffusers' `randn_tensor` (0.35): one draw per sample for a list of generators, on the
    generators' device, then moved to `device`."""
    rand_device = device
    batch_size = shape[0]
    device = device or torch.device("cpu")
    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
    if isinstance(generator, list):
        shape = (1,) + tuple(shape[1:])
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
            for i in range(batch_size)
        ]
        return torch.cat(latents, dim=0).to(device)
    return torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)


@dataclass
class AutoencoderKLOutput:
    latent_dist: "DiagonalGaussianDistribution"




class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H ,W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp((0.5 * self.logvar).float())
        self.var = torch.exp(self.logvar.float())
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            reduce_dim = list(range(1, self.mean.ndim))
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dim,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var +
                    self.var / other.var -
                    1.0 -
                    self.logvar +
                    other.logvar,
                    dim=reduce_dim,
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean

@dataclass
@dataclass
class DecoderOutput:
    sample: torch.FloatTensor
    posterior: Optional[DiagonalGaussianDistribution] = None

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, use_reentrant=False)
    else:
        return module(*inputs)


class GroupNorm(nn.GroupNorm):
    """Normalizes in float32 with float32 parameters, as autocast runs a group norm."""

    def forward(self, input):
        return F.group_norm(input.float(), self.num_groups, self.weight, self.bias, self.eps)


class Conv3d(nn.Conv3d):
    """Perform Conv3d on patches with numerical differences from nn.Conv3d within 1e-5. Only symmetric padding is supported.
    The input is cast to the stored weights' dtype."""

    def forward(self, input):
        input = input.to(self.weight.dtype)
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i in range(len(chunks)):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunks[i],
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, :self.padding[0]] = chunks[i - 1][:, :, -self.padding[0]:]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0]:] = chunks[i + 1][:, :, :self.padding[0]]
                else:
                    padded_chunk = chunks[i]
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = []
            for i in range(len(padded_chunks)):
                outputs.append(super().forward(padded_chunks[i]))
            self.padding = padding_bak
            return torch.cat(outputs, dim=-3)
        else:
            return super().forward(input)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        self.add_temporal_downsample = add_temporal_downsample
        stride = (2, 2, 2) if add_temporal_downsample else (1, 2, 2)  # THW
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)

    def forward(self, x: Tensor):
        spatial_pad = (0, 1, 0, 1, 0, 0)  # WHT
        x = nn.functional.pad(x, spatial_pad, mode="constant", value=0)

        temporal_pad = (0, 0, 0, 0, 0, 1) if self.add_temporal_downsample else (0, 0, 0, 0, 1, 1)
        x = nn.functional.pad(x, temporal_pad, mode="replicate")

        x = self.conv(x)
        return x


class DownsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
        shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)

        B, C, T, H, W = shortcut.shape
        shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
        return h + shortcut


class Upsample(nn.Module):
    def __init__(self, in_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        self.add_temporal_upsample = add_temporal_upsample
        self.scale_factor = (2, 2, 2) if add_temporal_upsample else (1, 2, 2)  # THW
        self.conv = Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class UpsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = Conv3d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(i_level >= np.log2(ffactor_spatial // ffactor_temporal))
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = DownsampleDCAE(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            use_checkpointing = bool(self.training and self.gradient_checkpointing)

            # downsampling
            h = self.conv_in(x)
            for i_level in range(len(self.block_out_channels)):
                for i_block in range(self.num_res_blocks):
                    h = forward_with_checkpointing(self.down[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
                if hasattr(self.down[i_level], "downsample"):
                    h = forward_with_checkpointing(self.down[i_level].downsample, h, use_checkpointing=use_checkpointing)

            # middle
            h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

            # end
            group_size = self.block_out_channels[-1] // (2 * self.z_channels)
            shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
            h = self.norm_out(h)
            h = swish(h)
            h = self.conv_out(h)
            h += shortcut
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # z to block_in
        block_in = block_out_channels[0]
        self.conv_in = Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = UpsampleDCAE(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # end
        self.norm_out = GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

        self.gradient_checkpointing = False


    def forward(self, z: Tensor) -> Tensor:
        with torch.no_grad():
            use_checkpointing = bool(self.training and self.gradient_checkpointing)
            # z to block_in
            repeats = self.block_out_channels[0] // (self.z_channels)
            h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)
            # middle
            h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
            h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)
            # upsampling
            for i_level in range(len(self.block_out_channels)):
                for i_block in range(self.num_res_blocks + 1):
                    h = forward_with_checkpointing(self.up[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
                if hasattr(self.up[i_level], "upsample"):
                    h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)
            # end
            h = self.norm_out(h)
            h = swish(h)
            h = self.conv_out(h)
        return h


class AutoencoderKLConv3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        only_encoder: bool = False,
        only_decoder: bool = False,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=in_channels, out_channels=out_channels, latent_channels=latent_channels,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            ffactor_spatial=ffactor_spatial, ffactor_temporal=ffactor_temporal, sample_size=sample_size,
            sample_tsize=sample_tsize, scaling_factor=scaling_factor, shift_factor=shift_factor,
        )
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        if not only_decoder:
            self.encoder = Encoder(
                in_channels=in_channels,
                z_channels=latent_channels,
                block_out_channels=block_out_channels,
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                downsample_match_channel=downsample_match_channel,
            )
        if not only_encoder:
            self.decoder = Decoder(
                z_channels=latent_channels,
                out_channels=out_channels,
                block_out_channels=list(reversed(block_out_channels)),
                num_res_blocks=layers_per_block,
                ffactor_spatial=ffactor_spatial,
                ffactor_temporal=ffactor_temporal,
                upsample_match_channel=upsample_match_channel,
            )

    @classmethod
    def from_config(cls, config: dict) -> "AutoencoderKLConv3D":
        return cls(**{k: v for k, v in config.items() if not k.startswith("_")})

    def encode(self, x: Tensor, return_dict: bool = True):
        if len(x.shape) != 5:  # (B, C, T, H, W)
            x = x[:, :, None]
        assert len(x.shape) == 5  # (B, C, T, H, W)
        if x.shape[2] == 1:
            x = x.expand(-1, -1, self.ffactor_temporal, -1, -1)
        else:
            assert x.shape[2] != self.ffactor_temporal and x.shape[2] % self.ffactor_temporal == 0

        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: Tensor, return_dict: bool = True, generator=None):
        decoded = self.decoder(z)
        if z.shape[-3] == 1:
            decoded = decoded[:, :, -1:]
        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
