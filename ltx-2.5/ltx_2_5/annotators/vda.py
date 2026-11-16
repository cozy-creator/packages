"""Video-Depth-Anything Small's network, trimmed to inference: DINOv2 ViT-S/14 and the temporal DPT head, under upstream's
module names so its checkpoint loads as it is. Adapted from DepthAnything/Video-Depth-Anything@4f5ae231
(`video_depth_anything/`), Copyright (2025) Bytedance Ltd. and/or its affiliates, whose DINOv2 is Meta's and whose
motion module is AnimateDiff's; Apache-2.0, see NOTICE. Attention is `scaled_dot_product_attention`, the same math as
upstream's xFormers and plain paths."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#: ViT-S/14 at upstream's 518 pretraining size (dinov2.py:339-350, 398-415), and the blocks the head reads
#: (video_depth.py:49-53, `vits`).
DIM, DEPTH, HEADS, PATCH, GRID = 384, 12, 6, 14, 37
LAYERS = (2, 5, 8, 11)
#: The head's widths for `vits` (Video-Depth-Anything run.py `model_configs`), its window (`num_frames=32`) and its
#: temporal attention heads (dpt_temporal.py:35).
FEATURES, OUT_CHANNELS, FRAMES, TEMPORAL_HEADS = 64, (48, 96, 192, 384), 32, 8
#: Frames per refinement batch in the head (dpt_temporal.py:53, `micro_batch_size`).
MICRO_BATCH = 4


class PatchEmbed(nn.Module):
    """dinov2_layers/patch_embed.py:26-82: a stride-14 convolution, flattened to tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """dinov2_layers/attention.py:29-81: fused QKV self-attention."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(DIM, DIM * 3)
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, HEADS, DIM // HEADS).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, tokens, DIM))


class LayerScale(nn.Module):
    """dinov2_layers/layer_scale.py:16-28."""

    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    """dinov2_layers/mlp.py:17-41."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(DIM, DIM * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(DIM * 4, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """dinov2_layers/block.py:36-107, in eval: pre-norm attention and MLP, each layer-scaled."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale()
        self.norm2 = nn.LayerNorm(DIM, eps=1e-6)
        self.mlp = Mlp()
        self.ls2 = LayerScale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVisionTransformer(nn.Module):
    """dinov2.py:44-321: the backbone, read at the blocks the head takes (`get_intermediate_layers`, normed, class
    token dropped). `mask_token` is stored and unused at inference."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, GRID * GRID + 1, DIM))
        self.mask_token = nn.Parameter(torch.zeros(1, DIM))
        self.blocks = nn.ModuleList(Block() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(DIM, eps=1e-6)

    def positions(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """dinov2.py:179-210 `interpolate_pos_encoding`: the 37x37 table, bicubic at upstream's offset scale factors."""
        if x.shape[1] - 1 == GRID * GRID and height == width:
            return self.pos_embed
        table = self.pos_embed.float()
        rows, cols = height // PATCH + 0.1, width // PATCH + 0.1
        grid = F.interpolate(table[:, 1:].reshape(1, GRID, GRID, DIM).permute(0, 3, 1, 2),
                             scale_factor=(rows / GRID, cols / GRID), mode="bicubic", antialias=False)
        assert (int(rows), int(cols)) == tuple(grid.shape[-2:])
        return torch.cat((table[:, :1], grid.permute(0, 2, 3, 1).view(1, -1, DIM)), dim=1).to(x.dtype)

    def intermediate(self, x: torch.Tensor) -> list[torch.Tensor]:
        _, _, height, width = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.positions(x, height, width)
        taken = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in LAYERS:
                taken.append(self.norm(x)[:, 1:])
        return taken


class PositionalEncoding(nn.Module):
    """motion_module.py:192-210: a stored sinusoidal table over the window's frames."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("pe", torch.zeros(1, FRAMES, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)].to(x.dtype)


class TemporalAttention(nn.Module):
    """motion_module.py:212-321 (attention.py:30-180 `CrossAttention`): self-attention across a window's frames at
    each position, eight heads, the table added to its input."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])
        self.pos_encoder = PositionalEncoding(dim)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        tokens = x.shape[1]
        x = self.pos_encoder(x.unflatten(0, (-1, frames)).transpose(1, 2).flatten(0, 1))  # (b f) d c -> (b d) f c
        q, k, v = (project(x).unflatten(-1, (TEMPORAL_HEADS, -1)).transpose(1, 2)
                   for project in (self.to_q, self.to_k, self.to_v))
        x = self.to_out[0](F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))
        return x.unflatten(0, (-1, tokens)).transpose(1, 2).flatten(0, 1)  # (b d) f c -> (b f) d c


class GEGLU(nn.Module):
    """attention.py:363-384."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim * 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """attention.py:296-338, GEGLU at 4x."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([GEGLU(dim), nn.Identity(), nn.Linear(dim * 4, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)
        return x


class TemporalTransformerBlock(nn.Module):
    """motion_module.py:137-189: two pre-norm temporal attentions, then the feed-forward."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attention_blocks = nn.ModuleList(TemporalAttention(dim) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(2))
        self.ff = FeedForward(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        for attention, norm in zip(self.attention_blocks, self.norms, strict=True):
            x = attention(norm(x), frames) + x
        return self.ff(self.ff_norm(x)) + x


class TemporalTransformer3DModel(nn.Module):
    """motion_module.py:68-134: group norm, a projection in, one block, a projection out, residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList([TemporalTransformerBlock(channels)])
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.proj_in(self.norm(x).permute(0, 2, 3, 1).reshape(batch, height * width, channels))
        for block in self.transformer_blocks:
            h = block(h, frames)
        return self.proj_out(h).reshape(batch, height, width, channels).permute(0, 3, 1, 2) + x


class TemporalModule(nn.Module):
    """motion_module.py:32-65, on frames flattened into the batch (`(b f) c h w`)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.temporal_transformer = TemporalTransformer3DModel(channels)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        return self.temporal_transformer(x, frames)


class ResidualConvUnit(nn.Module):
    """util/blocks.py:37-91, without batch norm."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(FEATURES, FEATURES, 3, padding=1)
        self.conv2 = nn.Conv2d(FEATURES, FEATURES, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.conv1(F.relu(x)))) + x


class FeatureFusionBlock(nn.Module):
    """util/blocks.py:94-162: adds a refined skip, refines, upsamples (to `size`, else 2x), projects."""

    def __init__(self) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(FEATURES, FEATURES, 1)
        self.resConfUnit1 = ResidualConvUnit()
        self.resConfUnit2 = ResidualConvUnit()

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None,
                size: tuple[int, ...] | None = None) -> torch.Tensor:
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        x = F.interpolate(x, size=size, scale_factor=None if size else 2, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class Scratch(nn.Module):
    """util/blocks.py:4-34 `_make_scratch` with dpt.py:107-124: each level to 64 channels, four fusion blocks, the
    output convolutions."""

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4 = OUT_CHANNELS
        self.layer1_rn = nn.Conv2d(c1, FEATURES, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(c2, FEATURES, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(c3, FEATURES, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(c4, FEATURES, 3, padding=1, bias=False)
        self.refinenet1 = FeatureFusionBlock()
        self.refinenet2 = FeatureFusionBlock()
        self.refinenet3 = FeatureFusionBlock()
        self.refinenet4 = FeatureFusionBlock()
        self.output_conv1 = nn.Conv2d(FEATURES, FEATURES // 2, 3, padding=1)
        self.output_conv2 = nn.Sequential(nn.Conv2d(FEATURES // 2, 32, 3, padding=1), nn.ReLU(),
                                          nn.Conv2d(32, 1, 1), nn.ReLU())


class DPTHeadTemporal(nn.Module):
    """dpt.py:47-159 with dpt_temporal.py:22-125: reassembles four token maps, mixes the coarse two and two fusion
    stages across frames, and refines to one depth channel at 14x the patch grid."""

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4 = OUT_CHANNELS
        self.projects = nn.ModuleList(nn.Conv2d(DIM, c, 1) for c in OUT_CHANNELS)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=4), nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2),
            nn.Identity(), nn.Conv2d(c4, c4, kernel_size=3, stride=2, padding=1)])
        self.scratch = Scratch()
        self.motion_modules = nn.ModuleList(TemporalModule(channels) for channels in (c3, c4, FEATURES, FEATURES))

    def forward(self, features: list[torch.Tensor], rows: int, cols: int, frames: int) -> torch.Tensor:
        maps = []
        for index, tokens in enumerate(features):
            x = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], rows, cols)
            maps.append(self.resize_layers[index](self.projects[index](x)))
        layer_1, layer_2, layer_3, layer_4 = maps
        layer_3 = self.motion_modules[0](layer_3, frames)
        layer_4 = self.motion_modules[1](layer_4, frames)
        s = self.scratch
        layer_1_rn, layer_2_rn = s.layer1_rn(layer_1), s.layer2_rn(layer_2)
        layer_3_rn, layer_4_rn = s.layer3_rn(layer_3), s.layer4_rn(layer_4)
        path_4 = self.motion_modules[2](s.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:]), frames)
        path_3 = self.motion_modules[3](s.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:]), frames)
        out = []
        for i in range(0, path_3.shape[0], MICRO_BATCH):
            part = slice(i, i + MICRO_BATCH)
            path_2 = s.refinenet2(path_3[part], layer_2_rn[part], size=layer_1_rn.shape[2:])
            path_1 = s.refinenet1(path_2, layer_1_rn[part])
            x = F.interpolate(s.output_conv1(path_1), (rows * PATCH, cols * PATCH), mode="bilinear", align_corners=True)
            out.append(s.output_conv2(x.float()).to(x.dtype))
        return torch.cat(out)


class VideoDepthAnything(nn.Module):
    """video_depth.py:35-68: relative depth for a window of frames, [B, T, 3, H, W] -> [B, T, H, W]."""

    def __init__(self) -> None:
        super().__init__()
        self.pretrained = DinoVisionTransformer()
        self.head = DPTHeadTemporal()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, _, height, width = x.shape
        features = self.pretrained.intermediate(x.flatten(0, 1))
        depth = self.head(features, height // PATCH, width // PATCH, frames)
        depth = F.relu(F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=True))
        return depth.squeeze(1).unflatten(0, (batch, frames))
