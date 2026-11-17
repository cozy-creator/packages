"""DWPose-l (`dw-ll_ucoco_384`), the whole-body pose network of DWPose's annotator: RTMPose-l, a CSPNeXt-l backbone and
an RTMCC head over 133 COCO-WholeBody keypoints at 288x384, under MMPose's module names so its checkpoint
(`dw-ll_ucoco_384.pth` `state_dict`) loads as it is. Configured as IDEA-Research/DWPose@0232d66f
`rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py` (SyncBN, so BatchNorm's default eps; SiLU). Adapted from
open-mmlab/mmdetection v3.3.0 (backbones/cspnext.py, layers/csp_layer.py, backbones/csp_darknet.py `SPPBottleneck`),
open-mmlab/mmcv v2.1.0 (`ConvModule`, `DepthwiseSeparableConvModule`) and open-mmlab/mmpose v1.3.2
(heads/coord_cls_heads/rtmcc_head.py, utils/rtmcc_block.py, utils/transformer.py `ScaleNorm`), Copyright (c)
OpenMMLab; Apache-2.0, see NOTICE. Inference only."""

from __future__ import annotations

from typing import cast

import math

import torch
import torch.nn.functional as F
from torch import nn

KEYPOINTS = 133
#: The network's input (width, height) and SimCC bins per pixel (the config's `codec`).
INPUT_SIZE, SPLIT = (288, 384), 2.0


class ConvModule(nn.Module):
    """mmcv conv_module.py: a convolution without bias, batch norm, SiLU."""

    def __init__(self, cin: int, cout: int, ksize: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, ksize, stride, ksize // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.activate = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.activate(self.bn(self.conv(x))))


class DepthwiseSeparableConvModule(nn.Module):
    """mmcv depthwise_separable_conv_module.py: a depthwise then a pointwise `ConvModule`."""

    def __init__(self, cin: int, cout: int, ksize: int) -> None:
        super().__init__()
        self.depthwise_conv = ConvModule(cin, cin, ksize, groups=cin)
        self.pointwise_conv = ConvModule(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.pointwise_conv(self.depthwise_conv(x)))


class CSPNeXtBlock(nn.Module):
    """csp_layer.py `CSPNeXtBlock` at expansion 1: a 3x3, then a 5x5 depthwise separable convolution."""

    def __init__(self, channels: int, add_identity: bool) -> None:
        super().__init__()
        self.conv1 = ConvModule(channels, channels, 3)
        self.conv2 = DepthwiseSeparableConvModule(channels, channels, 5)
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return cast(torch.Tensor, out + x if self.add_identity else out)


class ChannelAttention(nn.Module):
    """csp_layer.py `ChannelAttention`: a hard-sigmoid gate from the pooled channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)
        self.act = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, x * self.act(self.fc(self.global_avgpool(x))))


class CSPLayer(nn.Module):
    """csp_layer.py `CSPLayer` with CSPNeXt blocks and channel attention, at expand ratio 0.5."""

    def __init__(self, channels: int, blocks: int, add_identity: bool) -> None:
        super().__init__()
        mid = channels // 2
        self.main_conv = ConvModule(channels, mid, 1)
        self.short_conv = ConvModule(channels, mid, 1)
        self.final_conv = ConvModule(2 * mid, channels, 1)
        self.blocks = nn.Sequential(*(CSPNeXtBlock(mid, add_identity) for _ in range(blocks)))
        self.attention = ChannelAttention(2 * mid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.blocks(self.main_conv(x)), self.short_conv(x)), dim=1)
        return cast(torch.Tensor, self.final_conv(self.attention(x)))


class SPPBottleneck(nn.Module):
    """csp_darknet.py `SPPBottleneck`: max pools at 5, 9 and 13 beside the input."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        mid = channels // 2
        self.conv1 = ConvModule(channels, mid, 1)
        self.poolings = nn.ModuleList(nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in (5, 9, 13))
        self.conv2 = ConvModule(mid * 4, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return cast(torch.Tensor, self.conv2(torch.cat([x, *(pool(x) for pool in self.poolings)], dim=1)))


class CSPNeXt(nn.Module):
    """cspnext.py `CSPNeXt`, arch P5 at depth and width 1, read at its last stage (`out_indices=(4,)`)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvModule(3, 32, 3, stride=2), ConvModule(32, 32, 3), ConvModule(32, 64, 3))
        self.stage1 = nn.Sequential(ConvModule(64, 128, 3, stride=2), CSPLayer(128, 3, True))
        self.stage2 = nn.Sequential(ConvModule(128, 256, 3, stride=2), CSPLayer(256, 6, True))
        self.stage3 = nn.Sequential(ConvModule(256, 512, 3, stride=2), CSPLayer(512, 6, True))
        self.stage4 = nn.Sequential(ConvModule(512, 1024, 3, stride=2), SPPBottleneck(1024), CSPLayer(1024, 3, False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x))))))


class ScaleNorm(nn.Module):
    """transformer.py `ScaleNorm`: x over its RMS-scaled L2 norm, times a learned gain."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return cast(torch.Tensor, x / norm.clamp(min=1e-5) * self.g)


class Scale(nn.Module):
    """rtmcc_block.py `Scale`."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class RTMCCBlock(nn.Module):
    """rtmcc_block.py `RTMCCBlock`, self-attention without relative bias or rotary positions: a gated attention unit
    (squared-ReLU kernel) over the keypoint tokens, with a scaled residual."""

    def __init__(self, dim: int = 256, s: int = 128, expansion: int = 2) -> None:
        super().__init__()
        self.e, self.s = dim * expansion, s
        self.o = nn.Linear(self.e, dim, bias=False)
        self.uv = nn.Linear(dim, 2 * self.e + s, bias=False)
        self.gamma = nn.Parameter(torch.ones(2, s))
        self.beta = nn.Parameter(torch.zeros(2, s))
        self.ln = ScaleNorm(dim)
        self.act_fn = nn.SiLU()
        self.res_scale = Scale(dim)
        self.sqrt_s = math.sqrt(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v, base = torch.split(self.act_fn(self.uv(self.ln(x))), [self.e, self.e, self.s], dim=2)
        q, k = torch.unbind(base.unsqueeze(2) * self.gamma[None, None, :] + self.beta, dim=2)
        kernel = torch.square(F.relu(torch.bmm(q, k.permute(0, 2, 1)) / self.sqrt_s))
        return cast(torch.Tensor, self.res_scale(x) + self.o(u * torch.bmm(kernel, v)))


class RTMCCHead(nn.Module):
    """rtmcc_head.py `RTMCCHead.forward`: a 7x7 convolution to one map per keypoint, flattened (9x12), an MLP to 256,
    the GAU, then SimCC logits along x (576 bins) and y (768)."""

    def __init__(self) -> None:
        super().__init__()
        width, height = INPUT_SIZE
        flat = (width // 32) * (height // 32)
        self.final_layer = nn.Conv2d(1024, KEYPOINTS, kernel_size=7, padding=3)
        self.mlp = nn.Sequential(ScaleNorm(flat), nn.Linear(flat, 256, bias=False))
        self.gau = RTMCCBlock()
        self.cls_x = nn.Linear(256, int(width * SPLIT), bias=False)
        self.cls_y = nn.Linear(256, int(height * SPLIT), bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.gau(self.mlp(torch.flatten(self.final_layer(x), 2)))
        return self.cls_x(x), self.cls_y(x)


class RTMPose(nn.Module):
    """A top-down pose estimator: [B, 3, 384, 288] normalized RGB crops -> SimCC logits ([B, 133, 576], [B, 133, 768])."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = CSPNeXt()
        self.head = RTMCCHead()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return cast(tuple[torch.Tensor, torch.Tensor], self.head(self.backbone(x)))
