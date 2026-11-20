"""Inference layers adapted from Practical-RIFE 4.26 (see LICENSE.rife)."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def conv(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, 2, 1), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)

    def forward(self, image: Tensor) -> Tensor:
        image = F.leaky_relu(self.cnn0(image), 0.2)
        image = F.leaky_relu(self.cnn1(image), 0.2)
        image = F.leaky_relu(self.cnn2(image), 0.2)
        return cast(Tensor, self.cnn3(image))


class ResConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, feature: Tensor) -> Tensor:
        return F.leaky_relu(self.conv(feature) * self.beta + feature, 0.2)


class IFBlock(nn.Module):
    def __init__(self, inputs: int, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(conv(inputs, channels // 2), conv(channels // 2, channels))
        self.convblock = nn.Sequential(*(ResConv(channels) for _ in range(8)))
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(channels, 52, 4, 2, 1), nn.PixelShuffle(2))

    def forward(
        self, feature: Tensor, flow: Tensor | None, scale: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        feature = F.interpolate(
            feature, scale_factor=1 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            downsampled = F.interpolate(
                flow, scale_factor=1 / scale, mode="bilinear", align_corners=False
            )
            feature = torch.cat((feature, downsampled / scale), dim=1)
        result = self.lastconv(self.convblock(self.conv0(feature)))
        result = F.interpolate(
            result, scale_factor=float(scale), mode="bilinear", align_corners=False
        )
        return result[:, :4] * scale, result[:, 4:5], result[:, 5:]


class IFNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = IFBlock(15, 192)
        self.block1 = IFBlock(28, 128)
        self.block2 = IFBlock(28, 96)
        self.block3 = IFBlock(28, 64)
        self.block4 = IFBlock(28, 32)
        self.encode = Head()

    def forward(self, left: Tensor, right: Tensor, position: float) -> Tensor:
        height, width = left.shape[-2:]
        y = torch.linspace(-1, 1, height, device=left.device, dtype=left.dtype)
        x = torch.linspace(-1, 1, width, device=left.device, dtype=left.dtype)
        vertical, horizontal = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((horizontal, vertical), dim=-1)
        half_size = left.new_tensor([(width - 1) / 2, (height - 1) / 2])

        def warp(image: Tensor, flow: Tensor) -> Tensor:
            return F.grid_sample(
                image,
                grid + flow.permute(0, 2, 3, 1) / half_size,
                padding_mode="border",
                align_corners=True,
            )

        timestep = torch.full_like(left[:, :1], position)
        left_feature, right_feature = self.encode(left), self.encode(right)
        initial = torch.cat((left, right, left_feature, right_feature, timestep), dim=1)
        flow, mask, feature = cast(tuple[Tensor, Tensor, Tensor], self.block0(initial, None, 16))
        warped_left = warp(left, flow[:, :2])
        warped_right = warp(right, flow[:, 2:])
        blocks = (self.block1, self.block2, self.block3, self.block4)
        for block, scale in zip(blocks, (8, 4, 2, 1), strict=True):
            inputs = torch.cat(
                (
                    warped_left,
                    warped_right,
                    warp(left_feature, flow[:, :2]),
                    warp(right_feature, flow[:, 2:]),
                    timestep,
                    mask,
                    feature,
                ),
                dim=1,
            )
            delta, mask, feature = cast(tuple[Tensor, Tensor, Tensor], block(inputs, flow, scale))
            flow = flow + delta
            warped_left = warp(left, flow[:, :2])
            warped_right = warp(right, flow[:, 2:])
        weight = mask.sigmoid()
        return warped_left * weight + warped_right * (1 - weight)
