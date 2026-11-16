"""YOLOX-L, the person detector DWPose crops people with (`yolox_l.onnx` in `yzd-v/DWPose`, Megvii's 0.1.1rc0 release
export), under Megvii's module names so its checkpoint (`yolox_l.pth` `model`) loads as it is. Adapted from
Megvii-BaseDetection/YOLOX@0.1.1rc0 (yolox/models/network_blocks.py, darknet.py, yolo_pafpn.py, yolo_head.py;
exps/default/yolox_l.py: depth 1, width 1, SiLU), Copyright (c) Megvii Inc.; Apache-2.0, see NOTICE. Inference only:
the head returns what the ONNX export does (`decode_in_inference=False`, tools/export_onnx.py:77)."""

from __future__ import annotations

import torch
from torch import nn

#: yolox/exp/yolox_base.py:67-71 `init_yolo`: every BatchNorm at eps 1e-3.
BN_EPS = 1e-3
CLASSES = 80


class BaseConv(nn.Module):
    """network_blocks.py:28-53: convolution, batch norm, SiLU."""

    def __init__(self, cin: int, cout: int, ksize: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, ksize, stride, (ksize - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=BN_EPS)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """network_blocks.py:78-100, at expansion 1."""

    def __init__(self, channels: int, shortcut: bool) -> None:
        super().__init__()
        self.conv1 = BaseConv(channels, channels, 1)
        self.conv2 = BaseConv(channels, channels, 3)
        self.use_add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class CSPLayer(nn.Module):
    """network_blocks.py:146-179."""

    def __init__(self, cin: int, cout: int, n: int, shortcut: bool = True) -> None:
        super().__init__()
        hidden = cout // 2
        self.conv1 = BaseConv(cin, hidden, 1)
        self.conv2 = BaseConv(cin, hidden, 1)
        self.conv3 = BaseConv(2 * hidden, cout, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden, shortcut) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


class Focus(nn.Module):
    """network_blocks.py:182-204: space-to-depth, then a convolution."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = BaseConv(cin * 4, cout, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1))


class SPPBottleneck(nn.Module):
    """network_blocks.py:121-143: max pools at 5, 9 and 13 beside the input."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        hidden = cin // 2
        self.conv1 = BaseConv(cin, hidden, 1)
        self.m = nn.ModuleList(nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in (5, 9, 13))
        self.conv2 = BaseConv(hidden * 4, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(torch.cat([x, *(pool(x) for pool in self.m)], dim=1))


class CSPDarknet(nn.Module):
    """darknet.py:89-171 at depth 1, width 1: stride 8, 16 and 32 features (dark3, dark4, dark5)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = Focus(3, 64)
        self.dark2 = nn.Sequential(BaseConv(64, 128, 3, 2), CSPLayer(128, 128, 3))
        self.dark3 = nn.Sequential(BaseConv(128, 256, 3, 2), CSPLayer(256, 256, 9))
        self.dark4 = nn.Sequential(BaseConv(256, 512, 3, 2), CSPLayer(512, 512, 9))
        self.dark5 = nn.Sequential(BaseConv(512, 1024, 3, 2), SPPBottleneck(1024, 1024),
                                   CSPLayer(1024, 1024, 3, shortcut=False))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dark3 = self.dark3(self.dark2(self.stem(x)))
        dark4 = self.dark4(dark3)
        return dark3, dark4, self.dark5(dark4)


class YOLOPAFPN(nn.Module):
    """yolo_pafpn.py:11-107: top-down then bottom-up fusion of the three features."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = CSPDarknet()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(1024, 512, 1)
        self.C3_p4 = CSPLayer(1024, 512, 3, shortcut=False)
        self.reduce_conv1 = BaseConv(512, 256, 1)
        self.C3_p3 = CSPLayer(512, 256, 3, shortcut=False)
        self.bu_conv2 = BaseConv(256, 256, 3, 2)
        self.C3_n3 = CSPLayer(512, 512, 3, shortcut=False)
        self.bu_conv1 = BaseConv(512, 512, 3, 2)
        self.C3_n4 = CSPLayer(1024, 1024, 3, shortcut=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x2, x1, x0 = self.backbone(x)
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat([self.upsample(fpn_out0), x1], 1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat([self.upsample(fpn_out1), x2], 1))
        pan_out1 = self.C3_n3(torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1))
        pan_out0 = self.C3_n4(torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1))
        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    """yolo_head.py:18-209, in eval with `decode_in_inference=False`: per anchor point, box offsets, objectness and
    class probabilities, [B, anchors, 85]."""

    def __init__(self) -> None:
        super().__init__()
        self.stems = nn.ModuleList(BaseConv(c, 256, 1) for c in (256, 512, 1024))
        self.cls_convs = nn.ModuleList(nn.Sequential(BaseConv(256, 256, 3), BaseConv(256, 256, 3)) for _ in range(3))
        self.reg_convs = nn.ModuleList(nn.Sequential(BaseConv(256, 256, 3), BaseConv(256, 256, 3)) for _ in range(3))
        self.cls_preds = nn.ModuleList(nn.Conv2d(256, CLASSES, 1) for _ in range(3))
        self.reg_preds = nn.ModuleList(nn.Conv2d(256, 4, 1) for _ in range(3))
        self.obj_preds = nn.ModuleList(nn.Conv2d(256, 1, 1) for _ in range(3))

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        outputs = []
        for k, x in enumerate(features):
            x = self.stems[k](x)
            reg = self.reg_convs[k](x)
            outputs.append(torch.cat([self.reg_preds[k](reg), self.obj_preds[k](reg).sigmoid(),
                                      self.cls_preds[k](self.cls_convs[k](x)).sigmoid()], 1))
        return torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)


class YOLOX(nn.Module):
    """yolox/models/yolox.py: [B, 3, 640, 640] BGR in 0-255 -> [B, 8400, 85]."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = YOLOPAFPN()
        self.head = YOLOXHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
