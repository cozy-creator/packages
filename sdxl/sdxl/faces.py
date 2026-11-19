"""Faces in a reference picture (se-175): insightface's buffalo_l pack, its SCRFD-10G (`det_10g`) for faces and their
five landmarks, and its ArcFace (`w600k_r50`, IResNet-50 on WebFace600K) for a face's identity. Each is a torch port of
the ONNX graph insightface ships, its weights that graph's initializers (`scripts/faceid_ingest.py`): each batch norm
after a convolution comes folded into it, as the export left it; the others are their scale and shift.

A picture is read as insightface's `FaceAnalysis` reads it (python-package v0.7: `scrfd.py`, `arcface_onnx.py`,
`face_align.py`): SCRFD over the picture letterboxed into 640² (top left, on black), (x - 127.5) / 128; each anchor's box
and landmarks decoded from its stride, kept from score 0.5, NMS at 0.4; a face aligned by the similarity (Umeyama) that
takes its landmarks to ArcFace's five-point template, warped bilinear, and embedded from (x - 127.5) / 127.5,
L2-normalized. Only at 640²: ComfyUI_IPAdapter_plus's FaceID node reads a faceless picture again smaller, down to 320²,
but there SCRFD finds faces in animals (se-175's fox), and here a found face picks the adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from cozy_runtime.author import Config, Loader, Model, Serving, uses_components
from numpy.typing import NDArray
from PIL import Image as PILImage
from torch import nn

#: The pack's two networks, each a component.
DETECTION = "detection"
RECOGNITION = "recognition"
#: SCRFD's input side (insightface `FaceAnalysis.prepare`'s `det_size`).
SIDE = 640
#: SCRFD-10G's strides, anchors per place, score floor and NMS overlap (insightface `SCRFD`).
STRIDES = (8, 16, 32)
ANCHORS = 2
SCORE, OVERLAP = 0.5, 0.4
#: ArcFace's five landmarks (eyes, nose, mouth corners) in its 112² crop (insightface `face_align.arcface_dst`).
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                     [70.7299, 92.2041]], dtype=np.float64)  # fmt: skip
#: A face is a picture's subject when its box covers this much of the picture (about 100 px across at 1 MP, ArcFace's
#: 112² crop) and no other face is half its size: a portrait or a person, not a crowd or a face in the background.
SUBJECT = 0.01


def _conv(inputs: int, outputs: int, kernel: int = 3, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2)


# ---------------------------------------------------------------- SCRFD-10G


class _Basic(nn.Module):
    """mmdet's ResNetV1e BasicBlock; a strided block's shortcut averages 2x2 before its 1x1 projection (ResNet-D)."""

    def __init__(self, inputs: int, outputs: int, stride: int) -> None:
        super().__init__()
        self.conv1 = _conv(inputs, outputs, 3, stride)
        self.conv2 = _conv(outputs, outputs)
        self.downsample = (
            nn.Sequential(nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False), _conv(inputs, outputs, 1))
            if stride > 1 or inputs != outputs else None
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x if self.downsample is None else self.downsample(x)
        return F.relu(self.conv2(F.relu(self.conv1(x))) + shortcut)


class _Head(nn.Module):
    """One stride's head: three convolutions the score, box and landmarks share, the box scaled by its stride's own
    learned factor."""

    def __init__(self, inputs: int, width: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([_conv(inputs, width), _conv(width, width), _conv(width, width)])
        self.cls = _conv(width, ANCHORS)
        self.reg = _conv(width, ANCHORS * 4)
        self.kps = _conv(width, ANCHORS * 10)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for conv in self.convs:
            x = F.relu(conv(x))
        return torch.sigmoid(self.cls(x)), self.reg(x) * self.scale, self.kps(x)


class SCRFD(nn.Module):
    """SCRFD-10G with landmarks (`scrfd_10g_bnkps`): a 3-convolution stem and four BasicBlock stages (56, 88, 88, 224
    wide; 3, 4, 2, 3 deep), a PAFPN over the last three at 56, and a head per stride at 80."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.ModuleList([_conv(3, 28, 3, 2), _conv(28, 28), _conv(28, 56)])
        stages, inputs = [], 56
        for i, (width, depth) in enumerate(((56, 3), (88, 4), (88, 2), (224, 3))):
            blocks = [_Basic(inputs if j == 0 else width, width, 2 if i and j == 0 else 1) for j in range(depth)]
            stages.append(nn.Sequential(*blocks))
            inputs = width
        self.stages = nn.ModuleList(stages)
        self.lateral = nn.ModuleList([_conv(width, 56, 1) for width in (88, 88, 224)])
        self.fpn = nn.ModuleList([_conv(56, 56) for _ in range(3)])
        self.down = nn.ModuleList([_conv(56, 56, 3, 2) for _ in range(2)])
        self.pafpn = nn.ModuleList([_conv(56, 56) for _ in range(2)])
        self.heads = nn.ModuleList([_Head(56, 80) for _ in STRIDES])

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Each stride's scores [1, A, h, w], box distances [1, 4A, h, w] and landmark offsets [1, 10A, h, w], in
        input pixels / stride."""
        for conv in self.stem:
            x = F.relu(conv(x))
        x = F.max_pool2d(x, 2, 2)
        levels = []
        for stage in self.stages:
            x = stage(x)
            levels.append(x)
        laterals = [conv(level) for conv, level in zip(self.lateral, levels[1:], strict=True)]
        for i in (2, 1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:])
        outs = [conv(lateral) for conv, lateral in zip(self.fpn, laterals, strict=True)]
        for i, conv in enumerate(self.down):
            outs[i + 1] = outs[i + 1] + conv(outs[i])
        outs = [outs[0], *(conv(out) for conv, out in zip(self.pafpn, outs[1:], strict=True))]
        return [head(out) for head, out in zip(self.heads, outs, strict=True)]


# ---------------------------------------------------------------- ArcFace (IResNet-50)


class _Affine(nn.Module):
    """A batch norm in inference: its scale and shift per channel."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * self.weight.view(shape) + self.bias.view(shape)


class _IBasic(nn.Module):
    """arcface_torch's IBasicBlock: bn1, conv1 (bn2 folded), PReLU, conv2 (bn3 folded, the stride), and a strided
    block's 1x1 shortcut (its batch norm folded)."""

    def __init__(self, inputs: int, outputs: int, stride: int) -> None:
        super().__init__()
        self.bn1 = _Affine(inputs)
        self.conv1 = _conv(inputs, outputs)
        self.prelu = nn.PReLU(outputs)
        self.conv2 = _conv(outputs, outputs, 3, stride)
        self.downsample = nn.Conv2d(inputs, outputs, 1, stride) if stride > 1 or inputs != outputs else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x if self.downsample is None else self.downsample(x)
        out: torch.Tensor = self.conv2(self.prelu(self.conv1(self.bn1(x))))
        return out + shortcut


class IResNet(nn.Module):
    """arcface_torch's IResNet-50 (3, 4, 14, 3 blocks; 64 to 512 wide) from a 112² face to its 512 features."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = _conv(3, 64)
        self.prelu = nn.PReLU(64)
        layers, inputs = [], 64
        for width, depth in ((64, 3), (128, 4), (256, 14), (512, 3)):
            layers.append(nn.Sequential(*[_IBasic(inputs if j == 0 else width, width, 2 if j == 0 else 1)
                                          for j in range(depth)]))  # fmt: skip
            inputs = width
        self.layers = nn.ModuleList(layers)
        self.bn2 = _Affine(512)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = _Affine(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.conv1(x))
        for layer in self.layers:
            x = layer(x)
        out: torch.Tensor = self.features(self.fc(self.bn2(x).flatten(1)))
        return out


# ---------------------------------------------------------------- reading a picture


@dataclass(frozen=True)
class Face:
    """A face SCRFD found: its box (x1, y1, x2, y2) and five landmarks [5, 2], in the picture's pixels."""

    box: tuple[float, float, float, float]
    score: float
    landmarks: NDArray[np.float64]

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


def subject(faces: Sequence[Face], width: int, height: int) -> Face | None:
    """The face that is the picture's subject (`SUBJECT`), if one is."""
    ranked = sorted(faces, key=lambda face: face.area, reverse=True)
    if not ranked or ranked[0].area < SUBJECT * width * height:
        return None
    if len(ranked) > 1 and ranked[1].area * 2 >= ranked[0].area:
        return None
    return ranked[0]


def _rgb(picture: PILImage.Image) -> torch.Tensor:
    """[1, 3, H, W] f32 in 0..255."""
    rgb = picture.convert("RGB")
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).view(rgb.height, rgb.width, 3)
    return pixels.permute(2, 0, 1).unsqueeze(0).float()


def _letterboxed(rgb: torch.Tensor, side: int) -> tuple[torch.Tensor, float]:
    """`rgb` resized (bilinear, as cv2's INTER_LINEAR: no antialias, rounded to bytes) to fit `side`² at the top left,
    on black; and its scale."""
    height, width = rgb.shape[-2:]
    if height > width:
        size = (side, int(side * width / height))
    else:
        size = (int(side * height / width), side)
    scaled = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False).round().clamp(0, 255)
    canvas = torch.zeros(1, 3, side, side)
    canvas[..., : size[0], : size[1]] = scaled
    return canvas, size[0] / height


def _nms(boxes: NDArray[np.float32], scores: NDArray[np.float32]) -> list[int]:
    """insightface's NMS (its inclusive pixel areas), highest score first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        overlap = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[1:][overlap <= OVERLAP]
    return keep


def _decoded(levels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], side: int,
             scale: float) -> list[Face]:  # fmt: skip
    """SCRFD's outputs as faces in the picture's pixels (insightface `SCRFD.forward`, `detect`)."""
    scores, boxes, landmarks = [], [], []
    for stride, (cls, reg, kps) in zip(STRIDES, levels, strict=True):
        rows = side // stride
        ys, xs = np.mgrid[:rows, :rows]
        centers = np.repeat(np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride, ANCHORS, axis=0).astype(np.float32)
        score = cls.permute(0, 2, 3, 1).reshape(-1).float().cpu().numpy()
        distance = reg.permute(0, 2, 3, 1).reshape(-1, 4).float().cpu().numpy() * stride
        offset = kps.permute(0, 2, 3, 1).reshape(-1, 5, 2).float().cpu().numpy() * stride
        kept = score >= SCORE
        scores.append(score[kept])
        boxes.append(np.concatenate([centers - distance[:, :2], centers + distance[:, 2:]], axis=1)[kept])
        landmarks.append((centers[:, None, :] + offset)[kept])
    score = np.concatenate(scores)
    box = (np.concatenate(boxes) / scale).astype(np.float32)
    landmark = np.concatenate(landmarks) / scale
    return [Face((float(box[i, 0]), float(box[i, 1]), float(box[i, 2]), float(box[i, 3])), float(score[i]),
                 landmark[i].astype(np.float64)) for i in _nms(box, score)]  # fmt: skip


def _similarity(source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """The similarity [3, 3] (scale, rotation, translation) taking `source` points nearest `target` (Umeyama's least
    squares, scikit-image's `SimilarityTransform.estimate`, which insightface's `estimate_norm` calls)."""
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    s, t = source - mean_s, target - mean_t
    covariance = t.T @ s / len(source)
    sign = np.ones(2)
    if np.linalg.det(covariance) < 0:
        sign[1] = -1
    u, singular, v = np.linalg.svd(covariance)
    matrix = np.eye(3)
    rotation = u @ np.diag(sign) @ v
    factor = float(singular @ sign) / float(s.var(axis=0).sum())
    matrix[:2, :2] = factor * rotation
    matrix[:2, 2] = mean_t - factor * rotation @ mean_s
    return matrix


def aligned(picture: PILImage.Image, face: Face, size: int) -> torch.Tensor:
    """The face in its template's place at `size`² (a multiple of 112), [1, 3, size, size] bytes as f32: cv2's
    `warpAffine` (bilinear, black outside), as insightface's `norm_crop`."""
    rgb = _rgb(picture)
    height, width = rgb.shape[-2:]
    inverse = np.linalg.inv(_similarity(face.landmarks, TEMPLATE * size / 112))
    ys, xs = np.mgrid[:size, :size].astype(np.float64)
    source = inverse[:2, :2] @ np.stack([xs.ravel(), ys.ravel()]) + inverse[:2, 2:]
    # Pixel centres are whole numbers in cv2's coordinates: grid_sample's with its corners aligned.
    grid = np.stack([source[0] * 2 / (width - 1) - 1, source[1] * 2 / (height - 1) - 1], axis=-1)
    grid_t = torch.from_numpy(grid.reshape(1, size, size, 2)).float()
    warped = F.grid_sample(rgb, grid_t, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped.round().clamp(0, 255)


def crop(picture: PILImage.Image, face: Face, size: int) -> PILImage.Image:
    """`aligned` as a picture."""
    pixels = aligned(picture, face, size)[0].permute(1, 2, 0).to(torch.uint8).contiguous()
    return PILImage.frombytes("RGB", (size, size), bytes(pixels.numpy().tobytes()))


# ---------------------------------------------------------------- the model


class FaceGraph:
    def __init__(self, config: Config) -> None:
        # Built in its stored dtype (an as-is upload is never cast): insightface's graphs are f32.
        stored = config.tensor_dtype(f"{RECOGNITION}.fc.weight", default="f32")
        dtype = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[stored]
        self.components: dict[str, nn.Module] = {
            DETECTION: SCRFD().to(dtype).eval(),
            RECOGNITION: IResNet().to(dtype).eval(),
        }


class FaceAnalysis(Model[FaceGraph]):
    """insightface's buffalo_l: SCRFD-10G finds faces, ArcFace (IResNet-50) says whose."""

    serving = Serving(encoded_leaves=False, min_bits=16)

    graph: FaceGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(FaceGraph)

    @uses_components(DETECTION)
    def find(self, picture: PILImage.Image) -> list[Face]:
        """The picture's faces, highest score first."""
        net = self.graph.components[DETECTION]
        weight = next(net.parameters())
        canvas, scale = _letterboxed(_rgb(picture), SIDE)
        with torch.no_grad():
            levels = net(((canvas - 127.5) / 128.0).to(weight.device, weight.dtype))
        return _decoded(levels, SIDE, scale)

    @uses_components(RECOGNITION)
    def identities(self, faces: Sequence[torch.Tensor]) -> torch.Tensor:
        """Each aligned 112² face's ArcFace embedding, L2-normalized: [n, 512] f32."""
        net = self.graph.components[RECOGNITION]
        weight = next(net.parameters())
        pixels = (torch.cat(list(faces)) - 127.5) / 127.5
        with torch.no_grad():
            features: torch.Tensor = net(pixels.to(weight.device, weight.dtype)).float()
        return features.div(features.norm(dim=-1, keepdim=True))
