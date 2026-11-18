"""Depth for Union Control: Video-Depth-Anything Small as Lightricks' Union Control workflow runs it
(`LoadVideoDepthAnythingModel` "video_depth_anything_vits.pth", `VideoDepthAnythingProcess` 518 / 960 / fp32,
`VideoDepthAnythingOutput` "gray"; yuvraj108c/ComfyUI-Video-Depth-Anything@a0db08e6 node.py, whose
`infer_video_depth` is DepthAnything/Video-Depth-Anything@4f5ae231 video_depth.py:70-163). The checkpoint is upstream's
state dict as it is (`depth-anything/video-depth-anything-small`), in F32: the workflow's fp32."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from cozy_runtime.accel import sequence_parallel
from cozy_runtime.author import Config, Loader, Model, Serving, uses_components
from PIL import Image

from ..model import stored
from .reference import scaled
from .vda import VideoDepthAnything

COMPONENT = "model"
#: The node's settings: the network's short side, and the long side a frame is shrunk to first.
INPUT_SIZE, MAX_RES = 518, 960
#: Upstream's fixed windowing (video_depth.py:29-33): 32-frame windows overlapping by 10, whose first 10 are the last
#: window's keyframes; the first 2 align scale and shift, the next 8 cross-fade.
INFER_LEN, OVERLAP, KEYFRAMES, INTERP_LEN = 32, 10, [0, 12, 24, 25, 26, 27, 28, 29, 30, 31], 8
ALIGN = OVERLAP - INTERP_LEN
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
#: An RGB frame, and one frame's relative depth.
Pixels = npt.NDArray[np.uint8]
Depth = npt.NDArray[np.float32]


class Graph:
    def __init__(self, config: Config) -> None:
        self.components = {COMPONENT: stored(VideoDepthAnything(), COMPONENT, config)}


def _fitted(rgb: Pixels) -> Pixels:
    """node.py:13-34 `preprocess`: a frame longer than 960 shrinks to fit, on even sides, by area."""
    height, width = rgb.shape[:2]
    if max(height, width) <= MAX_RES:
        return rgb
    scale = MAX_RES / max(height, width)
    even = [side + side % 2 for side in (round(width * scale), round(height * scale))]
    fitted: Pixels = cv2.resize(rgb, (even[0], even[1]), interpolation=cv2.INTER_AREA)
    return fitted


def _network_size(height: int, width: int) -> tuple[int, int]:
    """(width, height) the network reads: upstream's cap for frames wider than 16:9 (video_depth.py:71-75), then
    `Resize` lower bound, aspect kept, on a multiple of 14 (util/transform.py:51-107)."""
    size = INPUT_SIZE
    ratio = max(height, width) / min(height, width)
    if ratio > 1.78:
        size = round(int(size * 1.777 / ratio) / 14) * 14
    scale = max(size / height, size / width)

    def multiple(x: float) -> int:
        y = int(np.round(x / 14) * 14)
        return y if y >= size else int(np.ceil(x / 14) * 14)

    return multiple(scale * width), multiple(scale * height)


def _normalized(rgb: Pixels, size: tuple[int, int]) -> torch.Tensor:
    """video_depth.py:77-89, 102: [0, 1] RGB, bicubic to `size`, ImageNet-normalized, CHW."""
    image = cv2.resize(rgb.astype(np.float32) / 255.0, size, interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(np.ascontiguousarray(((image - MEAN) / STD).transpose(2, 0, 1)).astype(np.float32))


def _scale_and_shift(prediction: Depth, target: Depth) -> tuple[float, float]:
    """Video-Depth-Anything utils/util.py:40-62 `compute_scale_and_shift_full`, its mask all ones: the least-squares
    `scale * prediction + shift` onto `target`, in F32."""
    prediction, target = prediction.astype(np.float32), target.astype(np.float32)
    a_00, a_01, a_11 = np.sum(prediction * prediction), np.sum(prediction), np.float32(prediction.size)
    b_0, b_1 = np.sum(prediction * target), np.sum(target)
    det = a_00 * a_11 - a_01 * a_01
    if det == 0:
        return 1.0, 0.0
    # As Python floats these round-trip exactly, and NumPy multiplies an F32 array by them in F32, as by upstream's.
    return float((a_11 * b_0 - a_01 * b_1) / det), float((-a_01 * b_0 + a_00 * b_1) / det)


def _align(aligned: list[Depth], keys: list[Depth], window: list[Depth]) -> None:
    """video_depth.py:121-159, one window at a time: the first window as it is; each later one scaled and shifted
    onto the aligned keyframes it opens with, its first 8 frames cross-faded into the 8 they repeat
    (utils/util.py:65-74 `get_interpolate_frames`)."""
    if not aligned:
        aligned += window
        keys += [window[k] for k in KEYFRAMES[:ALIGN]]
        return
    scale, shift = _scale_and_shift(np.concatenate(window[:ALIGN]), np.concatenate(keys))

    def fit(depth: Depth) -> Depth:
        depth = depth * scale + shift
        depth[depth < 0] = 0
        return depth

    post = [fit(depth) for depth in window[ALIGN:OVERLAP]]
    step = 1.0 / (INTERP_LEN - 1)
    weights = [0.0, *(i * step for i in range(1, INTERP_LEN - 1)), 1.0]
    aligned[-INTERP_LEN:] = [pre * (1 - w) + new * w for pre, new, w in zip(aligned[-INTERP_LEN:], post, weights, strict=True)]
    aligned += [fit(depth) for depth in window[OVERLAP:]]
    keys[1:] = [fit(window[k]) for k in KEYFRAMES[1:ALIGN]]


@sequence_parallel(degrees=(2, 4, 8))
class VideoDepth(Model[Graph]):
    serving = Serving(encoded_leaves=False, min_bits=16)
    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)

    @uses_components(COMPONENT)
    def gray(self, frames: list[Image.Image]) -> list[Image.Image]:
        """`frames` as the workflow's gray depth maps: each at 544 on its short side (`reference.scaled`) and within
        960 (`_fitted`), read in upstream's windows (`infer_video_depth`), aligned, then normalized over the whole clip
        to [0, 1] (node.py:47-56 `postprocess_gray`) and written as 8-bit gray RGB at the fitted size."""
        model = self.graph.components[COMPONENT]
        device = torch.device(str(self.device))
        rgb = [_fitted(np.asarray(frame, dtype=np.uint8)) for frame in scaled(frames)]
        height, width = rgb[0].shape[:2]
        size = _network_size(height, width)
        count, step = len(rgb), INFER_LEN - OVERLAP
        rgb += [rgb[-1]] * ((step - count % step) % step + OVERLAP)
        aligned: list[Depth] = []
        keys: list[Depth] = []
        previous: torch.Tensor | None = None
        for start in range(0, count, step):
            # A later window's first 10 frames are the last window's keyframes, so only its new frames are read.
            first = 0 if previous is None else OVERLAP
            window = torch.stack([_normalized(rgb[start + i], size) for i in range(first, INFER_LEN)]).to(device)
            if previous is not None:
                window = torch.cat([previous[KEYFRAMES], window])
            with torch.inference_mode():
                inferred = model(window[None])[0]
                inferred = F.interpolate(inferred[:, None], size=(height, width), mode="bilinear", align_corners=True)
            previous = window
            _align(aligned, keys, list(inferred[:, 0].float().cpu().numpy()))
        depths = aligned[:count]
        low = min(depth.min() for depth in depths)
        span = max(depth.max() for depth in depths) - low + 1e-6
        levels = (np.round((depth - low) / span * 255).clip(0, 255).astype(np.uint8) for depth in depths)
        return [Image.fromarray(np.repeat(level[:, :, None], 3, axis=2)) for level in levels]
