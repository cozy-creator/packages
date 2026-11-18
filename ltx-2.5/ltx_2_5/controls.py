"""A reference video as the Union Control IC-LoRA reads it: a control map per frame. Lightricks' Union Control workflow
(`ComfyUI-LTXVideo/example_workflows/2.5/LTX-2.5_ICLoRA_Union_Control_Distilled.json`) makes edges with
comfyui_controlnet_aux's `CannyEdgePreprocessor` at 92 / 200, resolution 512: its frame scaled so the short side is
512 on a 64-pixel grid (`resize_image`), then OpenCV's Canny, as three channels."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from PIL import Image

Pixels = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
Lines = np.ndarray[tuple[int, int], np.dtype[np.uint8]]

#: The workflow's annotator settings: low and high hysteresis thresholds, detect resolution.
LOW, HIGH, RESOLUTION = 92, 200, 512


def _fitted(rgb: Pixels) -> Pixels:
    height, width = rgb.shape[:2]
    scale = RESOLUTION / min(height, width)
    size = (int(np.round(width * scale / 64.0)) * 64, int(np.round(height * scale / 64.0)) * 64)
    return cast(Pixels, cv2.resize(rgb, size, interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA))


def edges(frames: list[Image.Image]) -> list[Image.Image]:
    made = []
    for frame in frames:
        lines = cast(Lines, cv2.Canny(_fitted(cast(Pixels, np.asarray(frame.convert("RGB")))), LOW, HIGH))
        made.append(Image.fromarray(np.repeat(lines[:, :, None], 3, axis=2)))
    return made
