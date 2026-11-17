"""A reference video's frames as Lightricks' Union Control workflow hands them to every annotator
(`LTX-2.5_ICLoRA_Union_Control_Distilled.json`, "Reference Video": `ResizeImageMaskNode` "scale shorter dimension",
544, lanczos; ComfyUI 0.16 nodes_post_processing.py `scale_shorter_dimension` and comfy/utils.py `lanczos`, which is
PIL's)."""

from __future__ import annotations

from PIL import Image

SHORT_SIDE = 544


def scaled(frames: list[Image.Image]) -> list[Image.Image]:
    """`frames` with their short side at 544, aspect kept (sides rounded as ComfyUI does); frames already there pass."""
    made = []
    for frame in frames:
        width, height = frame.size
        if height < width:
            size = (round(width / height * SHORT_SIDE), SHORT_SIDE)
        elif width < height:
            size = (SHORT_SIDE, round(height / width * SHORT_SIDE))
        else:
            size = (SHORT_SIDE, SHORT_SIDE)
        frame = frame.convert("RGB")
        made.append(frame if frame.size == size else frame.resize(size, Image.Resampling.LANCZOS))
    return made
