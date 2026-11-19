"""Dedicated upscale networks (spandrel's architectures): a picture enlarged and cleaned in about a second, never redrawn."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated

import msgspec
from cozy_runtime.author import App, Context, Telemetry
from cozy_runtime.fields import FileBound, Shape
from cozy_runtime.media import ImageFile
from PIL import Image

from .model import UpscaleNetwork

app = App()


class Resolution(StrEnum):
    HD = "1080p"
    QHD = "1440p"
    UHD = "2160p"


#: seedvr2's: each resolution's 16:9 canvas, which holds a picture's pixels (`picture_size`).
_RESOLUTIONS: dict[Resolution, tuple[int, int]] = {
    Resolution.HD: (1920, 1080),
    Resolution.QHD: (2560, 1440),
    Resolution.UHD: (3840, 2160),
}


class UpscaleImageInput(msgspec.Struct, kw_only=True):
    #: `upscale_image`: the picture enlarged, the box's own version.
    image: Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)]
    #: `picture_size`: the picture's short side, as seedvr2's Upscale takes it.
    resolution: Annotated[Resolution, Shape(pixels=_RESOLUTIONS)] = Resolution.UHD


class UpscaleImageOutput(msgspec.Struct):
    image: Annotated[
        ImageFile,
        FileBound(max_bytes=1 << 40, max_decoded_bytes=256 << 20, media_types=("image/png",)),
    ]
    width: int
    height: int


def picture_size(width: int, height: int, resolution: Resolution) -> tuple[int, int]:
    """seedvr2's: the picture at its own ratio, its short side at `resolution`'s edge, held to the pixels of that
    resolution's 16:9 canvas (21:9 at 2160p is 4399x1885)."""
    canvas = _RESOLUTIONS[resolution]
    scale = min(canvas[1] / min(width, height), math.sqrt(canvas[0] * canvas[1] / (width * height)))
    return max(2, round(width * scale)), max(2, round(height * scale))


def passes(width: int, height: int, target: tuple[int, int], scale: int) -> int:
    """The network's passes: at least one, then one more while the picture is still at least 2x short of the target. A pass
    more from a 225 px JPEG to 2160p (900 px, 2.4x short) is 15-17% closer to the clean picture by DISTS than Lanczos the rest
    of the way (se-151); a pass for the last few percent would cost 16 times the pixels for what Lanczos does."""
    count, short = 1, min(width, height) * scale
    while 2 * short <= min(target):
        count, short = count + 1, short * scale
    return count


#: The networks (se-151): IllustrationJaNai V3detail's DAT2 for drawings, the default, and 4xBHI_dat2_real for photos,
#: a `model.model=` away and in the box's Models picker by their shared family. On 225 px JPEGs at 2160p: V3detail DISTS .165
#: (mascots .096), V3denoise .177, BHI .184 (photos .180, V3detail's .252).
ART = "the-database/illustrationjanai-v3detail-dat2@0.1.0"
PHOTO = "phhofm/4xbhi-dat2-real@0.1.0"


@app.tool(label="Upscale", named="image", defaults={"model": ART}, memoize=True)
def upscale_image(ctx: Context, payload: UpscaleImageInput, model: UpscaleNetwork, tel: Telemetry) -> UpscaleImageOutput:
    """A picture enlarged by a dedicated upscale network, as a PNG: its passes (`passes`), then Lanczos to the size asked
    exactly. Transparency is enlarged by Lanczos beside it."""
    ctx.raise_if_cancelled()
    with tel.stage("reading picture"):
        picture = payload.image.read(mode="RGBA")
    width, height = picture_size(picture.width, picture.height, payload.resolution)
    count = passes(picture.width, picture.height, (width, height), model.scale)
    made = picture.convert("RGB")
    for n in range(count):
        ctx.raise_if_cancelled()
        with tel.stage(f"upscaling, pass {n + 1} of {count}"):
            made = model.upscale(made)
    with tel.stage("saving picture"):
        image = made.resize((width, height), Image.Resampling.LANCZOS)
        alpha = picture.getchannel("A")
        if alpha.getextrema() != (255, 255):
            image.putalpha(alpha.resize((width, height), Image.Resampling.LANCZOS))
        saved = ImageFile.from_image(image, format="png")
    tel.log("upscale_image", source=f"{picture.width}x{picture.height}", width=width, height=height, passes=count,
            scale=model.scale)  # fmt: skip
    return UpscaleImageOutput(saved, width, height)
