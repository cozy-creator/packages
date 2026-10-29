"""Remove a reference image's background without generating replacement subject pixels."""

from __future__ import annotations

from typing import Annotated

import msgspec
from PIL import Image

from cozy_runtime.author import App, AssetBound, Context, ImageAsset, Outputs, Telemetry

from .model import BackgroundRemovalModel

app = App()


class RemoveBackgroundInput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
    ]


class RemoveBackgroundOutput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/webp",),
        ),
    ]


def white_background(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Alpha-composite the original subject onto white with Pillow's standard operation."""
    return Image.composite(image, Image.new("RGB", image.size, "white"), mask)


@app.entrypoint(
    defaults={"model": "zhengpeng7/birefnet@0.1.0"},
    memoize=True,
)
def remove_background(
    ctx: Context,
    payload: RemoveBackgroundInput,
    model: BackgroundRemovalModel,
    out: Outputs,
    tel: Telemetry,
) -> RemoveBackgroundOutput:
    ctx.raise_if_cancelled()
    with tel.stage("reading reference", overall_range=(0.0, 0.1)):
        image = payload.image.read()
    with tel.stage("removing background", overall_range=(0.1, 0.9)):
        mask = model.mask(image)
    ctx.raise_if_cancelled()
    with tel.stage("saving character reference", overall_range=(0.9, 1.0)):
        result = out.save_image(white_background(image, mask), format="webp")
    return RemoveBackgroundOutput(result)
