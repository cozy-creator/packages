"""Remove a reference image's background without generating replacement subject pixels."""

from __future__ import annotations

from typing import Annotated

import msgspec
from PIL import Image

from cozy_runtime.author import App, Context, FileBound, ImageFile, Invocation, Operation, Outputs, Refuse, Telemetry

from .model import BackgroundRemovalModel

app = App()


class RemoveBackgroundInput(msgspec.Struct):
    image: Annotated[
        ImageFile,
        FileBound(max_decoded_bytes=256 << 20),
    ]


class RemoveBackgroundOutput(msgspec.Struct):
    image: Annotated[
        ImageFile,
        FileBound(
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
    """An image's subject cut out onto a white background, as a WebP."""
    ctx.raise_if_cancelled()
    with tel.stage("reading reference", overall_range=(0.0, 0.1)):
        image = payload.image.read()
    with tel.stage("removing background", overall_range=(0.1, 0.9)):
        mask = model.mask(image)
    ctx.raise_if_cancelled()
    with tel.stage("saving character reference", overall_range=(0.9, 1.0)):
        result = out.save_image(white_background(image, mask), format="webp")
    return RemoveBackgroundOutput(result)


@app.project_adapter("remove_background")
def adapt(op: Operation) -> Invocation | None:
    """A clean is a picture. One already transparent needs no run: this asset's files are what it reads."""
    picture = next((read for read in op.reads if "image" in read.files), None)
    if picture is None:
        raise Refuse(f"{op.asset} reads no picture to clean")
    if picture.files["image"].alpha >= 0.05:
        return None
    return Invocation(function="remove_background", input={"image": picture.files["image"].sha256})
