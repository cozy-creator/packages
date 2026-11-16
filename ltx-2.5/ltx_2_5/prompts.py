"""What LTX-2.5 reads, from what a user writes. One flowing paragraph is the model's own form (Lightricks' prompting
guide; the training captions its enhancer's system prompts describe), so the user's words go through as written, each
mention as its name. With pictures mentioned, the Ingredients IC-LoRA's two-part form (its model card's "How it works"):

    Reference sheet: <the panels in the sheet>

    Generated video: <the video>

and the sheet itself: one panel per picture, on black, no text, the pictures in the order they are first mentioned."""

from __future__ import annotations

import math

from cozy_runtime.fields import MENTION
from PIL import Image

#: Panels a row of the sheet holds; the card's sheets hold up to three a row.
ROW = 3
COLUMNS = {1: ("Center",), 2: ("Left", "Right"), 3: ("Left", "Middle", "Right")}


def names(text: str) -> str:
    """`text` with each mention as its name: `<amy.2>` is amy 2."""
    return MENTION.sub(lambda found: found[1].replace(".", " "), text).strip()


def places(count: int) -> list[str]:
    """Each panel's place on the sheet, as the card's examples name them ("Top Row Left")."""
    rows = math.ceil(count / ROW)
    tiers = {2: ("Top", "Bottom"), 3: ("Top", "Middle", "Bottom")}.get(rows)
    made = []
    for index in range(count):
        row, column = divmod(index, ROW)
        across = COLUMNS[min(ROW, count - row * ROW)][column]
        tier = f"{tiers[row]} Row" if tiers else f"Row {row + 1}"
        made.append(across if rows == 1 else f"{tier} {across}")
    return made


def sheet(pictures: list[Image.Image], width: int, height: int) -> Image.Image:
    """The pictures as panels on black at `width` x `height`, each as large as its cell holds it whole."""
    canvas = Image.new("RGB", (width, height))
    rows = math.ceil(len(pictures) / ROW)
    for index, picture in enumerate(pictures):
        row, column = divmod(index, ROW)
        cell_w, cell_h = width // min(ROW, len(pictures) - row * ROW), height // rows
        scale = min(cell_w / picture.width, cell_h / picture.height)
        fitted = picture.convert("RGB").resize((max(1, round(picture.width * scale)), max(1, round(picture.height * scale))),
                                               Image.Resampling.LANCZOS)
        canvas.paste(fitted, (column * cell_w + (cell_w - fitted.width) // 2, row * cell_h + (cell_h - fitted.height) // 2))
    return canvas


def rendered(video: str, panels: list[str]) -> str:
    """The text the model reads: the video as written, after the reference sheet when there is one."""
    if not panels:
        return video
    described = " ".join(f"{place}: {name}." for place, name in zip(places(len(panels)), panels, strict=True))
    return f"Reference sheet: {described}\n\nGenerated video: {video}"
