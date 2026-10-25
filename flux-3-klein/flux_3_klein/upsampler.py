"""FLUX.3 Klein's prompt upsampler (a Qwen3.5-9B fine-tune), served from its as-is upload: the
construction config is BFL's `config.json`, and the tokenizer, processor and chat template are the
checkpoint's assets.

Inference follows BFL's `flux3/pup.py`: greedy decoding, presence penalty 1.5 for text-to-image and 0
for edits, references resized to the upsampler's budget and round-tripped through JPEG q92.

Layout follows BFL's `flux3/bbox.py`: Klein is trained on captions that cite each element as `<id>`
and end with a JSON table of boxes, `[y_min, x_min, y_max, x_max]` on a 0-1000 canvas. Given boxes,
the upsampler writes that caption with its `*_bbox` prompts, and the given boxes are re-imposed on
its answer.
"""

from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Callable, Sequence
from importlib.resources import files
from typing import Any

import torch
from cozy_runtime import lm, vlm
from cozy_runtime.author import Config, Loader, Model, UnsupportedInput, uses_components
from PIL import Image, ImageOps
from transformers import AutoConfig
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5 import modeling_qwen3_5

COMPONENT = "model"
#: The drafter's component in Qwen3.5-9B's lane; only its MTP head (`mtp.*`) is built.
DRAFT = "model"
MAX_NEW_TOKENS = 3584
MAX_REFERENCES = 10
PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
#: By whether the request carries boxes.
I2I_SCAFFOLD = {
    False: (
        "Reference images follow in order (the first image, the second image, ...).\n"
        "User request: {instruction}\n"
        "Output the caption."
    ),
    True: (
        "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
        "User editing instruction: {instruction}\n"
        "Decide the edit, imagine the target image, and output the annotation record."
    ),
}
REFERENCE_MULTIPLE = 32
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, 11)}}
#: The layout canvas: boxes are integers 0-1000 on each axis.
SCALE = 1000
ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
#: A row: {"id", "bbox", "desc"}, or {"id", "from", "src_bbox", "tgt_bbox", "desc"} in an edit.
Row = dict[str, Any]


def canvas_box(box: Sequence[float]) -> list[int]:
    """`[x0, y0, x1, y1]` in 0-1 as `[y_min, x_min, y_max, x_max]` on the 0-1000 canvas."""
    x0, y0, x1, y1 = (round(min(max(float(v), 0.0), 1.0) * SCALE) for v in box)
    return [y0, x0, y1, x1]


def with_table(caption: str, rows: list[Row]) -> str:
    """Klein's layout caption: the caption, one space, the table."""
    return f"{caption.rstrip()} {json.dumps(rows, ensure_ascii=False)}"


def split(text: str) -> tuple[str, list[Any] | None]:
    """The caption and the rows: from the last '[' at which the rest parses as a JSON list. Not a
    regex: descs may hold brackets and escaped quotes."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


def valid_box(box: Any) -> list[Any] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == math.inf for v in box):
        return None
    y1, x1, y2, x2 = box
    # max <= 1 is 0-1 coordinates, not the 0-1000 canvas
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > SCALE or x2 > SCALE or max(box) <= 1:
        return None
    return list(box)


def detect(prompt: str) -> tuple[str, list[Row]] | None:
    """The prompt and its rows when it already carries boxes on the 0-1000 canvas: "<caption> [rows]"
    (Klein's caption) or the upsampler's own "ORIGINAL PROMPT: ... LOCKED BOUNDING-BOX ANNOTATIONS"
    input. A trailing list that is not all box rows (a caption ending in "[SALE]") is no table."""
    text, rows = split(prompt)
    if not rows or not all(
        isinstance(r, dict) and isinstance(r.get("id"), str) and ID.fullmatch(r["id"])
        and any(valid_box(r.get(k)) for k in ("bbox", "src_bbox", "tgt_bbox"))
        for r in rows
    ):
        return None
    locked = re.fullmatch(r"ORIGINAL PROMPT:\n(.*?)\n\nLOCKED BOUNDING-BOX ANNOTATIONS .*", text, re.S)
    return (locked.group(1) if locked else text), rows


def is_finished(prompt: str) -> bool:
    """A layout caption, not a request: its table is cited in full as `<id>`s. Klein reads it as it is."""
    found = detect(prompt)
    return found is not None and all(f"<{row['id']}>" in found[0] for row in found[1])


def upsampling_input(prompt: str, rows: list[Row]) -> str:
    return (
        f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n"
        "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
        "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
        f"provenance value, null, and coordinate exactly:\n{json.dumps(rows, ensure_ascii=False)}"
    )


def _valid_added(row: Any, caption: str) -> Row | None:
    """A row the upsampler added: kept only when its id is cited as `<id>` and it is well formed."""
    if not isinstance(row, dict):
        return None
    row_id, desc = row.get("id"), row.get("desc")
    if (not isinstance(row_id, str) or not ID.fullmatch(row_id) or f"<{row_id}>" not in caption
            or not isinstance(desc, str) or not desc.strip()):
        return None
    if (box := valid_box(row.get("bbox"))) is not None:
        return {"id": row_id, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = valid_box(src) if src is not None else None
    tgt_box = valid_box(tgt) if tgt is not None else None
    if (not isinstance(origin, str) or not re.fullmatch(r"ref_image_\d+|target", origin)
            or (src is not None and src_box is None) or (tgt is not None and tgt_box is None)
            or (src_box is None and tgt_box is None) or (origin == "target") == (src_box is not None)):
        return None
    return {"id": row_id, "from": origin, "src_bbox": src_box, "tgt_bbox": tgt_box, "desc": desc.strip()}


def _iou(a: list[Any], b: list[Any]) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def reconcile(text: str, locked: list[Row]) -> str:
    """The upsampler's answer with the given boxes re-imposed: a row re-planning a locked box under
    its own id (IoU >= 0.9) takes the locked id, in the rows and the caption's citations; locked rows
    are kept verbatim with the upsampler's desc; its own valid rows follow."""
    caption, rows = split(text)
    rows = rows or []
    locked_ids = {r["id"] for r in locked}
    row_ids = {r.get("id") for r in rows if isinstance(r, dict)}
    renamed: dict[str, str] = {}
    for lock in locked:
        lock_box = lock.get("bbox") or lock.get("tgt_bbox") or lock.get("src_bbox")
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") in renamed or row.get("id") in locked_ids:
                continue
            box = valid_box(row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox"))
            if box is not None and _iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")
    by_id = {r["id"]: r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)}
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append({**lock, "desc": desc.strip() if isinstance(desc, str) and desc.strip() else lock.get("desc", "")})
    for row in rows:
        added = _valid_added(row, caption)
        if added is not None and added["id"] not in {r["id"] for r in out}:
            out.append(added)
    return with_table(caption, out)


class DrafterGraph:
    def __init__(self, config: Config) -> None:
        # The head-only lane keeps Qwen3.5-9B's config (`drafter_head`).
        with no_init_weights():
            head = lm.Draft(AutoConfig.for_model(**config.mapping()), modeling_qwen3_5)
        self.components: dict[str, torch.nn.Module] = {DRAFT: head.to(torch.bfloat16).eval()}


class Drafter(Model[DrafterGraph], encoded_leaves="accept"):
    """Qwen3.5-9B's multi-token-prediction head (BFL's upsampler is that model fine-tuned, its
    head left out). The upsampler drafts tokens with it and keeps the ones its own decode
    agrees with: ~1.9 tokens a step."""

    graph: DrafterGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(DrafterGraph, factory=DrafterGraph)

    @uses_components(DRAFT)
    def drafting(self, body: Callable[[], str]) -> str:
        with lm.drafting(self.graph.components[DRAFT]):
            return body()


class UpsamplerGraph:
    def __init__(self, config: Config) -> None:
        # Every parameter in its stored dtype: Qwen3.5 keeps A_log and its gated norms in f32.
        self.components: dict[str, torch.nn.Module] = {COMPONENT: vlm.construct(config.mapping(), config)[0]}


def _floor(value: float) -> int:
    return max(REFERENCE_MULTIPLE, int(math.floor(value / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)


def reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    cap = MAX_PIXELS_BY_REFS[n_refs]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = _floor(width / beta), _floor(height / beta)
    if w * h > cap:
        if w == REFERENCE_MULTIPLE:
            h = _floor(cap / w)
        else:
            w = _floor(cap / h)
    return w, h


def policy_reference(image: Image.Image, n_refs: int) -> Image.Image:
    """RGB over white, LANCZOS to the size budget, JPEG q92: what the upsampler was trained on."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


class Upsampler(Model[UpsamplerGraph], encoded_leaves="accept", min_bits=4, phase="decode"):
    """Rewrites a short request, and its references when editing, into the caption Klein wants."""

    graph: UpsamplerGraph
    processor: Any
    #: BFL's system prompts (`system_prompts.json` of its repository), by task.
    system_prompts: dict[str, str]

    def execution_plan(self, plan: Any) -> str:
        return lm.install(plan, self.graph.components[COMPONENT])

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpsamplerGraph, factory=UpsamplerGraph)
        self.processor = vlm.processor(loader.assets)
        self.system_prompts = json.loads(files(__package__).joinpath("system_prompts.json").read_bytes())

    @uses_components(
        COMPONENT, memoize=True,
        memo_dependencies=(policy_reference, detect, upsampling_input, reconcile, _valid_added),
    )
    def upsample(self, prompt: str, images: list[Image.Image], canvas: str | None, table: str = "") -> str:
        """`table` is the request's rows as JSON, locked; without them, boxes the prompt carries
        lock themselves."""
        if len(images) > MAX_REFERENCES:
            raise UnsupportedInput(f"the upsampler takes at most {MAX_REFERENCES} references")
        model, processor = self.graph.components[COMPONENT], self.processor
        task = "i2i" if images else "t2i"
        locked: list[Row] | None = json.loads(table) if table else None
        if locked is None and (found := detect(prompt)):
            prompt, locked = found
        if locked:
            prompt = upsampling_input(prompt, locked)
        if images:
            references = [policy_reference(image, len(images)) for image in images]
            text = I2I_SCAFFOLD[bool(locked)].format(instruction=prompt)
            content: Any = [{"type": "image"} for _ in references] + [{"type": "text", "text": text}]
        else:
            references = []
            content = f"TARGET CANVAS: {canvas}\n{prompt}" if canvas else prompt
        messages = [
            {"role": "system", "content": self.system_prompts[f"{task}_bbox" if locked else task]},
            {"role": "user", "content": content},
        ]
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = processor(
            text=[chat], images=references or None, return_tensors="pt"
        ).to(torch.device(str(self.device)))
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        tokenizer = processor.tokenizer
        stops = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
        out = lm.generate(
            model, inputs, max_new_tokens=MAX_NEW_TOKENS, stop=stops, pad=tokenizer.pad_token_id,
            presence_penalty=PRESENCE_PENALTY[task],
        )
        caption = str(tokenizer.decode(out[0], skip_special_tokens=True)).strip()
        if not caption:
            raise UnsupportedInput("the prompt upsampler returned an empty caption", code="output_integrity")
        return reconcile(caption, locked) if locked else caption
