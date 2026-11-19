"""Bria's prompt-to-JSON VLMs, Qwen3-VL-4B fine-tunes served from their as-is uploads: `FIBO-vlm`
expands a prompt into a structured caption, `FIBO-edit-vlm` writes an edit's caption and instruction
from the image and the request.

Prompts, image limits, sampling and cleaning follow Bria's `FIBO-VLM-prompt-to-JSON@986ca8c4` and
`FIBO-edit-prompt-to-JSON@6d815165` blocks (`fibo_vlm_prompt_to_json.py`): `<generate>\\n{prompt}`, and
`<edit>\\nEditing instructions:\\n{prompt}` after the image, thumbnailed to 1024 px and seen at 3,136 to
1,003,520 pixels; sampled at temperature 0.2 (0.4 for an edit) and top-p 0.9, with each VLM's generation-config
top-k 20, from the render's seed, at most 4096 tokens; the answer kept to its trained fields (`clean`).
"""

from __future__ import annotations

import json
import math
from typing import cast

import torch
import torch.nn.functional as F
from cozy_runtime import lm, vlm
from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author import Config, Loader, Model, Serving, UnsupportedInput, memoize, uses_components
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    PreTrainedModel,
    Qwen3VLProcessor,
)
from transformers.initialization import no_init_weights

COMPONENT = "model"
MAX_NEW_TOKENS = 4096
THUMBNAIL = 1024
#: Bria's image limits, in place of the checkpoint's.
BRIA_IMAGE = {"size": {"shortest_edge": 4 * 28 * 28, "longest_edge": 1280 * 28 * 28}, "min_pixels": None,
              "max_pixels": None}
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
#: Bria's `sampling_temperature` defaults: the main block's, the edit block's; their `sampling_top_p`; and the top-k
#: of both VLMs' `generation_config.json`, which `generate` keeps.
TEMPERATURE, EDIT_TEMPERATURE, TOP_P, TOP_K = 0.2, 0.4, 0.9, 20
#: The fields `prepare_clean_caption` keeps, in its order: the main block's, and the edit block's.
FIELDS = ("short_description", "objects", "background_setting", "lighting", "aesthetics",
          "photographic_characteristics", "style_medium", "text_render", "context", "artistic_style")
EDIT_FIELDS = (*FIELDS, "edit_instruction")
_DROPPED = object()


def _kept(value: object) -> object:
    """`value` without its empty entries, inner first, as Bria's `remap(caption, visit=keep)`: None, "", {}, [] and
    NaN go, a container emptied by that too."""
    if isinstance(value, dict):
        value = {key: kept for key, inner in value.items() if (kept := _kept(inner)) is not _DROPPED}
    elif isinstance(value, list):
        value = [kept for inner in value if (kept := _kept(inner)) is not _DROPPED]
    empty = value is None or (isinstance(value, (str, dict, list)) and not value)
    empty = empty or (isinstance(value, float) and math.isnan(value))
    return _DROPPED if empty else value


def _score(value: float, bounds: tuple[float, float, float, float]) -> str:
    words = ("very low", "low", "medium", "high")
    return next((word for word, bound in zip(words, bounds, strict=True) if value < bound), "very high")


def clean(answer: dict[str, object], fields: tuple[str, ...], strings_as_json: bool) -> str:
    """Bria's `prepare_clean_caption`: `fields` in order, empty values dropped, its scores as words in `aesthetics`,
    compact `ujson`-style JSON. `strings_as_json`: a string value is read as JSON where it is JSON, as only the edit
    block's does."""
    scores: dict[str, object] = {}
    if isinstance(pick := answer.get("pickascore"), (int, float)):
        scores["preference_score"] = _score(pick, (0.78, 0.82, 0.87, 0.91))
    if isinstance(aesthetic := answer.get("aesthetic_score"), (int, float)):
        scores["aesthetic_score"] = _score(aesthetic, (5.5, 6, 7, 7.6))
    caption: dict[str, object] = {}
    for field in fields:
        if field not in answer:
            continue
        value = answer[field]
        if strings_as_json and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        caption[field] = value
    kept = _kept(caption)
    caption = kept if isinstance(kept, dict) else {}
    aesthetics = caption.get("aesthetics")
    if aesthetics is None:
        if scores:
            caption["aesthetics"] = scores
    elif isinstance(aesthetics, dict):
        aesthetics.update(scores)
    return json.dumps(caption, separators=(",", ":"))


class TiedHead(torch.nn.Module):
    """The language head of a tied checkpoint (`tie_word_embeddings`): logits from the input
    embedding's weight, which the checkpoint stores once. Holding the embedding outside the module
    tree keeps that weight under its one stored name, so Runtime fills it once."""

    embedding: torch.nn.Embedding

    def __init__(self, embedding: torch.nn.Embedding) -> None:
        super().__init__()
        object.__setattr__(self, "embedding", embedding)
        self.out_features = embedding.num_embeddings

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.embedding.weight)


class CaptionerGraph:
    def __init__(self, config: Config) -> None:
        mapping = config.mapping()
        with no_init_weights():
            model: PreTrainedModel = AutoModelForImageTextToText.from_config(AutoConfig.for_model(**mapping), dtype=torch.bfloat16)  # type: ignore[no-untyped-call]  # Transformers' auto classes are unannotated
        if model.config.get_text_config().tie_word_embeddings:
            model.lm_head = TiedHead(cast(torch.nn.Embedding, model.get_input_embeddings()))  # a token embedding
        self.components: dict[str, PreTrainedModel] = {COMPONENT: model.eval()}  # type: ignore[no-untyped-call]  # Transformers' eval is unannotated


class Captioner(Model[CaptionerGraph]):
    """One of Bria's prompt-to-JSON VLMs."""

    serving = Serving(min_bits=4, phase="decode")

    graph: CaptionerGraph
    processor: Qwen3VLProcessor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(CaptionerGraph)
        self.processor = cast(Qwen3VLProcessor, vlm.processor(loader.assets, image=BRIA_IMAGE))  # Qwen3-VL's own

    def execution_plan(self, plan: Plan) -> str:
        return lm.install(plan, self.graph.components[COMPONENT])

    @memoize
    @uses_components(COMPONENT)
    def caption(self, prompt: str, image: Image.Image | None, seed: int) -> str:
        """The VLM's answer for `prompt` (`<generate>`), or with `image` the edit's caption with its
        `edit_instruction` (`<edit>`), as compact JSON, before `clean`; sampled from `seed`."""
        content: list[dict[str, str]] = []
        images = []
        if image is None:
            content.append({"type": "text", "text": f"<generate>\n{prompt.strip()}"})
        else:
            image = image.convert("RGB")
            image.thumbnail((THUMBNAIL, THUMBNAIL))
            images.append(image)
            content.append({"type": "image"})
            content.append({"type": "text", "text": f"<edit>\nEditing instructions:\n{prompt.strip()}"})
        model, processor = self.graph.components[COMPONENT], self.processor
        chat = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True  # type: ignore[arg-type]  # content as parts
        )
        inputs = processor(text=[chat], images=images or None, padding=True, return_tensors="pt").to(
            torch.device(str(self.device))
        )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        tokenizer = processor.tokenizer
        stops = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
        out = lm.generate(model, inputs, max_new_tokens=MAX_NEW_TOKENS, stop=stops, pad=stops[-1],
                          temperature=TEMPERATURE if image is None else EDIT_TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                          seed=seed)
        text = str(tokenizer.decode(out[0], skip_special_tokens=True)).strip()
        text = text.replace(" End of texture answer.", "")
        try:
            caption = json.loads(text)
        except json.JSONDecodeError as error:
            raise UnsupportedInput(
                f"the prompt upsampler wrote no JSON caption ({error})",
                code="output_integrity",
            ) from error
        if not isinstance(caption, dict):
            raise UnsupportedInput("the prompt upsampler wrote no JSON object", code="output_integrity")
        return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
