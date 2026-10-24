"""Bria's prompt-to-JSON VLMs, Qwen3-VL-4B fine-tunes served from their as-is uploads: `FIBO-vlm`
expands a prompt into a structured caption, `FIBO-edit-vlm` writes an edit's caption and instruction
from the image and the request.

Prompts and image limits follow Bria's `FIBO-VLM-prompt-to-JSON` and `FIBO-edit-prompt-to-JSON`
blocks: `<generate>\\n{prompt}`, and `<edit>\\nEditing instructions:\\n{prompt}` after the image,
thumbnailed to 1024 px and seen at 3,136 to 1,003,520 pixels. Decoding is greedy (Bria samples at
temperature 0.2 and 0.4), so a caption is a function of its request.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from cozy_runtime import lm
from cozy_runtime.author import Config, Loader, Model, ModelAssets, UnsupportedInput, uses_components
from PIL import Image
from tokenizers import Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)
from transformers.initialization import no_init_weights

COMPONENT = "model"
MAX_NEW_TOKENS = 4096
THUMBNAIL = 1024
IMAGE_PIXELS = {"shortest_edge": 4 * 28 * 28, "longest_edge": 1280 * 28 * 28}
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


class TiedHead(torch.nn.Module):
    """The language head of a tied checkpoint (`tie_word_embeddings`): logits from the input
    embedding's weight, which the checkpoint stores once. Holding the embedding outside the module
    tree keeps that weight under its one stored name, so Runtime fills it once."""

    def __init__(self, embedding: torch.nn.Embedding) -> None:
        super().__init__()
        object.__setattr__(self, "embedding", embedding)
        self.out_features = embedding.num_embeddings

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.embedding.weight)


class CaptionerGraph:
    def __init__(self, config: Config) -> None:
        mapping: Mapping[str, Any] = config.mapping()
        with no_init_weights():
            model = AutoModelForImageTextToText.from_config(AutoConfig.for_model(**mapping), dtype=torch.bfloat16)
        if model.config.get_text_config().tie_word_embeddings:
            model.lm_head = TiedHead(model.get_input_embeddings())
        self.components: dict[str, torch.nn.Module] = {COMPONENT: model.eval()}


def build_processor(assets: ModelAssets) -> Qwen3VLProcessor:
    """The checkpoint's tokenizer, chat template and processors, parsed in memory, with Bria's
    image limits."""
    tokenizer_config = json.loads(assets.read("tokenizer_config.json"))
    for key in ("added_tokens_decoder", "tokenizer_file", "processor_class", "tokenizer_class",
                "chat_template", "name_or_path"):
        tokenizer_config.pop(key, None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer.json").decode()), **tokenizer_config
    )
    image = json.loads(assets.read("preprocessor_config.json"))
    video = json.loads(assets.read("video_preprocessor_config.json"))
    for section in (image, video):
        for key in ("processor_class", "image_processor_type", "video_processor_type", "min_pixels", "max_pixels"):
            section.pop(key, None)
    image["size"] = dict(IMAGE_PIXELS)
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(**image),
        video_processor=Qwen3VLVideoProcessor(**video),
        chat_template=assets.read("chat_template.jinja").decode(),
    )


class Captioner(Model[CaptionerGraph], encoded_leaves="accept", min_bits=4, phase="decode"):
    """One of Bria's prompt-to-JSON VLMs."""

    graph: CaptionerGraph
    processor: Qwen3VLProcessor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(CaptionerGraph, factory=CaptionerGraph)
        self.processor = build_processor(loader.assets)

    def execution_plan(self, plan: Any) -> str:
        return lm.install(plan, self.graph.components[COMPONENT])

    @uses_components(COMPONENT, memoize=True, memo_dependencies=(build_processor,))
    def caption(self, prompt: str, image: Image.Image | None) -> str:
        """The structured caption for `prompt` (`<generate>`), or with `image` the edit's caption
        with its `edit_instruction` (`<edit>`), as compact JSON."""
        content: list[dict[str, Any]] = []
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
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[chat], images=images or None, padding=True, return_tensors="pt").to(
            torch.device(str(self.device))
        )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        tokenizer = processor.tokenizer
        stops = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
        out = lm.generate(model, inputs, max_new_tokens=MAX_NEW_TOKENS, stop=stops, pad=stops[-1])
        text = str(tokenizer.decode(out[0], skip_special_tokens=True)).strip()
        text = text.replace(" End of texture answer.", "")
        try:
            caption = json.loads(text)
        except json.JSONDecodeError as error:
            raise UnsupportedInput(
                f"the prompt upsampler wrote no JSON caption ({error}); run with prompt_upsampling=false",
                code="output_integrity",
            ) from error
        if not isinstance(caption, dict):
            raise UnsupportedInput("the prompt upsampler wrote no JSON object", code="output_integrity")
        return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
