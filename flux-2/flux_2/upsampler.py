"""BFL's local prompt upsampler: FLUX.2 [dev]'s text encoder, Mistral-Small-3.2-24B, as the whole
vision-language model (`black-forest-labs/flux2` 50fe516, `text_encoder.py` `upsample_prompt`). BFL
uses it for [klein] too. Its system prompts, the references concatenated side by side under 768²
pixels and 512 new tokens are BFL's; decoding is greedy where BFL samples at temperature 0.15.
"""

from __future__ import annotations

from typing import Any

import torch
from cozy_runtime import lm
from cozy_runtime.author import Config, Loader, Model, UnsupportedInput, uses_components
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2 import (
    UPSAMPLING_MAX_IMAGE_SIZE,
    _validate_and_process_images,
    format_input,
)
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE_UPSAMPLING_I2I, SYSTEM_MESSAGE_UPSAMPLING_T2I
from PIL import Image, ImageOps
from transformers import initialization as transformer_init

from .model import processor, section, text_encoder

COMPONENT = "text_encoder"
MAX_NEW_TOKENS = 512


class UpsamplerGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            model = text_encoder(section(config.mapping(), COMPONENT))
        self.components: dict[str, torch.nn.Module] = {COMPONENT: model.eval()}


class Flux2Upsampler(Model[UpsamplerGraph], encoded_leaves="accept", min_bits=4, phase="decode"):
    """Rewrites a short request, and its references when editing, into a detailed FLUX.2 prompt."""

    graph: UpsamplerGraph
    processor: Any

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpsamplerGraph, factory=UpsamplerGraph)
        self.processor = processor(loader.assets)

    @uses_components(COMPONENT, memoize=True, memo_dependencies=(processor,))
    def upsample(self, prompt: str, images: list[Image.Image]) -> str:
        model, processor = self.graph.components[COMPONENT], self.processor
        system = SYSTEM_MESSAGE_UPSAMPLING_I2I if images else SYSTEM_MESSAGE_UPSAMPLING_T2I
        shown = None
        if images:
            upright = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
            shown = _validate_and_process_images([upright], Flux2ImageProcessor(), UPSAMPLING_MAX_IMAGE_SIZE)
        inputs = processor.apply_chat_template(
            format_input([prompt], system, shown), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(torch.device(str(self.device)))
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        tokenizer = processor.tokenizer
        out = lm.generate(model, inputs, max_new_tokens=MAX_NEW_TOKENS, stop=[tokenizer.eos_token_id],
                          pad=tokenizer.pad_token_id)
        caption = str(tokenizer.decode(out[0], skip_special_tokens=True)).strip()
        if not caption:
            raise UnsupportedInput("the prompt upsampler returned an empty caption", code="output_integrity")
        return caption
