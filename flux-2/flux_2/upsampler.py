"""BFL's local prompt upsampler: FLUX.2 [dev]'s text encoder, Mistral-Small-3.2-24B, as the whole
vision-language model (`black-forest-labs/flux2` 50fe516, `text_encoder.py` `upsample_prompt`). BFL
uses it for [klein] too. Its system prompts, the references concatenated side by side under 768²
pixels, 512 new tokens and its sampling are BFL's: temperature 0.15 (Mistral-Small-3.2's generation config) with
Transformers' default top-k 50 and top-p 1, from the render's seed.
"""

from __future__ import annotations

from typing import cast


import torch
from cozy_runtime import lm
from cozy_runtime.author import Config, Loader, Model, Serving, memoize, uses_components
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2 import (
    UPSAMPLING_MAX_IMAGE_SIZE,
    _validate_and_process_images,
    format_input,
)
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE_UPSAMPLING_I2I, SYSTEM_MESSAGE_UPSAMPLING_T2I
from PIL import Image, ImageOps
from transformers import initialization as transformer_init

from transformers import PixtralProcessor, PreTrainedModel

from .model import processor, text_encoder

COMPONENT = "text_encoder"
MAX_NEW_TOKENS = 512
#: BFL's `upsample_prompt(temperature=0.15)` under `do_sample=True`; Transformers' defaults for the rest.
TEMPERATURE, TOP_K, TOP_P = 0.15, 50, 1.0


class UpsamplerGraph:
    def __init__(self, config: Config) -> None:
        with transformer_init.no_init_weights():
            model = text_encoder(config.section(COMPONENT).mapping())
        self.components: dict[str, torch.nn.Module] = {COMPONENT: model.eval()}


class Flux2Upsampler(Model[UpsamplerGraph]):
    """Rewrites a short request, and its references when editing, into a detailed FLUX.2 prompt."""

    serving = Serving(min_bits=4, phase="decode")

    graph: UpsamplerGraph
    processor: PixtralProcessor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpsamplerGraph)
        self.processor = processor(loader.assets)

    @memoize
    @uses_components(COMPONENT)
    def upsample(self, prompt: str, images: list[Image.Image], seed: int) -> str:
        model, processor = self.graph.components[COMPONENT], self.processor
        system = SYSTEM_MESSAGE_UPSAMPLING_I2I if images else SYSTEM_MESSAGE_UPSAMPLING_T2I
        shown = None
        if images:
            upright = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
            shown = _validate_and_process_images([upright], Flux2ImageProcessor(), UPSAMPLING_MAX_IMAGE_SIZE)
        # Tokenized with return_dict, the template is a BatchFeature (Transformers types it as text).
        inputs = processor.apply_chat_template(  # type: ignore[attr-defined]
            format_input([prompt], system, shown), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(torch.device(str(self.device)))
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        tokenizer = processor.tokenizer
        # `model` is a causal LM (`text_encoder`).
        out = lm.generate(cast(PreTrainedModel, model), inputs, max_new_tokens=MAX_NEW_TOKENS, stop=[tokenizer.eos_token_id],
                          pad=tokenizer.pad_token_id, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, seed=seed)
        return str(tokenizer.decode(out[0], skip_special_tokens=True)).strip()
