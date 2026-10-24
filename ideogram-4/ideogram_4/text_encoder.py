"""Qwen3-VL-8B-Instruct, the frozen text encoder Ideogram 4 conditions on, from Qwen's own release
(`Qwen/Qwen3-VL-8B-Instruct`). Two slots read it: `text_encoder` builds what conditioning reads (the
36 language layers, whose hidden states Ideogram concatenates), and `upsampler` the whole language
model, which writes a structured caption from a plain prompt (Diffusers' local prompt enhancer:
Ideogram's system message, JSON constrained to the caption schema by `outlines`). Prompts are text
only, so the vision tower is never built.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from cozy_runtime.author import Config, Loader, Model, ModelAssets, UnsupportedInput, uses_components
from diffusers.pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from diffusers.pipelines.ideogram4.prompt_enhancer import (
    PROMPT_UPSAMPLE_TEMPERATURE,
    build_caption_logits_processor,
    generate_captions,
)
from tokenizers import Tokenizer
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedTokenizerFast
from transformers import initialization as transformer_init

COMPONENT = "model"
#: Ideogram 4's text budget (`max_text_tokens`).
MAX_TOKENS = 2048


def tokenizer(assets: ModelAssets) -> PreTrainedTokenizerFast:
    """Qwen's tokenizer and chat template (the same template Ideogram ships), parsed in memory."""
    config = json.loads(assets.read("tokenizer_config.json"))
    for key in ("added_tokens_decoder", "tokenizer_file", "name_or_path", "tokenizer_class", "processor_class",
                "chat_template", "auto_map"):
        config.pop(key, None)
    template = json.loads(assets.read("chat_template.json"))["chat_template"]
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer.json").decode()), chat_template=template, **config
    )


def language_model(config: Config) -> nn.Module:
    """Qwen3-VL as Qwen stores it (`model.language_model`, `lm_head`), without its vision tower."""
    with transformer_init.no_init_weights():
        model = AutoModelForImageTextToText.from_config(AutoConfig.for_model(**config.mapping()), dtype=torch.bfloat16)
    model.model.visual = None
    return model.eval()


def _conditioning_only(model: nn.Module) -> None:
    """Ideogram reads every decoder layer's output up to the last (layer 35, before the final
    norm): the norm and the head are never run, so they are not built."""
    model.model.language_model.norm = nn.Identity()
    model.lm_head = nn.Identity()


class EncoderGraph:
    def __init__(self, config: Config) -> None:
        model = language_model(config)
        _conditioning_only(model)
        self.components: dict[str, nn.Module] = {COMPONENT: model}


class QwenEncoder(Model[EncoderGraph], encoded_leaves="accept"):
    """Qwen3-VL-8B-Instruct as Ideogram 4's text encoder."""

    graph: EncoderGraph
    tokenizer: PreTrainedTokenizerFast

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(EncoderGraph, factory=EncoderGraph)
        self.tokenizer = tokenizer(loader.assets)

    @uses_components(COMPONENT, memoize=True, memo_dependencies=(tokenizer, language_model, _conditioning_only))
    def encode(self, caption: str) -> torch.Tensor:
        """`Ideogram4Pipeline.encode_prompt` for one caption, unpadded: the hidden states of
        Ideogram's 13 layers concatenated per token, `[tokens, 53248]` in bf16."""
        model = self.graph.components[COMPONENT]
        device = next(model.parameters()).device
        messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        count = int(ids.shape[1])
        if count > MAX_TOKENS:
            raise UnsupportedInput(f"the caption is {count} tokens; Ideogram 4 reads at most {MAX_TOKENS}", code="prompt")
        with torch.inference_mode():
            states = Ideogram4Pipeline._get_text_encoder_hidden_states(
                model.model, ids.to(device), torch.ones_like(ids, device=device), torch.arange(count, device=device)[None]
            )
        return torch.stack(states).permute(1, 2, 3, 0).reshape(count, -1).cpu()


class UpsamplerGraph:
    def __init__(self, config: Config) -> None:
        self.components: dict[str, nn.Module] = {COMPONENT: language_model(config)}


class QwenUpsampler(Model[UpsamplerGraph], encoded_leaves="accept", min_bits=4, phase="decode"):
    """Qwen3-VL-8B-Instruct writing Ideogram 4's structured caption from a plain prompt."""

    graph: UpsamplerGraph
    tokenizer: PreTrainedTokenizerFast
    constrained: Any = None

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpsamplerGraph, factory=UpsamplerGraph)
        self.tokenizer = tokenizer(loader.assets)
        # The caption schema's grammar, compiled once; `generate_captions` resets it for each caption.
        self.constrained = build_caption_logits_processor(self.graph.components[COMPONENT], self.tokenizer)

    @uses_components(COMPONENT, memoize=True, memo_dependencies=(tokenizer, language_model))
    def caption(self, prompt: str, *, width: int, height: int, seed: int) -> str:
        """Diffusers' `upsample_prompt` for one prompt: Ideogram's caption system message and the
        target aspect ratio, sampled at temperature 1 from `seed`, constrained to the caption's JSON
        schema; returned as compact JSON."""
        model = self.graph.components[COMPONENT]
        device = next(model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            (text,) = generate_captions(
                model, self.tokenizer, self.constrained, prompt, height, width,
                temperature=PROMPT_UPSAMPLE_TEMPERATURE, max_new_tokens=MAX_TOKENS, generator=generator, device=device,
            )
        try:
            caption = json.loads(text)
        except json.JSONDecodeError as error:
            raise UnsupportedInput(
                f"the prompt upsampler wrote no JSON caption ({error}); run with prompt_upsampling=false",
                code="output_integrity",
            ) from error
        return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
