"""Lightricks' prompt enhancer for LTX-2.5: Gemma 4 E2B-it rewriting a prompt into the model's caption style, with
upstream's own text-to-video and image-to-video system prompts and decoding (ltx_core/text_encoders/gemma/encoders
`enhance_t2v`/`enhance_i2v`, `GEMMA4_ENHANCE_GENERATION_KWARGS`; Diffusers 0.43 `LTX2InContextPipeline.enhance_prompt`
reproduces them, and its copies of the system prompts are byte-identical to upstream's)."""

from __future__ import annotations

import json

import torch
from cozy_runtime.author import Config, Loader, Model, ModelAssets, Serving, uses_components
from diffusers.pipelines.ltx2.prompt_enhancement import (
    _pad_inputs_for_attention_alignment,
    _prepare_enhance_image,
    clean_response,
)
from diffusers.pipelines.ltx2.utils import (
    GEMMA4_PROMPT_ENHANCEMENT_CONFIG,
    LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT,
    LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT,
)
from PIL import Image
from tokenizers import Tokenizer
from transformers import (
    AddedToken, Gemma4AudioFeatureExtractor, Gemma4Config, Gemma4ForConditionalGeneration, Gemma4ImageProcessor,
    Gemma4Processor, Gemma4VideoProcessor, GemmaTokenizer,
)

from .model import stored

COMPONENT = "model"
#: Tokenizer settings that name files or classes, not the tokenizer's own behaviour.
METADATA = ("tokenizer_class", "auto_map", "name_or_path", "tokenizer_file", "vocab_file", "processor_class", "backend")


def processor(assets: ModelAssets) -> Gemma4Processor:
    """The upstream processor from the checkpoint's files as its mirror keeps them (`processor_config.json`,
    `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`), built in memory as AutoProcessor builds it."""
    def read(name: str) -> bytes:
        return assets.read(f"{COMPONENT}/{name}")
    settings = json.loads(read("processor_config.json"))
    images, video, audio = settings.pop("image_processor"), settings.pop("video_processor"), settings.pop("feature_extractor")
    for part, key in ((images, "image_processor_type"), (video, "video_processor_type"), (audio, "feature_extractor_type")):
        part.pop(key, None)
    settings.pop("processor_class", None)
    words = json.loads(read("tokenizer_config.json"))
    added = words.pop("added_tokens_decoder", {})
    for key in METADATA:
        words.pop(key, None)
    chat = read("chat_template.jinja").decode()
    tokenizer = GemmaTokenizer(tokenizer_object=Tokenizer.from_str(read("tokenizer.json").decode()),
                               added_tokens_decoder={int(i): AddedToken(**token) for i, token in added.items()}, **words)
    tokenizer.chat_template = chat
    return Gemma4Processor(feature_extractor=Gemma4AudioFeatureExtractor(**audio), image_processor=Gemma4ImageProcessor(**images),
                           tokenizer=tokenizer, video_processor=Gemma4VideoProcessor(**video), chat_template=chat, **settings)


class Graph:
    def __init__(self, config: Config) -> None:
        settings = config.section(COMPONENT).mapping()
        text = config.section(COMPONENT).section("text_config").mapping()
        # Canonical JSON writes 30.0 as 30; Gemma's float-only field takes it back as a float.
        if type(text.get("final_logit_softcapping")) is int:
            text["final_logit_softcapping"] = float(text["final_logit_softcapping"])
        settings["text_config"] = text
        self.components = {COMPONENT: stored(Gemma4ForConditionalGeneration(Gemma4Config(**settings)), COMPONENT, config)}


class Enhancer(Model[Graph]):
    serving = Serving(min_bits=16, phase="decode")
    graph: Graph
    processor: Gemma4Processor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph)
        self.processor = processor(loader.assets)

    @uses_components(COMPONENT)
    def rewrite(self, prompt: str, picture: Image.Image | None) -> str:
        """`prompt` as LTX-2.5's training captions read: the image-to-video form opens on `picture`, the video's first
        frame; greedy, so a prompt always reads the same."""
        model = self.graph.components[COMPONENT]
        config = GEMMA4_PROMPT_ENHANCEMENT_CONFIG
        if picture is None:
            messages: list[dict[str, object]] = [
                {"role": "system", "content": LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": f"{config.user_prompt_prefix}: {prompt}"},
            ]
            image = None
        else:
            messages = [
                {"role": "system", "content": LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"User Raw Input Prompt: {prompt}."}]},
            ]
            image = _prepare_enhance_image(picture)
        template = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=template, images=image, return_tensors="pt").to(torch.device(str(self.device)))
        pad = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        inputs = _pad_inputs_for_attention_alignment(inputs, pad_token_id=pad)
        with torch.inference_mode():
            made = model.generate(**inputs, max_new_tokens=config.max_new_tokens, **config.generation_kwargs)
        return str(clean_response(self.processor.tokenizer.decode(made[0][len(inputs.input_ids[0]):], skip_special_tokens=True)))
