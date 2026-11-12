"""Pruna's prompt upsampler for P-H3-Image (a Qwen3.5-9B fine-tune): a short request to the full prompt the
model renders best, or for an edit a brief of what to keep and what to change."""

from __future__ import annotations

import math
import re
from typing import cast

import torch
from cozy_runtime import lm, vlm
from cozy_runtime.accel.kernels import Plan
from cozy_runtime.author import Config, Loader, Model, Serving, memoize, uses_components
from PIL import Image
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedModel, ProcessorMixin
from transformers.initialization import no_init_weights

from .model import VISION_PIXELS, resized, size_at_area, stored

#: The upsampler's components, by slot.
UPSAMPLER_COMPONENTS = ("prompt_upsampler", "prompt_upsampler_processor")
ASPECT_RATIOS = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16}
MAX_NEW_TOKENS = {"t2i": 1024, "edit": 1536}
_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape",
             "non_diegetic_music")  # fmt: skip
_SECTION = re.compile(r"(?m)^(" + "|".join(_SECTIONS) + r"):[ \t]*$")
_RETENTION = re.compile(
    r"^<(Picture|Subject) (\d+)>(?: \(([^)]*)\))?:\s*(fully_preserved|partially_preserved|attribute_transfer|"
    r"weak_reference)\s*[-–—]\s*(.*)$"
)


def aspect_label(width: int, height: int) -> str:
    """The aspect ratio the upsampler knows closest to `width / height`."""
    return min(ASPECT_RATIOS, key=lambda label: abs(math.log(width / height / ASPECT_RATIOS[label])))


def is_edit_prompt(prompt: str) -> bool:
    """An edit prompt in a form P-H3-Image edits from: the six-section brief, or one `Edit <Picture 1>.` paragraph
    that says what it keeps."""
    text = prompt.strip()
    names = _SECTION.findall(text)
    if names:
        parts = _SECTION.split(text)
        body = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        retention = [line.strip() for line in body.get("retention_analysis", "").splitlines() if line.strip()]
        return (tuple(names) == _SECTIONS and text.count("[keyframe completion]") == 1 and bool(retention)
                and all(_RETENTION.match(line) for line in retention))  # fmt: skip
    return text.startswith("Edit <Picture 1>.") and "Keep" in text and "\n" not in text


class UpsamplerGraph:
    def __init__(self, config: Config) -> None:
        section = config.section("prompt_upsampler")
        with no_init_weights():
            model: PreTrainedModel = AutoModelForImageTextToText.from_config(  # type: ignore[no-untyped-call]  # unannotated in Transformers
                AutoConfig.for_model(**section.mapping()), dtype=torch.bfloat16
            )
            model.tie_weights()
        self.components: dict[str, nn.Module] = {"prompt_upsampler": stored(model, "prompt_upsampler", config).eval()}  # type: ignore[no-untyped-call]  # Transformers' eval is untyped
        self.processor_config = config.section("prompt_upsampler_processor").mapping()


class PH3Upsampler(Model[UpsamplerGraph]):
    """Pruna's prompt upsampler (a Qwen3.5-9B fine-tune): a short request to the full prompt P-H3-Image renders
    best, or for an edit a brief of what to keep and what to change. Greedy; it reasons first."""

    serving = Serving(min_bits=4, phase="decode")

    graph: UpsamplerGraph
    processor: ProcessorMixin

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpsamplerGraph)
        self.processor = vlm.processor(loader.assets, "prompt_upsampler_processor/", config=self.graph.processor_config)

    def unload(self, loader: Loader) -> None:
        return None

    def execution_plan(self, plan: Plan) -> str:
        return lm.install(plan, cast(PreTrainedModel, self.graph.components["prompt_upsampler"]))

    @memoize
    @uses_components("prompt_upsampler")
    def upsample(self, request: str, pictures: list[Image.Image], aspect: str) -> tuple[str, str]:
        """The prompt for `request` and "", or `request` itself and why, when the answer is cut off or not in the
        form P-H3-Image reads."""
        model = cast(PreTrainedModel, self.graph.components["prompt_upsampler"])
        tokenizer = self.processor.tokenizer
        task = "edit" if pictures else "t2i"
        shown = [resized(p.convert("RGB"), size_at_area(*p.size, VISION_PIXELS)) for p in pictures]
        content: str | list[dict[str, str]]
        if shown:
            content = [part for index in range(len(shown))
                       for part in ({"type": "text", "text": f"<Picture {index + 1}>: "}, {"type": "image"})]  # fmt: skip
            content.append({"type": "text", "text": f"Edit <Picture 1>. {request.strip()}"})
        else:
            content = f"Aspect ratio: {aspect}\nRequest: {request.strip()}"
        text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=shown or None, return_tensors="pt").to(torch.device(str(self.device)))
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        out = lm.generate(model, inputs, max_new_tokens=MAX_NEW_TOKENS[task], stop=[end], pad=tokenizer.pad_token_id)
        ids = [int(token) for token in out[0].tolist()]
        ended = end in ids
        answer = str(tokenizer.decode(ids[: ids.index(end)] if ended else ids, skip_special_tokens=False))
        prompt = answer.partition("</think>")[2].strip()
        valid = ended and answer.count("</think>") == 1 and len(prompt.split()) >= 5
        if valid and (not shown or is_edit_prompt(prompt)):
            return prompt, ""
        return request, "cut off at its token limit" if not ended else "not in the form P-H3-Image reads"
