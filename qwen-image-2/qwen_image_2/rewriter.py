"""Qwen-Image-2.1's prompt enhancers, as Qwen ships them (QwenLM/Qwen-Image-2.1@ab96caa9 `prompt_rewrite/`): two
Qwen3.5-9B fine-tunes, `Qwen-Image-2.1-PE-I2I`, which rewrites an edit instruction from its pictures, and
`Qwen-Image-2.1-PE-T2I`, which expands a text-to-image request. Each thinks, then answers one JSON object:
`rewritten_prompt`, `wh_ratio` and, for an edit, `ratio_follow`. Their system prompts are the checkpoints' own
`system_prompt.txt` (byte-identical to `prompt_rewrite/prompts/`), bundled here.

Sampling is upstream's (`pe_core.PROFILES`): temperature 1.0, top-p 0.95, presence penalty 1.5 (t2i) or 0 (edit).
Runtime's sampler has no top-k, so upstream's top-k 20 is not applied. The seed is the render's: a new render rewrites
anew, and a given seed repeats its rewrite. The decode is plain and graphed.

Upstream lets each think for up to 24000 tokens: 25-143 s of a 4090 an edit (runs 37228-37379). Here the thought
stops at `THINK_TOKENS` and the model answers, closed as Qwen3 closes a thinking budget, and an edit's rewrite is asked
to be short (`_BRIEF`). Qwen3.5-9B's multi-token-prediction head (`Drafter`) drafts the decode: where the enhancer's
weights stream (a card or cap below its 11 GiB), each pass over them serves every drafted token it keeps. On a 4090,
against upsampling off, an edit (pose, a new scene, a new shot; runs 37863-38112):

    thought               added   rewrite        the gains (designed scene, no stray props) / clothes held
    none                  +37 s   never parses   none: it reasons in the answer until it is cut off
    128, <=80 words        +9 s   1250 chars     all / 3 of 6
    128                   +14 s   2300 chars     all / 3 of 6
    256                   +15 s   1950 chars     all / 3 of 6
    512                   +19 s   1860 chars     all / 5 of 6
    upstream's (~3000)    +69 s   2030 chars     all / 6 of 6
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal, TypeVar

import torch
from cozy_runtime import lm
from cozy_runtime.author import Config, Loader, Model, uses_components
from cozy_runtime.vlm import COMPONENT, Question, VisionLM
from PIL import Image
from transformers import AutoConfig
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5 import modeling_qwen3_5

T = TypeVar("T")

#: How far the rewrite goes beyond the words: `balanced` is Qwen's own rewrite; the others say so after the
#: request, where the enhancer reads intent.
Creativity = Literal["faithful", "balanced", "creative"]

_STEER: dict[str, dict[str, str]] = {
    "edit": {
        "faithful": "Change only what I asked, exactly as asked, and keep everything else as it is.",
        "creative": (
            "Make it a striking new picture: be bold and imaginative with the expression, pose, styling, "
            "setting, light and camera, well beyond what I wrote, keeping the people and products recognizable."
        ),
    },
    "t2i": {
        "faithful": "Stay close to what I wrote and add only what the picture needs.",
        "creative": "Be bold and imaginative: a vivid, unexpected take on this, well beyond what I wrote.",
    },
}


@dataclass(frozen=True)
class _Profile:
    presence_penalty: float
    system: str


def _profile(task: str, presence_penalty: float) -> _Profile:
    text = files(__package__).joinpath(f"prompt_rewrite/system_prompt_{task}.txt").read_text(encoding="utf-8")
    return _Profile(presence_penalty, text.strip())


PROFILES = {"edit": _profile("edit", 0.0), "t2i": _profile("t2i", 1.5)}
TEMPERATURE, TOP_P = 1.0, 0.95
THINK_TOKENS = 128
#: After the request, where the enhancer reads intent: it writes about 40% less (runs 38013-38036).
_BRIEF = {"edit": "Write the rewrite in at most 80 words.", "t2i": ""}
#: The answer: a rewrite runs 600-2600 characters (runs 37180-37379), well inside this.
MAX_ANSWER = 2048
#: Qwen3's words for closing a thought at its budget (qwen.readthedocs.io quickstart, `early_stopping_text`).
CLOSE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
#: Each picture as the enhancer was trained on it: RGB, at most 1 MP (`pe_core.load_image`).
MAX_PIXELS = 1024 * 1024
_RATIO = re.compile(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*")
_FOLLOW = re.compile(r"\s*<image(\d+)>\s*")


@dataclass(frozen=True)
class Rewrite:
    prompt: str
    #: The canvas the enhancer chose: `W:H`, or the 1-based picture whose shape the output follows.
    ratio: str | None
    follow: int | None


def reference(image: Image.Image) -> Image.Image:
    """A picture as the enhancer reads it: over white, as the image model's vision encoder sees it, and
    shrunk to `MAX_PIXELS` as upstream's `load_image` does."""
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    width, height = image.size
    if width * height > MAX_PIXELS:
        scale = (MAX_PIXELS / (width * height)) ** 0.5
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    return image


def _spans(answer: str) -> list[str]:
    """Every balanced top-level `{...}` in `answer`, braces inside JSON strings skipped (`pe_core._balanced_spans`)."""
    spans, depth, start, quoted, escaped = [], 0, -1, False, False
    for i, ch in enumerate(answer):
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch == "{":
            start = i if depth == 0 else start
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(answer[start : i + 1])
    return spans


def parse(answer: str, pictures: int) -> Rewrite | None:
    """The answer's last JSON object with a `rewritten_prompt` (upstream also reads its training's misspelt
    `rewrited_prompt`), and its canvas: a valid `wh_ratio`, else a `ratio_follow` naming one of the pictures."""
    for span in reversed(_spans(answer)):
        try:
            found = json.loads(span)
        except json.JSONDecodeError:
            continue
        if not isinstance(found, dict):
            continue
        prompt = found.get("rewritten_prompt") or found.get("rewrited_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        ratio = _RATIO.fullmatch(str(found.get("wh_ratio") or ""))
        follow = _FOLLOW.fullmatch(str(found.get("ratio_follow") or ""))
        followed = int(follow[1]) if follow and 1 <= int(follow[1]) <= pictures else None
        valid = ratio is not None and float(ratio[1]) > 0 and float(ratio[2]) > 0
        return Rewrite(prompt.strip(), f"{ratio[1]}:{ratio[2]}" if valid else None, None if valid else followed)
    return None


class DrafterGraph:
    def __init__(self, config: Config) -> None:
        # The head-only lane keeps Qwen3.5-9B's config (`drafter_head`).
        with no_init_weights():
            head = lm.Draft(AutoConfig.for_model(**config.mapping()), modeling_qwen3_5)
        self.components: dict[str, torch.nn.Module] = {"model": head.to(torch.bfloat16).eval()}


class Drafter(Model[DrafterGraph], encoded_leaves="accept"):
    """Qwen3.5-9B's multi-token-prediction head: the enhancers are that model fine-tuned without it. Their decode
    drafts tokens with it and keeps the ones it agrees with."""

    graph: DrafterGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(DrafterGraph, factory=DrafterGraph)

    @uses_components("model")
    def drafting(self, body: Callable[[], T]) -> T:
        with lm.drafting(self.graph.components["model"]):
            return body()


class Enhancer(VisionLM):
    """Runtime's Qwen-VL model, answering after a thought of at most `think` tokens."""

    @uses_components(COMPONENT)
    def answer(
        self, question: Question, *, system: str, think: int, presence_penalty: float, seed: int,
        progress: Callable[[float], None],
    ) -> tuple[str, int, int]:
        """The answer, and the tokens its thought and the answer took."""
        content = [*({"type": "image"} for _ in question.images), {"type": "text", "text": question.text}]
        chat: str = self.processor.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, enable_thinking=True,
        )
        tokenizer = self.processor.tokenizer
        configured = self.pipe.components[COMPONENT].generation_config.eos_token_id
        ends = [tokenizer.eos_token_id, *(configured if isinstance(configured, list) else [configured])]
        ends = [token for token in dict.fromkeys(ends) if isinstance(token, int)]
        budget = think + MAX_ANSWER
        thought, thought_tokens = "", 0
        if think:
            tokens = self._decode(chat, question.images, ends + [tokenizer.convert_tokens_to_ids("</think>")], think,
                                  presence_penalty, seed, lambda n: progress(n / budget))
            thought_tokens = tokens.shape[1]
            thought = str(self.processor.batch_decode(tokens, skip_special_tokens=True)[0])
        # No thought is the template's own empty one; one that stopped short of its budget ended itself.
        thought, ended, _ = thought.partition("</think>")
        closed = chat + thought.rstrip() + (CLOSE if think and not ended and thought_tokens >= think else "\n</think>\n\n")
        tokens = self._decode(closed, question.images, ends, MAX_ANSWER, presence_penalty, seed,
                              lambda n: progress((thought_tokens + n) / budget))
        return str(self.processor.batch_decode(tokens, skip_special_tokens=True)[0]), thought_tokens, tokens.shape[1]

    def _decode(self, text: str, images: list[Image.Image], stop: list[int], most: int, presence_penalty: float,
                seed: int, progress: Callable[[int], None]) -> torch.Tensor:
        model = self.pipe.components[COMPONENT]
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        return lm.generate(model, inputs, max_new_tokens=most, stop=stop, pad=stop[0], presence_penalty=presence_penalty,
                           temperature=TEMPERATURE, top_p=TOP_P, seed=seed, progress=progress)


def rewrite(
    enhancer: Enhancer, drafter: Drafter, prompt: str, images: list[Image.Image], creativity: Creativity, seed: int,
    progress: Callable[[float], None],
) -> tuple[Rewrite | None, str, int, int]:
    """The enhancer's rewrite of `prompt` (`<imageN>` for two or more pictures, as its system prompt asks), drafted
    by `drafter`, its raw answer and the tokens its thought and its answer took; `progress` is told the share of its
    budget decoded."""
    task = "edit" if images else "t2i"
    profile = PROFILES[task]
    steer = _STEER[task].get(creativity, "")
    question = Question("\n\n".join(filter(None, (prompt.strip(), steer, _BRIEF[task]))),
                        [reference(image) for image in images])
    answer, thought_tokens, answer_tokens = drafter.drafting(lambda: enhancer.answer(
        question, system=profile.system, think=THINK_TOKENS, presence_penalty=profile.presence_penalty, seed=seed,
        progress=lambda done: progress(min(done, 1.0)),
    ))
    return parse(answer, len(images)), answer, thought_tokens, answer_tokens
