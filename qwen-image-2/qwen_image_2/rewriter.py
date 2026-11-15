"""Qwen-Image-2.1's prompt enhancers, as Qwen ships them (QwenLM/Qwen-Image-2.1@ab96caa9 `prompt_rewrite/`): two
Qwen3.5-9B fine-tunes, `Qwen-Image-2.1-PE-I2I`, which rewrites an edit instruction from its pictures, and
`Qwen-Image-2.1-PE-T2I`, which expands a text-to-image request. Each thinks, then answers one JSON object:
`rewritten_prompt`, `wh_ratio` and, for an edit, `ratio_follow`. Each answers under its checkpoint's own
`system_prompt.txt` (upstream's `prompt_rewrite/prompts/`).

Sampling is upstream's (`pe_core.PROFILES`): temperature 1.0, top-p 0.95, presence penalty 1.5 (t2i) or 0 (edit).
Runtime's sampler has no top-k, so upstream's top-k 20 is not applied. The seed is the render's: a new render rewrites
anew, and a given seed repeats its rewrite. The decode is plain and graphed.

Upstream lets each think for up to 24000 tokens: 25-143 s of a 4090 an edit (runs 37228-37379). Here the thought
stops at `THINK_TOKENS` and the model answers, closed as Qwen3 closes a thinking budget, and an edit's rewrite is asked
to be short (`_BRIEF`). On a 4090, against upsampling off, an edit (pose, a new scene, a new shot; runs 37863-38112):

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
from typing import Literal

import torch
from cozy_runtime import lm
from cozy_runtime.author import Loader, uses_components
from cozy_runtime.vlm import COMPONENT, Question, VisionLM
from PIL import Image

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


#: Upstream's presence penalty for each task.
PRESENCE_PENALTY = {"edit": 0.0, "t2i": 1.5}
#: The checkpoint's system prompt: Qwen's PE-T2I and PE-I2I each carry their own.
SYSTEM_PROMPT = "system_prompt.txt"
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
        return Rewrite(prompt.strip(), f"{ratio[1]}:{ratio[2]}" if valid else None, None if valid else followed)  # type: ignore[index]  # `valid`: matched
    return None


class Enhancer(VisionLM):
    """Runtime's Qwen-VL model, answering under its checkpoint's system prompt after a thought of at most `think`
    tokens."""

    system: str

    def load(self, loader: Loader) -> None:
        super().load(loader)
        self.system = loader.assets.read(SYSTEM_PROMPT).decode("utf-8").strip()

    @uses_components(COMPONENT)
    def answer(
        self, question: Question, *, think: int, presence_penalty: float, seed: int,
        progress: Callable[[int, int], None],
    ) -> tuple[str, int, int]:
        """The answer, and the tokens its thought and the answer took; `progress(step, total)` is told
        the tokens decoded of its budget."""
        content = [*({"type": "image"} for _ in question.images), {"type": "text", "text": question.text}]
        chat: str = self.processor.apply_chat_template(
            [{"role": "system", "content": self.system}, {"role": "user", "content": content}],  # type: ignore[arg-type, unused-ignore]  # content as parts
            add_generation_prompt=True, tokenize=False, enable_thinking=True,
        )
        tokenizer = self.processor.tokenizer
        configured = self.pipe.components[COMPONENT].generation_config.eos_token_id
        ends = [tokenizer.eos_token_id, *(configured if isinstance(configured, list) else [configured])]
        ends = [token for token in dict.fromkeys(ends) if isinstance(token, int)]
        budget = think + MAX_ANSWER

        def decoded(count: int) -> None:
            progress(min(count, budget) - 1, budget)

        thought, thought_tokens = "", 0
        if think:
            tokens = self._decode(chat, question.images, ends + [tokenizer.convert_tokens_to_ids("</think>")], think,
                                  presence_penalty, seed, decoded)
            thought_tokens = tokens.shape[1]
            thought = str(self.processor.batch_decode(tokens, skip_special_tokens=True)[0])  # type: ignore[no-untyped-call, unused-ignore]  # Transformers' own, unannotated
        # No thought is the template's own empty one; one that stopped short of its budget ended itself.
        thought, ended, _ = thought.partition("</think>")
        closed = chat + thought.rstrip() + (CLOSE if think and not ended and thought_tokens >= think else "\n</think>\n\n")
        tokens = self._decode(closed, question.images, ends, MAX_ANSWER, presence_penalty, seed,
                              lambda n: decoded(thought_tokens + n))
        return str(self.processor.batch_decode(tokens, skip_special_tokens=True)[0]), thought_tokens, tokens.shape[1]  # type: ignore[no-untyped-call, unused-ignore]  # Transformers' batch_decode is untyped in some releases

    def _decode(self, text: str, images: list[Image.Image], stop: list[int], most: int, presence_penalty: float,
                seed: int, progress: Callable[[int], None]) -> torch.Tensor:
        model = self.pipe.components[COMPONENT]
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        return lm.generate(model, inputs, max_new_tokens=most, stop=stop, pad=stop[0], presence_penalty=presence_penalty,
                           temperature=TEMPERATURE, top_p=TOP_P, seed=seed, progress=progress)


def rewrite(
    enhancer: Enhancer, prompt: str, images: list[Image.Image], creativity: Creativity, seed: int,
    progress: Callable[[int, int], None],
) -> tuple[Rewrite | None, str, int, int]:
    """The enhancer's rewrite of `prompt` (`<imageN>` for two or more pictures, as its system prompt asks), its raw
    answer and the tokens its thought and its answer took; `progress(step, total)` is told the tokens of its budget
    decoded."""
    task = "edit" if images else "t2i"
    steer = _STEER[task].get(creativity, "")
    question = Question("\n\n".join(filter(None, (prompt.strip(), steer, _BRIEF[task]))),
                        [reference(image) for image in images])
    answer, thought_tokens, answer_tokens = enhancer.answer(
        question, think=THINK_TOKENS, presence_penalty=PRESENCE_PENALTY[task], seed=seed,
        progress=progress,
    )
    return parse(answer, len(images)), answer, thought_tokens, answer_tokens
