"""Qwen-Image-2.1's prompt enhancers, run as Qwen runs them (QwenLM/Qwen-Image-2.1@ab96caa9 `prompt_rewrite/`,
`pe_core.py` and `run_vllm.py`): two Qwen3.5-9B fine-tunes, `Qwen-Image-2.1-PE-I2I`, which rewrites an edit instruction
from its pictures, and `Qwen-Image-2.1-PE-T2I`, which expands a text-to-image request. Each reads the request under its
checkpoint's own `system_prompt.txt`, thinks, and ends with one JSON object: `rewritten_prompt`, `wh_ratio` and, for an
edit, `ratio_follow`. Nothing of ours is added to the request or to the system prompt (Paul, 11-18: "just run the prompt
enhancer as alibaba / qwen intended to be run").

Sampling and budget are upstream's (`pe_core.PROFILES`): temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence
penalty 1.5 (t2i) or 0 (edit), and one generation of at most 16256 (t2i) or 24000 (edit) tokens, thought and answer
together. The seed is the render's: a new render rewrites anew, and a given seed repeats its rewrite.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from cozy_runtime import lm
from cozy_runtime.author import Loader, uses_components
from cozy_runtime.vlm import COMPONENT, Question, VisionLM
from PIL import Image

#: Upstream's sampling (`pe_core.Profile`); its min-p is 0, which samples as none.
TEMPERATURE, TOP_P, TOP_K = 1.0, 0.95, 20
#: Upstream's presence penalty and token budget, thought and answer together, for each task (`pe_core.PROFILES`).
PRESENCE_PENALTY = {"edit": 0.0, "t2i": 1.5}
MAX_NEW_TOKENS = {"edit": 24000, "t2i": 16256}
#: The checkpoint's system prompt: Qwen's PE-T2I and PE-I2I each carry their own.
SYSTEM_PROMPT = "system_prompt.txt"
#: Each picture as the enhancer was trained on it: at most 1 MP (`pe_core.load_image`).
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
    """A picture as the enhancer reads it, upstream's `load_image`: RGB (PIL's `convert`, which drops an alpha), shrunk
    to `MAX_PIXELS`."""
    image = image.convert("RGB")
    width, height = image.size
    if width * height > MAX_PIXELS:
        scale = (MAX_PIXELS / (width * height)) ** 0.5
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    return image


def split(text: str) -> tuple[str, str]:
    """The thought and the answer of a decode that began inside the template's `<think>` (`pe_core.split_thinking`):
    with no `</think>` the budget ran out mid-thought, and there is no answer."""
    if "</think>" in text:
        thought, _, answer = text.partition("</think>")
        return thought.partition("<think>")[2].strip() if "<think>" in thought else thought.strip(), answer.strip()
    if "<think>" in text:
        return text.partition("<think>")[2].strip(), ""
    return "", text.strip()


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
    """Runtime's Qwen-VL model, answering under its checkpoint's own system prompt."""

    system: str

    def load(self, loader: Loader) -> None:
        super().load(loader)
        self.system = loader.assets.read(SYSTEM_PROMPT).decode("utf-8").strip()

    @uses_components(COMPONENT)
    def answer(
        self, question: Question, *, budget: int, presence_penalty: float, seed: int,
        progress: Callable[[int, int], None],
    ) -> tuple[str, int]:
        """One generation of at most `budget` tokens after the template's `<think>`, as text, and its tokens;
        `progress(step, total)` is told the tokens decoded."""
        content = [*({"type": "image"} for _ in question.images), {"type": "text", "text": question.text}]
        chat: str = self.processor.apply_chat_template(
            [{"role": "system", "content": self.system}, {"role": "user", "content": content}],  # type: ignore[arg-type, unused-ignore]  # content as parts
            add_generation_prompt=True, tokenize=False, enable_thinking=True,
        )
        model = self.pipe.components[COMPONENT]
        tokenizer = self.processor.tokenizer
        configured = model.generation_config.eos_token_id
        ends = [tokenizer.eos_token_id, *(configured if isinstance(configured, list) else [configured])]
        ends = [token for token in dict.fromkeys(ends) if isinstance(token, int)]
        inputs = self.processor(text=[chat], images=question.images or None, return_tensors="pt").to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        tokens = lm.generate(model, inputs, max_new_tokens=budget, stop=ends, pad=ends[0], presence_penalty=presence_penalty,
                             temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, seed=seed,
                             progress=lambda count: progress(min(count, budget) - 1, budget))
        return str(self.processor.batch_decode(tokens, skip_special_tokens=True)[0]), tokens.shape[1]  # type: ignore[no-untyped-call, unused-ignore]  # Transformers' batch_decode is untyped in some releases


def rewrite(
    enhancer: Enhancer, prompt: str, images: list[Image.Image], seed: int, progress: Callable[[int, int], None],
) -> tuple[Rewrite | None, str, int]:
    """The enhancer's rewrite of `prompt` (`<imageN>` for two or more pictures, as its system prompt asks), its answer
    and the tokens it decoded; `progress(step, total)` is told the tokens of its budget decoded."""
    task = "edit" if images else "t2i"
    text, tokens = enhancer.answer(
        Question(prompt.strip(), [reference(image) for image in images]),
        budget=MAX_NEW_TOKENS[task], presence_penalty=PRESENCE_PENALTY[task], seed=seed, progress=progress,
    )
    _, answer = split(text)
    return parse(answer, len(images)), answer, tokens
