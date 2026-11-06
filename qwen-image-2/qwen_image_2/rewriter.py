"""Qwen-Image-2.1's prompt enhancers, as Qwen ships them (QwenLM/Qwen-Image-2.1@ab96caa9 `prompt_rewrite/`): two
Qwen3.5-9B fine-tunes, `Qwen-Image-2.1-PE-I2I`, which rewrites an edit instruction from its pictures, and
`Qwen-Image-2.1-PE-T2I`, which expands a text-to-image request. Each thinks, then answers one JSON object:
`rewritten_prompt`, `wh_ratio` and, for an edit, `ratio_follow`. Their system prompts are the checkpoints' own
`system_prompt.txt` (byte-identical to `prompt_rewrite/prompts/`), bundled here.

Sampling is upstream's (`pe_core.PROFILES`): temperature 1.0, top-p 0.95, presence penalty 1.5 (t2i) or 0
(edit), thinking on, 16256 or 24000 new tokens. Runtime's sampler has no top-k, so upstream's top-k 20 is not
applied. The seed is the render's: a new render rewrites anew, and a given seed repeats its rewrite. The decode is
plain and graphed: Runtime's drafted decode ran eager, about 2.5x slower (run 37180), and leaves with varena v2.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from cozy_runtime.vlm import Question, VisionLM
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


@dataclass(frozen=True)
class _Profile:
    presence_penalty: float
    max_new_tokens: int
    system: str


def _profile(task: str, presence_penalty: float, max_new_tokens: int) -> _Profile:
    text = files(__package__).joinpath(f"prompt_rewrite/system_prompt_{task}.txt").read_text(encoding="utf-8")
    return _Profile(presence_penalty, max_new_tokens, text.strip())


PROFILES = {"edit": _profile("edit", 0.0, 24000), "t2i": _profile("t2i", 1.5, 16256)}
TEMPERATURE, TOP_P = 1.0, 0.95
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


def rewrite(
    enhancer: VisionLM, prompt: str, images: list[Image.Image], creativity: Creativity, seed: int,
    progress: Callable[[float], None],
) -> tuple[Rewrite | None, str]:
    """The enhancer's rewrite of `prompt` (`<imageN>` for two or more pictures, as its system prompt asks), and
    its raw answer; `progress` is told the share of its token budget decoded."""
    task = "edit" if images else "t2i"
    profile = PROFILES[task]
    steer = _STEER[task].get(creativity, "")
    question = Question("\n\n".join(filter(None, (prompt.strip(), steer))), [reference(image) for image in images])
    answer = enhancer.ask(
        [question], system=profile.system, thinking=True, stop=[], max_new_tokens=profile.max_new_tokens,
        presence_penalty=profile.presence_penalty, temperature=TEMPERATURE, top_p=TOP_P, seed=seed,
        progress=lambda tokens: progress(min(tokens / profile.max_new_tokens, 1.0)),
    )[0]
    return parse(answer, len(images)), answer
