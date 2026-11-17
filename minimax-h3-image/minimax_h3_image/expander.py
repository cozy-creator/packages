"""MiniMax-H3-Image-PE, MiniMax's prompt expander (a Qwen3.8-27B fine-tune): a short request to the detailed English
caption MiniMax-H3-Image renders best, and the canvas it suits.

Text to image runs MiniMax's T2I PE as released (`prompt/t2i/`): its system prompt, its task prompt with the request,
thinking at `xhigh`, sampled at temperature 1.0 and top-p 0.95, at most 4096 tokens with the thought, answering
`{"expanded_caption", "aspect_ratio"}`. Runtime's sampler has no top-k, so upstream's top-k 20 is not applied, and no
grammar holds the answer to its JSON schema: an answer that doesn't parse renders the request as written, with a
warning.

An edit runs MiniMax's edit PE prompt (`prompt/edit/task_prompt.txt`) on the same checkpoint, as Diffusers'
`minimax-h3-image-pe-blocks` do: MiniMax releases no edit PE weights, only the prompt "for a vision-language model of
your choice", and this checkpoint keeps Qwen3.8's vision tower. Each picture is labelled `<Picture n>` with its ratio
before it; the answer is `{"expanded_caption", "target_aspect_ratio"}`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

from cozy_runtime.author import Loader, uses_components
from cozy_runtime.vlm import COMPONENT, Question, VisionLM
from PIL import Image

#: The T2I PE's canvases (`prompt/t2i/request_params.json`'s schema).
RATIOS = ("1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16")
TEMPERATURE, TOP_P = 1.0, 0.95
#: Upstream's `max_tokens`, the thought included.
MAX_TOKENS = 4096
#: What the PE sees of a picture: about one megapixel, as the text encoder's vision tower reads it.
VISION_PIXELS = 1 << 20
_RATIO = re.compile(r"\s*(\d+)\s*:\s*(\d+)\s*")


@dataclass(frozen=True)
class Expansion:
    caption: str
    #: `W:H` the PE chose, or None.
    ratio: str | None


def reduced(width: int, height: int) -> str:
    ratio = Fraction(width, height)
    return f"{ratio.numerator}:{ratio.denominator}"


def _spans(answer: str) -> list[str]:
    """Every balanced top-level `{...}` in `answer`, braces inside JSON strings skipped."""
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


def parse(answer: str, ratio_key: str) -> Expansion | None:
    """The answer's last JSON object with an `expanded_caption`, and its ratio when it is a positive `W:H`."""
    for span in reversed(_spans(answer)):
        try:
            found = json.loads(span)
        except json.JSONDecodeError:
            continue
        if not isinstance(found, dict):
            continue
        caption = found.get("expanded_caption")
        if not isinstance(caption, str) or not caption.strip():
            continue
        ratio = _RATIO.fullmatch(str(found.get(ratio_key) or ""))
        valid = ratio is not None and int(ratio[1]) > 0 and int(ratio[2]) > 0
        return Expansion(caption.strip(), f"{ratio[1]}:{ratio[2]}" if valid else None)  # type: ignore[index]  # `valid`: matched
    return None


def shown(picture: Image.Image) -> Image.Image:
    """A picture as the PE reads it: RGB, at most `VISION_PIXELS`."""
    picture = picture.convert("RGB")
    width, height = picture.size
    if width * height > VISION_PIXELS:
        scale = math.sqrt(VISION_PIXELS / (width * height))
        picture = picture.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    return picture


class H3ImagePE(VisionLM):
    """MiniMax-H3-Image-PE on Runtime's Qwen-VL model, with MiniMax's released prompts from its checkpoint."""

    t2i_system: str
    t2i_task: str
    edit_task: str

    def load(self, loader: Loader) -> None:
        super().load(loader)
        read = loader.assets.read
        self.t2i_system = read("prompt/t2i/system_prompt.txt").decode("utf-8").strip()
        self.t2i_task = read("prompt/t2i/task_prompt.txt").decode("utf-8")
        self.edit_task = read("prompt/edit/task_prompt.txt").decode("utf-8")

    def question(self, request: str, pictures: list[Image.Image], aspect_ratio: str | None) -> tuple[Question, str, str]:
        """`request` as the PE reads it, its system turn and its answer's ratio key: with pictures, the edit PE, told
        `aspect_ratio` (`adaptive` when None); else the T2I PE."""
        if not pictures:
            return Question(self.t2i_task.format(request)), self.t2i_system, "aspect_ratio"
        count = len(pictures)
        parts: list[str | Image.Image] = [
            self.edit_task,
            f"Reference Image Count:\n{count} input image(s), labeled <Picture 1> through <Picture {count}>.",
        ]
        for index, picture in enumerate(pictures, 1):
            parts += [f"Reference Media:\n- <Picture {index}> [aspect_ratio {reduced(*picture.size)}]", shown(picture)]
        parts.append(
            f"\nUser Editing Instruction (JSON string):\n{json.dumps(request, ensure_ascii=False)}\n\n"
            f"User Aspect Ratio Selection:\n{aspect_ratio or 'adaptive'}\n\n"
            "Apply the instructions above to the supplied inputs and reference images. "
            "Output only the required two-field JSON object."
        )
        return Question(parts=parts), "", "target_aspect_ratio"

    @uses_components(COMPONENT)
    def expand(
        self, request: str, pictures: list[Image.Image], aspect_ratio: str | None, *, seed: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[Expansion | None, str]:
        """The expansion of `request` (None when the answer doesn't parse) and the raw answer (`question`)."""
        question, system, key = self.question(request, pictures, aspect_ratio)
        answer = self.ask([question], system=system, thinking=True, stop=[], max_new_tokens=MAX_TOKENS,
                          presence_penalty=0.0, temperature=TEMPERATURE, top_p=TOP_P, seed=seed, progress=progress)[0]  # fmt: skip
        return parse(answer, key), answer
