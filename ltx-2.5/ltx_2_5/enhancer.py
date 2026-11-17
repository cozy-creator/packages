"""Lightricks' prompt enhancer for LTX-2.5: Gemma 4 E2B-it, served as the Runtime's VisionLM, rewriting a prompt into the
model's caption style with upstream's own text-to-video and image-to-video system prompts, messages and decoding
(ltx_core/text_encoders/gemma/encoders `enhance_t2v`/`enhance_i2v`; Diffusers 0.43 `LTX2InContextPipeline.enhance_prompt`
reproduces them, and its copies of the system prompts are byte-identical to upstream's)."""

from __future__ import annotations

from collections.abc import Sequence

from cozy_runtime.accel import sequence_parallel
from cozy_runtime.vlm import Question, VisionLM
from diffusers.pipelines.ltx2.prompt_enhancement import _prepare_enhance_image, clean_response
from diffusers.pipelines.ltx2.utils import (
    GEMMA4_PROMPT_ENHANCEMENT_CONFIG, LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT, LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT,
)
from PIL import Image


@sequence_parallel(degrees=(2, 4, 8))
class Enhancer(VisionLM):
    """Replicated beside the sequence-sharded DiT."""

    def rewrite(self, prompt: str, picture: Image.Image | None, labels: Sequence[str] = ()) -> str | None:
        """`prompt` as LTX-2.5's training captions read: the image-to-video form opens on `picture`, the video's first
        frame. Greedy, at most upstream's 600 new tokens, every repeated 5-gram blocked (`no_repeat_ngram_size=5`), so a
        prompt always reads the same. Each of `labels` (a reference's name in the prompt: `Image 1`, `Audio 1`, a
        video's) binds an identity, a voice or a motion, so the system prompt asks for them as written and for no looks
        of a referenced subject; None when the rewrite lost one, and the prompt goes as written."""
        config = GEMMA4_PROMPT_ENHANCEMENT_CONFIG
        if picture is None:
            question, system = Question(f"{config.user_prompt_prefix}: {prompt}"), LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
        else:
            question = Question(f"User Raw Input Prompt: {prompt}.", [_prepare_enhance_image(picture)])
            system = LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT
        if labels:
            system += (f"\n\nKeep each of these labels exactly as written wherever it appears: {', '.join(labels)}. Never "
                       "describe how a labelled subject looks: its reference supplies that.")
        answer, = self.ask([question], system=system, thinking=False, stop=[], max_new_tokens=config.max_new_tokens,
                           presence_penalty=0.0, temperature=0.0, top_p=1.0, seed=config.seed,
                           no_repeat_ngram=int(config.generation_kwargs["no_repeat_ngram_size"]))
        rewritten = str(clean_response(answer))
        return rewritten if all(label in rewritten for label in labels) else None
