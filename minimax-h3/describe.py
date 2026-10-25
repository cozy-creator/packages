"""H3's reference descriptions, asked of Qwen3.5 (`qwen3.5/ask`): a supplied character's identity
in one sentence, a place's look, and a motion clip's people, beats, camera and cuts."""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, cast

import msgspec
from cozy_runtime.author import (
    AssetBound,
    Context,
    ImageAsset,
    Telemetry,
    UnsupportedInput,
    VideoAsset,
    invocable,
)
from cozy_runtime.vlm import AskPrompt, AskResult
from qwen3_5 import ask

IDENTITY_PROMPT = (
    "Describe only how the single main character in this image looks, in exactly one concise "
    "sentence. Include distinctive visible identity, species or character type, hair, face, colors, "
    "clothing, armor, accessories, and carried or worn items when visible. Say nothing about what "
    "the character is doing or where they are: do not describe the background, setting, scenery, "
    "other characters, pose, posture, action, facial expression, camera, lighting, image quality, or "
    "art style. Do not invent hidden details or a name. Return only the sentence, without a label "
    "or introduction."
)
# Owner ruling (se-066): appearance only. A caption naming any of these is regenerated once with
# the words pointed out; clauses still naming one are then dropped.
SCENE_OR_POSE = re.compile(
    r"\b(?:background|backdrop|setting|scene|scenery|surroundings|room|indoors|outdoors|studio|wall|"
    r"mirror|selfie|stand(?:s|ing)?|sit(?:s|ting)?|seated|pos(?:e|es|ed|ing)|lean(?:s|ing)?|"
    r"kneel(?:s|ing)?|crouch(?:es|ing)?|walk(?:s|ing)?|looking|facing|gaz(?:es|ing)|confident(?:ly)?)\b",
    re.IGNORECASE,
)
SENTENCE_TOKENS = 160
PLACE_PROMPT = (
    "Describe only the place in this image, in exactly one concise sentence: the kind of place, "
    "its architecture, materials, colors, furnishings and lighting. Say nothing about any person, "
    "character or animal, and nothing about the camera, image quality or art style. Return only the "
    "sentence, without a label or introduction."
)
#: A clip is watched as video: about 2 fps, at least 16 and at most 32 frames (a 3 s chop still
#: gets 16), each at most 448 px on its long edge.
CLIP_READING = {"fps": 2.0, "min_frames": 16, "max_frames": 32, "max_edge": 448}
CLIP_PARTS = (
    "PEOPLE: one line per person, \"P1: <where in the frame at the start> - <build, hair, "
    "clothing>\".\n"
    "BEATS: one line per beat in time order, \"<start>-<end> s: <who does what to whom>; <which "
    "way each person turns, falls or moves>; <each person's facing relative to the camera>\".\n"
    "CAMERA: the framing and each camera move with its time.\n"
)
CLIP_CUTS = "CUTS: the time of each cut, or none.\n"
CLIP_RULES = (
    "Name people only as P1, P2 and so on. Describe only what is visible; do not guess names, "
    "intentions or the place's story."
)
CLIP_TOKENS = 700
#: A frame whose ffmpeg scene score (`select`, on frames 270 px high) reaches CUT_SCENE starts a
#: new shot; detections within CUT_GAP seconds of a cut are that cut. On se-072's 15 clips the
#: non-cuts peak at 0.16 and the 10 cuts start at 0.26: 0.2 finds all 10 and no false cut.
CUT_SCENE = 0.2
CUT_GAP = 0.25
SECTION = re.compile(r"^\s*(PEOPLE|BEATS|CAMERA|CUTS)\s*:\s*", re.IGNORECASE | re.MULTILINE)


class CutsInput(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, max_decoded_bytes=16 << 30)]
    #: The clip as H3 reads it ends here (its 15 s trim does not travel with the asset).
    seconds: Annotated[float, msgspec.Meta(gt=0)]


class CutsOutput(msgspec.Struct):
    #: Cut times in seconds; None when ffmpeg cannot read the clip.
    cuts: list[float] | None


def _ask() -> Callable[..., Awaitable[AskResult]]:
    # Managed dependencies install an asynchronous caller for the source entrypoint.
    return cast(Callable[..., Awaitable[AskResult]], ask)


async def characters(images: list[ImageAsset], tel: Telemetry) -> list[str]:
    """One sentence on each character's appearance, all in one call; empty where the image shows
    none. A sentence naming a setting or pose word is asked again once with those words pointed
    out, and any clause still naming one is dropped."""
    asked = await _ask()(
        prompts=[AskPrompt(IDENTITY_PROMPT, images=[image]) for image in images], max_tokens=SENTENCE_TOKENS
    )
    texts = [" ".join(text.split()) for text in asked.texts]
    named = {i: sorted({w.lower() for w in SCENE_OR_POSE.findall(text)}) for i, text in enumerate(texts)}
    named = {i: words for i, words in named.items() if words}
    if not named:
        return texts
    for i, words in named.items():
        tel.log("description named setting or pose; regenerating", words=", ".join(words), first=texts[i][:400])
    retries = [
        AskPrompt(
            f'{IDENTITY_PROMPT} An earlier answer was "{texts[i]}". It mentioned {", ".join(words)}, which '
            "describe the pose or surroundings, not the character. Rewrite it without them.",
            images=[images[i]],
        )
        for i, words in named.items()
    ]
    again = await _ask()(prompts=retries, max_tokens=SENTENCE_TOKENS)
    for i, text in zip(named, again.texts, strict=True):
        texts[i] = strip_scene_or_pose(" ".join(text.split()))
    return texts


async def places(images: list[ImageAsset]) -> list[str]:
    """One sentence on each place's look, people left out, all in one call."""
    asked = await _ask()(
        prompts=[AskPrompt(PLACE_PROMPT, images=[image]) for image in images], max_tokens=SENTENCE_TOKENS
    )
    return [" ".join(text.split()) for text in asked.texts]


async def clip(video: VideoAsset, tel: Telemetry) -> str:
    """A motion clip's people, beats, camera and cuts, watched as timestamped video in one pass.
    ffmpeg finds the cuts (`find_cuts`): they fill `CUTS` and the model is told where the shot
    changes; without ffmpeg the model judges them itself."""
    seconds = float(video.probe().duration)
    cuts: list[float] | None = None
    if shutil.which("ffmpeg") is None:
        tel.log("no ffmpeg on this machine: the model judges cuts itself (cozy tools install ffmpeg)", level="warning")
    else:
        found = cast(Callable[..., Awaitable[CutsOutput]], find_cuts)
        cuts = (await found(payload=CutsInput(video, seconds))).cuts
    asked = await _ask()(
        prompt=clip_prompt(cuts), videos=[video], max_tokens=CLIP_TOKENS, video_end=seconds, **CLIP_READING
    )
    text = with_cuts(asked.texts[0], cuts)
    if not text:
        raise UnsupportedInput("the clip description is empty", code="output_integrity")
    return text


@invocable(memoize=True)
async def find_cuts(ctx: Context, *, payload: CutsInput, tel: Telemetry) -> CutsOutput:
    """The clip's cut times from ffmpeg's scene score, memoized by the clip: a repeat costs nothing.
    ffmpeg's own progress (`-progress`) is the `find_cuts` stage, and the scan ends at its exit.
    None, with a warning, when ffmpeg cannot read the clip or exits before its end."""

    def scan() -> tuple[str, str]:
        """The scene marks, and why ffmpeg failed ("" when it read the clip to its end)."""
        with tempfile.TemporaryDirectory() as scratch:
            Path(scratch, "clip").write_bytes(payload.video.read_bytes())
            with open(Path(scratch, "errors"), "w+") as errors, subprocess.Popen(
                ["ffmpeg", "-v", "error", "-nostdin", "-nostats", "-progress", "pipe:1", "-t", f"{payload.seconds:.3f}",
                 "-i", "clip", "-an", "-vf",
                 f"setpts=PTS-STARTPTS,scale=-2:270,select='gte(scene,{CUT_SCENE})',metadata=print:file=marks",
                 "-f", "null", "-"],
                cwd=scratch, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors, text=True,
            ) as ffmpeg:
                assert ffmpeg.stdout is not None
                ended = False
                try:
                    for line in ffmpeg.stdout:
                        name, _, value = line.strip().partition("=")
                        if name == "out_time_us" and value.isdigit():
                            tel.progress(min(int(value) / 1e6 / payload.seconds, 1.0), stage="find_cuts")
                        ended = ended or line.strip() == "progress=end"
                except BaseException:
                    ffmpeg.kill()
                    raise
                code = ffmpeg.wait()
                errors.seek(0)
                said = errors.read()[-400:]
            if code or not ended:
                return "", f"ffmpeg {f'exited {code}' if code else 'stopped before the end'}: {said}"
            marks = Path(scratch, "marks")
            return marks.read_text() if marks.exists() else "", ""

    marks, failed = await asyncio.to_thread(scan)
    if failed:
        tel.log("ffmpeg cannot read the clip: the model judges cuts itself", level="warning", error=failed)
        return CutsOutput(None)
    cuts: list[float] = []
    for t in map(float, re.findall(r"pts_time:([\d.]+)", marks)):
        if 0 < t < payload.seconds and (not cuts or t - cuts[-1] >= CUT_GAP):
            cuts.append(t)
    return CutsOutput(cuts)


def clip_prompt(cuts: list[float] | None) -> str:
    """The instruction for the whole clip: told where ffmpeg found the shot changes, or, with
    `cuts` None (no ffmpeg), asked to judge them itself."""
    if cuts is None:
        shots, parts = "", CLIP_PARTS + CLIP_CUTS
    elif cuts:
        at = ", ".join(f"{t:.1f} s" for t in cuts)
        shots, parts = f"The shot changes at {at}; people may differ across shots. ", CLIP_PARTS
    else:
        shots, parts = "It is one continuous shot. ", CLIP_PARTS
    return (
        "This clip's motion will be performed again by other characters in another place. "
        f"{shots}Describe its structure, not its story, in exactly these {'four' if cuts is None else 'three'} "
        f"parts, each starting on its own line:\n{parts}{CLIP_RULES}"
    )


def with_cuts(reply: str, cuts: list[float] | None) -> str:
    """The model's PEOPLE / BEATS / CAMERA with `CUTS` from the detected cuts; the reply as it is
    when the model judged the cuts itself."""
    if cuts is None:
        return reply
    pieces = SECTION.split(reply)
    parts = {name.upper(): body.strip() for name, body in zip(pieces[1::2], pieces[2::2], strict=True)}
    parts.pop("CUTS", None)
    found = ", ".join(f"{t:.1f} s" for t in cuts) or "none"
    body = "\n\n".join(f"{name}:\n{text}" for name, text in parts.items()) if parts else reply.strip()
    return f"{body}\n\nCUTS:\n{found}"


def strip_scene_or_pose(text: str) -> str:
    """Drop each comma-separated clause that still names a setting or pose."""
    kept = [clause for clause in text.rstrip(". ").split(", ") if not SCENE_OR_POSE.search(clause)]
    return ", ".join(kept) + "." if kept else ""
