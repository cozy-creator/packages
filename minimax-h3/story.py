"""Shared reference assets and H3 segment prompts: authored text plus the global sections."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from itertools import pairwise
from typing import Annotated, Literal, NamedTuple

import msgspec
from cozy_runtime.author import (
    AssetBound,
    AudioAsset,
    AudioPreparation,
    ImageAsset,
    InvalidRequest,
    VideoAsset,
    VideoPreparation,
)

from minimax_h3.official import MAX_VIDEO_REFERENCES, Task, validate_reference_policy

#: MiniMax's full-reference sections in their canonical order. A `style` section is read and
#: moved into the description's opening, where the guide puts it (ref guide 5.2).
SECTIONS = (
    "subject_definitions",
    "summary",
    "style",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
# A header starts a line; any case, and `_`, `-` and space are the same separator.
_HEADER = re.compile(
    r"^[ \t]*(" + "|".join(name.replace("_", "[ _-]+") for name in SECTIONS) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_TOKEN = re.compile(r"<([^<>\n]+)>")
_MEDIA = re.compile(r"\s*(picture|audio|video)\s+(\d+)\s*", re.IGNORECASE)
_SHOT = re.compile(r"\[Shot\s*(\d+)\]", re.IGNORECASE)
_CUT = re.compile(r"\bcut(?:s|ting)?\b", re.IGNORECASE)
#: Reference clips: a video longer than 15 s is truncated to its first 15 s (Runtime records the
#: adjustment); audio outside 2-15 s refuses. Every input that carries a reference declares these.
REFERENCE_VIDEO = VideoPreparation(min_seconds=2, max_seconds=15, excess="truncate")
REFERENCE_AUDIO = AudioPreparation(min_seconds=2, max_seconds=15, excess="refuse")
#: A reference video's upload: a source of any length is admitted and truncated, not refused.
VIDEO_MAX_BYTES = 1 << 30
#: Read bounds. A video streams, one decoded frame at a time, straight to its motion size
#: (`read_reference_video`), so a frame of any real source fits. A prepared image is at most
#: 16,777,216 RGB8 pixels; audio is 15 s, mono or stereo.
VIDEO_READ_BYTES = 16 << 30
IMAGE_READ_BYTES = 64 << 20
AUDIO_READ_BYTES = 64 << 20
#: Each kind's file field, its H3 reference kind and its `<Picture|Audio|Video N>` label. A
#: `frame` is a keyframe, never a reference: it has a file and nothing else.
FILE = {"character": "image", "background": "image", "audio": "audio", "video": "video",
        "anchor_video": "video", "world_state": "image", "frame": "image"}
_SLOT = {"character": "picture", "background": "picture", "audio": "audio", "video": "video",
         "anchor_video": "video", "world_state": "picture"}
_POLICY = {"character": "image", "background": "image", "audio": "audio", "video": "video",
           "anchor_video": "video", "world_state": "image"}
#: The kinds read from a clip.
CLIPS = ("video", "anchor_video")
_SPEAKER = re.compile(r"\s*\((S\d+)\)")
_PREFIX = re.compile(r"\s*\[([^\]\n]*)\]")
#: The summary's task types (ref guide 3); a summary without one opens with the first.
TASKS = ("reference generation", "keyframe completion", "video editing", "video continuation",
         "audio reuse", "audio reference")
#: A clip guides a description shorter than this (the guide asks 350-500 words): H3 then fills
#: the gaps from the clip and copies its people and setting (runs 10706, 10783 vs 10707).
SHORT_DESCRIPTION = 150


class StoryReference(msgspec.Struct):
    """A supplied file (`FILE[kind]`), or a character or background image generated from its
    description.

    - `video`: a reference clip, its motion and camera (or the source of a `[video editing]`).
    - `anchor_video`: a clip whose frames are anchored in the output, which follows it frame by
      frame, as an edit's source (Alissonerdx's head swap and style transfer; ComfyUI's Add
      Guide): every latent frame at the render's own canvas, on every step, and the render takes
      its canvas (`guide_rows`). Never named in a prompt, one per call, no longer than the
      render. It costs what a second render does: per `reference_rows`, about 5.5x the denoise
      for a 12 s clip beside a 15 s render.
    - `frame`: a keyframe, the first or last frame of a segment (`at`; a segment's `first_frame`
      or `last_frame` names one explicitly). Never named in a prompt, never a reference.

    `world_state` is internal: the picture a continuing segment without references takes from
    its predecessor's last frame (`h3._continuation_pictures`), as a soft picture of the set.
    """

    name: str
    kind: Literal["character", "background", "audio", "video", "anchor_video", "frame"]
    description: str = ""
    retention_analysis: Annotated[str, msgspec.Meta(max_length=1024)] | None = None
    image: Annotated[
        ImageAsset | None,
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
    ] = None
    audio: Annotated[
        AudioAsset | None,
        AssetBound(max_bytes=256 << 20, max_decoded_bytes=AUDIO_READ_BYTES),
        REFERENCE_AUDIO,
    ] = None
    video: Annotated[
        VideoAsset | None,
        AssetBound(max_bytes=VIDEO_MAX_BYTES, max_decoded_bytes=VIDEO_READ_BYTES),
        REFERENCE_VIDEO,
    ] = None
    #: A video's own sound: dropped unless set, so the clip guides motion alone.
    use_audio: bool = False
    #: A character's voice: a timbre reference wherever the character is named.
    voice: Annotated[
        AudioAsset | None,
        AssetBound(max_bytes=256 << 20, max_decoded_bytes=AUDIO_READ_BYTES),
        REFERENCE_AUDIO,
    ] = None
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None
    #: A `frame`'s default segment: the first segment's first frame, or the last one's last.
    at: Literal["first", "last"] = "first"

    def __post_init__(self) -> None:
        # Prompts quote the name as <name>: any text reads as a name once its brackets and
        # extra whitespace go.
        self.name = " ".join(self.name.replace("<", " ").replace(">", " ").split())
        if not self.name:
            raise ValueError("a reference name needs at least one visible character")


def voices(references: Sequence[StoryReference]) -> tuple[list[StoryReference], dict[str, StoryReference]]:
    """Every reference with each character's voice after it, as the audio reference
    "<Name>'s voice", and each voice's character."""
    pool: list[StoryReference] = []
    owners: dict[str, StoryReference] = {}
    for ref in references:
        pool.append(ref)
        if ref.voice is not None:
            voice = StoryReference(f"{ref.name}'s voice", "audio", audio=ref.voice)
            pool.append(voice)
            owners[voice.name] = ref
    return pool, owners


def reference_seed(reference: StoryReference, request_id: str) -> int:
    if reference.seed is not None:
        return reference.seed
    identity = f"{request_id}/h3/reference/{reference.name.casefold()}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


async def resolve_reference_images(
    references: Sequence[StoryReference],
    generate: Callable[[StoryReference], Awaitable[ImageAsset]],
    prepare_character: Callable[[StoryReference], Awaitable[ImageAsset]] | None = None,
    show: Callable[[StoryReference, ImageAsset], None] | None = None,
) -> dict[str, ImageAsset | AudioAsset | VideoAsset]:
    """Prepare supplied characters, reuse other supplied files, and generate missing images
    once, concurrently. `show` gets each made image in request order, as soon as it and every
    earlier one exist."""
    images: dict[str, ImageAsset | AudioAsset | VideoAsset] = {}
    for ref in voices(references)[0]:
        supplied = getattr(ref, FILE[ref.kind])
        if supplied is not None:
            images[ref.name] = supplied
    pending = [
        ref for ref in references
        if ref.kind in ("character", "background") and (
            ref.image is None or ref.kind == "character" and prepare_character is not None
        )
    ]

    async def one(reference: StoryReference) -> ImageAsset:
        if reference.image is not None:
            assert prepare_character is not None
            return await prepare_character(reference)
        return await generate(reference)

    tasks = [asyncio.create_task(one(reference)) for reference in pending]
    ready = 0
    try:
        for finished in asyncio.as_completed(tasks):
            await finished
            while ready < len(tasks) and tasks[ready].done():
                reference, image = pending[ready], tasks[ready].result()
                images[reference.name] = image
                if show is not None:
                    show(reference, image)
                ready += 1
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return images


def image_prompt(reference: StoryReference, style: str = "") -> str:
    parts = [reference.description.strip()]
    if style.strip():
        parts.append(f"style:\n{style.strip()}")
    if reference.kind == "character":
        parts.append(
            "Generate exactly one image: a character design sheet on a plain white background, "
            "with three side-by-side panels of equal width showing the same character. Left: "
            "a full-body front view. Middle: a full-body back view. Right: a close-up of the "
            "head and shoulders, facing the camera with a neutral expression, showing the face "
            "in sharp detail: eyes, eyebrows, nose, lips, skin, hairline and hairstyle. Show "
            "the entire character in both full-body views, in a natural neutral pose. Keep the "
            "character's face, appearance, proportions and clothing consistent across all "
            "three panels. Single subject isolated against a seamless pure white studio "
            "background. Show the whole subject clearly, with no scenery, text, labels or "
            "other people."
        )
    else:
        parts.append(
            "Generate an empty environment reference image showing only the architecture, "
            "materials, lighting and defining features of the location. Absolutely no people, "
            "characters, human figures, faces, bodies, crowds, pedestrians, silhouettes or "
            "human-like subjects may appear anywhere in the image. Do not include labels, "
            "text, panels or character references."
        )
    parts.append(
        "Follow the visual style explicitly requested in the description or style section. "
        "If no visual style is specified, use a realistic photographic look. "
        "Use anime styling only when explicitly requested."
    )
    return "\n\n".join(parts)


def validate_references(references: Sequence[StoryReference]) -> dict[str, StoryReference]:
    # The render children enforce this same policy after model preparation; refuse here first.
    # Videos attach only where named, so the H3 limit on them is per call (`_segment_call`).
    pool = voices(references)[0]
    story = [ref for ref in pool if ref.kind != "frame"]
    videos = sum(ref.kind in CLIPS for ref in story)
    if story:
        _policy(
            [_POLICY[ref.kind] for ref in story if ref.kind not in CLIPS]
            + ["video"] * min(videos, MAX_VIDEO_REFERENCES)
        )
    by_name: dict[str, StoryReference] = {}
    for reference in pool:
        key = name_key(reference.name)
        if key in by_name:
            raise InvalidRequest(
                f"duplicate reference name: {reference.name}", fields=["references"]
            )
        field = FILE[reference.kind]
        if reference.voice is not None and reference.kind != "character":
            raise InvalidRequest(
                f"only a character takes a voice, not {reference.kind} {reference.name}",
                fields=["references"],
            )
        for other in {"image", "audio", "video"} - {field}:
            if getattr(reference, other) is not None:
                raise InvalidRequest(
                    f"{reference.kind} reference {reference.name} takes {field}, not {other}",
                    fields=["references"],
                )
        if reference.kind == "frame":
            if reference.image is None:
                raise InvalidRequest(f"frame {reference.name} needs an image", fields=["references"])
        elif reference.kind == "world_state":
            pass  # internal: a predecessor's last frame, taken when it lands (`h3._continuation_pictures`)
        elif getattr(reference, field) is None and (field != "image" or not reference.description.strip()):
            raise InvalidRequest(
                f"reference {reference.name} needs {'an image or a nonempty description' if field == 'image' else f'its {field}'}",
                fields=["references"],
            )
        if reference.use_audio and reference.kind not in CLIPS:
            raise InvalidRequest(f"only a clip has audio to use, not {reference.kind} {reference.name}",
                                 fields=["references"])
        if reference.retention_analysis is not None and not reference.retention_analysis.strip():
            raise InvalidRequest(
                f"reference {reference.name} has blank retention_analysis; omit it for the default",
                fields=["references"],
            )
        by_name[key] = reference
    return by_name


def _policy(kinds: list[str], prefix: str = "") -> None:
    try:
        validate_reference_policy(kinds)
    except ValueError as exc:
        raise InvalidRequest(f"{prefix}{exc}", code="reference_policy", fields=["references"]) from exc


def encoder_prompt(task: Task, prompt: str) -> str:
    """The text H3's encoder receives: the prompt verbatim, plus `N/A` music unless it names it.

    H3 invents score and speech for audio no section describes. The line takes the task's
    upstream layout: fl2va's single-line fields, ref2va's heading-over-value sections.
    """
    if "non_diegetic_music" in prompt.casefold() or "non_diegetic_music" in _sections(prompt):
        return prompt
    line = "non_diegetic_music: N/A" if task.startswith("fl2va") else "non_diegetic_music:\nN/A"
    return "\n\n".join(filter(None, (prompt.rstrip(), line)))


class SegmentCall(NamedTuple):
    prompt: str
    references: list[StoryReference]
    warnings: list[str]


def name_key(name: str) -> str:
    """How a `<Name>` token matches a reference: any case, `_`, `-` and space alike."""
    return " ".join(re.sub(r"[_-]", " ", name).split()).casefold()


def compile_segments(
    prompts: Sequence[str],
    references: Sequence[StoryReference],
    *,
    style: str = "",
    overall_soundscape: str = "",
    non_diegetic_music: str = "N/A",
    segment_music: Sequence[str | None] | None = None,
    forced: Sequence[frozenset[str]] | None = None,
) -> list[SegmentCall]:
    """Each child prompt uses its music override, inline score, or overall score. A segment
    also attaches its `forced` references unnamed (a continuation's picture). Frames are
    keyframes, never references: they are left out."""
    validate_references(references)
    references = [ref for ref in references if ref.kind != "frame"]
    forced = forced or [frozenset()] * len(prompts)
    overrides = [None] * len(prompts) if segment_music is None else segment_music
    calls, texts = [], []
    for index, (text, override) in enumerate(zip(prompts, overrides, strict=True)):
        music = non_diegetic_music if override is None else override
        text = _fill(
            text, {"non_diegetic_music": music.strip() or "N/A"},
            replace_music=override is not None,
        )
        texts.append(text)
        calls.append(_segment_call(
            text, references, style.strip(), overall_soundscape.strip(), index, forced[index]
        ))
    sent = {ref.name for call in calls for ref in call.references}
    # Anchored frames enter as Add Guide rows, never through the text, so they need no mention:
    # a clip no prompt names anchors the first segment (a one-segment call's only one).
    unnamed = frozenset(ref.name for ref in references if ref.kind == "anchor_video" and ref.name not in sent)
    if calls and unnamed:
        calls[0] = _segment_call(
            texts[0], references, style.strip(), overall_soundscape.strip(), 0, forced[0] | unnamed
        )
    for ref in references:
        if not calls or ref.name in sent or ref.kind == "anchor_video":
            continue
        if ref.kind == "video":
            calls[0].warnings.append(
                f"<{ref.name}> is never named in a prompt, so H3 never sees it; name the clip "
                "where its motion applies."
            )
    return calls


def _numbering(references: Sequence[StoryReference]) -> dict[tuple[str, int], StoryReference]:
    """`<Picture N>`, `<Audio N>` and `<Video N>`: each kind counted in the given order, as H3
    labels them. A `use_audio` clip's soundtrack takes an `<Audio N>` just before its video. A
    `frame` is no reference and takes none."""
    counts = dict.fromkeys(("picture", "audio", "video"), 0)
    slots: dict[tuple[str, int], StoryReference] = {}
    for ref in references:
        if ref.kind == "frame":
            continue
        for slot in ("audio", "video") if ref.kind in CLIPS and ref.use_audio else (_SLOT[ref.kind],):
            counts[slot] += 1
            slots[slot, counts[slot]] = ref
    return slots


def _segment_call(
    text: str,
    references: Sequence[StoryReference],
    style: str,
    soundscape: str,
    index: int,
    forced: frozenset[str] = frozenset(),
) -> SegmentCall:
    """Backgrounds and `forced` always; characters, world states, audio and videos only where
    the text names them, all in request order, as H3 packs and labels them. With no references
    at all, none: the segment renders from its text and keyframes (FL2VA).

    A token names a reference by its name first (any case, `_`, `-` and space alike), else as
    the global `<Picture N>`/`<Audio N>`/`<Video N>`; these are renumbered for the call. A
    character's voice follows the character. The prompt takes MiniMax's full-reference form
    (ref guide): a summary task type, each clip bound to the characters who perform it, and
    `style` as the description's opening sentence.
    """
    pool, owners = voices(references)
    if not pool:
        return SegmentCall(text, [], [])
    by_key = {name_key(ref.name): ref for ref in pool}
    by_slot = _numbering(pool)
    text, style = _open_with_style(text, style)
    words = len(_body(text, "detailed_description").split()) - len(style.split())

    def lookup(token: re.Match[str]) -> StoryReference | None:
        named = by_key.get(name_key(token.group(1)))
        media = _MEDIA.fullmatch(name_key(token.group(1)))
        if named is not None or media is None:
            return named
        return by_slot.get((media.group(1).lower(), int(media.group(2))))

    def part(token: re.Match[str], ref: StoryReference) -> tuple[str, str]:
        """The reference and which of its labels a token names: a clip's sound or picture."""
        media = _MEDIA.fullmatch(name_key(token.group(1)))
        named = by_key.get(name_key(token.group(1))) is not None
        return ref.name, media.group(1).lower() if media and not named else _SLOT[ref.kind]

    def mentioned(span: str) -> list[StoryReference]:
        found = {ref.name: ref for token in _TOKEN.finditer(span) if (ref := lookup(token))}
        return list(found.values())

    unmatched = dict.fromkeys(
        token.group(0)
        for token in _TOKEN.finditer(text)
        if _MEDIA.fullmatch(name_key(token.group(1))) and lookup(token) is None
    )
    warnings = (
        [f"Segment {index + 1}: {', '.join(unmatched)} match no reference; left as written."]
        if unmatched else []
    )
    named = {ref.name for ref in mentioned(text)}
    named |= {voice.name for voice in pool if voice.name in owners and owners[voice.name].name in named}
    # An anchored clip packs after every other reference, so when it leaves the packed
    # references (`guide_rows`) every other label stays as H3 numbers it.
    used = sorted(
        (ref for ref in pool if ref.kind == "background" or ref.name in named | forced),
        key=lambda ref: ref.kind == "anchor_video",
    )
    if sum(ref.kind == "anchor_video" for ref in used) > 1:
        raise InvalidRequest(
            f"segment {index + 1} anchors more than one video's frames", fields=["references"]
        )
    if all(ref.kind == "audio" for ref in used):
        # H3's Ref2VA needs a picture: a segment that names none renders from its text and
        # keyframes (FL2VA), which takes no audio reference.
        if used:
            warnings.append(f"Segment {index + 1} names only audio and no picture, so it renders "
                            "without references.")
        return SegmentCall(text, [], warnings)
    _policy([_POLICY[ref.kind] for ref in used], f"segment {index + 1}: ")
    sections = _sections(text)
    _, start, end = sections.get("detailed_description", (0, 0, 0))
    markers = list(_SHOT.finditer(text, start, end))
    spans = pairwise([marker.start() for marker in markers] + [end])
    shots = [
        (m.group(1), {ref.name for ref in mentioned(text[a:b])})
        for m, (a, b) in zip(markers, spans, strict=True)
    ]
    _, start, end = sections.get("subject_definitions", (0, 0, 0))
    bound = {
        (person.name, clip.name)
        for named_here in map(mentioned, text[start:end].splitlines())
        for person in named_here if person.kind == "character"
        for clip in named_here if clip.kind == "video"
    }
    _, start, end = sections.get("retention_analysis", (0, 0, 0))
    # The text's own retention line for a reference opens with it; the template's then gives way.
    retained = {
        part(first, ref)
        for line in text[start:end].splitlines()
        if (first := _TOKEN.match(line.lstrip(" \t-*"))) and (ref := lookup(first))
    }
    given = _tasks(text)
    editing = bool({"video editing", "video continuation"} & {task.casefold() for task in given})

    def speaker(owner: StoryReference, span: str) -> str:
        """The `(Sx)` the text gives the voice's character, never a new one (guide §5.4)."""
        for token in _TOKEN.finditer(span):
            if lookup(token) is owner and (sx := _SPEAKER.match(span, token.end())):
                return f" ({sx.group(1)})"
        return ""

    child = {(ref.name, kind): number for (kind, number), ref in _numbering(used).items()}

    def label(ref: StoryReference, kind: str | None = None) -> str:
        kind = kind or _SLOT[ref.kind]
        return f"<{kind.capitalize()} {child[ref.name, kind]}>"

    def renumber(token: re.Match[str]) -> str:
        ref = lookup(token) if _MEDIA.fullmatch(name_key(token.group(1))) else None
        if ref is None or part(token, ref) not in child:
            return token.group(0)
        return label(ref, part(token, ref)[1])

    def names(refs: Sequence[StoryReference]) -> str:
        tokens = [f"<{ref.name}>" for ref in refs]
        return ", ".join(tokens[:-1]) + " and " + tokens[-1] if len(tokens) > 1 else "".join(tokens)

    characters = [ref for ref in used if ref.kind == "character"]
    places = [ref for ref in used if ref.kind == "background"]
    clips = [ref for ref in used if ref.kind == "video"]

    def performers(clip: StoryReference) -> list[StoryReference]:
        """The characters a motion clip's shots name, else every character in the call."""
        cast = set().union(*(found for _, found in shots if clip.name in found))
        return [ref for ref in characters if ref.name in cast] or characters

    motion = {} if editing else {clip.name: performers(clip) for clip in clips}
    voiced = {ref.name: speaker(owners[ref.name], text) for ref in used if ref.name in owners}
    # A subject_definitions line of the text's own that opens with a reference, and cites its
    # picture for an image, defines it: the template adds no second line, and the reference's
    # description joins the text's line.
    _, start, end = sections.get("subject_definitions", (0, 0, 0))
    own: dict[tuple[str, str], int] = {}
    for definition in re.finditer(r"[^\n]+", text[start:end]):
        first = _TOKEN.match(definition.group(0).lstrip(" \t-*"))
        if not first or (ref := lookup(first)) is None or ref not in used:
            continue
        cited = FILE[ref.kind] != "image" or any(
            lookup(token) is ref and _MEDIA.fullmatch(name_key(token.group(1)))
            for token in _TOKEN.finditer(definition.group(0))
        )
        if cited:
            own.setdefault(part(first, ref), start + definition.end())
    for (name, kind), at in sorted(own.items(), key=lambda item: -item[1]):
        ref = by_key[name_key(name)]
        if kind == _SLOT[ref.kind] and ref.description and ref.description.strip() not in text[start:at]:
            text = text[:at] + f" {ref.description.strip()}" + text[at:]
    text = _TOKEN.sub(renumber, text)
    definitions: list[str] = []
    retention: list[str] = []
    tasks = set()
    for ref in used:
        slot = label(ref)
        line_label = slot
        defined = len(definitions)
        if ref.name in owners:
            owner = f"<{owners[ref.name].name}>"
            definitions.append(f"{slot} is the voice-timbre reference for {owner}{voiced[ref.name]}.")
            default = (f"reference - {owner}'s voice follows {slot}'s timbre and delivery without "
                       "copying the original signal.")
        elif ref.kind == "audio":
            definitions.append(f"{slot} is the supplied audio reference for <{ref.name}>.")
            default = f"partially_copy - {slot} is copied into the target audio, with other sounds around it."
        elif ref.kind in CLIPS:
            source = editing or ref.kind == "anchor_video"
            role = "source video" if source else "motion and camera reference"
            who = f" for {names(motion[ref.name])}" if motion.get(ref.name) else ""
            definitions.append(f"{slot} is the {role}{who}." if _MEDIA.fullmatch(ref.name)
                               else f"<{ref.name}> is the {role} in {slot}{who}.")
            if ref.use_audio:
                sound = label(ref, "audio")
                if (ref.name, "audio") not in own:
                    definitions.append(f"{sound} is the synchronized audio track of {slot}.")
                if (ref.name, "audio") not in retained:
                    retention.append(f"{sound}: partially_copy - the synchronized audio track of {slot} "
                                     "is copied into the target audio, with other sounds around it.")
                    tasks.add("audio reuse")
            if source:
                line_label = f"{slot} (source video editing)"
                default = ("fully_preserved - its framing, setting and subjects are kept except where "
                           "the description changes them.")
            else:
                line_label = f"{slot} (motion, choreography and camera movement)"
                cast = names(motion[ref.name]) or "the target video's subjects"
                where = f" in {names(places)}" if places else ""
                default = (f"attribute_transfer - its movements, timing and camera work are performed "
                           f"by {cast}{where}.")
        elif ref.kind == "world_state":
            definitions.append(
                f"<{ref.name}> is the world state shown in {slot}: how the set is arranged and "
                "where each person and object stands."
            )
            default = (f"attribute_transfer - keep the set's arrangement and each person's and "
                       f"object's position from {slot}, not its exact frame, framing or moment.")
            line_label = f"<{ref.name}>"
        else:
            moves = [f"<{clip.name}>" if not _MEDIA.fullmatch(clip.name) else label(clip)
                     for clip in clips
                     if ref in motion.get(clip.name, ()) and (ref.name, clip.name) not in bound]
            binding = f", whose movements come from {' and '.join(moves)}" if moves else ""
            definitions.append(f"<{ref.name}> is the {ref.kind} shown in {slot}{binding}.")
            preserved = (
                "keep the face, hair, and outfit of the referenced character. Ignore the white background"
                if ref.kind == "character"
                else "the architecture, materials and defining features of the referenced environment"
            )
            default = f"fully_preserved - {preserved}."
            appears = [f"[Shot {n}]" for n, found in shots if ref.name in found]
            line_label = f"<{ref.name}>" + (f" (appears in {', '.join(appears)})" if appears else "")
        if (ref.name, _SLOT[ref.kind]) in own:
            del definitions[defined]
        elif ref.description:
            definitions[defined] += f" {ref.description}"
        line = ref.retention_analysis or default
        if ref.kind == "audio":
            tasks.add("audio reuse" if line.split(" ", 1)[0] in ("fully_copy", "partially_copy")
                      else "audio reference")
        if ref.retention_analysis or (ref.name, _SLOT[ref.kind]) not in retained:
            retention.append(f"{line_label}: {line}")
    if clips and not editing and words < SHORT_DESCRIPTION:
        warnings.append(
            f"Segment {index + 1}: {names(clips)} guide{'s' * (len(clips) == 1)} a description of "
            f"{words} words. H3 copies a clip's own people and setting unless the description "
            "establishes each character, the background and the beats in order; MiniMax's guide "
            "asks 350-500 words."
        )
    if "summary" in (sections := _sections(text)):
        _, start, end = sections["summary"]
        tag = " + ".join([*(given or ["reference generation"]), *sorted(tasks - set(given))])
        if prefix := _PREFIX.match(text, start, end) if given else None:
            text = text[: prefix.start(1)] + tag + text[prefix.end(1):]
        elif body := text[start:end].lstrip():
            text = text[: end - len(body)] + f"[{tag}] " + body + text[end:]
        else:
            text = text[:start] + f"\n[{tag}]" + text[start:]
    additions = {
        "subject_definitions": "\n".join(definitions),
        "retention_analysis": "\n".join(retention),
        "overall_soundscape": soundscape,
    }
    return SegmentCall(_fill(text, additions), used, warnings)


WRITER_TOKENS = 1400
WRITER_PARTS = ("ROLES", "DESCRIPTION", "SOUNDSCAPE")
_PART = re.compile(r"^[ \t*#]*(" + "|".join(WRITER_PARTS) + r")[ \t*]*:[ \t*]*", re.MULTILINE)
_ROLE = re.compile(r"^\W*<([^<>\n]+)>[^:=\n]*[:=][ \t]*(.+)$", re.MULTILINE)
_PERSON = re.compile(r"\bP\d+\b")
_CLIP_WORDS = re.compile(r"\s*\b(?:from|in|of) (?:the|this) (?:reference )?(?:clip|video)\b", re.IGNORECASE)
WRITER_TASK = """You write one shot list for MiniMax-H3, a video model. The video performs the motion of a \
reference clip again, with the request's characters in the request's place.

{references}

The request; keep every event, name and detail it asks for:
{request}

Write exactly these three parts, each starting with its label on its own line, and nothing else:
ROLES: one line per character who performs the clip's motion, "<Name>: <who they are in the clip, \
by what they do there>", matching the clip's people to the request's characters by what the request \
says each one does.
DESCRIPTION: 350 to 500 words of English in playback order, as shots: "[Shot 1] ..." and each later \
shot "[Shot N] At MM:SS.mmm, ...". At each character's first appearance give their position in the \
frame and their look, and establish the place from its look. Then follow the clip's beats in order: \
who does what to whom, which way each turns, falls or moves, each person's facing relative to the \
camera, and the camera's framing and moves. Keep everything the request asks for. The clip gives only \
motion, timing and camera: never describe its people's looks or its setting, never call anyone P1 \
or P2, and never mention the clip itself. Write times as MM:SS.mmm, such as 00:02.400. Name each \
of {tokens} by that token, exactly as written, at its first appearance and wherever it acts. Write \
at least 350 words. Do not restate the style.
SOUNDSCAPE: one or two sentences of ambience and physical sounds, without music or speech."""


def writer_request(
    text: str, attached: Sequence[StoryReference], clips: dict[str, str], style: str, soundscape: str
) -> str | None:
    """What the writer is asked for a call that attaches a described clip and lacks what MiniMax's
    guide asks for: a 350-500-word description, each character bound to the clip, a soundscape.
    None when nothing is missing."""
    if not _missing(text, attached, clips, soundscape):
        return None
    lines = ["References, each named by its token:"]
    for ref in attached:
        if ref.kind in ("character", "background"):
            role = "a character" if ref.kind == "character" else "the place"
            lines.append(f"<{ref.name}>: {role}. " + (f"Look: {ref.description}" if ref.description else ""))
        elif clips.get(ref.name):
            lines.append(f"<{ref.name}>: the motion clip, P1, P2 and so on being its people:\n{clips[ref.name]}")
    if style:
        lines.append(f"\nStyle: {style}")
    tokens = [f"<{ref.name}>" for ref in attached if ref.kind in ("character", "background")]
    listed = ", ".join(tokens[:-1]) + " and " + tokens[-1] if len(tokens) > 1 else "".join(tokens)
    return WRITER_TASK.format(references="\n".join(lines), request=text.strip(), tokens=listed)


def apply_written(
    text: str,
    reply: str,
    references: Sequence[StoryReference],
    attached: Sequence[StoryReference],
    clips: dict[str, str],
    soundscape: str,
) -> tuple[str, list[str]]:
    """The writer's reply where the text lacks it: the guide's binding line per unbound character
    ("<Ann> is the character whose appearance comes from <Picture 1> and whose movements come
    from ... in <Video 1>"), the description when the text's is short, the soundscape when it has
    none. The text's own full sections stay as written; a description that drops a named
    reference or names the clip's people is not used."""
    missing = _missing(text, attached, clips, soundscape)
    found = list(_PART.finditer(reply))
    parts = {
        m.group(1): reply[m.end(): nxt.start() if nxt else len(reply)].strip()
        for m, nxt in zip(found, [*found[1:], None], strict=True)
    }
    warnings = []
    clip = next(ref for ref in attached if ref.kind == "video" and clips.get(ref.name))
    by_key = {name_key(ref.name): ref for ref in attached if ref.kind == "character"}
    picture = {ref.name: n for (kind, n), ref in _numbering(voices(references)[0]).items() if kind == "picture"}
    roles = [
        f"<{ref.name}> is the character whose appearance comes from <Picture {picture[ref.name]}> and "
        f"whose movements come from {role} in <{clip.name}>."
        for name, said in _ROLE.findall(parts.get("ROLES", ""))
        if (ref := by_key.get(name_key(name))) and ref.name in missing.get("roles", ())
        and (role := _role(said))
    ]
    if roles:
        text = _fill(text, {"subject_definitions": "\n".join(roles)})
    if "description" in missing:
        written = parts.get("DESCRIPTION", "")
        named = {name_key(token.group(1)) for token in _TOKEN.finditer(written)}
        asked = {name_key(token.group(1)) for token in _TOKEN.finditer(text)}
        dropped = [ref for ref in attached if ref.kind == "character" and name_key(ref.name) in asked - named]
        places = [f"<{ref.name}>" for ref in attached if ref.kind == "background" and name_key(ref.name) not in named]
        if places:
            # A place told in words only: the reference is still where it all happens.
            written = f"The video takes place in {' and '.join(places)}.\n{written}"
        why = (f"it is {len(written.split())} words" if len(written.split()) < SHORT_DESCRIPTION
               else f"it never names <{dropped[0].name}>" if dropped
               else "it names the clip's people" if _PERSON.search(written) else "")
        if why:
            warnings.append(f"The written description was not used: {why}; the text's own is sent.")
        else:
            text = _replace(text, "detailed_description", written)
    if "soundscape" in missing and parts.get("SOUNDSCAPE"):
        text = _replace(text, "overall_soundscape", parts["SOUNDSCAPE"])
    return text, warnings


def _role(said: str) -> str:
    """A writer's role phrase as a binding reads it: no P1/P2, no "from the clip", lowercase."""
    role = _CLIP_WORDS.sub("", _PERSON.sub("", said)).strip(" ,.;:-")
    return role[:1].lower() + role[1:] if role[:1].isupper() and not role[1:2].isupper() else role


def _missing(
    text: str, attached: Sequence[StoryReference], clips: dict[str, str], soundscape: str
) -> dict[str, tuple[str, ...]]:
    """What a call with a described clip lacks: characters bound to no clip in
    `subject_definitions`, a description shorter than `SHORT_DESCRIPTION` words, a soundscape."""
    if not any(ref.kind == "video" and clips.get(ref.name) for ref in attached):
        return {}
    keys = {name_key(ref.name): ref for ref in attached}
    bound = set()
    for line in _body(text, "subject_definitions").splitlines():
        found = [keys.get(name_key(t.group(1))) for t in _TOKEN.finditer(line)]
        if any(ref is not None and ref.kind == "video" for ref in found) or re.search(r"<\s*video\s+\d+\s*>", line, re.I):
            bound |= {ref.name for ref in found if ref is not None and ref.kind == "character"}
    missing: dict[str, tuple[str, ...]] = {}
    if unbound := tuple(ref.name for ref in attached if ref.kind == "character" and ref.name not in bound):
        missing["roles"] = unbound
    if len(_body(text, "detailed_description").split()) < SHORT_DESCRIPTION:
        missing["description"] = ()
    if not soundscape and _body(text, "overall_soundscape").strip().upper() in ("", "N/A"):
        missing["soundscape"] = ()
    return missing


def _replace(text: str, name: str, body: str) -> str:
    """`name`'s body replaced, its heading kept; a headerless description gains its heading."""
    found = _sections(text)
    if name not in found:
        return _fill(text, {name: body})
    head, start, end = found[name]
    rest = text[end:].lstrip()
    return text[:head] + f"{name}:\n{body}" + (f"\n\n{rest}" if rest else "\n")


def _tasks(text: str) -> list[str]:
    """The summary's `[task + task]` prefix, when it opens with one."""
    _, start, end = _sections(text).get("summary", (0, 0, 0))
    prefix = _PREFIX.match(text, start, end)
    given = [task.strip() for task in prefix.group(1).split("+")] if prefix else []
    return given if all(task.casefold() in TASKS for task in given) else []


def _body(text: str, name: str) -> str:
    _, start, end = _sections(text).get(name, (0, 0, 0))
    return text[start:end]


def _open_with_style(text: str, style: str) -> tuple[str, str]:
    """The segment's own `style` section, else the global style, as the description's opening
    sentence (ref guide 5.2) rather than a section of its own; and that style."""
    found = _sections(text)
    if "style" in found:
        head, start, end = found["style"]
        style = text[start:end].strip() or style
        text = text[:head] + text[end:]
    if not style:
        return text, style
    found = _sections(text)
    if "detailed_description" not in found:
        return _fill(text, {"detailed_description": style}), style
    head, start, end = found["detailed_description"]
    if head == start:
        return f"{style}\n{text}", style
    rest = text[start:end].lstrip(" \t")
    return text[:start] + "\n" + style + ("" if rest.startswith("\n") else "\n") + rest + text[end:], style


def _shot_texts(text: str) -> list[str]:
    """The description's `[Shot N]` spans, or the whole description as one shot."""
    _, start, end = _sections(text).get("detailed_description", (0, 0, 0))
    marks = [m.start() for m in _SHOT.finditer(text, start, end)] or [start]
    return [text[a:b] for a, b in pairwise([*marks, end])]


def opening_cut_warnings(
    prompts: Sequence[str], references: Sequence[StoryReference], carries_picture: Sequence[bool]
) -> list[str]:
    """A hint, not a gate: a segment carrying the previous picture whose first shot cuts to
    characters the previous segment's last shot does not name. The carried frames open the
    new clip, so H3 keeps the old characters and the cut is lost."""
    by_key = {name_key(r.name): r.name for r in references if r.kind == "character"}
    by_slot = {key: ref.name for key, ref in _numbering(references).items() if ref.kind == "character"}

    def characters(shot: str) -> set[str]:
        found = set()
        for token in _TOKEN.finditer(shot):
            media = _MEDIA.fullmatch(token.group(1))
            name = (by_key.get(name_key(token.group(1))) if media is None
                    else by_slot.get((media.group(1).lower(), int(media.group(2)))))
            if name is not None:
                found.add(name)
        return found

    warnings = []
    for index, (previous, text) in enumerate(pairwise(prompts), start=1):
        if not carries_picture[index] or not text.strip():
            continue
        opening = _shot_texts(text)[0]
        new = sorted(characters(opening) - characters(_shot_texts(previous)[-1]))
        if new and _CUT.search(opening):
            names = ", ".join(f"<{name}>" for name in new)
            warnings.append(
                f"Segment {index + 1} carries the previous picture but opens with a cut to "
                f"{names}, not in segment {index}'s last shot; the carried frames keep its "
                "characters. For a clean picture cut with continuous sound, set "
                "context_frames: false."
            )
    return warnings


def _sections(text: str) -> dict[str, tuple[int, int, int]]:
    """Each section's first header: (header start, body start, body end).

    Text before the first header is the description when no header names one.
    """
    headers = list(_HEADER.finditer(text))
    found: dict[str, tuple[int, int, int]] = {}
    for i, header in enumerate(headers):
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        name = re.sub(r"[ _-]+", "_", header.group(1).lower())
        found.setdefault(name, (header.start(), header.end(), body_end))
    lead = headers[0].start() if headers else len(text)
    if "detailed_description" not in found and text[:lead].strip():
        found["detailed_description"] = (0, 0, lead)
    return found


def _fill(text: str, additions: dict[str, str], *, replace_music: bool = False) -> str:
    """Append each addition inside its section, or add the section in canonical order.

    Only these insertions change the text. An `N/A` body gives way to described content;
    a segment's own music is kept unless a structured music override is given. A description
    without a heading gains `detailed_description:` once sections are inserted ahead of it.
    """
    additions = {name: additions[name] for name in SECTIONS if additions.get(name)}
    if not text.strip():
        return "\n\n".join(f"{name}:\n{value}" for name, value in additions.items())
    found = _sections(text)
    edits: list[tuple[int, int, int, str]] = []
    for order, (name, addition) in enumerate(additions.items()):
        if name in found:
            _, start, end = found[name]
            body = text[start:end].strip()
            if body and name == "non_diegetic_music" and replace_music:
                at = text.index(body, start)
                edits.append((at, order, at + len(body), addition))
                continue
            if body and (name == "non_diegetic_music" or addition.upper() == "N/A"):
                continue
            if body.upper() == "N/A":
                at = text.index(body, start)
                edits.append((at, order, at + len(body), addition))
            else:
                at = start + len(text[start:end].rstrip())
                edits.append((at, order, at, "\n" + addition))
            continue
        later = [found[n][0] for n in SECTIONS[SECTIONS.index(name) + 1:] if n in found]
        section = f"{name}:\n{addition}"
        if later:
            edits.append((min(later), order, min(later), section + "\n\n"))
        else:
            at = len(text.rstrip())
            edits.append((at, order, at, "\n\n" + section))
    if found.get("detailed_description", (1, 0, 0))[:2] == (0, 0) and any(at == 0 for at, *_ in edits):
        edits.append((0, len(additions), 0, "detailed_description:\n"))
    for start, _, end, insert in sorted(edits, reverse=True):
        text = text[:start] + insert + text[end:]
    return text
