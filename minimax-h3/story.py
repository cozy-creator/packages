"""A shot piece's references and prompt: its text plus MiniMax's full-reference sections."""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, NamedTuple

from cozy_runtime.author import (
    MENTION,
    AudioFile,
    AudioPreparation,
    ImageFile,
    InvalidRequest,
    Reference,
    VideoFile,
    VideoPreparation,
    name_key,
)

from minimax_h3.official import Task, validate_reference_policy

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
_MEDIA = re.compile(r"\s*(picture|audio|video)\s+(\d+)\s*", re.IGNORECASE)
_SHOT = re.compile(r"\[Shot\s*(\d+)\]", re.IGNORECASE)
#: Reference clips: a video of any length from 2 s, of which each piece reads its own window
#: (`shot`); audio outside 2-15 s refuses. Every input that carries a reference declares these.
REFERENCE_VIDEO = VideoPreparation(min_seconds=2)
REFERENCE_AUDIO = AudioPreparation(min_seconds=2, max_seconds=15, excess="refuse")
#: A reference video's upload.
VIDEO_MAX_BYTES = 1 << 30
#: Read bounds. A video streams, one decoded frame at a time, straight to its motion size
#: (`read_reference_video`), so a frame of any real source fits. A prepared image is at most
#: 16,777,216 RGB8 pixels; audio is 15 s, mono or stereo.
VIDEO_READ_BYTES = 16 << 30
IMAGE_READ_BYTES = 64 << 20
AUDIO_READ_BYTES = 64 << 20
#: Each kind's file field, its H3 reference kind and its `<Picture|Audio|Video N>` label.
FILE = {"character": "image", "prop": "image", "background": "image", "audio": "audio", "video": "video",
        "anchor_video": "video", "world_state": "image"}
_SLOT = {"character": "picture", "prop": "picture", "background": "picture", "audio": "audio", "video": "video",
         "anchor_video": "video", "world_state": "picture"}
_POLICY = {"character": "image", "prop": "image", "background": "image", "audio": "audio", "video": "video",
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


class StoryReference(Reference, kw_only=True):
    """One reference file (`FILE[kind]`) a piece attaches.

    - `background`: the environment, the shot's setup plate.
    - `video`: reference footage, its motion and camera (or the source of a `[video editing]`).
    - `anchor_video`: a plate, followed frame by frame (Alissonerdx's head swap and style
      transfer; ComfyUI's Add Guide): every latent frame at the render's own canvas, on every
      step, and the render takes its canvas (`guide_rows`). Never named in a prompt, one per call,
      no longer than the render. It costs what a second render does: per `reference_rows`, about
      5.5x the denoise for a 12 s clip beside a 15 s render.
    - `world_state`: a soft picture of the set, read at 512 px (or a text-only shot's previous
      piece's last frame), with its layout as its `description`.
    """

    kind: Literal["character", "prop", "background", "audio", "video", "anchor_video", "world_state"]
    description: str = ""
    image: ImageFile | None = None
    audio: AudioFile | None = None
    video: VideoFile | None = None
    #: A video's own sound: dropped unless set, so the clip guides motion alone.
    use_audio: bool = False
    #: A character's voice: a timbre reference wherever the character is named.
    voice: AudioFile | None = None


def voices(references: Sequence[StoryReference]) -> tuple[list[StoryReference], dict[str, StoryReference]]:
    """Every reference with each character's voice after it, as the audio reference
    "<Name>'s voice", and each voice's character."""
    pool: list[StoryReference] = []
    owners: dict[str, StoryReference] = {}
    for ref in references:
        pool.append(ref)
        if ref.voice is not None:
            voice = StoryReference(name=f"{ref.name}'s voice", kind="audio", audio=ref.voice)
            pool.append(voice)
            owners[voice.name] = ref
    return pool, owners


def validate_references(references: Sequence[StoryReference]) -> None:
    """One name per reference, voices only on characters. The H3 limits are per call
    (`_piece_call`)."""
    names: set[str] = set()
    for reference in voices(references)[0]:
        if (key := name_key(reference.name)) in names:
            raise InvalidRequest(f"two assets are named <{reference.name}>", code="asset_name", fields=["assets"])
        names.add(key)
        if reference.voice is not None and reference.kind != "character":
            raise InvalidRequest(f"only a character takes a voice, not {reference.kind} <{reference.name}>",
                                 fields=["assets"])  # fmt: skip


def _policy(kinds: list[str], prefix: str = "") -> None:
    try:
        validate_reference_policy(kinds)
    except ValueError as exc:
        raise InvalidRequest(f"{prefix}{exc}", code="reference_policy", fields=["assets"]) from exc


def encoder_prompt(task: Task, prompt: str) -> str:
    """The text H3's encoder receives: the prompt verbatim, plus `N/A` music unless it names it.

    H3 invents score and speech for audio no section describes. The line takes the task's
    upstream layout: fl2va's single-line fields, ref2va's heading-over-value sections.
    """
    if "non_diegetic_music" in prompt.casefold() or "non_diegetic_music" in _sections(prompt):
        return prompt
    line = "non_diegetic_music: N/A" if task.startswith("fl2va") else "non_diegetic_music:\nN/A"
    return "\n\n".join(filter(None, (prompt.rstrip(), line)))


class PieceCall(NamedTuple):
    prompt: str
    references: list[StoryReference]
    warnings: list[str]


def piece_call(
    text: str,
    references: Sequence[StoryReference],
    *,
    style: str,
    soundscape: str,
    music: str,
    index: int,
    forced: frozenset[str],
) -> PieceCall:
    """A Ref2VA piece's text and references: `text` with its music section, then MiniMax's
    reference sections (`_piece_call`)."""
    text = _fill(text, {"non_diegetic_music": music.strip() or "N/A"})
    return _piece_call(text, references, style.strip(), soundscape.strip(), index, forced)


def _numbering(references: Sequence[StoryReference]) -> dict[tuple[str, int], StoryReference]:
    """`<Picture N>`, `<Audio N>` and `<Video N>`: each kind counted in the given order, as H3
    labels them. A `use_audio` clip's soundtrack takes an `<Audio N>` just before its video."""
    counts = dict.fromkeys(("picture", "audio", "video"), 0)
    slots: dict[tuple[str, int], StoryReference] = {}
    for ref in references:
        for slot in ("audio", "video") if ref.kind in CLIPS and ref.use_audio else (_SLOT[ref.kind],):
            counts[slot] += 1
            slots[slot, counts[slot]] = ref
    return slots


def _piece_call(
    text: str,
    references: Sequence[StoryReference],
    style: str,
    soundscape: str,
    index: int,
    forced: frozenset[str] = frozenset(),
) -> PieceCall:
    """Backgrounds and `forced` always; characters, world states, audio and videos only where
    the text names them, all in request order, as H3 packs and labels them. With no references
    at all, none: the piece renders from its text and keyframes (FL2VA).

    A token names a reference by its name first (any case, `_`, `-` and space alike), else as
    the global `<Picture N>`/`<Audio N>`/`<Video N>`; these are renumbered for the call. A
    character's voice follows the character. The prompt takes MiniMax's full-reference form
    (ref guide): a summary task type, each clip bound to the characters who perform it, and
    `style` as the description's opening sentence.
    """
    pool, owners = voices(references)
    if not pool:
        return PieceCall(text, [], [])
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
        found = {ref.name: ref for token in MENTION.finditer(span) if (ref := lookup(token))}
        return list(found.values())

    unmatched = dict.fromkeys(
        token.group(0)
        for token in MENTION.finditer(text)
        if _MEDIA.fullmatch(name_key(token.group(1))) and lookup(token) is None
    )
    warnings = (
        [f"Piece {index + 1}: {', '.join(unmatched)} match no reference; left as written."]
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
            f"piece {index + 1} anchors more than one video's frames", fields=["assets"]
        )
    if all(ref.kind == "audio" for ref in used):
        # H3's Ref2VA needs a picture: a piece that names none renders from its text and
        # keyframes (FL2VA), which takes no audio reference.
        if used:
            warnings.append(f"Piece {index + 1} names only audio and no picture, so it renders "
                            "without references.")
        return PieceCall(text, [], warnings)
    _policy([_POLICY[ref.kind] for ref in used], f"piece {index + 1}: ")
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
        if (first := MENTION.match(line.lstrip(" \t-*"))) and (ref := lookup(first))
    }
    given = _tasks(text)
    editing = bool({"video editing", "video continuation"} & {task.casefold() for task in given})

    def speaker(owner: StoryReference, span: str) -> str:
        """The `(Sx)` the text gives the voice's character, never a new one (guide §5.4)."""
        for token in MENTION.finditer(span):
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
        first = MENTION.match(definition.group(0).lstrip(" \t-*"))
        if not first or (ref := lookup(first)) is None or ref not in used:
            continue
        cited = FILE[ref.kind] != "image" or any(
            lookup(token) is ref and _MEDIA.fullmatch(name_key(token.group(1)))
            for token in MENTION.finditer(definition.group(0))
        )
        if cited:
            own.setdefault(part(first, ref), start + definition.end())
    for (name, kind), at in sorted(own.items(), key=lambda item: -item[1]):
        ref = by_key[name_key(name)]
        if kind == _SLOT[ref.kind] and ref.description and ref.description.strip() not in text[start:at]:
            text = text[:at] + f" {ref.description.strip()}" + text[at:]
    text = MENTION.sub(renumber, text)
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
            # People are named only where the call has a character (se-089): a world state of a test pattern
            # that said "each person" grew people (25176).
            definitions.append(
                f"<{ref.name}> is the world state shown in {slot}: how the set is arranged and "
                + ("where each person and object stands." if characters else "where everything in it stands.")
            )
            kept = "each person's and object's position" if characters else "where everything in it stands"
            default = (f"attribute_transfer - keep the set's arrangement and {kept} "
                       f"from {slot}, not its exact frame, framing or moment.")
            line_label = f"<{ref.name}>"
        else:
            moves = [f"<{clip.name}>" if not _MEDIA.fullmatch(clip.name) else label(clip)
                     for clip in clips
                     if ref in motion.get(clip.name, ()) and (ref.name, clip.name) not in bound]
            binding = f", whose movements come from {' and '.join(moves)}" if moves else ""
            definitions.append(f"<{ref.name}> is the {ref.kind} shown in {slot}{binding}.")
            preserved = {
                "character": "keep the face, hair, and outfit of the referenced character. Ignore the white background",
                "prop": "keep the shape, materials and markings of the referenced object",
            }.get(ref.kind, "the architecture, materials and defining features of the referenced environment")
            default = f"fully_preserved - {preserved}."
            appears = [f"[Shot {n}]" for n, found in shots if ref.name in found]
            line_label = f"<{ref.name}>" + (f" (appears in {', '.join(appears)})" if appears else "")
        if (ref.name, _SLOT[ref.kind]) in own:
            del definitions[defined]
        elif ref.description:
            definitions[defined] += f" {ref.description}"
        if ref.kind == "audio":
            tasks.add("audio reuse" if default.split(" ", 1)[0] in ("fully_copy", "partially_copy")
                      else "audio reference")
        if (ref.name, _SLOT[ref.kind]) not in retained:
            retention.append(f"{line_label}: {default}")
    if clips and not editing and words < SHORT_DESCRIPTION:
        warnings.append(
            f"Piece {index + 1}: {names(clips)} guide{'s' * (len(clips) == 1)} a description of "
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
    return PieceCall(_fill(text, additions), used, warnings)

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
    """The piece's own `style` section, else the global style, as the description's opening
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


def _fill(text: str, additions: dict[str, str]) -> str:
    """Append each addition inside its section, or add the section in canonical order.

    Only these insertions change the text. An `N/A` body gives way to described content;
    the text's own music is kept. A description
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
