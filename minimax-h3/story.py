"""A shot segment's references and prompt: its text plus MiniMax's full-reference sections."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Literal, NamedTuple

import msgspec
from cozy_runtime.author import AudioFile, AudioPreparation, ImageFile, VideoFile

from minimax_h3.official import MAX_AUDIO_REFERENCES, MAX_IMAGE_REFERENCES, MAX_REFERENCES, MAX_VIDEO_REFERENCES, Task

#: A mention: `<`, any text without brackets or line breaks, `>`.
MENTION = re.compile(r"<([^<>\n]+)>")
#: MiniMax's full-reference sections in their canonical order. Anything else a person writes, such
#: as a `style:` line, is words where it stands.
SECTIONS = (
    "subject_definitions",
    "summary",
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
#: Reference clips: a video of any length, of which each segment reads its own window (`shot`); audio
#: past 15 s is cut with a warning. Every input that carries a reference declares these.
#: Below this a clip has too little to guide a segment, which then goes without it (`shot_windows`).
MIN_CLIP_SECONDS = 2
REFERENCE_AUDIO = AudioPreparation(max_seconds=15, excess="truncate")
#: Read bounds. A video streams, one decoded frame at a time, straight to its motion size
#: (`read_reference_video`), so a frame of any real source fits. A prepared image is at most
#: 16,777,216 RGB8 pixels; audio is 15 s, mono or stereo.
VIDEO_READ_BYTES = 16 << 30
IMAGE_READ_BYTES = 64 << 20
AUDIO_READ_BYTES = 64 << 20
#: Each kind's file field, its H3 reference kind and its `<Picture|Audio|Video N>` label.
FILE = {"character": "image", "audio": "audio", "video": "video", "anchor_video": "video", "world_state": "image"}
_SLOT = {"character": "picture", "audio": "audio", "video": "video", "anchor_video": "video", "world_state": "picture"}
_POLICY = {"character": "image", "audio": "audio", "video": "video", "anchor_video": "video", "world_state": "image"}
#: The kinds read from a clip.
CLIPS = ("video", "anchor_video")
_SPEAKER = re.compile(r"\s*\((S\d+)\)")
#: A picture's retention line when the writer gave it none. A clip's or a voice's look is not what it is there for.
_KEEP_LOOK = "keep the look shown in the reference."
#: A text's own retention line: its reference, then what it keeps.
_OWN_RETENTION = re.compile(r"[ \t*-]*<([^<>]+)>(?:[ \t]*\(appears in [^)]*\))?[ \t]*:(.+)", re.I)
_PREFIX = re.compile(r"\s*\[([^\]\n]*)\]")
#: The summary's task types (ref guide 3); a summary without one opens with the first.
TASKS = ("video editing", "video continuation", "keyframe completion", "reference generation",
         "audio reuse", "audio reference")


def name_key(name: str) -> str:
    """How a token matches a name: any case, with `_`, `-` and space alike."""
    return " ".join(re.sub(r"[_-]", " ", name).split()).casefold()


class StoryReference(msgspec.Struct, kw_only=True):
    """One reference file (`FILE[kind]`) a segment attaches, named as its text names it.

    - `character`: a picture, which is always a character, with its voice when it has one.
    - `audio`: a sound the soundtrack reuses.
    - `video`: reference footage, its motion and camera.
    - `anchor_video`: a plate, followed frame by frame (Alissonerdx's head swap and style
      transfer; ComfyUI's Add Guide): every latent frame at the render's own canvas, on every
      step, and the render takes its canvas (`guide_rows`). Never named in a prompt, one per call,
      no longer than the render. It costs what a second render does: per `reference_rows`, about
      5.5x the denoise for a 12 s clip beside a 15 s render.
    - `world_state`: a picture cozy adds at a join, the previous segment's last frame or a plate's
      first, read at 512 px.
    """

    name: str
    kind: Literal["character", "audio", "video", "anchor_video", "world_state"]
    description: str = ""
    image: ImageFile | None = None
    audio: AudioFile | None = None
    video: VideoFile | None = None
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


def _within_limits(
    used: Sequence[StoryReference], added: frozenset[str]
) -> tuple[list[StoryReference], list[StoryReference]]:
    """A call's references inside what H3 takes, and those left out: what cozy added itself first,
    then the last named. A plate stays, since it is the render's frames."""
    limits = {"image": MAX_IMAGE_REFERENCES, "video": MAX_VIDEO_REFERENCES, "audio": MAX_AUDIO_REFERENCES}
    kept = list(used)
    left: list[StoryReference] = []
    ours = [ref for ref in used if ref.name in added]
    others = [ref for ref in used if ref.name not in added and ref.kind != "anchor_video"]
    for ref in [*ours, *reversed(others)]:
        kind = _POLICY[ref.kind]
        if sum(_POLICY[kept_ref.kind] == kind for kept_ref in kept) > limits[kind] or len(kept) > MAX_REFERENCES:
            kept.remove(ref)
            left.append(ref)
    return kept, left


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


def _named(
    token: re.Match[str], by_key: Mapping[str, StoryReference], by_slot: Mapping[tuple[str, int], StoryReference]
) -> StoryReference | None:
    """The reference a token names: by its name first (any case, `_`, `-` and space alike), else as `<Picture N>`,
    `<Audio N>` or `<Video N>`."""
    named = by_key.get(name_key(token.group(1)))
    media = _MEDIA.fullmatch(name_key(token.group(1)))
    if named is not None or media is None:
        return named
    return by_slot.get((media.group(1).lower(), int(media.group(2))))


def segment_call(
    text: str,
    references: Sequence[StoryReference],
    *,
    soundscape: str,
    music: str,
    index: int,
    forced: frozenset[str],
    retained: Mapping[str, str] = {},
    added: frozenset[str] = frozenset(),
    keyframes: bool = False,
    continues: bool = False,
) -> SegmentCall:
    """A Ref2VA segment's text and references: `text` with its music and soundscape sections, whose mentions name
    references too, then what only cozy knows or a field set (`_segment_call`). `retained`: `own_retention` of the
    shot's text. `added`: the references cozy adds at a join. `keyframes`: it has a first or last frame. `continues`: it
    goes on from a take."""
    text = _fill(text, {"overall_soundscape": soundscape.strip() or "N/A", "non_diegetic_music": music.strip() or "N/A"})
    return _segment_call(text, references, "", index, forced, retained, added, keyframes, continues)


def _numbering(references: Sequence[StoryReference]) -> dict[tuple[str, int], StoryReference]:
    """`<Picture N>`, `<Audio N>` and `<Video N>`: each kind counted in the given order, as H3
    labels them."""
    counts = dict.fromkeys(("picture", "audio", "video"), 0)
    slots: dict[tuple[str, int], StoryReference] = {}
    for ref in references:
        counts[_SLOT[ref.kind]] += 1
        slots[_SLOT[ref.kind], counts[_SLOT[ref.kind]]] = ref
    return slots


def _segment_call(
    text: str,
    references: Sequence[StoryReference],
    soundscape: str,
    index: int,
    forced: frozenset[str] = frozenset(),
    author: Mapping[str, str] = {},
    added: frozenset[str] = frozenset(),
    keyframes: bool = False,
    continues: bool = False,
) -> SegmentCall:
    """`forced` always; every other reference only where the text names it, its summary included, all in
    request order, as H3 packs and labels them. With no references left, none: the segment renders from its
    text and keyframes (FL2VA), and says so if it had some to name.

    A token names a reference by its name first (any case, `_`, `-` and space alike), else as
    the global `<Picture N>`/`<Audio N>`/`<Video N>`; these are renumbered for the call. A
    character's voice follows the character. To the text as written go only what cozy alone
    knows or a field set: each name's `<Picture N>` in `subject_definitions` (its own line
    replaces that), each clip bound to the characters who perform it, the soundscape, and the `summary`'s task tag, computed from the request (ref guide 3), replacing a
    different one with a warning, in a section of its own when the text has none. Retention is the writer's: a reference's own line from
    `author` on a segment that lost it, one line for each reference cozy `added` at a join, and `_KEEP_LOOK` for a picture with no line.
    """
    pool, owners = voices(references)
    if not pool:
        return SegmentCall(text, [], [])
    by_key = {name_key(ref.name): ref for ref in pool}
    by_slot = _numbering(pool)

    def lookup(token: re.Match[str]) -> StoryReference | None:
        return _named(token, by_key, by_slot)

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
        [f"Segment {index + 1}: {', '.join(unmatched)} match no reference; left as written."]
        if unmatched else []
    )
    named = {ref.name for ref in mentioned(text)}
    named |= {voice.name for voice in pool if voice.name in owners and owners[voice.name].name in named}
    # An anchored clip packs after every other reference, so when it leaves the packed
    # references (`guide_rows`) every other label stays as H3 numbers it.
    used = sorted(
        (ref for ref in pool if ref.name in named | forced),
        key=lambda ref: ref.kind == "anchor_video",
    )
    used, left = _within_limits(used, added)
    if left:
        warnings.append(f"Segment {index + 1}: H3 takes at most {MAX_IMAGE_REFERENCES} pictures, {MAX_VIDEO_REFERENCES} "
                        f"videos, {MAX_AUDIO_REFERENCES} sounds, so it goes without "
                        f"{', '.join(f'<{ref.name}>' for ref in left)}.")  # fmt: skip
    if all(ref.kind == "audio" for ref in used):
        # H3's Ref2VA needs a picture: a segment that names none renders from its text and
        # keyframes (FL2VA), which takes no audio reference.
        warnings.append(f"Segment {index + 1} names only audio and no picture, so it renders without references."
                        if used else f"Segment {index + 1} names none of the shot's references, so it renders without them.")
        return SegmentCall(text, [], warnings)
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
    editing = any(ref.kind == "anchor_video" for ref in used)

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
    tasks = {"video editing"} if editing else set()
    tasks |= {"video continuation"} if continues else set()
    tasks |= {"keyframe completion"} if keyframes else set()
    tasks |= {"reference generation"} if any(ref.kind not in ("audio", "anchor_video") for ref in used) else set()
    for ref in used:
        slot = label(ref)
        defined = len(definitions)
        if ref.name in owners:
            owner = f"<{owners[ref.name].name}>"
            definitions.append(f"{slot} is the voice-timbre reference for {owner}{voiced[ref.name]}.")
            tasks.add("audio reference")
        elif ref.kind == "audio":
            definitions.append(f"{slot} is the supplied audio reference for <{ref.name}>.")
            tasks.add("audio reuse")
        elif ref.kind in CLIPS:
            source = editing or ref.kind == "anchor_video"
            role = "source video" if source else "motion and camera reference"
            who = f" for {names(motion[ref.name])}" if motion.get(ref.name) else ""
            definitions.append(f"{slot} is the {role}{who}." if _MEDIA.fullmatch(ref.name)
                               else f"<{ref.name}> is the {role} in {slot}{who}.")
        elif ref.kind == "world_state":
            # People are named only where the call has a character (se-089): a world state of a test pattern
            # that said "each person" grew people (25176).
            definitions.append(
                f"<{ref.name}> is the world state shown in {slot}: how the set is arranged and "
                + ("where each person and object stands." if characters else "where everything in it stands.")
            )
            kept = "each person's and object's position" if characters else "where everything in it stands"
            retention.append(f"<{ref.name}>: attribute_transfer - keep the set's arrangement and {kept} "
                             f"from {slot}, not its exact frame, framing or moment.")
        else:
            moves = [f"<{clip.name}>" if not _MEDIA.fullmatch(clip.name) else label(clip)
                     for clip in clips
                     if ref in motion.get(clip.name, ()) and (ref.name, clip.name) not in bound]
            binding = f", whose movements come from {' and '.join(moves)}" if moves else ""
            definitions.append(f"<{ref.name}> is the {ref.kind} shown in {slot}{binding}.")
        if (ref.name, _SLOT[ref.kind]) in own:
            del definitions[defined]
        elif ref.description:
            # A description ends its line with a blank one, however it was typed.
            definitions[defined] += f" {ref.description.strip()}\n"
        if (ref.name, _SLOT[ref.kind]) not in retained:
            if line := author.get(name_key(ref.name)):
                retention.append(f"<{ref.name}>: {line}")
            elif FILE[ref.kind] == "image" and ref.kind != "world_state":
                retention.append(f"<{ref.name}>: {_KEEP_LOOK}")
    tag = " + ".join(task for task in TASKS if task in tasks)
    given = _tasks(text)
    if given and {task.casefold() for task in given} != tasks:
        warnings.append(f"Segment {index + 1}: the summary's [{' + '.join(given)}] is not what its references are, so it "
                        f"goes as [{tag}]")  # fmt: skip
    sections = _sections(text)
    if "summary" in sections:
        _, start, end = sections["summary"]
        if prefix := _PREFIX.match(text, start, end) if given else None:
            text = text[: prefix.start(1)] + tag + text[prefix.end(1):]
        elif body := text[start:end].lstrip():
            text = text[: end - len(body)] + f"[{tag}] " + body + text[end:]
        else:
            text = text[:start] + f"\n[{tag}]" + text[start:]
    additions = {
        "subject_definitions": "\n".join(definitions),
        "summary": "" if "summary" in sections else f"[{tag}]",
        "retention_analysis": "\n".join(retention),
        "overall_soundscape": soundscape or "N/A",
    }
    return SegmentCall(_fill(text, additions), used, warnings)

def _tasks(text: str) -> list[str]:
    """The summary's `[task + task]` prefix, when it opens with one."""
    _, start, end = _sections(text).get("summary", (0, 0, 0))
    prefix = _PREFIX.match(text, start, end)
    given = [task.strip() for task in prefix.group(1).split("+")] if prefix else []
    return given if all(task.casefold() in TASKS for task in given) else []


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


def description_span(text: str) -> tuple[int, int]:
    """Where a text's description is (`_sections`): its body under its heading, or the text before
    the first header when none names one."""
    _, start, end = _sections(text).get("detailed_description", (0, 0, 0))
    return start, end


def own_retention(text: str) -> dict[str, str]:
    """What a text's own retention lines keep, by reference name (`name_key`), each without its `(appears in ...)`: a
    later segment has its own shots. Its first segment has these lines as written; `_segment_call` writes them on the others,
    only for the references that segment uses."""
    _, start, end = _sections(text).get("retention_analysis", (0, 0, 0))
    found = (_OWN_RETENTION.match(line) for line in text[start:end].splitlines())
    return {name_key(line.group(1)): line.group(2).strip() for line in found if line}


def without_sections(text: str, names: Sequence[str]) -> str:
    """`text` with the named sections, headers and all, taken out."""
    found = _sections(text)
    for start, end in sorted(((found[name][0], found[name][2]) for name in names if name in found), reverse=True):
        text = text[:start] + text[end:]
    return text


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
