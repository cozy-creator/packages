"""A MiniMax H3 shot, shared by `generate`, `edit` and `upscale` (`h3.py`): its fields, its text on the video's clock,
the references each segment reads, its segments, and the joins and carry between them."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from itertools import accumulate, pairwise
from typing import Annotated, Literal, NamedTuple

import msgspec
import numpy as np
import numpy.typing as npt
import torch
from cozy_runtime.author import InvalidRequest, OutputError, Outputs, Seed
from cozy_runtime.fields import (
    Advanced,
    AspectRatio,
    Audio,
    Count,
    FileBound,
    Image,
    Pickup,
    Placeholder,
    Prompt,
    Range,
    Ratios,
    Snap,
    Total,
    Video,
)
from cozy_runtime.media import (
    AudioFile,
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaEvent,
    DecodedMediaHeader,
    DecodedMediaStream,
    DecodedVideoFormat,
    DecodedVideoFrame,
    File,
    ImageFile,
    Latents,
    VideoFile,
)
from PIL import Image as PILImage
from safetensors.torch import load as load_tensors

from minimax_h3 import shot_latents
from minimax_h3.continuation import AUDIO_TICKS_PER_S, Carry, ContinuationPlan, audio_ticks, carries, decode_context
from minimax_h3.official import FPS, MAX_FRAMES, clip_size, shown_width, within_ratio
from story import _HEADER, MENTION, MIN_CLIP_SECONDS, StoryReference, description_span, name_key, without_sections

#: An output's or latents' byte bound: TensorD's output allowance reads it (256 MiB where none is declared), and the
#: disk is the real limit.
ANY_SIZE = 1 << 40
MIN_DURATION_S = 5
MAX_DURATION_S = 15
#: H3's own canvas, 16:9's.
NATIVE = (1344, 768)
#: Turbo's dense steps of eight, and the full model's and Regenerate's tiers (`TURBO_DENSE_STEPS`, `FULL_RECIPES`,
#: `regenerate.PRESETS`): draft is the fastest, full the closest to the model's reference.
Quality = Literal["draft", "standard", "full"]
#: The full model's step counts; turbo always runs its LoRA's 8.
FullSteps = Literal[30, 40, 50]
#: The frames each later segment carries from the one before (`continuation.CONTEXT_FRAMES`).
CarriedFrames = Literal[22, 39, 56]
ShotImage = Annotated[ImageFile, FileBound(max_decoded_bytes=256 << 20)]
#: Generated frames are square-pixel RGB at 24 fps, tagged BT.709 limited-range in the container.
VIDEO_COLOR = {"color_primaries": 1, "color_transfer": 1, "color_matrix": 1, "color_range": 1}
AUDIO_LAYOUTS = {1: ("mono", ("FC",)), 2: ("stereo", ("FL", "FR"))}


class Beat(msgspec.Struct, kw_only=True):
    """What happens, and for how long."""

    duration: Annotated[int, Range(1, MAX_DURATION_S, unit="s"), Total()] = MAX_DURATION_S
    description: Prompt[Image | Video | Audio] = Prompt()


class ShotVideo(Video, frozen=True, kw_only=True):
    """A take, and its latents when H3 made it and its machine still has them."""

    latents: Latents | None = None
    #: The take is one of the reading box's own versions.
    own: bool = False


class ShotPickup(Pickup, frozen=True, kw_only=True):
    """Pick up from here: the take, the frame the new part starts on (H3's `17k + 5` join grid, past its shortest
    piece: `shot_latents.nearest_join`), and the seconds into this shot's beats it starts at."""

    take: ShotVideo
    out_frame: Annotated[int, Snap(17, offset=5, first=124)]  # `official.MIN_FRAMES`, 5 s on the grid
    #: Unset: the cut's time on this box's own take, whose clock the beats share, else 0.
    beats_from: float | None = None


class Shot(msgspec.Struct, kw_only=True):
    """One continuous shot of any length, with sound: a summary every segment reads, then what happens, beat by beat.
    An empty soundscape or music goes as N/A."""

    summary: Prompt[Image | Video | Audio] = Prompt()
    beats: Annotated[list[Beat], Count(min=1)]
    soundscape: Annotated[Prompt[Image | Video | Audio], Placeholder("N/A (silent)")] = Prompt()
    music: Annotated[Prompt[Image | Video | Audio], Placeholder("N/A (silent)")] = Prompt()
    aspect_ratio: Annotated[AspectRatio, Ratios("16:9")] | None = None
    first_frame: ShotImage | None = None
    last_frame: ShotImage | None = None
    turbo: bool = True
    quality: Quality = "standard"
    #: The full model's steps: asking for them runs it.
    steps: Annotated[FullSteps | None, Advanced()] = None
    carried_frames: CarriedFrames = 22
    seed: Seed | None = None
    pickup: ShotPickup | None = None


class ShotResult(msgspec.Struct):
    video: Annotated[VideoFile, FileBound(max_bytes=ANY_SIZE, media_types=("video/mp4",))]
    warnings: list[str]
    #: The whole video's latents, which a pickup of it resumes and an upscale starts from.
    latents: Latents | None = None
    #: With a pickup: how many seconds of the take this keeps, where the new part starts.
    kept_s: float | None = None


#: A segment holds 15 s; a later one 14 s new, beside what it carries.
_CARRY = Carry(22, True)
#: A kept video reaches its shot's end if less than this is left.
_REACHES_S = 1.0
#: A field that ends a text's description (MiniMax's FL2VA lines, Ref2VA's sections), and a description's own heading.
_AFTER = re.compile(r"^[ \t]*(?:audio|overall[ _-]*soundscape|non[ _-]*diegetic[ _-]*music)[ \t]*:", re.I | re.M)
_DESCRIPTION = re.compile(r"^[ \t]*(?:integrated[ _-]*multimodal|detailed)[ _-]*description[ \t]*:[ \t]*\n?", re.I | re.M)
#: A timestamp: at a line's start, after an optional `[Shot N]`; a time mid-sentence is plain text. It says when, and the
#: model reads the words for what: cozy never takes one for a cut.
_TIMED = re.compile(r"^[ \t]*(\[Shot[ \t]*\d+\][ \t]*)?At[ \t]+(\d+):(\d+(?:\.\d+)?)[ \t]*,?[ \t]*", re.I | re.M)
_OWN_SHOT = re.compile(r"\[Shot[ \t]*\d+\]", re.I)
FIRST_SHOT = re.compile(r"\[Shot[ \t]*1\]", re.I)
#: A join lands on H3's 17-frame grid, so a picked-up video can end just short of the timestamp the next call opens on:
#: within this, that block opens it, and the earlier ones stay history.
JOIN_S = 17 / FPS
_SOUNDSCAPE = re.compile(r"^[ \t]*(?:audio|overall[ _-]*soundscape)[ \t]*:", re.I | re.M)
_REF2VA_HEADING = re.compile(r"^[ \t]*(?:summary|detailed[ _-]*description)[ \t]*:[ \t]*\n?", re.I | re.M)
#: A line of a person's text the planner would read as ours: a timestamp or a section. Their line breaks reach the model.
_OURS = re.compile(rf"(?:{_TIMED.pattern})|(?:{_AFTER.pattern})|(?:{_HEADER.pattern})", re.I | re.M)
#: Names cozy gives what it adds to a segment.
_ADDED = frozenset({"plate", "plate opening"})

Mentioned = Image | Video | Audio


class Said(NamedTuple):
    """One of a shot's texts as written, its references, and the time it covers on the video's clock: a beat's, or
    the whole shot's."""

    text: str
    references: Mapping[str, Mentioned]
    start: float = 0.0
    end: float = math.inf


class Script(NamedTuple):
    """What `written` makes of a shot: MiniMax's text on the video's clock, where its last beat ends, its texts, each
    mention's name in the text, the references in H3's order, and each clip's window."""

    prompt: str
    duration_s: float
    said: list[Said]
    names: dict[str, str]
    order: list[str]
    clips: dict[str, tuple[VideoFile, float, float]]
    people: bool
    soundscape: str
    music: str
    warnings: list[str]


def _spoken(key: str) -> str:
    """A mention's name as MiniMax reads it: `<kyogre.2>` as `kyogre 2`, a clip's range left out."""
    return " ".join(key.partition("@")[0].replace(".", " ").split())


def h3_reference(name: str, value: Mentioned, voice: AudioFile | None = None) -> StoryReference:
    """A mention as H3 reads it: a picture is a character, with its `voice` when something binds one; a clip is motion
    and camera; a sound is a voice-timbre reference, whose speaker the text names (Paul, 11-06: "it's brought in as a
    voice-reference")."""
    if isinstance(value, Image):
        return StoryReference(name=name, kind="character", image=value.file, description=value.description, voice=voice)
    if isinstance(value, Video):
        return StoryReference(name=name, kind="video", video=value.file)
    return StoryReference(name=name, kind="voice", audio=value.file)


def here(script: Script, start: float, end: float) -> list[StoryReference]:
    """A segment's references, from `start` to `end` on the video's clock: those of the whole shot's texts and of the
    beats its time covers, in H3's order."""
    keys: dict[str, Mentioned] = {}
    for said in script.said:
        if said.start < end and said.end > start:
            keys |= said.references
    return [h3_reference(script.names[key], keys[key]) for key in script.order if key in keys]


def _named(texts: Sequence[Said]) -> tuple[list[str], dict[str, str]]:
    """Every key in H3's order (first mention in the beats, then music, soundscape and summary, as Calcifer numbered
    them before) and its name in the text, one per key."""
    order = list(dict.fromkeys(key for said in texts for key in said.references))
    taken = set(_ADDED)
    names: dict[str, str] = {}
    for key in order:
        name, number = _spoken(key), 1
        while name_key(name) in taken:
            number += 1
            name = f"{_spoken(key)} {number}"
        taken.add(name_key(name))
        names[key] = name
    return order, names


def _in_names(said: Said, names: Mapping[str, str]) -> str:
    """A text with each mention as its name."""
    return MENTION.sub(lambda found: f"<{names[found[1]]}>" if found[1] in said.references else found[0], said.text)


def kept(text: str) -> str:
    """A person's text as it goes into our form: each line as written, but one that would read as ours doesn't start
    its line, so it can't set a time or end the description."""
    return _OURS.sub(lambda line: f"… {line.group(0).lstrip()}", text.strip())


def clock(seconds: float) -> str:
    """H3's `MM:SS.mmm`."""
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:06.3f}"


def written(shot: Shot, *, offset: float = 0.0, length: float | None = None) -> Script:
    """MiniMax's text for a shot: its summary and its beats on one timeline, the video's, `offset` seconds behind the
    beats' own (a pickup's `beats_from` less the cut). `length` is a plate's: the beats are cut at it, or the last runs
    on to it. One beat with no summary goes as written."""
    warned: list[str] = []
    times = list(pairwise(accumulate((float(beat.duration) for beat in shot.beats), initial=0.0)))
    if length is not None:
        last = next((index for index, (start, _) in enumerate(times) if start >= length), len(times))
        if last < len(times):
            warned.append(f"the plate is {length:.2f} s, so beat {last + 1}{' and after' if last + 1 < len(times) else ''} "
                          "isn't rendered")  # fmt: skip
        times = times[:last]
        end = times[-1][1]
        if end > length:
            warned.append(f"beat {len(times)} is cut at the plate's end, {length:.2f} s")
        elif end < length:
            warned.append(f"the beats run {end:g} s and the plate {length:.2f} s: beat {len(times)} runs on to its end")
        times[-1] = (times[-1][0], length)
    beats = [(beat, max(0.0, start - offset), end - offset) for beat, (start, end) in zip(shot.beats, times) if end - offset > 0]
    said = [Said(beat.description.text, beat.description.references, start, end) for beat, start, end in beats]
    whole = [Said(prompt.text, prompt.references) for prompt in (shot.music, shot.soundscape, shot.summary)]
    order, names = _named([*said, *whole])
    music, soundscape, summary = (_in_names(text, names) for text in whole)
    told: list[str] = []
    for text, (_, start, _) in zip(said, beats, strict=True):
        # A beat with no words yet is only a length: a box opens with one.
        if said_here := kept(_in_names(text, names)):
            told.append(f"{'' if told else '[Shot 1] '}{f'At {clock(start)}, ' if start else ''}{said_here}")
    summary = kept(summary)
    if len(shot.beats) == 1 and not summary:
        prompt = _in_names(said[0], names) if said else ""
    elif told:
        prompt = "\n\n".join(filter(None, (summary and f"summary:\n{summary}", "detailed_description:\n" + "\n".join(told))))
    else:
        prompt = f"[Shot 1] {summary}"
    values = {key: value for text in [*said, *whole] for key, value in text.references.items()}
    clips: dict[str, tuple[VideoFile, float, float]] = {}
    for key, value in values.items():
        if isinstance(value, Video):
            # A clip lines up with the first text that mentions it: the shot's start for a whole-shot text, else its
            # beat's, and it's cut at that beat's end.
            first = min((text for text in [*said, *whole] if key in text.references), key=lambda text: text.start)
            clips[names[key]] = (value.file, first.start, first.end)
    return Script(prompt, beats[-1][2] if beats else 0.0, [*said, *whole], names, order, clips,
                  any(isinstance(value, Image) for value in values.values()), soundscape, music, warned)  # fmt: skip


def _delivered(seconds: int, first: bool, carry: Carry) -> int:
    """The frames a segment of `seconds` new seconds delivers on H3's 17n+5 grid."""
    return ContinuationPlan(seconds, Carry() if first else carry).delivered_frames


class ShotPlan(NamedTuple):
    """A shot's segments: each one's whole seconds and the time on the shot's clock it opens on."""

    seconds: list[int]
    opens: list[float]
    warnings: list[str]


def plan_segments(duration_s: float, marks: Iterable[float], carried: bool = False, carry: Carry = _CARRY,
                  split: bool = False) -> ShotPlan:  # fmt: skip
    """A shot's segments: whole blocks of its text (from one of `marks`, its timestamps, to the next), as many as fit
    each segment, so the fewest, and each later segment opens on a timestamp. A segment holds 15 s, or 14 when it
    continues a video: `carried`, or any but the first. A block longer than that renders that long and warns, and what
    follows comes straight after it; with `split` (an edit, whose plate the text holds throughout) it's split across
    segments instead. Lengths are H3's whole seconds on its 17n+5 grid, each the nearest to the next timestamp from what
    is delivered so far, and at least 5 s; the last reaches the end."""
    fresh = not carried
    # A later segment's new seconds: its carried frames share the 362-frame window.
    most_next = (MAX_FRAMES - carry.head_frames) // FPS

    def limit(index: int) -> int:
        return _delivered(MAX_DURATION_S, True, carry) if index == 0 and fresh else _delivered(most_next, False, carry)

    edges = sorted({mark for mark in marks if 0 < mark < duration_s})
    groups: list[list[float]] = []
    for start, end in pairwise([0.0, *edges, duration_s]):
        if groups and math.ceil((end - groups[-1][0]) * FPS - 1e-6) <= limit(len(groups) - 1):
            groups[-1][1] = end
            continue
        while split and math.ceil((end - start) * FPS - 1e-6) > limit(len(groups)):
            # Whole segments of it, leaving at least the shortest piece for the rest.
            most = MAX_DURATION_S if len(groups) == 0 and fresh else most_next
            take = min(_delivered(most, not groups and fresh, carry), math.floor((end - start) * FPS) - _delivered(MIN_DURATION_S, False, carry))
            groups.append([start, start + take / FPS])
            start += take / FPS
        groups.append([start, end])
    seconds: list[int] = []
    warnings: list[str] = []
    past = 0  # the frames delivered beyond the last timestamp reached
    for index, (start, end) in enumerate(groups):
        first = index == 0 and fresh
        most = MAX_DURATION_S if first else most_next
        asked = math.ceil((end - start) * FPS - 1e-6)
        wanted = min(asked - past, _delivered(most, first, carry))
        lengths = range(MIN_DURATION_S, most + 1)
        if asked > _delivered(most, first, carry):
            length, past = most, 0
            held = f"{_delivered(most, first, carry) / FPS:.1f} s, the most one segment holds"
            warnings.append(f"the text at {clock(start)} runs {end - start:g} s and renders {held}: update its timing. "
                            "What follows comes straight after it, so the video is shorter than its beats.")  # fmt: skip
        elif index == len(groups) - 1:
            length = next((n for n in lengths if _delivered(n, first, carry) >= wanted), most)
            if _delivered(length, first, carry) - wanted > FPS:
                warnings.append(f"a segment renders at least {MIN_DURATION_S} s, so this runs "
                                f"{(_delivered(length, first, carry) - wanted) / FPS:.1f} s longer than asked.")  # fmt: skip
        else:
            length = min(lengths, key=lambda n: (abs(_delivered(n, first, carry) - wanted), n))
            past = _delivered(length, first, carry) - wanted
            if past > FPS:
                warnings.append(f"a segment renders at least {MIN_DURATION_S} s, so the text at {clock(end)} starts about "
                                f"{past / FPS:.1f} s late: update its timing.")  # fmt: skip
        seconds.append(length)
    return ShotPlan(seconds, [start for start, _ in groups], warnings)


def prompt_parts(prompt: str) -> tuple[str, str, str]:
    """A prompt's heading (T2VA's), its body, and the audio and music fields that hold on every segment."""
    prompt = prompt.strip()
    head = found.group(0) if (found := _DESCRIPTION.match(prompt)) else ""
    after = _AFTER.search(prompt, len(head))
    return head, prompt[len(head) : after.start() if after else None].strip(), prompt[after.start() :].strip() if after else ""


class Block(NamedTuple):
    """A timed block of a text's description: its time, whether its line marks a shot, and where its line and its
    words start. The text before the first is the opening block."""

    at: float
    marked: bool
    line: int
    words: int


def blocks(body: str, duration_s: float) -> tuple[list[Block], list[str]]:
    """The timed blocks of a text's description. A time out of order or past the end stays plain text, with a warning."""
    found_blocks: list[Block] = []
    warnings: list[str] = []
    last = 0.0
    for found in _TIMED.finditer(body, *description_span(body)):
        at = int(found[2]) * 60 + float(found[3])
        if at > last and at < duration_s:
            found_blocks.append(Block(at, bool(found[1]), found.start(), found.end()))
            last = at
        elif at > 0:
            why = "out of order" if at <= last else f"past the end of this {duration_s:g} s video"
            warnings.append(f"the time {clock(at)} is {why}: it stays plain text. Update its timing.")
    return found_blocks, warnings


def segment_texts(prompt: str, starts: Sequence[float], *, continued: bool = False, carried: bool = True,
                  duration_s: float = math.inf, carry: Carry = _CARRY) -> list[str]:  # fmt: skip
    """Each segment's text: the prompt's, with nothing added, and only its own blocks (`plan_segments` and `starts`:
    what opens each), so no segment redraws what an earlier one set up (0.2.4 gate, runs 24980/24981: back in the food
    cart). The first is as written, past 15 s without what comes after it. A later one keeps the text's sections, its
    summary among them, less the retention, which tells the whole video's flow (`story` restores each reference's own
    retention line): the block it opens on, with no time, then the rest with their times on its clock, whose head is the
    carried frames, and its `[Shot N]` renumbered from 1. With `continued`, the first segment goes on from a take too, on
    the block in force at its start, carrying the take's frames unless not `carried`. The audio and music fields close
    every segment."""
    head, body, tail = prompt_parts(prompt)
    timed_blocks, _ = blocks(body, duration_s)
    begin, stop = description_span(body)
    ends = [*(block.line for block in timed_blocks[1:]), stop][: len(timed_blocks)]
    timed = [(0.0, False, body[begin : timed_blocks[0].line if timed_blocks else stop].strip()),
             *((block.at, block.marked, body[block.words : end].strip()) for block, end in zip(timed_blocks, ends, strict=True))]  # fmt: skip
    texts = []
    for index, (start, end) in enumerate(zip(starts, [*starts[1:], math.inf], strict=True)):
        if not index and not continued:
            cut = next((block.line for block in timed_blocks if block.at >= end), None)
            text = body if cut is None else (body[:cut].rstrip() + "\n" + body[stop:].lstrip()).rstrip()
        else:
            opens = max(at for at, _, _ in timed if at <= start + (JOIN_S if not index else 0))
            lines = []
            for at, marked, said in timed:
                if at == opens or start < at < end:
                    lead = carry.head_frames / FPS if index or carried else 0.0
                    time = f"At {clock(at - start + lead)}, " if at > max(opens, start) else ""
                    lines.append(f"{'[Shot 1] ' if marked else ''}{time}{said if time else said[:1].upper() + said[1:]}")
            number = iter(range(1, len(body) + 2))
            description = _OWN_SHOT.sub(lambda _: f"[Shot {next(number)}]", "\n".join(line for line in lines if line))
            text = "\n\n".join(filter(None, (body[:begin].rstrip() + ("\n" if begin else "") + description, body[stop:].strip())))
            text = without_sections(text, ("retention_analysis",))
        text = head + text
        texts.append(f"{text}\n\n{tail}" if tail else text)
    return texts


#: Where a segment reads a clip: the clip's frame its window starts at, and the clip's frame it stops at.
ClipWindow = tuple[int, int]


def shot_windows(clips: Sequence[tuple[StoryReference, float, float]], spans: Sequence[tuple[int, int]]
                 ) -> tuple[list[dict[str, ClipWindow]], list[str]]:  # fmt: skip
    """Where each segment's window of each clip starts and stops, and the warnings. A clip lines up with the video's
    clock from its own start (`clips`: each clip, and the seconds its text starts and ends at), second for second, and
    is cut at its text's end. A segment reads the video frames it samples, its carried head included (`spans`: first
    frame, count), from the clip's frame there, so a plate stays frame-aligned across an edge. A segment left with less
    than `MIN_CLIP_SECONDS` of a clip goes without it."""
    least = MIN_CLIP_SECONDS * FPS
    windows: list[dict[str, ClipWindow]] = [{} for _ in spans]
    total = spans[-1][0] + spans[-1][1]
    warnings = []
    for ref, start_s, end_s in clips:
        assert ref.video is not None
        probe = ref.video.probe()
        length = math.floor(probe.duration * FPS + Fraction(1, 2))
        offset = round(start_s * FPS)
        stop = length if math.isinf(end_s) else min(length, round(end_s * FPS) - offset)
        if abs(probe.pixel_aspect_ratio - 1) > Fraction(1, 50):
            warnings.append(f"<{ref.name}> has {probe.pixel_aspect_ratio} pixels; it's read at its display shape")
        if within_ratio(shown := shown_width(probe.width, probe.pixel_aspect_ratio), probe.height) != (shown, probe.height):
            warnings.append(f"<{ref.name}> is past 4:1; its middle is used")
        for window, (start, frames) in zip(windows, spans, strict=True):
            if min(start + frames, offset + stop) - max(start, offset) >= least:
                window[ref.name] = (max(0, start - offset), stop)
        if ref.kind == "video" and (span := sum(ref.name in window for window in windows)) > 1:
            warnings.append(f"<{ref.name}> guides {span} segments, each from its own span: put each timestamp where the clip "
                            "shows it, or the shot runs ahead of the clip")  # fmt: skip
        if ref.kind == "anchor_video":
            continue  # the plate is the shot's length, give or take H3's grid
        room = (total if math.isinf(end_s) else round(end_s * FPS)) - offset
        if length < room:
            warnings.append(f"<{ref.name}> is {length / FPS:.1f} s and its text {room / FPS:.1f} s, which runs on without it")
        elif length > room:
            warnings.append(f"<{ref.name}> is {length / FPS:.1f} s long; its text uses its first {room / FPS:.1f} s")
    return windows, warnings


def fl2va_text(text: str, soundscape: str, music: str) -> str:
    """An FL2VA segment's text: its own, with no `summary:` or `detailed_description:` heading (Ref2VA's form, not
    FL2VA's), with the soundscape as its `audio` line and the music as its `non_diegetic_music` line where it has none."""
    text = _REF2VA_HEADING.sub("", text).strip()
    lines = []
    if not _SOUNDSCAPE.search(text):
        lines.append(f"audio: {soundscape.strip() or 'N/A'}")
    if "non_diegetic_music" not in text.casefold():
        lines.append(f"non_diegetic_music: {music.strip() or 'N/A'}")
    return "\n\n".join([text.rstrip(), *lines])


def segment_seed(seed: int | None, request_id: str, index: int) -> int:
    """The shot's seed, one more per segment; or chosen once per request, so a retried attempt keeps it."""
    if seed is not None:
        return seed + index
    return int.from_bytes(hashlib.sha256(f"{request_id}/h3/shot/{index}".encode()).digest()[:4], "big")


def is_native(width: int, height: int) -> bool:
    """Whether a shape is about H3's own 1344x768."""
    return abs(Fraction(width, height) - Fraction(*NATIVE)) <= Fraction(1, 20)


def canvas(frame: tuple[int, int], joined: bool) -> tuple[tuple[int, int], list[str]]:
    """The (width, height) a shot opening or ending on a frame of this shape renders at: the frame's shape within 4:1,
    on H3's canvas. A shot of several segments (`joined`) carries its canvas across each join: a shape it can't carry
    goes toward 16:9 until one carries, with a warning (GPU 11)."""
    width, height = within_ratio(*frame)
    native = math.log(NATIVE[0] / NATIVE[1])
    for step in range(33):
        ratio = math.exp(math.log(width / height) * (1 - step / 32) + native * step / 32)
        shape = (round(height * ratio), height) if ratio < width / height else (width, round(width / ratio))
        tall, wide = clip_size(*shape, guide=True)
        if not joined or carries(tall, wide):
            moved = [] if not step else [f"a {frame[0]}x{frame[1]} frame can't carry from one segment to the next, so the "
                                         f"shot is {wide}x{tall}"]  # fmt: skip
            return (wide, tall), moved
    return NATIVE, []


def fitted(name: str, size: tuple[int, int], shape: tuple[int, int]) -> list[str]:
    """A warning when a first or last frame of `size` is cropped to the shot's `shape`."""
    if abs(Fraction(*size) - Fraction(*shape)) <= Fraction(1, 100):
        return []
    return [f"{name} is {size[0]}x{size[1]}: its middle is used for the shot's {shape[0]}x{shape[1]}"]


def frame_at(video: VideoFile, frames: int, out: Outputs) -> ImageFile:
    """A video's last frame before `frames`: the one a new part starts from."""
    decoded = video.read(start=Fraction(max(frames - 6, 0), FPS), end=Fraction(frames, FPS), audio=False)
    return out.save_image(PILImage.frombytes("RGB", (decoded.width, decoded.height), decoded.frames_rgb[-1]))


def unusable(take: ShotVideo) -> str:
    """Why a pickup can't resume from its take's latents, as `regenerate_shot` reads them; empty when it can. H3's own
    videos have sound, so one without it isn't the video they were made with."""
    if take.file.probe().soundtrack is None:
        return "its video has no sound, so it isn't the video they were made with"
    assert take.latents is not None
    try:
        data = take.latents.read_bytes()
        shot_latents.delivered(data)
        decode_context(shot_latents.context(data))
    except InvalidRequest as refused:
        return refused.message
    except (KeyError, ValueError, RuntimeError):  # bytes no safetensors reader takes
        return "it is not a minimax-h3 latents file"
    return ""


def sound_chunks(stream: DecodedMediaStream) -> list[npt.NDArray[np.float32]]:
    """A decoded video's sound, `(channels, samples)` per chunk, none of its pictures."""
    with stream as events:
        return [np.stack([np.frombuffer(pcm, dtype=np.float32) for pcm in event.pcm_f32le])
                for event in events if isinstance(event, DecodedAudioChunk)]  # fmt: skip


def cut_latents(take: ShotVideo, frames: int, out: Outputs, carry: Carry) -> File:
    """A take's latents as a shot of `frames` (a join of the grid, so what follows is in phase), with the carry there and
    the sound that stays."""
    assert take.latents is not None
    data = take.latents.read_bytes()
    parts = sound_chunks(take.file.stream(end=Fraction(audio_ticks(frames), AUDIO_TICKS_PER_S), pictures=False))
    # A soundtrack that starts past the join has nothing to keep yet: `cut` pads it with silence.
    sound = np.concatenate(parts, axis=1) if parts else np.zeros((2, 0), dtype=np.float32)
    return out.save_bytes(shot_latents.cut(data, frames, carry.frames, torch.from_numpy(sound)), media_type="application/octet-stream")


def shot_latents_file(out: Outputs, parts: Sequence[File], frames: int, context: File, text: str) -> File:
    """The whole video's latents: each part's delivered latents end to end (each segment cuts its carried head, whole
    VAE cycles, so they join in phase; a picked-up take's own latents first), in the latent space they name, with its
    last segment's carry and the text it was made from."""
    blobs = [part.read_bytes() for part in parts]
    spaces = {shot_latents.metadata(data).get("vae_space", "") for data in blobs}
    if len(spaces) != 1 or "" in spaces:
        raise OutputError("the video's segments name no single latent space", code="output_integrity")
    latents = torch.cat([load_tensors(data)["latents"] for data in blobs], dim=2)
    return out.save_bytes(shot_latents.save(latents, frames, spaces.pop(), context.read_bytes(), text),
                          media_type="application/octet-stream")  # fmt: skip


def upscale_prompt(script: Script) -> str:
    """The text a video was made from, for an upscale that re-renders it (Paul, 10-31): the whole video as one segment's
    text (`segment_texts`) after each picture's description, closing with the soundscape and music as its segments do
    (`fl2va_text`). Names lose their brackets: the re-render attaches no reference."""
    pictures = {script.names[key]: value for said in script.said for key, value in said.references.items() if isinstance(value, Image)}
    lines = " ".join(f"{name}: {value.description.strip().rstrip('.')}." for name, value in pictures.items() if value.description.strip())
    text = segment_texts(script.prompt, [0.0])[0]
    if lines:
        text = text.replace("[Shot 1] ", f"{lines} [Shot 1] ", 1) if "[Shot 1] " in text else f"{lines}\n{text}"
    return re.sub(r"<([^<>]+)>", r"\1", fl2va_text(text, script.soundscape, script.music))


def _kept_frames(take: VideoFile, frames: int, size: tuple[int, int]) -> Iterator[DecodedMediaEvent]:
    """A take's first `frames` on the 24-fps clock, each cover-scaled and centre-cropped to `size`, silent."""
    yield DecodedMediaHeader(
        video=DecodedVideoFormat(width=size[0], height=size[1], time_base=Fraction(1, FPS), pixel_aspect_ratio=Fraction(1),
                                 nominal_frame_rate=Fraction(FPS), **VIDEO_COLOR),
        audio=None,
    )  # fmt: skip
    count, origin, last = 0, None, b""
    with take.stream(end=Fraction(frames, FPS), audio=False) as events:
        for event in events:
            if not isinstance(event, DecodedVideoFrame):
                continue
            origin = event.start_time if origin is None else origin
            picture = PILImage.frombytes("RGB", (event.width, event.height), event.rgb)
            last = _cover(picture, size)
            # Onto the 24-fps clock: a faster source's extra frames go, a slower one's hold.
            for index in range(count, min(frames, round((event.start_time - origin) * FPS) + 1)):
                yield _frame_event(last, index, size)
                count = index + 1
    for index in range(count, frames):
        yield _frame_event(last, index, size)


def _cover(picture: PILImage.Image, size: tuple[int, int]) -> bytes:
    scale = max(size[0] / picture.width, size[1] / picture.height)
    scaled = picture.resize((max(size[0], round(picture.width * scale)), max(size[1], round(picture.height * scale))),
                            PILImage.Resampling.BICUBIC)  # fmt: skip
    left, top = (scaled.width - size[0]) // 2, (scaled.height - size[1]) // 2
    return scaled.crop((left, top, left + size[0], top + size[1])).tobytes()


def _frame_event(rgb: bytes, index: int, size: tuple[int, int]) -> DecodedVideoFrame:
    return DecodedVideoFrame(width=size[0], height=size[1], rgb=rgb, pts=index, duration=1, time_base=Fraction(1, FPS),
                             pixel_aspect_ratio=Fraction(1), **VIDEO_COLOR)  # fmt: skip


def join_kept(out: Outputs, take: VideoFile, frames: int, new: VideoFile) -> tuple[VideoFile, list[str]]:
    """A take's first `frames`, then `new`, as one video, when no latents carry the take into it: the kept start is
    encoded at the new part's size and joined by packet copy (`save_video_concat`), under one soundtrack, the take's to
    the cut, then the new part's."""
    probe = new.probe()
    sound = probe.soundtrack
    warnings = [] if (take.probe().width, take.probe().height) == (probe.width, probe.height) else [
        f"the kept start is {take.probe().width}x{take.probe().height}, so it's scaled to the new part's {probe.width}x{probe.height}"
    ]  # fmt: skip
    start = out.save_video_stream(_kept_frames(take, frames, (probe.width, probe.height))).video
    if sound is None:
        raise OutputError("the new part has no soundtrack to join under", code="output_integrity")
    kept_format = take.probe().soundtrack
    samples = frames * sound.sample_rate // FPS
    if kept_format is not None and (kept_format.channels, kept_format.sample_rate) == (sound.channels, sound.sample_rate):
        parts = sound_chunks(take.stream(end=Fraction(frames, FPS), pictures=False))
        before = np.concatenate(parts, axis=1)[:, :samples] if parts else np.zeros((sound.channels, 0), dtype=np.float32)
    else:
        before = np.zeros((sound.channels, 0), dtype=np.float32)
        if kept_format is not None:
            warnings.append("the kept start's sound isn't the new part's format, so it's silent")
    before = np.pad(before, ((0, 0), (0, samples - before.shape[1])))
    after = sound_chunks(new.stream(pictures=False))
    waveform = np.concatenate([before, *after], axis=1).astype(np.float32)

    def soundtrack(counts: Sequence[int]) -> Iterator[DecodedMediaEvent]:
        layout, names = AUDIO_LAYOUTS[sound.channels]
        yield DecodedMediaHeader(video=None, audio=DecodedAudioFormat(
            channels=sound.channels, sample_rate=sound.sample_rate, channel_layout=layout, channel_names=names,
            time_base=Fraction(1, sound.sample_rate)))  # fmt: skip
        yield DecodedAudioChunk(channels=sound.channels, sample_count=int(waveform.shape[1]), sample_rate=sound.sample_rate,
                                channel_layout=layout, channel_names=names,
                                pcm_f32le=tuple(np.ascontiguousarray(channel).tobytes() for channel in waveform),
                                pts=0, time_base=Fraction(1, sound.sample_rate))  # fmt: skip

    return out.save_video_concat([start, new], soundtrack).video, warnings
