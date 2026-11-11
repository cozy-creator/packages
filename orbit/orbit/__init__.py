"""Orbit: a picture's subject seen from all the way round. MiniMax H3 under MATLOWAI's ORB360 LoRA turns the camera
once, clockwise, around the frozen subject, from the caption ORB360 was trained on, starting and ending on the picture,
so it loops (README.md)."""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import Annotated, NamedTuple

import msgspec
import numpy as np
import numpy.typing as npt
from cozy_runtime.author import App, Context, Outputs, Seed, Telemetry, call
from cozy_runtime.fields import FileBound, Image, ImagePreparation, Placeholder, Prompt, Range
from cozy_runtime.media import (
    DecodedMediaEvent,
    DecodedMediaHeader,
    DecodedVideoFrame,
    ImageFile,
    References,
    VideoFile,
)
from h3 import PackageShot, ShotLoRA, render_shot
from minimax_h3.official import FPS, frames_for
from minimax_h3.shot import ANY_SIZE, MAX_DURATION_S, MIN_DURATION_S, Beat, Shot
from PIL import Image as PILImage

app = App()

#: MATLOWAI's ORB360 step 1500 (hf://MATLOWAI/MiniMax-H3-ORB360-CardSpin), trained on H3's Ref2VA DiT, clockwise only.
ORB360 = "matlowai/orb360@0.1.1"
#: What the video is, for its viewer: one clockwise turn whose end is its start.
TURN = {"degrees": 360, "direction": "cw", "loops": True}
#: The share of the turn the loop's join redraws, from about 330 degrees (se-114: a longer one reversed and stalled).
JOINED = 1 / 12


class Orbit(msgspec.Struct, kw_only=True):
    #: The view the turn starts and ends on: a picture put in this field.
    picture: Annotated[ImageFile, ImagePreparation(max_pixels=16 << 20)]
    #: What the picture shows, in a word or two: the caption names it.
    subject: Annotated[str, Placeholder("motorcycle")] = ""
    #: One turn's length. 15 s turns smoothest; 5 s ghosts drawn characters.
    duration: Annotated[int, Range(MIN_DURATION_S, MAX_DURATION_S, unit="s")] = MAX_DURATION_S
    seed: Seed | None = None


class OrbitResult(msgspec.Struct):
    #: One turn, frame 0 at 0 degrees, no sound; frame `n` would be frame 0 again, so it isn't kept.
    video: Annotated[VideoFile, FileBound(max_bytes=ANY_SIZE, media_types=("video/mp4",))]
    warnings: list[str] = msgspec.field(default_factory=list)


class Turned(NamedTuple):
    """A turn as its frames, each a video's frame by index, in order; and its join: the frame after the last, which
    would be frame 0 again, against frame 0 (PSNR on 96x54 grey, the demo's loopcut.py)."""

    frames: list[tuple[VideoFile, int]]
    seam_db: float
    warnings: list[str]


def named(subject: str) -> str:
    """The subject as a mention's name: no brackets, dots or line breaks."""
    return " ".join(re.sub(r"[<>.\n]", " ", subject).split()) or "subject"


def backdrop(picture: ImageFile) -> str:
    """`white` or `plain` when the picture's edges are one even color (a character drawn on white), else "": a real
    scene's caption on a plain backdrop grows snow and ground (demo run 46150)."""
    pixels = np.asarray(picture.read(max_pixels=1 << 16), dtype=np.int16)
    edge = max(2, round(min(pixels.shape[:2]) * 0.04))
    rim = np.concatenate([pixels[:edge].reshape(-1, 3), pixels[-edge:].reshape(-1, 3),
                          pixels[:, :edge].reshape(-1, 3), pixels[:, -edge:].reshape(-1, 3)])  # fmt: skip
    middle = np.median(rim, axis=0)
    if (np.abs(rim - middle).max(axis=1) < 16).mean() < 0.85:
        return ""
    return "white" if middle.min() > 200 else "plain"


def caption(subject: str, frames: int, plain: str) -> str:
    """ORB360's trained caption (its card's `orbit_realscene_caption.txt`, or the demo's plain-backdrop wording), its
    120-degree milestones on the clip's frames. H3 opens the summary with the task its references make."""
    s, last = f"<{subject}>", frames - 1
    if plain:
        backdrop_words = "plain white" if plain == "white" else "plain"
        where = f"against the {backdrop_words} background shown in <Picture 1>"
        kept = f"its {backdrop_words} background and light"
        scene = (f"The shot begins from <Picture 1>, with its {backdrop_words} background and studio light. The camera "
                 f"performs exactly one smooth clockwise 360-degree orbit around the completely stationary {s}, at constant "
                 "radius, constant elevation, constant focal length, constant speed, and constant framing, ending at the "
                 "starting viewpoint. The subject remains perfectly rigid and stationary. The background stays the same "
                 f"{'plain, empty white' if plain == 'white' else 'plain and empty'} all the way around, with no ground, "
                 "sky, landscape or weather. Everything is frozen in time: hair, clothing and loose parts do not move.")
    else:
        where = "in the real environment shown in <Picture 1>"
        kept = "its environment, weather and light"
        scene = ("The shot begins from <Picture 1>, with its exact environment, ground, background, weather and light. The "
                 f"camera performs exactly one smooth clockwise 360-degree orbit around the completely stationary {s}, at "
                 "constant radius, constant elevation, constant focal length, constant speed, and constant framing, ending "
                 "at the starting viewpoint. The subject remains perfectly rigid and stationary. The surrounding real scene "
                 "stays in place in the world and is revealed consistently as the camera circles, continuing the same "
                 "location, terrain and sky seen in <Picture 1>. Everything in the scene is frozen in time: falling snow, "
                 "rain, dust and particles hang motionless in mid-air; fur, hair, feathers, grass, leaves, water and clouds "
                 "do not move.")
    milestones = []
    for k in range(4):
        frame = last * k // 3
        view = "; the viewpoint corresponds to <Picture 1>" if k in (0, 3) else ""
        milestones.append(f"At {frame / FPS:.6f} seconds (zero-based frame {frame}), the camera has advanced {120 * k} "
                          f"degrees clockwise from its starting azimuth{view}.")  # fmt: skip
    return "\n".join([
        "subject_definitions:",
        f"{s} is the exact same rigid subject shown in <Picture 1>. Its geometry, proportions, materials, colors, "
        "markings, and surface details remain unchanged.",
        "<Picture 1> is the first-frame viewpoint of [Shot 1].",
        "",
        "summary:",
        f"ORB360_CW. A frozen-moment 360-degree orbit of {s} {where}, like a single instant stopped in time.",
        "",
        "retention_analysis:",
        f"{s} (appears in [Shot 1]): fully_preserved - its geometry, proportions, materials, colors, markings and surface "
        "details are retained, and it stays completely frozen.",
        f"<Picture 1> ([Shot 1] first frame and last frame): fully_preserved - the orbit starts and ends on this view, "
        f"including {kept}.",
        "",
        "detailed_description:",
        f"[Shot 1] {scene} No articulation, deformation, secondary motion, wind, zoom, dolly, camera roll, cuts, lighting "
        "changes, or scene changes.",
        "",
        "All four milestones below belong to the same uninterrupted [Shot 1], not separate shots. The video runs at 24 "
        "fps. Angles describe camera motion clockwise as viewed from above, relative to the starting camera azimuth; the "
        "subject itself does not rotate. The camera moves continuously at constant angular speed between these "
        "milestones, without pausing, reversing or completing extra turns.",
        *milestones,
        "",
        "The entire subject is a single frozen three-dimensional scene throughout the shot. Every visible eyelid keeps "
        "its reference position: no blinking. Each eye keeps a fixed gaze relative to the head and never tracks the "
        "moving camera. Facial expression, mouth, head, hands, feet and body pose remain unchanged. Hair strands, "
        "clothing, ribbons, accessories and loose parts are rigidly fixed in the same positions. There is no breathing, "
        "sway, settling or secondary motion. Only the camera changes position. Camera radius, elevation, focal length "
        "and look-at point stay fixed. Lighting stays fixed in world space; normal viewpoint-dependent shading and "
        "reflections remain physically consistent.",
        "",
        "overall_soundscape:",
        "N/A",
        "",
        "non_diegetic_music:",
        "N/A",
    ])  # fmt: skip


Grey = npt.NDArray[np.float32]


def grey(frame: DecodedVideoFrame) -> Grey:
    """A frame as loopcut.py compares it: 96x54 grey."""
    small = PILImage.frombytes("RGB", (frame.width, frame.height), frame.rgb).convert("L").resize((96, 54))
    return np.asarray(small, dtype=np.float32)


def psnr(a: Grey, b: Grey) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255**2 / mse)


def frames_of(video: VideoFile) -> Iterator[DecodedVideoFrame]:
    with video.stream(audio=False) as events:
        for event in events:
            if isinstance(event, DecodedVideoFrame):
                yield event


def frame_image(video: VideoFile, index: int, out: Outputs) -> ImageFile:
    """Frame `index` of `video`, saved as a picture a call can take."""
    decoded = video.read(start=Fraction(index, FPS), end=Fraction(index + 1, FPS), audio=False)
    return out.save_image(PILImage.frombytes("RGB", (decoded.width, decoded.height), decoded.frames_rgb[0]))


def orbit_shot(payload: Orbit, subject: str, text: str, *, pinned: bool) -> PackageShot:
    """The ORB360 shot on Ref2VA: the picture as `<subject>`, and with `pinned` as its first and last frame too."""
    picture = payload.picture
    beat = Beat(duration=payload.duration, description=Prompt(text=text, references=References({subject: Image(file=picture)})))
    ends = picture if pinned else None
    return PackageShot(shot=Shot(beats=[beat], first_frame=ends, last_frame=ends, seed=payload.seed),
                       loras=[ShotLoRA(model=ORB360)])  # fmt: skip


async def pinned(payload: Orbit, subject: str, text: str, out: Outputs, tel: Telemetry) -> Turned:
    """Ref2VA under ORB360 with the picture as its first and last frame: its last frame is the turn's end."""
    with tel.scope("Turn"):
        made = await call(render_shot, payload=orbit_shot(payload, subject, text, pinned=True))
    thumbs = [grey(frame) for frame in frames_of(made.video)]
    return Turned([(made.video, i) for i in range(len(thumbs) - 1)], psnr(thumbs[-1], thumbs[0]), made.warnings)


async def bridged(payload: Orbit, subject: str, text: str, out: Outputs, tel: Telemetry) -> Turned:
    """se-114's B: the Ref2VA turn under ORB360, cut where its tail comes closest to frame 0, its last twelfth redrawn
    by a 5 s FL2VA shot without the LoRA from there to frame 0, fitted into those frames."""
    with tel.scope("Turn"):
        made = await call(render_shot, payload=orbit_shot(payload, subject, text, pinned=False))
    out.publish("video", made.video, label="Turn, before its loop is joined")
    thumbs = [grey(frame) for frame in frames_of(made.video)]
    tail = range(max(1, len(thumbs) - 16), len(thumbs))
    turn = max(tail, key=lambda i: psnr(thumbs[0], thumbs[i]))
    start = turn - max(1, round(turn * JOINED))
    arc = round(360 * (turn - start) / turn)
    words = (f"[Shot 1] The camera continues a slow, smooth clockwise orbit around the completely stationary {subject}, "
             "moving from the first frame's viewpoint to the last frame's viewpoint: about "
             f"{arc} degrees of arc over the whole shot, at constant angular speed, constant radius, constant elevation and "
             f"constant focal length, without pausing or reversing. The {subject} and the whole scene stay frozen in time "
             "and perfectly rigid. No zoom, dolly, camera roll, cuts, lighting changes or scene changes.\n\n"
             "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A")  # fmt: skip
    beat = Beat(duration=MIN_DURATION_S, description=Prompt(text=words))
    shot = Shot(beats=[beat], first_frame=frame_image(made.video, start, out), last_frame=frame_image(made.video, 0, out),
                seed=payload.seed)  # fmt: skip
    with tel.scope("Loop join"):
        bridge = await call(render_shot, payload=PackageShot(shot=shot))
    joined = [grey(frame) for frame in frames_of(bridge.video)]
    span = turn - start
    picks = [(i * (len(joined) - 1) * 2 + span) // (2 * span) for i in range(span)]
    tel.log("orbit join", turn=turn, start=start, bridge_frames=len(joined), arc=arc)
    frames = [(made.video, i) for i in range(start)] + [(bridge.video, i) for i in picks]
    return Turned(frames, psnr(joined[-1], thumbs[0]), [*made.warnings, *bridge.warnings])


#: How a turn's end meets its start.
METHODS: dict[str, Callable[[Orbit, str, str, Outputs, Telemetry], Awaitable[Turned]]] = {"pinned": pinned, "bridged": bridged}
#: se-116's pod runs: `pinned` turned fully and closed at 36-39 dB on the motorcycle and swordswoman, in one call.
ENDS = "pinned"


def silent(frames: Sequence[tuple[VideoFile, int]]) -> Iterator[DecodedMediaEvent]:
    """The frames named, in order, as one stream without sound."""
    header = False
    i = 0
    while i < len(frames):
        video = frames[i][0]
        wanted: dict[int, int] = {}
        while i < len(frames) and frames[i][0] is video:
            wanted[frames[i][1]] = wanted.get(frames[i][1], 0) + 1
            i += 1
        with video.stream(audio=False) as events:
            index = -1
            for event in events:
                if isinstance(event, DecodedMediaHeader):
                    if not header:
                        header = True
                        yield replace(event, audio=None)
                elif isinstance(event, DecodedVideoFrame):
                    index += 1
                    for _ in range(wanted.get(index, 0)):
                        yield event
                    if index >= max(wanted):
                        break


@app.tool(label="Orbit", named="video", accelerator=False)
async def orbit(ctx: Context, payload: Orbit, out: Outputs, tel: Telemetry) -> OrbitResult:
    """The picture's subject seen from all the way round: one clockwise camera turn around it, frozen in time, that
    loops. H3 renders it under ORB360 from its trained caption, which names the subject and the angle every third of
    the turn, its end meeting its start (`ENDS`)."""
    ctx.raise_if_cancelled()
    subject = named(payload.subject)
    frames = frames_for(payload.duration)
    plain = backdrop(payload.picture)
    turned = await METHODS[ENDS](payload, subject, caption(subject, frames, plain), out, tel)
    with tel.stage("writing the turn"):
        saved = out.save_video_stream(silent(turned.frames))
    tel.log("orbit", method=ENDS, backdrop=plain or "scene", frames=saved.frame_count, seam_db=round(turned.seam_db, 2))
    return OrbitResult(saved.video, list(dict.fromkeys(turned.warnings)))
