# Segments

`generate` renders one or more segments and joins them. A segment is one H3 call: free-text
`prompt`, `duration_s` (5–15, required) and optional `seed`, the booleans `context_frames`
(carry the previous picture) and `context_audio` (carry the previous sound), both defaulting to
`true`, its `first_frame`/`last_frame` (`frame` references by name), and its own `turbo` and
`steps`. The model only ever sees one segment's prompt, so the workflow copies the global
information into it.

A segment renders on FL2VA (the `transformer`) from its text and keyframes when it uses no
reference and continues nothing. It renders on Ref2VA (`transformer_ref`) when it names a
reference, has a background, or continues the previous segment; a continuation that names no
reference takes the previous segment's last frame as a soft picture of the set. On Ref2VA a
segment's first and last frames are anchored as ComfyUI's Add Guide anchors them.

## Request

- `references` (optional): characters, backgrounds, audio, videos, anchor videos and frames,
  shared by every segment. A character, audio or video goes into the segments whose prompt
  names it; a background into every segment; frames and anchor videos need no name.
- `clean_references` (default `true`): BiRefNet removes the background of each supplied
  character image once, and H3 receives it composited onto white. The cleaned image
  appears in the reference outputs and is reused for every segment; `false` sends it as
  supplied. Supplied backgrounds, audio and videos are used as supplied. References generated
  from descriptions keep their existing path.
- `describe_references` (default `true`): a vision-language model (Qwen3.5, `qwen3.5/ask`)
  writes one sentence about each supplied character or background image that has no
  `description` (a character's identity, hair, face, clothing and items; a place's look,
  people left out), and watches each clip as video for its people, beats, camera and cuts.
  It runs once, alongside background removal; the sentences join the subject definitions.
  Each segment with a clip then gains, from one writing pass over all of them, what its
  text lacks of MiniMax's guide format: each character's role in the clip, a 350-500-word
  description establishing the characters, the background and the clip's beats in order,
  and a soundscape. What the text already has in full is kept as written. Every result is
  memoized on the machine, so a repeat of the same files skips the model. If describing or
  writing fails, the run continues with the text as written and reports a warning.
- `style`: look shared by generated reference images and every segment (optional).
- `overall_soundscape`: sound shared by every segment (optional).
- `non_diegetic_music`: score shared by every segment (default `N/A`).
  Each segment may override it with its own `non_diegetic_music` string. Omitted or
  `null` inherits the shared score; blank or `N/A` requests no score. An explicit
  segment field replaces an inline music section. Without that field, inline music
  takes precedence over the shared score.
- `segments`: the ordered calls.
- `context_frames`: the picture window a continuing segment carries (0, 22, 39 or 56; default 22).
- `turbo` (default `true`) and `steps`: turbo runs the PDD-8 LoRA's eight evaluations; off,
  the full model runs `steps` 30 (default), 40 or 50. Steps given beside turbo are ignored with
  a warning. Each segment may set its own `turbo` and `steps`, so turbo and full-model
  segments mix in one video.

Omit CLI `--gpus` for this workflow: each model then uses its supported GPU count.
BiRefNet uses one GPU; H3 can still use all eight on a supported eight-GPU machine.
An explicit `--gpus=N` applies to every GPU child, so a value greater than one
cannot serve BiRefNet's single-GPU operation.

```yaml
turbo: true
segments:
  - duration_s: 6
    prompt: "<Rover> approaches <Bridge>."
  - duration_s: 6
    context_frames: false
    prompt: "Cut to <Mara> watching from the far side of <Bridge>."
  - duration_s: 6
    prompt: "<Mara> waves as <Rover> lands beside her."
```

Segment 2 is a clean picture cut over continuous sound: music, room tone and the rhythm
of speech carry on, the picture does not.

Unknown fields are ignored, not refused; a missing required field, such as a misspelled
`duration_s`, still refuses.

```json
{
  "style": "Soft watercolor animation with muted pastel colors.",
  "overall_soundscape": "A low hum of laboratory machinery fills the hall.",
  "references": [
    {"name": "Subject-4", "kind": "character", "description": "An adult woman in a white lab coat."},
    {"name": "Background", "kind": "background", "image": "./lab.png"}
  ],
  "segments": [
    {
      "duration_s": 6,
      "prompt": "summary:\n[reference generation] <Subject-4> explores.\n\ndetailed_description:\n[Shot 1] <Subject-4> is walking through the laboratory <Background>, exploring and looking around."
    }
  ]
}
```

## What the workflow does to each segment

Write prompts in MiniMax's [full-reference format](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md):
`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
`overall_soundscape`, `non_diegetic_music`. Any section may be missing. Text before the
first heading counts as the description. Headings match in any case, with `_`, `-` or a
space between words.

1. **References.** Name them as `<Name>` (any case; `_`, `-` and space alike). Each segment
   gets every background, and the characters, audio and videos it names, in the order
   `references` lists them, which is the order H3 reads them. A character's `voice` goes
   wherever the character does, right after it.
2. **Numbering.** `<Picture k>`, `<Audio k>` and `<Video k>` count each kind over that
   attached set; a `use_audio` clip's sound takes an `<Audio k>` just before its
   `<Video k>`, as H3 labels it. A token such as `<Video 2>` names the reference called
   "Video 2" if there is one, else the global second video; either way it is renumbered
   for the call, so a segment naming only `<Video 2>` sends it as its `<Video 1>`. A token
   that matches nothing stays as written, with a warning.
3. **Sections.** The workflow adds `subject_definitions` (one line per attached
   reference), `retention_analysis` (one line per attached reference, with the
   `[Shot N]` markers that name it), `overall_soundscape` and `non_diegetic_music`. Audio
   follows MiniMax's guide (see [Voices and audio](#voices-and-audio)).
   - Existing subject, retention, and soundscape sections gain the global content.
   - If the section is missing, it is added in the order above, under the canonical heading;
     a description without a heading gains `detailed_description:`.
   - A body of `N/A` gives way to described global content in those sections.
   - A `summary` that does not open with `[task types]` gains `[reference generation]`.
   - `style` (the segment's own `style:` section, else the global one) becomes the
     description's opening sentence, where the guide puts it.
   - A segment's own music is kept as written, unless its explicit `non_diegetic_music`
     field replaces the music section.
4. **Everything else** reaches the model byte for byte.

H3's ref2va needs a picture. A segment with no background and no named character
therefore receives every image and audio reference, never a video, with a warning.

## Action videos

A `kind: video` reference guides action: choreography, timing, a fight's beats. Like a
character, it goes only to the calls whose text names it, so one clip can guide one segment
or several. Per call: at most three clips of at least 2 s, square pixels, any resolution or
frame rate; H3 puts each on its own canvas at 24 fps. A clip longer than 15 s is truncated to
its first 15 s, and the result's `warnings` says so. A clip guides at half the canvas edge and
every second latent frame, and only during the dense denoise steps, so a 15 s render with a
clip costs ~1.3x the render without one (5.5x at the full sizing).
A clip's sound is dropped unless the reference sets `use_audio: true`; then it conditions
the audio as `<Audio k>`, "the synchronized audio track of `<Video k>`", copied by default
(on a silent clip, `use_audio` is ignored with a warning). Each clip is bound to the characters its shots
name (all of the call's characters if no shot names it): their lines say their movements
come from it, and it gains `<Video k> (motion, choreography and camera movement):
attribute_transfer - its movements, timing and camera work are performed by <...> in
<background>`; a `<Video k>` line of your own replaces it. With `[video editing]` or
`[video continuation]` in the summary, the clip is the source video instead, kept
`fully_preserved` except where the description changes it.

A `kind: anchor_video` clip is the source the output follows frame by frame, as edit LoRAs
such as Alissonerdx's head swap and style transfer need. It works as ComfyUI's Add Guide does,
not as a reference: every latent frame at the render's own canvas (the render takes the clip's
aspect ratio) sits on the frame it anchors, on every denoise step, and Qwen never sees it, so
the text says what it shows. It is the call's source video whatever the summary says, and it
costs what a second render does (~5.5x for a 12 s clip beside a 15 s render). A segment takes
at most one, beside a picture or another video, no longer than the segment, and no `frame` as
its first frame. It needs no mention in the prompt: a segment that names it follows it from its
first frame, and one no segment names anchors the first segment.

A `kind: frame` image is a keyframe: `at: first` (default) starts segment 1 on it and `at: last`
ends the last segment on it; a segment's `first_frame:` or `last_frame:` names one explicitly,
and then its `at` does not apply. On FL2VA a frame is the official keyframe; on Ref2VA it is
anchored as Add Guide anchors an image, the last frame at the target's last pixel frame. A
segment that continues the previous one starts where that one ends, so it takes no first
frame.

Binding alone does not stop H3 from copying a clip's own people and setting: the
description must establish each character's look, the background and the beats in order
(MiniMax's guide asks 350-500 words). `describe_references` writes it from the clip and the
segment's text; a segment whose clip still guides a shorter description gets a warning, as
does a declared clip no prompt names.

`clean_references` leaves videos alone; `describe_references` describes them (above).

Write the role where you name the subject, as MiniMax's guide does. Name clips anything;
`Video 1`, `Video 2`… read naturally:

```json
{
  "references": [
    {"name": "Subject 1", "kind": "character", "image": "./ann.png"},
    {"name": "Subject 2", "kind": "character", "image": "./ben.png"},
    {"name": "Dojo", "kind": "background", "description": "A wooden dojo at dusk."},
    {"name": "Video 1", "kind": "video", "video": "./kick.mp4"},
    {"name": "Video 2", "kind": "video", "video": "./throw.mp4"}
  ],
  "segments": [
    {"duration_s": 6, "prompt": "subject_definitions:\n<Subject 1>'s motion comes from <Video 1>.\n\ndetailed_description:\n[Shot 1] <Subject 1> attacks <Subject 2> in <Dojo> with the spinning kick of <Video 1>."},
    {"duration_s": 6, "prompt": "subject_definitions:\n<Subject 2>'s motion comes from <Video 2>.\n\ndetailed_description:\n[Shot 1] <Subject 2> blocks and throws <Subject 1> as in <Video 2>."}
  ]
}
```

Segment 2 sends `throw.mp4` as its `<Video 1>`. Attach files by path in the request, or:

```sh
cozy run minimax-h3/generate --input fight.json \
  --asset references.3.video=kick.mp4 --asset references.4.video=throw.mp4
```

A clip (`video` or `anchor_video`) can be a segment of a longer source: `#t=<start>,<end>` on
its path, in whole seconds, `m:ss` or `h:mm:ss`, either end optional (`#t=5`, `#t=,0:12`). The
`cozy` CLI cuts it once, frame-accurately, keeps the cut in `~/.cozy/cache/clips` and sends only
the cut; the 15 s truncation applies to the cut.

```yaml
  - {name: Video 1, kind: video, video: "videos/sentry-program.m4v#t=3:19,3:32"}
```

## Voices and audio

A character's `voice` (2–15 s of clean speech) is a timbre reference: the character speaks
new lines in that voice rather than replaying the clip. It renders as
`<Audio k> is the voice-timbre reference for <Name> (Sx).` and
`<Audio k>: reference - <Name>'s voice follows <Audio k>'s timbre and delivery without
copying the original signal.` The speaker ID is the one the text gives that character
(`<Name> (S1) says ...`); with none in the text, none is added. Write the new line as
`<Name> (S1) says <d>[English] ...</d>`.

A `kind: audio` reference is used as is: `partially_copy`, its signal copied into the
segment with other sounds around it, such as a dialogue line or a piece of music. When the
clip carries a character's words, write those exact words where the character says them,
`<Name> (S1) says <d>[English] the exact words</d>`, so the lips follow them. The
reference's own `retention_analysis` replaces the default, e.g. `fully_copy - ...` for the
whole track, or `reference - ...` to borrow only its style. The summary's `[task types]`
gain `audio reference` or `audio reuse` to match.

## Unwanted speech

H3 has no negative prompt. It fills audio time that no section describes with invented,
often non-English speech. Prevent it in the text:

- Describe the concrete non-voice sounds in `overall_soundscape`: room tone, weather,
  footsteps, cloth, impacts, breathing. "Environmental ambience continues across
  segments." names no sound. In six-second clips with five seeds each, that line gave
  18.5% speech; a concrete description gave 0%.
- Where nobody should speak, write "walks silently, her lips closed". After a line, add
  "then closes her lips". Characters who never speak get no `(S1)` ID.
- Use `N/A` in `overall_soundscape` only for total silence.

## Writing segments that join well

From community chains ([H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context#writing-prompts-for-a-chain)):

- Describe the continuation of what is on screen. H3 renders contradictions as unions:
  if segment N ends on A and segment N+1 describes B and C, you get all three.
- To change setup, open the segment by holding the previous closing framing for about
  2 s with no dialogue (an airlock), then cut.
- Give every hold micro-motion: a breath, a weight shift, an eyeline change. A held
  framing with nothing happening renders as a freeze.
- Timecodes in a continuation land early by the context window. `duration_s` counts new
  time only, but H3 samples the context first: a beat written for 4.0 s lands at 3.08 s
  with context 22 (0.92 s early; 1.63 s at 39, 2.33 s at 56).
- Keep context 22. A 56-frame window samples 2.33 s of old frames in every segment,
  costing compute and new time.
- Draft in Turbo; render long final chains with `turbo: false`. Turbo dulls the audio
  and softens the picture, and each segment inherits the previous one's output, so the
  loss compounds down a chain.
- In very long pieces, restart continuity with `context_frames: false` at a natural
  scene or musical transition.

## References and execution

Each generated character reference uses one Qwen Image 2.1 call and returns exactly one image.
That image is a 16:9, 2 MP character design sheet on one plain white canvas, in three
panels: a full-body front view, a full-body back view, and a neutral front-facing
head-and-shoulders close-up that gives H3 the face in detail. Appearance, proportions
and clothing stay consistent across the panels. The authored character description is
included verbatim apart from surrounding whitespace. Background references stay one 1:1,
1 MP environment image.

The global `style` text also reaches every Qwen character and background reference
request. For example, `style: realistic` guides both reference images and H3 video
segments. The sheet layout does not prescribe an art style: references follow
the explicitly requested style, default to a realistic photographic look when no
style is specified, and use anime styling only when explicitly requested.

A reference has:

- `name` and `kind` (`character`, `background`, `audio`, `video`, `anchor_video` or `frame`);
- its file: `image` (a character, background or frame), `audio` or `video`; a frame takes `at`;
- optional `description`, `retention_analysis` and `seed`; on a character `voice`, on a video
  or anchor video `use_audio`.

A supplied image skips generation. Otherwise Qwen-Image-2.1 generates the image once
from the description. Names normalise: whitespace collapses and `<`/`>` are dropped. Two
names that match the same token are refused. Attach files with a JSON path, or with
`--asset references.0.image=...`, `--asset references.0.voice=...`,
`--asset references.3.audio=...` or `--asset references.4.video=...`. Audio and voices are WAV, FLAC, Ogg or MP3. Audio references are 2–15 s each, 15 s in total, at most
three. Allowed: up to nine images and twelve image and audio references in all; videos count
per call. An audio-only set is refused.

```sh
cozy run minimax-h3/generate --input story.json
```

## Continuation

Each segment after the first may carry two things from its predecessor, chosen separately:

| `context_frames` | `context_audio` | Result |
| --- | --- | --- |
| `true` (default) | `true` (default) | Motion and sound continue |
| `false` | `true` | Clean picture cut; music, room tone and speech rhythm continue |
| `true` | `false` | Motion continues; the sound starts fresh |
| `false` | `false` | A fresh start |

The picture is the previous segment's last `context_frames` frames (global: 22, 39 or 56;
0 carries no picture anywhere, while sound still follows each segment's `context_audio`).
The sound is always its last second. Both are the previous segment's own latents, never a
decoded and re-encoded copy, so colour and sharpness do not drift down a chain.

The carried picture becomes the literal opening of the new clip, so a prompt that opens
with a cut to other characters loses to it: H3 keeps the old characters. Use
`context_frames: false` for a cut; the run warns when a carrying segment's first shot cuts
to characters the previous segment's last shot does not name.

A segment delivers at least its `duration_s`: the native grid snaps it up, and nothing
sampled after the carried frames is discarded. A carried picture of 22/39/56 frames allows
at most 14/13/12 new seconds, and carried sound alone (a one-second head) 14; longer
segments are shortened and reported. Segments with both carries off are independent cuts.
Turbo uses eight PDD evaluations, and the full model 30/40/50 steps.

Outputs arrive as they are made. Prepared and generated reference images are published in
the request's reference order, each once it and every earlier one exist, and the video after
every segment as the joined film so far: one file,
`<run>-video.mp4`, appended in place (a fragmented MP4, one init plus one fragment per
segment) that `cozy run play <run>` follows live. The final MP4 is the last revision, byte
for byte, bounded at 256 MiB. A failed or canceled run fails, but keeps the references and
the film through its last good segment. Loudness is set per segment as it is joined, so a
later, louder segment never changes a fragment already published.
