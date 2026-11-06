# MiniMax H3

Video with sound from MiniMax H3, as three tools. Each is a box on a canvas and the same call for `cozy run`:

- **`generate`** (MiniMax H3) renders one continuous shot of any length from an optional `summary` and at least one
  beat, with an optional first and last frame. `turbo: true` (the default) runs the turbo LoRA's 8 steps; `turbo: false`
  or `steps` (30, 40 or 50, under Advanced) the full model. `quality` (draft, standard, full) trades speed for fidelity.
- **`edit`** (MiniMax H3 Edit) edits a plate, a video it follows frame by frame, by its beats, under one of Alissonerdx's
  guide-edit LoRAs: character swap, head swap, style transfer, VFX edit or motion transfer.
- **`upscale`** (MiniMax H3 Upscale) re-renders a take at 1080p, 1440p or 2160p with NVIDIA's Regenerate-2K ([UPSCALE.md](UPSCALE.md)).

```yaml
# shot.yaml — cozy run minimax-h3/generate --input shot.yaml (in a project: each <name> is that box)
summary: <theo> and <juno> fight on <dock>, on 35 mm film, in a warm grade, with no crowd.   # every segment reads it
beats:                                        # what happens, in order, each 1 to 15 s; together, the shot's length
  - {duration: 6, description: The two circle each other.}
  - {duration: 7, description: <juno> feints left and sweeps <theo>'s legs, as in <walk@24-120>.}
  - {duration: 14, description: The camera drops to a low angle. <theo> rolls and stands. <juno> says, in <juno_voice>, <d><English>Again.<d>}
soundscape: gulls, surf, wind, <rain>
first_frame: <open>                           # and/or last_frame
turbo: true
```

```yaml
# edit.yaml — cozy run minimax-h3/edit --input edit.yaml
plate: <take3@48-336>                         # frames 48 to 336 of take3: the edit's timeline
edit: character_swap                          # its trigger (`body_swap:`) opens each beat, at the LoRA's strength
beats: [{duration: 12, description: <kyogre> in place of the man.}]
```

**References.** Each text mentions what it uses, and a segment gets only what its own beats, the summary, the soundscape
and the music mention. A picture is always a character, defined with its description. An audio box `<X_voice>`
mentioned beside `<X>` is X's voice timbre; any other sound is reused in the soundtrack. A video is motion and camera:
it lines up with the start of the first beat that mentions it, second for second, and is cut at that beat's end (in the
summary, with the shot's start). H3 numbers references by first mention in the beats, music, soundscape, then summary.

**First and last frame** are centre-cropped and scaled to the shot: its `aspect_ratio` (16:9) when set, else the
first frame's shape, moved toward 16:9 when the shot has several segments and that shape can't carry across a join.

**Pick up from here** fills `pickup` (`take`, `out_frame` on H3's `17k + 5` join grid, and `beats_from`): the take's
first `out_frame` frames are kept and the beats from `beats_from` seconds render after them; unset, `beats_from` is the
cut's time on the box's own take and 0 on another's. With the take's latents it goes on from them and the video
decodes as one; without them a new video opens on the cut frame and is joined after the kept start. A `last_frame`
ends the final segment; a `first_frame` is ignored. `kept_s` in the result says where the new part starts.

**Edit.** The plate's length is the edit's: the last beat is cut at its end or runs on to it. With no picture in the
first segment, the plate's own first frame is the picture. The plate sets the shape; there is no summary, soundscape
or music, and joins carry 22 frames.

The result is `video` (768p mp4), `latents` (its own, which a pickup resumes and `upscale` starts from) and `warnings`.

- **Beats.** The summary and each beat become MiniMax's sections and `At` times, on one timeline, and a line of a person's that would
  read as ours is kept from it. A beat with no words is only a length. An empty `soundscape` or `music` says `N/A`, which is the
  default, since H3 otherwise invents score and speech. Dialogue goes in a beat, `<d>[English] Hello<d>`.
- **Text as written** goes in [MiniMax's form](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md):
  open with `[Shot 1]`, or get a warning. Added only: each name's `<Picture N>` in `subject_definitions`, the `summary`'s task tag,
  a `retention_analysis` line for each picture without one (`<name>: keep the look shown in the reference.`), `non_diegetic_music`
  from its field, and `overall_soundscape` from its, or `N/A`, which the guide reads as silence. Your own line replaces any of these. A
  `style:` line means nothing to H3: it goes where it stands.
- **Sections.** With a `detailed_description:`, timestamps are read only inside it. `summary:`, `overall_soundscape:` and
  `non_diegetic_music:` go on every segment.
- **Task tag.** With references, every segment's `summary` opens with its `[task + task]` from the guide, computed from the
  request: a plate is `video editing`, a pickup `video continuation`, a first or last frame `keyframe completion`,
  a picture or reference video `reference generation`, an audio reference `audio reuse`, a voice `audio reference`. A different tag you wrote is replaced, with a warning; with no `summary`, one holds just the tag.
  Text alone has none.
- **Timestamps.** `At MM:SS.mmm` or `At MM:SS`, any case, at the start of a line, after an optional `[Shot N]`. A time
  mid-sentence is plain text. Write `[Shot N] At …, the shot cuts to …` for a cut, as the guide does, and a plain `At …`
  for a change inside a take (a second marker made run 26176 jump; run 26187, a plain time, did not). The model reads
  what a timestamp means: cozy never decides it is a cut. A time out of order or past the end stays plain text, with a warning.
- **Segments.** The text from one timestamp to the next is a block, and the text before the first is the opening block. A
  shot renders in the fewest segments of whole blocks: the first up to 15 s, each later one up to 14 s new (340 frames, run
  25217; the 22 frames it carries from the last share H3's 15 s window). Each later segment starts on a block's timestamp,
  carrying the last segment's frames. A segment's text is only its own blocks: the one that opens it loses its time, the
  rest are timed on its clock, whose head is the carried frames, and its `[Shot N]` renumber from 1. So nothing redraws
  the scene at a join (restating an opening pulled runs 24980/24981 back there, with a cut), and each block says only
  what happens then: the carried frames and pictures hold the rest. A later segment keeps the `summary`, as the scene it plays in, so say the scene there and not the sequence, and put camera moves in the text they belong to: in the summary they run again in every segment. It takes each
  reference's own `retention_analysis` line without its `(appears in …)`, or the default line for a picture. A shot of
  several segments decodes their latents as one video, so no join steps the picture.
- **Timing that doesn't fit.** Nothing is refused. A block too long for a segment renders at the limit and warns with its
  time: what follows comes straight after it, so the video is shorter than asked (an edit splits it across segments). A join lands on H3's whole-second grid, within about 0.7 s of its timestamp.
- **Pictures.** A segment gets only the references its own texts mention: a character, sound or clip mentioned in one
  beat goes only to the segments holding that beat, and a segment that names none renders without them, with a warning.
  A plate needs no naming. Each is a definition in the text of the segments that name it, with its description. A
  character's picture should be on a blank background: an opaque one is said, since its background goes into the shot.
  A later segment sees the previous segment's last frame as a world state after the shot's own pictures, read at 512 px:
  it keeps who stands where without opening on that frame.
- **Footage.** A clip gives motion and camera; a plate is followed frame by frame, and takes no first keyframe: the edit
  LoRA decides what changes. Each segment reads its own span of a clip, second for second, its carried head included,
  so a plate stays frame-aligned across an edge. A segment left with under 2 s of a clip skips it; a clip shorter or
  longer than its text warns. Put each timestamp where the footage shows it.
- **Quality.** Turbo runs its eight steps with 1, 4 or 8 dense (draft, standard, full: Sol-Engine `sol_bsa` to NVIDIA's
  HyperFlow quality), the rest sparse. The full model runs NVIDIA's 30-step tiers: full 15 dense steps, the rest sparse
  at tau 0.5, with EasyCache; standard 10 dense at tau 1.0; draft that with FirstBlockCache.
- **Length.** A segment renders whole seconds on H3's 17n+5 frame grid, so the video can run up to a second past its
  beats.
- **Models.** `minimaxai/minimax-h3@1.0.0-rc.7` and `alibaba-pai/minimax-h3-turbo-lora@1.0.1`.
  `model.base_model=<checkpoint>` and `--lora <org/lora>[,strength]` serve every segment; a
  LoRA trained on the other transformer than a segment runs is applied to it, with a warning. Edit's LoRAs are
  Alissonerdx's ([Minimax-H3-ComfyUI](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI)), each at strength 1.0.

GPU: one RTX 4090 works (int8, streaming); an 80 GB card is faster. Measured on one H100 at
30 steps: about 3 minutes for 5 s and 16 minutes for 14 s; turbo runs 8 steps.
