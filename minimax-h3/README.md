# MiniMax H3

Video with sound from MiniMax H3, as two functions on the shot contract (`cozy_runtime.author`'s
`ShotRequest`), so a project renders shots with it, or with any other package that takes the same
fields:

- **`generate`** renders one continuous video of any length. `turbo: true` (the default) runs the
  turbo LoRA's 8 steps; `turbo: false` the full model at `steps` 30, 40 or 50 (refused with turbo).
- **`upscale`** takes a whole video to 1080p, 1440p or 2160p ([UPSCALE.md](UPSCALE.md)).

```yaml
# video.yaml — cozy run minimax-h3/generate --input video.yaml
prompt: |                                     # as written, in MiniMax's form; <Name> names a reference
  [Shot 1] <Theo> and <Juno> circle each other on <Dock>.
  At 00:06.000, <Juno> feints left.
  [Shot 2] At 00:13.000, the shot cuts to a low angle as she sweeps <Theo>'s legs.
  At 00:27.000, <Theo> rolls and stands.
duration_s: 40                                # any length; past 15 s it needs timestamps
style: 35 mm film, warm grade; no crowd       # holds in every piece
soundscape: gulls, surf, wind
music: N/A
references:                                   # in this order H3 numbers them; every one is used
  - {name: Theo, kind: character, image: ./theo.png}
  - {name: Juno, kind: character, image: ./juno.png, description: a red coat, voice: ./juno.wav}
  - {name: Lantern, kind: prop, image: ./lantern.png}
  - {name: Dock, kind: environment, image: ./dock-b.png}
  - {name: Stage, kind: environment, image: ./stage.png, follow: loose,
     description: "Theo by the hut, left; Juno by the crates, right."}
  - {name: Walk, kind: video, video: "./walk.mp4#t=0:03,0:09", use_audio: false}
  - {name: Rain, kind: audio, audio: ./rain.wav}
first_frame: ./open.png                       # and/or last_frame
# plate: {video: ./take3.mp4}                 # the video being edited (no first_frame with it)
# continue_from: {video: ./v1.mp4, latents: ./v1.latents}   # resume a video this package made
turbo: true
```

The result is `video` (768p mp4), `latents` (its own, for `upscale` and `continue_from`) and
`warnings`. `follow: loose` keeps a picture's arrangement, not its frame (read at 512 px: a world
state). `continue_from` puts the request on that video's timeline: `duration_s` is the whole length,
text before its end is history, and the result is the whole video, decoded as one. With a plate,
add its edit LoRA (`--lora alissonerdx/minimax-h3-character-swap`, …) and start the prompt with that
LoRA's trigger (`body_swap:`). The latents also carry the text the video was made from, which
`upscale` uses when given no prompt.

- **Text.** The prompt goes as written, in [MiniMax's form](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md):
  open with `[Shot 1]` and write `retention_analysis` lines yourself, or get a warning. Added only: each name's
  `<Picture N>` in `subject_definitions`, the `summary`'s task tag, `style` from its field, `non_diegetic_music: N/A` when none is written, since H3
  otherwise invents score and speech, and `overall_soundscape` from the `soundscape` field, or `N/A` with a warning,
  since the guide reads that as silence. Your own line replaces any of these.
- **Sections.** With a `detailed_description:`, timestamps are read only inside it. `summary:` goes on the first piece
  only; `overall_soundscape:` and `non_diegetic_music:` go on every piece.
- **Task tag.** With references, every piece's `summary` opens with its `[task + task]` from the guide, computed from the
  request: a plate is `video editing`, `continue_from` `video continuation`, a first or last frame `keyframe completion`,
  a picture or reference video `reference generation`, an audio reference or a clip's own sound `audio reuse`, a voice
  `audio reference`. A different tag you wrote is replaced, with a warning; with no `summary`, one holds just the tag.
  A later piece's is just its tag. Text alone has none.
- **Timestamps.** `At MM:SS.mmm` or `At MM:SS`, any case, at the start of a line, after an optional `[Shot N]`. A time
  mid-sentence is plain text. Write `[Shot N] At …, the shot cuts to …` for a cut, as the guide does, and a plain `At …`
  for a change inside a take (a second marker made run 26176 jump; run 26187, a plain time, did not). The model reads
  what a timestamp means: cozy never decides it is a cut. A time out of order or past the end stays plain text, with a warning.
- **Pieces.** The text from one timestamp to the next is a block, and the text before the first is the opening block. A
  shot renders in the fewest pieces of whole blocks: the first up to 15 s, each later one up to 14 s new (340 frames, run
  25217; the 22 frames it carries from the last share H3's 15 s window). Each later piece starts on a block's timestamp,
  carrying the last piece's frames. A piece's text is only its own blocks: the one that opens it loses its time, the
  rest are timed on its clock, whose head is the carried frames, and its `[Shot N]` renumber from 1. So nothing redraws
  the scene at a join (restating an opening pulled runs 24980/24981 back there, with a cut), and each block says only
  what happens then: the carried frames and pictures hold the rest. A later piece drops the `summary`, and takes each
  reference's own `retention_analysis` line without its `(appears in …)`. It sees the previous piece's last frame as a
  picture after the shot's own (a place takes that frame instead). A shot of several pieces decodes their latents as one
  video, so no join steps the picture.
- **Timing that doesn't fit.** Nothing is refused. With no timestamps a video past 15 s renders 15 s and warns. A block too
  long for a piece renders at the limit and warns with its time: what follows comes straight after it, so the video is
  shorter than `duration_s`. A join lands on H3's whole-second grid, within about 0.7 s of its timestamp.
- **Pictures.** Characters, props and the environment (`<Dock>`) are definitions in every
  piece's text, each with its `description` when given. A project cleans character and prop
  pictures (background removal) before they reach `shot`, so a direct call should pass clean
  ones; nothing describes them automatically. A world state is a soft picture of the set
  read at 512 px, with its `layout` in its definition: the shot keeps who stands where without
  opening on that frame.
  A later piece's environment is the previous piece's last frame, with the environment picture
  beside it as a soft world state: the place holds without the picture's framing pulling the
  shot back with a cut (runs 20996, 21955, 21957).
- **Footage.** `reference_footage` gives motion and camera; a `plate` (`{video}`) is followed
  frame by frame, and takes no first keyframe: an edit LoRA decides what changes, and without one
  the shot comes back close to the clip (run 21018). A clip can be as long as the shot: each piece
  reads its own span of it, second for second with the shot, its carried head included, so a
  plate stays frame-aligned across an edge. A shorter clip guides the shot to its end (a piece
  left with under 2 s of it skips it), and a longer one's excess is unused; both warn. Put each
  timestamp where the footage shows it: one ahead of the clip runs the shot ahead of it, and the
  next piece goes on from where the shot is (its last frame), not from the clip's moment (run
  24603 cut there; 24964 did not).
  `keyframes` anchor the first and last frames.
- **Presets.** `preset` trades speed for fidelity. Turbo runs its eight steps with 8, 4 or 1 dense
  (quality, balanced, fast: NVIDIA's HyperFlow quality and Sol-Engine `sol_bsa`), the rest sparse.
  The full model runs NVIDIA's 30-step tiers: quality 15 dense steps, the rest sparse at tau 0.5,
  with EasyCache; balanced 10 dense at tau 1.0; fast that with FirstBlockCache. Above 768p it
  also picks the upscale's.
- **Result.** `video` (768p mp4), `frames` (one per `writes` entry) and `warnings`. A piece
  renders whole seconds on H3's 17n+5 frame grid, so the video can run up to a second past
  `duration_s`; the edit trims it.
- **Models.** `minimaxai/minimax-h3@1.0.0-rc.7` and `alibaba-pai/minimax-h3-turbo-lora@1.0.1`.
  `model.base_model=<checkpoint>` and `--lora <org/lora>[,strength]` serve every piece; a
  LoRA trained on the other transformer than a piece runs is applied to it, with a warning.

GPU: one RTX 4090 works (int8, streaming); an 80 GB card is faster. Measured on one H100 at
30 steps: about 3 minutes for 5 s and 16 minutes for 14 s; turbo runs 8 steps.
