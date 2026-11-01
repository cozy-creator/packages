# MiniMax H3

Video with sound from MiniMax H3, as two node functions for Cozy Creator's projects:

- **`shot`** renders one continuous shot of any length (the `video/shot` interface).
- **`upscale`** takes a whole shot to 1080p, 1440p or 2160p ([UPSCALE.md](UPSCALE.md)).

A project composes shots into a film: it renders the pictures, keeps world states, cuts the
edit and calls these per shot. Either runs alone:

```yaml
# shot.yaml — cozy run minimax-h3/shot --input shot.yaml
prompt: <Theo> and <Juno> spar on <Dock>.     # holds for the whole shot
duration_s: 40                                # any length; past 15 s it needs beats
beats:                                        # each holds until the next
  - {at: 0, prompt: They circle each other.}
  - {at: 13, prompt: <Juno> sweeps <Theo>'s legs.}
  - {at: 27, prompt: <Theo> rolls and stands.}
style: 35 mm film, warm grade; no crowd    # holds in every piece; prompt is only the opening
soundscape: gulls, surf, wind
music: N/A
assets:                                       # every one is used, named or not
  - {name: Theo, kind: character, image: ./theo.png}
  - {name: Juno, kind: character, image: ./juno.png, description: a red coat, voice: ./juno.wav}
  - {name: Lantern, kind: prop, image: ./lantern.png}
  - {name: Dock, kind: environment, image: ./dock-b.png}      # the setup plate
audio: [{name: Rain, audio: ./rain.wav}]
reference_footage: [{name: Walk, video: "./walk.mp4#t=0:03,0:09", use_audio: false}]
world_state: [{name: Stage, image: ./stage.png, layout: Theo by the hut, left; Juno by the crates, right.}]
keyframes: {first: ./open.png}                # and/or last
writes: [last, 12.5]                          # frames to return
turbo: true                                   # false: the full model at steps 30, 40 or 50
preset: balanced                              # quality | balanced | fast
```

- **Pieces.** A shot renders in the fewest pieces that reach its length: the first up to
  15 s, each later one up to 14 s new (the 22 frames it carries from the last share H3's 15 s
  window), so 1 + ⌈(d − 15) / 14⌉ for d > 15. Each piece continues the last from its carried
  latents. Edges go between beats: no beat starts from 2 s before an edge to 2.5 s after it,
  because a carried head delays a new action (run 20751: beats on an edge started about 6 s
  late). Where that can't hold, the shot warns. Within that, each piece is as long as it can
  be, since a longer generation is more consistent. A shot of several pieces decodes their
  latents as one video, so no join steps the picture. An 8 s shot with beats at 0 and 5 is
  one generation; the 40 s shot above is pieces of 15, 14 and 11 s.
- **Text.** Each piece is one `[Shot 1]` paragraph. The first: the shot's `prompt`, the beat it
  is in and each later beat ("Then …"), ending "One continuous shot with no cuts.". A later one
  describes where it starts, never the opening again (restating it pulls the shot back there,
  with a cut): the carried frames as they are (everything where it was, the camera still
  moving, nothing new for two seconds), the beat in force ("Already under way, not begun
  again: …" when it began earlier) and each later beat; it sees the previous piece's last frame
  as a picture after the shot's own (a place takes that frame instead). Beats carry no times: H3 marks cuts with timed
  `[Shot N]`. A prompt with its own `integrated_multimodal_description:`/`audio:`/
  `non_diegetic_music:` fields keeps them.
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
  beat where the footage shows it: a beat ahead of the clip runs the shot ahead of it, and the
  next piece goes on from where the shot is (its last frame, and the take restated), not from
  the clip's moment (run 24603 cut there; 24964 did not).
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
