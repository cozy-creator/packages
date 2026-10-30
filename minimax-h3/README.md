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
style: 35 mm film, warm grade
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
```

- **Pieces.** A shot renders in the fewest pieces that reach its length: the first up to
  15 s, each later one up to 14 s (the 22 frames it carries from the last share H3's 15 s
  window), so 1 + ⌈(d − 15) / 14⌉ for d > 15. Each piece continues the last from its carried
  latents, never decoded. Within that count, boundaries go on beat times where they fit
  (within a second; H3's frame grid puts a boundary up to 0.7 s off a whole second). Other
  beats are ordered action inside a piece. An 8 s shot with beats at 0 and 5 is one
  generation; the 40 s shot above is three pieces of 13 s, starting at 0, 13.7 and 27.1 s.
- **Text.** Each piece is one `[Shot 1]` in plain English: the shot's `prompt`, the beat it
  is in (repeated as continuing action when a beat spans a boundary), each later beat as
  "Then, …", and "One continuous shot with no cuts." (first piece) or "The same shot
  continues without a cut…" (the rest). Beats carry no times: H3 marks cuts with timed
  `[Shot N]`. A prompt with its own `integrated_multimodal_description:`/`audio:`/
  `non_diegetic_music:` fields keeps them; the lines end its description.
- **Pictures.** Characters, props and the environment (`<Dock>`) are definitions in every
  piece's text. A world state is a soft picture of the set read at 512 px, with its `layout`
  in its definition: the shot keeps who stands where without opening on that frame.
- **Footage.** `reference_footage` gives motion and camera; a `plate` (`{video}`) is followed
  frame by frame, for a shot of one piece. `keyframes` anchor the first and last frames.
- **Result.** `video` (768p mp4), `frames` (one per `writes` entry) and `warnings`. A piece
  renders whole seconds on H3's 17n+5 frame grid, so the video can run up to a second past
  `duration_s`; the edit trims it.
- **Models.** `minimaxai/minimax-h3@1.0.0-rc.6` and `alibaba-pai/minimax-h3-turbo-lora@1.0.1`.
  `model.base_model=<checkpoint>` and `--lora <org/lora>[,strength]` serve every piece; a
  LoRA trained on the other transformer than a piece runs is applied to it, with a warning.

GPU: one RTX 4090 works (int8, streaming); an 80 GB card is faster. Measured on one H100 at
30 steps: about 3 minutes for 5 s and 16 minutes for 14 s; turbo runs 8 steps.
