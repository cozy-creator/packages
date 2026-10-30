# MiniMax H3

Video with sound from MiniMax H3: 5 to 15 seconds at 768p, optionally taken to 1080p, 1440p
or 2160p by SeedVR2 or NVIDIA's Regenerate.

`generate` makes every video: one segment is a single clip, several are joined into one video
([LONG_FORM.md](LONG_FORM.md)). `upscale` takes an existing video to 1080p, 1440p or 2160p with
the same `resolution` and `upscale` fields ([UPSCALE.md](UPSCALE.md)).

```yaml
# cozy run minimax-h3/generate --input story.yaml
turbo: true                   # the PDD-8 turbo LoRA's 8 steps (default); false: the full model
steps: 30                     # turbo false only: 30 (default), 40 or 50
resolution: 1080p             # 768p (default) | 1080p | 1440p | 2160p, the whole video
upscale: {method: seedvr2}    # or {method: regenerate, strength: 0.909}
style: 35 mm film, warm grade
overall_soundscape: gulls, surf, wind
references:                   # optional
  - {name: Mira,  kind: character,  image: ./mira.png}
  - {name: Dock,  kind: background, image: ./dock.png}
  - {name: Rain,  kind: audio,      audio: ./rain.wav}
  - {name: Walk,  kind: video,      video: ./walk.mp4}           # motion and camera reference
  - {name: Src,   kind: anchor_video, video: ./src.mp4}         # the output follows it frame by frame
  - {name: Open,  kind: frame, image: ./open.png, at: first}    # segment 1's first frame
  - {name: Close, kind: frame, image: ./close.png, at: last}    # the last segment's last frame
segments:
  - prompt: <Mira> walks along <Dock> at dawn.
    duration_s: 10
    seed: 7
  - prompt: She stops and looks out to sea.
    duration_s: 5             # continues segment 1 (context_frames, context_audio: true)
    turbo: false              # each segment may set its own turbo and steps
  - prompt: Cut to <Mira> on the harbour wall at noon.
    duration_s: 6
    context_frames: false     # a cut: picture and sound start fresh
    context_audio: false
```

- **Segments.** Each is 5 to 15 seconds and continues the previous one's picture and sound
  unless `context_frames`/`context_audio` turn that off. A segment renders on FL2VA (the
  `transformer`) from its text and keyframes, or on Ref2VA (`transformer_ref`) when it names a
  reference, has a background, or continues the previous segment.
- **References** are used where a prompt names them as `<Name>`; a background always.
  - `frame`: a keyframe. `at: first` is segment 1's first frame and `at: last` the last
    segment's last; a segment's `first_frame:` or `last_frame:` names one explicitly. A segment
    that continues the previous one takes no first frame.
  - `anchor_video`: the output follows this clip frame by frame (ComfyUI's Add Guide), as edit
    LoRAs such as head swap and style transfer need; one per segment, no longer than it.
  - Frames and anchor videos need no `<Name>`.
- **Supplied characters** have their backgrounds removed (BiRefNet) and are described in one
  sentence (Qwen3.5); `clean_references: false` and `describe_references: false` turn either
  off. Write prompts in MiniMax's
  [full-reference format](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
- **Upscale** ([UPSCALE.md](UPSCALE.md)): `resolution` above 768p upscales the whole video; the
  768p original is returned in `originals`. The video is an MP4 in `~/.cozy/outputs`.
- **Models.** `generate` defaults to `minimaxai/minimax-h3@1.0.0-rc.6` and `alibaba-pai/minimax-h3-turbo-lora@1.0.1`;
  `model.base_model=<checkpoint>` serves every segment. LoRAs: one `--lora <org/lora>[,strength]`
  per LoRA (strength 1 by default), in the order they stack, or a `lora:` list in the request
  file. A LoRA trained on the other transformer than a segment runs is applied to it, with a
  warning naming both.

```sh
cozy run minimax-h3/generate --input story.yaml --lora alissonerdx/minimax-h3-live-wallpaper,1.0
```

GPU: one NVIDIA card with 80 GB or more (H100, RTX PRO 6000, H200); the 8-bit lane is 70 GiB,
and `--on new` rents a card that holds it. Measured on one H100 at 30 steps: about 3 minutes
for 5 s of video and 16 minutes for 14 s; turbo runs 8 steps. Most computers rent: add
`--on new`.

`encode_text` and `assemble_video` are the workflow's parts, callable alone.
