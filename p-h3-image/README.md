# P-H3-Image

Pruna AI's image model ([PrunaAI/P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image)), mirrored as
`prunaai/p-h3-image`. **`generate`** (the P-H3-Image tool) renders in 8 steps without CFG: text to image, or an edit of
the pictures its `instructions` mention. The first mentioned is edited and the rest are references (up to 8), each
read as `<Picture N>`.

```sh
cozy run p-h3-image/generate instructions='A 1970s film frame of a night market in the rain' aspect_ratio=16:9
cozy run p-h3-image/generate instructions='Give the woman in <portrait> the hat from <hat>'   # in a project
```

- **Size.** Text to image defaults to 8 MP at 1:1 (2912x2912). `aspect_ratio` and `megapixels` give the card's
  presets (16:9 is 3872x2176), 1 to 16 MP on a 32 px grid. An edit defaults to Picture 1's size held to 4-8 MP; the
  VAE reads each picture at its own size held to 1-8 MP.
- **`prompt_upsampling`** (on): Pruna's upsampler (Qwen3.5-9B) writes the full prompt, or the edit's brief of what to
  keep. Off, the instructions render as written; an edit's opens with `Edit <Picture 1>.`. The result's `prompt` is
  what rendered.
- **Model.** A 33B DiT, Qwen3-VL-4B's hidden states through Pruna's text connector, and a 24-channel VAE with a
  one-frame ViT decoder. Diffusers 0.41 has no P-H3-Image pipeline; this is a port of its `p_h3_image` modular
  pipeline (main), using the Diffusers classes Pruna's checkpoint names.
- **Lanes.** `bf16-full` is Pruna's upload as it is. `bf16` replaces the DiT's time embedder and AdaLN projections
  with exact tables of the rows its 8 steps read (`cozy run p-h3-image/pruned <bf16-full> prunaai/p-h3-image`).
  `int8`, `fp8` and `nvfp4` are recipes over `bf16` (`cozy model quantize <bf16 checkpoint> --lane int8 --package
  p-h3-image`). A 24 GB card serves `int8`, streaming.

License: the MiniMax H3 Community License, which P-H3-Image inherits (its LICENSE and NOTICE are in the mirror).
