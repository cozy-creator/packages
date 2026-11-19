# MiniMax-H3-Image

MiniMax's image model ([MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)), mirrored as
`minimaxai/minimax-h3-image`. **`generate`** (the MiniMax-H3-Image tool) renders in 6 steps without CFG: text to image,
or an edit of the pictures its `instructions` mention, up to 16, each read as `<Picture N>`.

```sh
cozy run minimax-h3-image/generate instructions='A 1970s film frame of a night market in the rain' aspect_ratio=16:9
cozy run minimax-h3-image/generate instructions='Put the hat from <hat> on the woman in <portrait>'   # in a project
```

- **Size.** 4 to 20 MP on a 32 px grid, 16 MP by default, the model's own (1:1 is 4096x4096, 16:9 5472x3072, 21:9
  6272x2688), 1:1 through 21:9 either way. Unset, the ratio is the expander's choice, else Picture 1's, else 1:1. Each
  picture is read at its own size, its short edge held to 1024-2048 px.
- **`prompt_upsampling`** (on): MiniMax's prompt expander, MiniMax-H3-Image-PE, writes the detailed caption the model
  renders best and picks the canvas; an edit runs MiniMax's edit PE prompt on the same checkpoint. One added line of
  system prompt has it write what was asked instead of a sanitized caption; its one exception is nobody under 18 and no
  real person nude or in sexual content. Off, the instructions render as written. The result's `prompt` is what
  rendered.
- **Model.** A 33B DiT (H3's, without audio, one output head per step), MiniMax-H3's text encoder unchanged (the
  `text_encoder` slot: minimax-h3's own default, llmfan46's Heretic abliteration of Qwen3-VL-32B, so a machine with H3
  fetches nothing more for it; MiniMax's is `model.text_encoder=minimaxai/minimax-h3@1.0.0-rc.8`), and a VAE whose encoder is H3's and
  whose decoder is a 2D ViT. A request fetches the expander only when it upsamples and the VAE encoder only when it
  has pictures. A port of Diffusers 0.43's `minimax_h3_image` pipeline.
- **Lanes.** `bf16-full` is MiniMax's Diffusers upload as it is. `bf16` replaces the DiT's time embedder and AdaLN
  projections with exact tables of the rows its 6 steps read (`cozy run ./minimax-h3-image/prepare --model
  source=<DiT> --model vae=<VAE>`).
  `int8` (the default), `w4a8` and `nvfp4` are recipes over `bf16` (`cozy model quantize <bf16 checkpoint>
  --lane int8 --package minimax-h3-image`). The expander's `nvfp4` is MiniMax's upload (`expander_config` drops
  ModelOpt's `quantization_config`); `expander_bf16` decodes it to `bf16`, and its `int8` and `w4a8` are recipes over
  that. `abliterate` projects a refusal direction out of a `bf16` lane's residual writers.
- **Speed** (16.8 MP, int8, warm): RTX PRO 6000 12.4 s (6 x 1.6 s, decode 0.75 s); ComfyUI's int8 15.5 s on the same
  card. RTX 4090 39.7 s (the text encoder and DiT page; 6 x 2.7 s).

License: the MiniMax-H3-Image Community License (its LICENSE and NOTICE are in the mirror). It grants no rights in the
EU, the UK, South Korea or the USA, and needs MiniMax's written authorization above USD 1M annual revenue or for a
hosted service (Section IV).
